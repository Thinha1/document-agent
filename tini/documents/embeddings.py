"""Embedding provider seam and remote API adapters used by document RAG."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from tini.config import Settings

GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class EmbeddingError(RuntimeError):
    """An embedding provider rejected a request or returned invalid vectors."""


class EmbeddingProvider(Protocol):
    provider: str
    model: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str], *, title: str) -> list[tuple[float, ...]]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class GeminiEmbeddingProvider:
    provider = "gemini"

    def __init__(self, api_key: str, model: str, dimensions: int, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = (base_url or GEMINI_API_ROOT).rstrip("/")

    def embed_documents(self, texts: Sequence[str], *, title: str) -> list[tuple[float, ...]]:
        prepared = [f"title: {title} | text: {text}" for text in texts]
        return self._embed_many(prepared, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> tuple[float, ...]:
        vectors = self._embed_many(
            [f"task: search result | query: {text}"], task_type="RETRIEVAL_QUERY"
        )
        return vectors[0]

    def _embed_many(self, texts: Sequence[str], task_type: str) -> list[tuple[float, ...]]:
        output: list[tuple[float, ...]] = []
        for offset in range(0, len(texts), 64):
            batch = texts[offset : offset + 64]
            requests = []
            for text in batch:
                request: dict[str, object] = {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self.dimensions,
                }
                if self.model != "gemini-embedding-2":
                    request["taskType"] = task_type
                requests.append(request)
            payload = self._post(
                f"models/{quote(self.model)}:batchEmbedContents", {"requests": requests}
            )
            try:
                output.extend(tuple(item["values"]) for item in payload["embeddings"])
            except (KeyError, TypeError) as exc:
                raise EmbeddingError("Gemini returned an invalid embedding response") from exc
        return _validate_vectors(output, len(texts), self.dimensions)

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}/{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise EmbeddingError(f"Gemini embedding request failed ({exc.code}): {detail}") from exc
        except (OSError, ValueError) as exc:
            raise EmbeddingError(f"Gemini embedding request failed: {exc}") from exc


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        provider: str = "openai_compatible",
    ):
        from openai import OpenAI

        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: Sequence[str], *, title: str) -> list[tuple[float, ...]]:
        return self._embed_many(list(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed_many([text])[0]

    def _embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        try:
            response = self.client.embeddings.create(
                model=self.model, input=texts, dimensions=self.dimensions
            )
        except Exception as exc:
            raise EmbeddingError(f"OpenAI-compatible embedding request failed: {exc}") from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [tuple(item.embedding) for item in ordered]
        return _validate_vectors(vectors, len(texts), self.dimensions)


class CachingEmbedder:
    """Wraps any provider so a chunk is embedded once per (provider, model,
    dimensions, title, text).

    Ingestion embeds BEFORE the index decides whether a file changed, so without
    this an unchanged re-ingest re-embedded every chunk (338 chunks, 23 s on a
    local GPU, real money on a hosted API) and a one-paragraph edit re-embedded
    the whole document. Only document chunks are cached; queries pass through.
    The cache is a local SQLite file of vectors keyed by a hash — the chunk text
    itself is not stored in it."""

    def __init__(self, inner: EmbeddingProvider, path: Path):
        self.inner = inner
        self.path = Path(path)
        self.provider = inner.provider
        self.model = inner.model
        self.dimensions = inner.dimensions
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _key(self, title: str, text: str) -> str:
        raw = f"{self.provider}|{self.model}|{self.dimensions}|{title}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        return conn

    def embed_documents(self, texts: Sequence[str], *, title: str) -> list[tuple[float, ...]]:
        keys = [self._key(title, text) for text in texts]
        found: dict[str, tuple[float, ...]] = {}
        with self._lock:
            conn = self._connect()
            try:
                for start in range(0, len(keys), 500):
                    batch = list(set(keys[start : start + 500]))
                    marks = ",".join("?" * len(batch))
                    for key, blob in conn.execute(
                        f"SELECT key, vector FROM vectors WHERE key IN ({marks})", batch
                    ):
                        vector = struct.unpack(f"<{len(blob) // 4}f", blob)
                        if len(vector) == self.dimensions:
                            found[key] = vector
                missing = [i for i, key in enumerate(keys) if key not in found]
                if missing:
                    fresh = self.inner.embed_documents([texts[i] for i in missing], title=title)
                    rows = []
                    for index, vector in zip(missing, fresh, strict=True):
                        found[keys[index]] = tuple(vector)
                        rows.append((keys[index], struct.pack(f"<{len(vector)}f", *vector)))
                    with conn:
                        conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows)
                self.hits += len(keys) - len(missing)
                self.misses += len(missing)
            finally:
                conn.close()
        return [found[key] for key in keys]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.inner.embed_query(text)


def make_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    provider = _make_embedding_provider(settings)
    if provider is not None and settings.embedding_cache:
        return CachingEmbedder(provider, settings.home / "embed_cache.sqlite")
    return provider


def _make_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"", "none"}:
        return None
    if not settings.embedding_model:
        raise ValueError("TINI_EMBEDDING_MODEL is required when embeddings are enabled")
    if not settings.embedding_api_key:
        raise ValueError("TINI_EMBEDDING_API_KEY or the provider API key is required")
    if settings.embedding_dimensions <= 0:
        raise ValueError("TINI_EMBEDDING_DIMENSIONS must be positive")
    if provider == "gemini":
        return GeminiEmbeddingProvider(
            settings.embedding_api_key,
            settings.embedding_model,
            settings.embedding_dimensions,
            settings.embedding_base_url,
        )
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleEmbeddingProvider(
            settings.embedding_api_key,
            settings.embedding_model,
            settings.embedding_dimensions,
            settings.embedding_base_url,
            provider,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _validate_vectors(
    vectors: Sequence[Sequence[float]], expected_count: int, dimensions: int
) -> list[tuple[float, ...]]:
    if len(vectors) != expected_count:
        raise EmbeddingError(f"Expected {expected_count} vectors, received {len(vectors)}")
    clean: list[tuple[float, ...]] = []
    for vector in vectors:
        values = tuple(float(value) for value in vector)
        if len(values) != dimensions:
            raise EmbeddingError(f"Expected {dimensions} dimensions, received {len(values)}")
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingError("Embedding contains a non-finite value")
        clean.append(values)
    return clean
