"""Embedding configuration stays independent from chat generation."""

from __future__ import annotations

import json
from types import SimpleNamespace

from tini.config import Settings
from tini.documents.embeddings import (
    GeminiEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    make_embedding_provider,
)
from tini.ops.settings_api import settings_info


def _clear_embedding_env(monkeypatch):
    for name in (
        "TINI_EMBEDDING_PROVIDER",
        "TINI_EMBEDDING_MODEL",
        "TINI_EMBEDDING_API_KEY",
        "TINI_EMBEDDING_BASE_URL",
        "TINI_EMBEDDING_DIMENSIONS",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_embedding_model_is_independent_from_chat_model(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("TINI_PROVIDER", "custom_openai")
    monkeypatch.setenv("TINI_MODEL", "softdream")
    monkeypatch.setenv("TINI_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("TINI_EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("TINI_EMBEDDING_API_KEY", "dedicated-embedding-key")
    monkeypatch.setenv("TINI_EMBEDDING_DIMENSIONS", "768")

    settings = Settings()

    assert settings.provider == "custom_openai"
    assert settings.model == "softdream"
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-2"
    assert settings.embedding_dimensions == 768
    assert settings.embedding_api_key == "dedicated-embedding-key"


def test_gemini_embedding_can_reuse_gemini_key(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("TINI_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "shared-gemini-key")

    assert Settings().embedding_api_key == "shared-gemini-key"


def test_dedicated_embedding_key_wins_over_provider_key(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("TINI_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "shared-key")
    monkeypatch.setenv("TINI_EMBEDDING_API_KEY", "dedicated-key")

    assert Settings().embedding_api_key == "dedicated-key"


def test_dashboard_exposes_embedding_status_without_exposing_key(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("TINI_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("TINI_EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("TINI_EMBEDDING_API_KEY", "never-return-this-key")

    payload = settings_info()["embedding"]

    assert payload == {
        "provider": "gemini",
        "model": "gemini-embedding-2",
        "base_url": "",
        "dimensions": 768,
        "api_key_set": True,
    }
    assert "never-return-this-key" not in repr(payload)


class _GeminiResponse:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps({"embeddings": [{"values": [0.5] * self.dimensions}]}).encode()


def test_gemini_embedding_2_uses_native_batch_api_and_retrieval_prefixes(monkeypatch):
    captured = []

    def fake_urlopen(request, timeout):
        captured.append((request, json.loads(request.data), timeout))
        return _GeminiResponse(3)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = GeminiEmbeddingProvider("secret", "gemini-embedding-2", 3)

    document = provider.embed_documents(["Annual compensation review."], title="Benefits")
    query = provider.embed_query("salary policy")

    assert document == [(0.5, 0.5, 0.5)]
    assert query == (0.5, 0.5, 0.5)
    assert captured[0][0].full_url.endswith("models/gemini-embedding-2:batchEmbedContents")
    assert captured[0][1]["requests"][0]["content"]["parts"][0]["text"].startswith(
        "title: Benefits | text:"
    )
    assert captured[1][1]["requests"][0]["content"]["parts"][0]["text"].startswith(
        "task: search result | query:"
    )
    assert "taskType" not in captured[0][1]["requests"][0]


def test_embedding_provider_factory_rejects_partial_configuration():
    settings = Settings(
        embedding_provider="gemini",
        embedding_model="",
        embedding_api_key="key",
    )

    try:
        make_embedding_provider(settings)
    except ValueError as exc:
        assert "TINI_EMBEDDING_MODEL" in str(exc)
    else:
        raise AssertionError("partial embedding configuration should fail")


def test_openai_compatible_embedding_uses_separate_base_url(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3])]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    provider = OpenAICompatibleEmbeddingProvider(
        "embedding-key", "local-embed", 3, "http://localhost:4000/v1"
    )

    assert provider.embed_query("deployment policy") == (0.1, 0.2, 0.3)
    assert captured["api_key"] == "embedding-key"
    assert captured["base_url"] == "http://localhost:4000/v1"
    assert captured["model"] == "local-embed"
    assert captured["dimensions"] == 3
