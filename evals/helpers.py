"""Shared eval plumbing: a scripted fake LLM client for offline tests, and a
real-Tini factory for live ones."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace


def _has_key() -> bool:
    """True when the ACTIVE provider (TINI_PROVIDER) has its key set, so live
    evals run on whatever the user actually configured (anthropic, openrouter,
    gemini, ...), not only on ANTHROPIC_API_KEY."""
    from tini.config import load_settings
    from tini.loop.models import PROVIDERS

    settings = load_settings()
    provider = PROVIDERS.get(settings.provider)
    return bool(settings.api_key or (provider and os.getenv(provider.key_env)))


HAS_KEY = _has_key()


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, args: dict, call_id: str = "tu_1"):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)


def response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        content=blocks,
    )


class ScriptedClient:
    """Plays back a fixed list of responses — the 'model' for offline tests."""

    def __init__(self, script: list):
        self._script = list(script)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        return self._script.pop(0)


def make_tini(home: Path, client=None, **settings_overrides):
    """Build a Tini with an isolated home dir; optionally swap in a fake client."""
    from tini.app import Tini
    from tini.config import Settings

    # Never inherit an orchestration toggle from a maintainer's .env. A stale
    # value would consume an extra scripted model response and make the test
    # pass locally but fail in CI (or vice versa).
    settings_overrides.setdefault("graph_workflows", False)
    # Same reasoning for the document index: a maintainer's .env may point
    # TINI_DOCUMENT_INDEX at a real Qdrant server (self-hosted RAG setup, see
    # docs/document-ingestion.md) and TINI_EMBEDDING_PROVIDER at a real
    # embedder. Without this, every test built through make_tini() silently
    # writes into that real server and calls that real embedder instead of
    # the isolated sqlite/no-vector index its assertions assume — found when
    # ingestion tests left "default"/"engineering" workspace junk in a real
    # local Qdrant collection.
    settings_overrides.setdefault("document_index", "sqlite")
    settings_overrides.setdefault("embedding_provider", "none")
    # ...and the workspace: a maintainer's .env may name the workspace their real
    # corpus lives in (TINI_DOCUMENT_WORKSPACE=denso), while these tests ingest
    # into "default" — the search tool would then look in the wrong workspace.
    settings_overrides.setdefault("document_workspace", "default")
    # ...and reranking: a maintainer's .env may have TINI_RERANK_PROVIDER=jev
    # set (trying it out, see tini/documents/rerank.py) without a working key
    # yet, which makes make_document_index() raise for every test built
    # through this helper — found the hard way when it broke over 50
    # unrelated tests the moment a real .env picked up that setting.
    settings_overrides.setdefault("rerank_provider", "none")
    settings = Settings(home=home, **settings_overrides)
    if client is not None and not settings.api_key:
        settings.api_key = "offline"  # never read the real key for scripted runs
    return Tini(settings=settings, client=client)


def pdf_with_pages(texts: list[str]) -> bytes:
    """A hand-built PDF with one real text page per string (see test_document_ocr)."""
    count = len(texts)
    font_num = 3 + count * 2
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(count))
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        font_num: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(texts):
        page_num, content_num = 3 + i * 2, 4 + i * 2
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/MediaBox [0 0 600 200] /Contents {content_num} 0 R >>"
        ).encode()
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects[content_num] = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    total = max(objects) + 1
    offsets = [0] * total
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {total}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, total):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(total).encode() + b" /Root 1 0 R >>\nstartxref\n" + str(xref).encode() + b"\n%%EOF"
    return bytes(out)
