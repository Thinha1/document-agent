"""SearXNG search and the retrieval-only tool surface, fully offline."""

from __future__ import annotations

import io
import json
import urllib.request

import pytest

from tini.config import Settings
from tini.runtime.session import RAG_SOUL, load_soul
from tini.tools import build_registry, search


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_searxng_uses_json_search_endpoint(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured.update(url=request.full_url, headers=dict(request.header_items()), timeout=timeout)
        payload = {"results": [{
            "title": "Internal runbook",
            "content": "Restart the worker safely.",
            "url": "https://docs.internal/runbook",
            "engine": "internal-wiki",
        }]}
        return Response(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    results = search._searxng("worker restart", "http://searxng:8080", 5)

    assert "http://searxng:8080/search?" in captured["url"]
    assert "q=worker+restart" in captured["url"]
    assert "format=json" in captured["url"]
    assert captured["timeout"] == 20
    assert results[0]["engine"] == "internal-wiki"


@pytest.mark.parametrize(
    "url",
    ["", "searxng:8080", "ftp://searxng", "http://user:pass@searxng:8080"],
)
def test_searxng_rejects_invalid_base_urls(url):
    with pytest.raises(ValueError):
        search._search_url(url)


def test_search_tool_requires_self_hosted_endpoint(monkeypatch):
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    output = search.make_tool().fn("anything")

    assert "SEARXNG_BASE_URL" in output
    assert "Tavily" not in output
    assert "DuckDuckGo" not in output


def test_registry_without_a_document_index_exposes_only_web_search(tmp_path):
    settings = Settings(home=tmp_path)
    settings.ensure_home()

    registry = build_registry(settings)

    assert set(registry._tools) == {"search_web"}
    assert "read-only" in RAG_SOUL


def test_rag_soul_is_created_without_overwriting_custom_persona(tmp_path):
    settings = Settings(home=tmp_path)
    settings.ensure_home()
    soul = tmp_path / "SOUL.md"
    assert load_soul(settings) == RAG_SOUL

    soul.write_text("my custom persona", encoding="utf-8")
    assert load_soul(settings) == "my custom persona"
