"""Read-only search through a user-controlled SearXNG instance.

SearXNG is the only web-search adapter. The agent never falls back to a public
endpoint, so operators control which engines and internal sources are queried.
The instance must enable JSON responses in ``search.formats``.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping
from urllib.parse import urlsplit

from tini.tools.registry import Tool


def _search_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("SEARXNG_BASE_URL must be an http(s) URL")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("SEARXNG_BASE_URL must not contain credentials, a query, or a fragment")
    return base_url + "/search"


def _searxng(query: str, base_url: str, max_results: int) -> list[dict[str, str]]:
    limit = max(1, min(int(max_results), 20))
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    request = urllib.request.Request(
        f"{_search_url(base_url)}?{params}",
        headers={"Accept": "application/json", "User-Agent": "tini-agent/0.1"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise TypeError("SearXNG returned an invalid JSON response")
    return [
        {
            "title": str(item.get("title") or ""),
            "content": str(item.get("content") or "")[:500],
            "url": str(item.get("url") or ""),
            "engine": str(item.get("engine") or ""),
        }
        for item in results[:limit]
        if isinstance(item, Mapping)
    ]


def probe_searxng(base_url: str) -> None:
    """Verify connectivity and JSON search support without writing state."""
    _searxng("health check", base_url, 1)


def make_tool() -> Tool:
    def search_web(query: str, max_results: int = 5) -> str:
        base_url = os.getenv("SEARXNG_BASE_URL", "").strip()
        if not base_url:
            return (
                "Search is not configured. Set SEARXNG_BASE_URL to your self-hosted "
                "SearXNG instance and enable JSON in search.formats."
            )
        try:
            results = _searxng(query, base_url, max_results)
        except Exception as exc:
            hint = " Enable JSON in SearXNG search.formats." if "403" in str(exc) else ""
            return f"SearXNG search failed ({exc}).{hint}"
        if not results:
            return "No results found in the configured SearXNG sources."
        lines = [f"Search results for '{query}' (via self-hosted SearXNG):"]
        for index, result in enumerate(results, 1):
            engine = f" [{result['engine']}]" if result["engine"] else ""
            lines.append(
                f"{index}. {result['title']}{engine}\n"
                f"   {result['content']}\n"
                f"   {result['url']}"
            )
        return "\n".join(lines)

    return Tool(
        name="search_web",
        description=(
            "Search the sources configured in the operator's self-hosted SearXNG instance. "
            "Returns titles, snippets, source engines, and URLs. Use it for external or "
            "federated lookup; use the RAG retriever for indexed document chunks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "number of results (default 5)",
                },
            },
            "required": ["query"],
        },
        fn=search_web,
    )
