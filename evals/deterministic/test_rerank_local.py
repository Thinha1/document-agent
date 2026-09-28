"""The self-hosted reranker (provider `tei`): request/response handling and,
above all, the rule that it can only ever talk to a local address.

A reranker receives the question and the candidate chunks' TEXT on every search.
The hosted provider (Jev) has its endpoint fixed in code for that reason; this
one has a URL setting, so the same guarantee has to come from validating the
address. Network-free: urlopen is faked, same style as test_rerank.py.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from tini.config import Settings
from tini.documents import rerank as rerank_module
from tini.documents.models import SearchHit
from tini.documents.rerank import (
    TEI_MAX_TEXTS_PER_CALL,
    RerankError,
    RerankingIndex,
    TeiReranker,
    local_rerank_url,
    make_reranker,
)


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _server(monkeypatch, scores_for=lambda texts: [0.0] * len(texts), *, shuffle=False):
    """Fake TEI: records every request; answers like the real one (a list of
    {index, score}, sorted best-first — NOT in input order)."""
    seen = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data)
        seen.append((request, body))
        scores = scores_for(body["texts"])
        entries = [{"index": i, "score": s} for i, s in enumerate(scores)]
        entries.sort(key=lambda e: -e["score"])
        return _Response(entries)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


# --- the address policy ---------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8081",
        "http://localhost:8081",
        "http://LOCALHOST:8081/",
        "http://[::1]:8081",
        "http://reranker:80",
        "https://reranker",
    ],
)
def test_local_addresses_are_accepted(url):
    assert local_rerank_url(url).startswith(("http://", "https://"))


@pytest.mark.parametrize(
    "url",
    [
        "http://api.typesafe.ai",
        "http://192.168.1.20:8081",  # another machine on the LAN
        "http://10.0.0.5",
        "http://0.0.0.0:8081",
        "http://localhost.evil.example",  # looks local, is not
        "http://localhost@evil.example",  # userinfo trick: the host is evil.example
        "http://evil.example#@localhost",
        "http://127.0.0.1.evil.example",
        "http://evil.example/127.0.0.1",
        "http://user:pass@127.0.0.1:8081",  # credentials are refused too
        "ftp://127.0.0.1",
        "127.0.0.1:8081",  # no scheme: not parseable as a host
        "",
        "http://",
        "http://[::1",  # malformed
        "http://127.0.0.1:99999999",  # malformed port
        "http://127.0.0.1:8081/rerank",  # a bare address only
        "http://127.0.0.1:8081/?next=http://evil.example",
    ],
)
def test_anything_else_is_refused(url):
    with pytest.raises(ValueError):
        local_rerank_url(url)


def test_a_remote_url_cannot_be_smuggled_in_through_settings():
    settings = Settings(rerank_provider="tei", rerank_url="http://api.evil.example")
    with pytest.raises(ValueError, match="not local"):
        make_reranker(settings)


def test_the_default_url_is_loopback_and_needs_no_key(monkeypatch):
    monkeypatch.delenv("TINI_RERANK_URL", raising=False)
    monkeypatch.delenv("TINI_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    reranker = make_reranker(Settings(rerank_provider="tei"))
    assert isinstance(reranker, TeiReranker)
    assert reranker.url == "http://127.0.0.1:8081"


# --- request / response ----------------------------------------------------------


def test_request_matches_the_tei_rerank_shape(monkeypatch):
    seen = _server(monkeypatch, lambda texts: [0.1, 0.9])
    TeiReranker("http://127.0.0.1:8081").rerank("when founded?", ["alpha", "beta"])

    request, body = seen[0]
    assert request.full_url == "http://127.0.0.1:8081/rerank"
    assert request.get_method() == "POST"
    assert body == {"query": "when founded?", "texts": ["alpha", "beta"], "truncate": True}
    assert request.get_header("Authorization") is None, "a local server needs no credentials"


def test_scores_come_back_in_input_order_though_the_server_sorts_best_first(monkeypatch):
    _server(monkeypatch, lambda texts: [0.2, 0.9, 0.5])
    assert TeiReranker("http://127.0.0.1:8081").rerank("q", ["a", "b", "c"]) == [0.2, 0.9, 0.5]


def test_no_passages_makes_no_request(monkeypatch):
    seen = _server(monkeypatch)
    assert TeiReranker("http://127.0.0.1:8081").rerank("q", []) == []
    assert seen == []


def test_more_candidates_than_one_call_allows_are_split_and_reassembled(monkeypatch):
    seen = _server(monkeypatch, lambda texts: [float(t) for t in texts])
    texts = [str(n) for n in range(TEI_MAX_TEXTS_PER_CALL + 10)]  # 42

    scores = TeiReranker("http://127.0.0.1:8081").rerank("q", texts)

    assert [len(body["texts"]) for _, body in seen] == [TEI_MAX_TEXTS_PER_CALL, 10]
    assert scores == [float(n) for n in range(42)], "batching must not reorder or lose a score"


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "nope"},  # not a list
        [{"index": 0, "score": 0.5}],  # scored 1 of 2
        [{"index": 0, "score": 0.5}, {"index": 5, "score": 0.1}],  # index out of range
        [{"index": 0, "score": 0.5}, {"index": 1}],  # no score
        [{"index": 0, "score": 0.5}, {"index": "1", "score": 0.1}],  # wrong type
        [{"index": 0, "score": 0.5}, {"index": 0, "score": 0.9}],  # scored one twice, other never
    ],
)
def test_a_reply_that_cannot_be_matched_to_the_candidates_is_an_error(monkeypatch, payload):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _Response(payload))
    with pytest.raises(RerankError):
        TeiReranker("http://127.0.0.1:8081").rerank("q", ["a", "b"])


def test_a_server_that_is_down_is_a_rerank_error_naming_the_address(monkeypatch):
    def refuse(request, timeout=None):
        raise ConnectionRefusedError("actively refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(RerankError, match="127.0.0.1:8081"):
        TeiReranker("http://127.0.0.1:8081").rerank("q", ["a"])


def test_overloaded_or_warming_up_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(rerank_module, "_sleep", lambda s: None)
    calls = []

    def flaky(request, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError(request.full_url, 503, "warming", {}, io.BytesIO(b"loading"))
        return _Response([{"index": 0, "score": 0.7}])

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    assert TeiReranker("http://127.0.0.1:8081").rerank("q", ["a"]) == [0.7]
    assert len(calls) == 3


def test_a_client_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(rerank_module, "_sleep", lambda s: None)
    calls = []

    def bad(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 413, "big", {}, io.BytesIO(b"too many texts"))

    monkeypatch.setattr("urllib.request.urlopen", bad)
    with pytest.raises(RerankError, match="413"):
        TeiReranker("http://127.0.0.1:8081").rerank("q", ["a"])
    assert len(calls) == 1


# --- inside RerankingIndex -------------------------------------------------------


class _Pool:
    def __init__(self, texts):
        self.hits = [
            SearchHit(
                document_id="d", chunk_id=f"c{i}", title="t", source_uri="file:///t",
                text=text, ordinal=i, score=1.0 - i / 100,
            )
            for i, text in enumerate(texts)
        ]
        self.requested = None

    def search(self, query, top_k=6, workspace_id="default"):
        self.requested = top_k
        return self.hits[:top_k]


def test_the_local_reranker_reorders_a_wider_pool_and_keeps_the_top_k(monkeypatch):
    _server(monkeypatch, lambda texts: [0.9 if "answer" in t else 0.1 for t in texts])
    pool = _Pool(["noise one", "noise two", "noise three", "the answer is here"])
    index = RerankingIndex(pool, TeiReranker("http://127.0.0.1:8081"), candidates=4)

    hits = index.search("q", top_k=2)

    assert pool.requested == 4, "the wider pool is fetched, then cut to top_k"
    assert hits[0].text == "the answer is here"
    assert len(hits) == 2 and index.errors == 0


def test_a_dead_reranker_falls_back_to_vector_order_and_is_counted(monkeypatch):
    def refuse(request, timeout=None):
        raise ConnectionRefusedError("down")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    pool = _Pool(["first", "second", "third"])
    index = RerankingIndex(pool, TeiReranker("http://127.0.0.1:8081"), candidates=3)

    hits = index.search("q", top_k=2)

    assert [h.text for h in hits] == ["first", "second"]
    assert index.errors == 1 and "not reachable" in index.last_error
