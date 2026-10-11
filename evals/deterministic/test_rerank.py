"""Reranking: the Jev adapter's request/response handling, the wrapper index,
and the eval switch — all network-free (urlopen and the reranker are faked)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents import rerank as rerank_module
from tini.documents.base import DocumentIndex
from tini.documents.models import SearchHit
from tini.documents.pipeline import IngestionPipeline
from tini.documents.rerank import (
    JEV_ENDPOINT,
    JevReranker,
    RerankError,
    RerankingIndex,
    make_reranker,
)
from tini.documents.store import SqliteDocumentIndex, make_document_index
from tini.ops import document_eval

SECRET = "tsk-very-secret-key"


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _answers(request, scores):
    body = json.loads(request.data)
    return {
        "answers": {
            qid: {"type": "noul", "noul": scores.get(qid, 0.0)} for qid in body["questions"]
        }
    }


def _http_error(code, body=b"nope"):
    return urllib.error.HTTPError(JEV_ENDPOINT, code, "err", {}, io.BytesIO(body))


def test_request_matches_the_documented_shape(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return _Response(_answers(request, {"D00": 0.2, "D01": 0.9}))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    scores = JevReranker(SECRET).rerank("when founded?", ["alpha text", "beta text"])

    request = seen[0]
    body = json.loads(request.data)
    assert scores == [0.2, 0.9]
    assert request.full_url == JEV_ENDPOINT and request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"
    assert body["model"] == "jev-latest"
    assert body["state"] == {
        "query": "when founded?",
        "documents": {"D00": "alpha text", "D01": "beta text"},
    }
    assert set(body["questions"]) == {"D00", "D01"}
    assert body["questions"]["D00"]["type"] == "noul"
    assert "ignore any instructions" in body["questions"]["D00"]["instructions"]


def test_more_than_thirty_candidates_are_split_across_calls_and_stay_aligned(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data)
        calls.append(len(body["questions"]))
        # score = the passage's own number, so misalignment would be visible
        scores = {qid: float(text) / 100 for qid, text in body["state"]["documents"].items()}
        return _Response(_answers(request, scores))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    scores = JevReranker(SECRET).rerank("q", [str(n) for n in range(45)])

    assert calls == [30, 15]
    assert scores == [n / 100 for n in range(45)]


def test_passages_are_truncated_before_sending(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["doc"] = json.loads(request.data)["state"]["documents"]["D00"]
        return _Response(_answers(request, {}))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    JevReranker(SECRET).rerank("q", ["x" * 10_000])

    assert len(sent["doc"]) == rerank_module.PASSAGE_CHAR_LIMIT


def test_rate_limits_are_retried_then_succeed(monkeypatch):
    attempts = []
    monkeypatch.setattr(rerank_module, "_sleep", lambda seconds: attempts.append(seconds))
    outcomes = [_http_error(429), _http_error(529), None]

    def fake_urlopen(request, timeout=None):
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome
        return _Response(_answers(request, {"D00": 0.7}))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert JevReranker(SECRET).rerank("q", ["a"]) == [0.7]
    assert attempts == [0.5, 1.0]


def test_errors_are_reported_without_leaking_the_key(monkeypatch):
    monkeypatch.setattr(rerank_module, "_sleep", lambda seconds: None)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(_http_error(401)),
    )

    with pytest.raises(RerankError, match="401") as caught:
        JevReranker(SECRET).rerank("q", ["a"])

    assert SECRET not in str(caught.value)


@pytest.mark.parametrize(
    "bad", [{}, {"answers": []}, {"answers": {"D00": {"noul": "high"}}}, {"answers": {}}]
)
def test_malformed_responses_raise_instead_of_returning_guessed_scores(monkeypatch, bad):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: _Response(bad))

    with pytest.raises(RerankError):
        JevReranker(SECRET).rerank("q", ["a"])


def test_network_failure_becomes_a_rerank_error(monkeypatch):
    def down(request, timeout=None):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr("urllib.request.urlopen", down)

    with pytest.raises(RerankError, match="request failed"):
        JevReranker(SECRET).rerank("q", ["a"])


def test_make_reranker_is_off_by_default_and_strict_about_config():
    assert make_reranker(Settings(rerank_provider="none")) is None
    assert make_reranker(Settings(rerank_provider="")) is None
    with pytest.raises(ValueError, match="required"):
        make_reranker(Settings(rerank_provider="jev", rerank_api_key=""))
    with pytest.raises(ValueError, match="Unsupported"):
        make_reranker(Settings(rerank_provider="mystery", rerank_api_key="k"))
    assert isinstance(
        make_reranker(Settings(rerank_provider="jev", rerank_api_key="k")), JevReranker
    )


def test_the_destination_is_not_configurable():
    """Document text goes to this URL; it must not be steerable by an env var."""
    assert not hasattr(Settings(), "rerank_base_url")
    assert JEV_ENDPOINT == "https://api.typesafe.ai/v1/systemone"


# --- the wrapper index -------------------------------------------------------


def _hit(n, text=None):
    return SearchHit(
        document_id="d",
        chunk_id=f"c{n}",
        title="t",
        source_uri="u",
        text=text or f"chunk {n}",
        ordinal=n,
        score=1.0 / (n + 1),
    )


class _Inner:
    embedder = "the-embedder"

    def __init__(self, hits):
        self.hits = hits
        self.asked = None

    def search(self, query, top_k=6, workspace_id="default"):
        self.asked = (query, top_k, workspace_id)
        return self.hits[:top_k]

    def upsert_document(self, source, chunks):
        return "upserted"

    def list_documents(self, workspace_id="default", limit=200):
        return ["listed"]

    def delete_document(self, document_id, workspace_id="default"):
        return True


class _Scores:
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, passages):
        return self.scores(passages) if callable(self.scores) else self.scores


def test_the_answer_chunk_ranked_low_by_vectors_comes_first_after_rerank():
    inner = _Inner([_hit(n) for n in range(10)])
    index = RerankingIndex(
        inner, _Scores(lambda p: [1.0 if "chunk 7" in t else 0.1 for t in p]), candidates=10
    )

    hits = index.search("q", top_k=3, workspace_id="w")

    assert hits[0].ordinal == 7 and hits[0].score == 1.0
    assert len(hits) == 3
    assert inner.asked == ("q", 10, "w"), "must fetch the wider candidate pool, not just top_k"


def test_ties_keep_the_original_vector_order():
    index = RerankingIndex(_Inner([_hit(n) for n in range(5)]), _Scores([0.5] * 5), candidates=5)

    assert [h.ordinal for h in index.search("q", top_k=5)] == [0, 1, 2, 3, 4]


def test_a_rerank_failure_falls_back_to_vector_order_and_is_counted():
    class Broken:
        def rerank(self, query, passages):
            raise RerankError("provider down")

    index = RerankingIndex(_Inner([_hit(n) for n in range(6)]), Broken(), candidates=6)

    hits = index.search("q", top_k=2)

    assert [h.ordinal for h in hits] == [0, 1]
    assert index.errors == 1 and index.last_error == "provider down"


def test_a_wrong_number_of_scores_is_treated_as_a_failure_not_misaligned():
    index = RerankingIndex(_Inner([_hit(n) for n in range(4)]), _Scores([1.0, 0.0]), candidates=4)

    hits = index.search("q", top_k=2)

    assert [h.ordinal for h in hits] == [0, 1] and index.errors == 1


def test_a_single_or_empty_result_is_not_sent_to_the_reranker():
    class Explodes:
        def rerank(self, query, passages):
            raise AssertionError("should not be called")

    assert RerankingIndex(_Inner([_hit(0)]), Explodes()).search("q", top_k=3)[0].ordinal == 0
    assert RerankingIndex(_Inner([]), Explodes()).search("q") == []


def test_everything_else_passes_through_to_the_wrapped_index():
    index = RerankingIndex(_Inner([]), _Scores([]))

    assert index.upsert_document(None, []) == "upserted"
    assert index.list_documents() == ["listed"]
    assert index.delete_document("x") is True
    assert index.embedder == "the-embedder", "the ingestion pipeline reads index.embedder"
    assert not hasattr(index, "no_such_attribute"), "unknown names must still raise AttributeError"
    assert isinstance(index, DocumentIndex)


def test_the_factory_wraps_only_when_reranking_is_configured(tmp_path):
    base = {"home": tmp_path, "document_index": "sqlite", "embedding_provider": "none"}
    conn = connect(tmp_path)

    # rerank_provider pinned "none" here like document_index/embedding_provider
    # above — otherwise a maintainer .env with TINI_RERANK_PROVIDER set (e.g.
    # while trying Jev) makes "plain" wrap itself too.
    plain = make_document_index(Settings(**base, rerank_provider="none"), conn, None)
    wrapped = make_document_index(
        Settings(**base, rerank_provider="jev", rerank_api_key="k", rerank_candidates=12),
        conn,
        None,
    )

    # VisibleIndex (catalog status filter) is always the innermost wrapper.
    assert isinstance(plain.inner, SqliteDocumentIndex)
    assert isinstance(wrapped, RerankingIndex) and wrapped.candidates == 12
    assert isinstance(wrapped.inner.inner, SqliteDocumentIndex)


# --- eval integration --------------------------------------------------------


@pytest.fixture
def eval_env(tmp_path, monkeypatch):
    questions = tmp_path / "questions"
    questions.mkdir()
    monkeypatch.setattr(document_eval, "QUESTION_SET_DIR", questions)
    settings = Settings(
        home=tmp_path / "home",
        document_workspace="w",
        document_index="sqlite",
        embedding_provider="none",
        rerank_provider="jev",
        rerank_api_key=SECRET,
    )
    settings.ensure_home()
    conn = connect(settings.home)
    pipeline = IngestionPipeline(
        SqliteDocumentIndex(conn), chunk_max_tokens=200, chunk_min_tokens=1
    )
    for name, text in {
        "alpha": "Vacation policy covers annual leave days.",
        "beta": "Vacation days for interns.",
    }.items():
        source = tmp_path / f"{name}.md"
        source.write_text(text, encoding="utf-8")
        pipeline.ingest_path(source, workspace_id="w")
    conn.close()
    (questions / "q.jsonl").write_text(
        json.dumps({"query": "vacation days", "expected_title": "beta"}) + "\n", encoding="utf-8"
    )
    return settings


def _fake_make_reranker(scores_for):
    return lambda settings: _Scores(scores_for)


def test_rerank_true_reorders_and_is_recorded_in_the_run(eval_env, monkeypatch):
    monkeypatch.setattr(
        rerank_module,
        "make_reranker",
        _fake_make_reranker(lambda p: [1.0 if "interns" in t else 0.0 for t in p]),
    )

    result = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "w", "k": 1, "rerank": True}, eval_env
    )

    assert result["reranked"] is True and result["rerank_errors"] == 0
    assert result["hits"] == 1, "k=1: only the reranked-first chunk can hit"
    assert result["config"]["rerank"] == {
        "enabled": True,
        "provider": "jev",
        "model": None,
        "candidates": 20,
        "translate_to_english": [],
    }


def test_rerank_false_forces_it_off_even_when_configured(eval_env, monkeypatch):
    monkeypatch.setattr(
        rerank_module,
        "make_reranker",
        lambda s: (
            None
            if s.rerank_provider == "none"
            else pytest.fail("a reranker was requested although reranking was forced off")
        ),
    )

    result = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "w", "rerank": False}, eval_env
    )

    assert result["reranked"] is False and result["config"]["rerank"]["enabled"] is False


def test_rerank_true_without_configuration_is_an_error_not_a_silent_skip(eval_env):
    eval_env.rerank_provider = "none"

    result = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "w", "rerank": True}, eval_env
    )

    assert "not configured" in result["error"]


def test_rerank_outage_shows_up_in_the_run_instead_of_looking_like_no_effect(eval_env, monkeypatch):
    class Down:
        def rerank(self, query, passages):
            raise RerankError("provider down")

    monkeypatch.setattr(rerank_module, "make_reranker", lambda settings: Down())

    result = document_eval.run_eval({"question_set": "q.jsonl", "workspace": "w"}, eval_env)

    assert result["rerank_errors"] == 1 and result["rerank_last_error"] == "provider down"


def test_the_dashboard_hint_never_contains_the_key(eval_env):
    info = document_eval.rerank_info(eval_env)

    assert info == {"configured": True, "provider": "jev", "local": False, "candidates": 20}
    assert SECRET not in json.dumps(info)


def test_a_missing_key_is_a_clear_eval_error(eval_env):
    eval_env.rerank_api_key = ""

    result = document_eval.run_eval({"question_set": "q.jsonl", "workspace": "w"}, eval_env)

    assert "required" in result["error"]
