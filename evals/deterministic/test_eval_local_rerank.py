"""DETERMINISTIC EVAL — the local reranker is the default, and the dashboard
eval treats it as configured, reports answer-in-chunks, and refuses an empty
workspace. Network-free: the reranker is a stub."""

from __future__ import annotations

import json

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents import rerank as rerank_module
from tini.documents.pipeline import IngestionPipeline
from tini.documents.rerank import TeiReranker
from tini.documents.store import make_document_index
from tini.ops import document_eval


@pytest.fixture
def settings(tmp_path):
    return Settings(
        home=tmp_path / "home",
        document_workspace="eval-ws",
        document_index="sqlite",
        embedding_provider="none",
        rerank_provider="tei",
        rerank_api_key="",
    )


@pytest.fixture
def question_dir(tmp_path, monkeypatch):
    directory = tmp_path / "questions"
    directory.mkdir()
    monkeypatch.setattr(document_eval, "QUESTION_SET_DIR", directory)
    return directory


def _ingest(settings, tmp_path, name, text):
    settings.ensure_home()
    source = tmp_path / f"{name}.md"
    source.write_text(text, encoding="utf-8")
    conn = connect(settings.home)
    try:
        IngestionPipeline(make_document_index(settings, conn, None)).ingest_path(
            source, workspace_id=settings.document_workspace
        )
    finally:
        conn.close()


def _write_set(directory, cases):
    (directory / "qa.jsonl").write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")


def test_the_default_provider_is_the_local_one(monkeypatch):
    monkeypatch.delenv("TINI_RERANK_PROVIDER", raising=False)
    monkeypatch.delenv("TINI_RERANK_URL", raising=False)

    settings = Settings()

    assert settings.rerank_provider == "tei"
    assert isinstance(rerank_module.make_reranker(settings), TeiReranker)


def test_a_local_reranker_needs_no_key_but_a_hosted_one_does():
    local = document_eval.rerank_info(Settings(rerank_provider="tei", rerank_api_key=""))
    hosted_without_key = document_eval.rerank_info(Settings(rerank_provider="jev", rerank_api_key=""))
    hosted = document_eval.rerank_info(Settings(rerank_provider="jev", rerank_api_key="k"))
    off = document_eval.rerank_info(Settings(rerank_provider="none"))

    assert local["configured"] is True and local["local"] is True
    assert hosted_without_key["configured"] is False
    assert hosted["configured"] is True and hosted["local"] is False
    assert off["configured"] is False


def test_rerank_true_is_accepted_for_the_local_provider(tmp_path, settings, question_dir, monkeypatch):
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(question_dir, [{"id": "a", "query": "vacation days", "expected_title": "vacation",
                               "expected_keywords": ["twenty"]}])
    monkeypatch.setattr(TeiReranker, "rerank", lambda self, query, passages: [1.0] * len(passages))

    result = document_eval.run_eval(
        {"question_set": "qa.jsonl", "workspace": "eval-ws", "rerank": True, "save": False}, settings
    )

    assert "error" not in result
    assert result["reranked"] is True and result["rerank_errors"] == 0
    assert result["config"]["rerank"]["provider"] == "tei"


def test_answer_in_chunks_counts_only_cases_with_every_keyword_present(tmp_path, settings, question_dir):
    settings.rerank_provider = "none"
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(question_dir, [
        {"id": "all", "query": "vacation days", "expected_title": "vacation", "expected_keywords": ["twenty", "vacation"]},
        {"id": "part", "query": "vacation days", "expected_title": "vacation", "expected_keywords": ["twenty", "absent"]},
        {"id": "nokw", "query": "vacation days", "expected_title": "vacation"},
    ])

    result = document_eval.run_eval(
        {"question_set": "qa.jsonl", "workspace": "eval-ws", "save": False}, settings
    )

    # the case without keywords is not scored; a half-present answer does not count
    assert result["answer_in_chunks"] == {"found": 1, "total": 2, "rate": 0.5}


def test_a_workspace_with_no_documents_is_an_error_not_a_zero_score(tmp_path, settings, question_dir):
    settings.rerank_provider = "none"
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(question_dir, [{"id": "a", "query": "vacation", "expected_title": "vacation"}])

    result = document_eval.run_eval(
        {"question_set": "qa.jsonl", "workspace": "other-ws", "save": False}, settings
    )

    assert "other-ws" in result["error"] and "no indexed documents" in result["error"]


def test_a_set_written_for_another_workspace_is_refused_not_scored_as_all_misses(tmp_path, settings, question_dir):
    settings.rerank_provider = "none"
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(question_dir, [{"id": "a", "query": "vacation", "expected_title": "some-other-document"}])

    result = document_eval.run_eval(
        {"question_set": "qa.jsonl", "workspace": "eval-ws", "save": False}, settings
    )

    assert "None of the documents" in result["error"] and "eval-ws" in result["error"]


def test_questions_about_a_hidden_document_are_expired_not_scored_as_misses(tmp_path, settings, question_dir):
    from tini.documents import catalog

    settings.rerank_provider = "none"
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _ingest(settings, tmp_path, "expenses", "Meals are reimbursed up to fifty dollars per day.")
    conn = connect(settings.home)
    try:
        uri = next(d.source_uri for d in make_document_index(settings, conn, None).list_documents("eval-ws") if d.title == "expenses")
        catalog.record_ingest(conn, {"workspace": "eval-ws", "source_uri": uri, "title": "expenses",
                                     "content_hash": "h", "text": "Meals", "profile": {}})
        catalog.update(conn, "eval-ws", uri, {"status": "superseded"})
    finally:
        conn.close()
    _write_set(question_dir, [
        {"id": "a", "query": "vacation days", "expected_title": "vacation", "expected_keywords": ["twenty"]},
        {"id": "b", "query": "meals reimbursed", "expected_title": "expenses", "expected_keywords": ["fifty"]},
    ])

    result = document_eval.run_eval(
        {"question_set": "qa.jsonl", "workspace": "eval-ws", "save": False}, settings
    )

    # b is not scored: only a counts, so a hidden document cannot drag recall down
    assert result["total"] == 1 and result["hits"] == 1 and result["recall"] == 1.0
    assert result["answer_in_chunks"] == {"found": 1, "total": 1, "rate": 1.0}
    assert [c["id"] for c in result["cases"]] == ["a"]
    # ...and is checked the other way round: the superseded document did not come back
    assert result["expired"]["documents"] == ["expenses"]
    assert [(c["id"], c["leaked"]) for c in result["expired"]["cases"]] == [("b", False)]
    assert result["expired"]["leaked"] == 0
