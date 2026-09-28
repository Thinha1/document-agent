"""The dashboard's retrieval-eval runner scores a labeled question set against a
real index, network-free (sqlite + no embedder, question set in tmp_path)."""

from __future__ import annotations

import json

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import make_document_index
from tini.ops import document_eval


@pytest.fixture
def settings(tmp_path):
    return Settings(
        home=tmp_path / "home",
        document_workspace="eval-ws",
        document_index="sqlite",
        embedding_provider="none",
        # Pinned like the other backends above — a maintainer .env with
        # TINI_RERANK_PROVIDER set (e.g. trying Jev, no key yet) would
        # otherwise make every test in this file raise in make_document_index.
        rerank_provider="none",
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


def _write_set(directory, name, cases):
    (directory / name).write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")


def test_lists_only_non_empty_question_sets(question_dir, settings):
    _write_set(question_dir, "a.jsonl", [{"id": "x", "query": "q", "expected_title": "t"}])
    (question_dir / "empty.jsonl").write_text("", encoding="utf-8")

    assert document_eval.list_question_sets(settings) == [
        {"name": "a.jsonl", "count": 1, "source": "built-in"}
    ]


def test_scores_hits_misses_and_keyword_coverage(tmp_path, settings, question_dir):
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _ingest(settings, tmp_path, "expenses", "Meals are reimbursed up to fifty dollars per day.")
    _write_set(
        question_dir,
        "qa.jsonl",
        [
            {
                "id": "hit",
                "query": "vacation days accrue",
                "expected_title": "vacation",
                "expected_keywords": ["twenty", "nonexistent"],
            },
            {
                "id": "miss",
                "query": "vacation days accrue",
                "expected_title": "expenses",
                "expected_keywords": ["meals"],
            },
        ],
    )

    result = document_eval.run_eval(
        {"question_set": "qa.jsonl", "workspace": "eval-ws", "k": 1}, settings
    )

    by_id = {row["id"]: row for row in result["cases"]}
    assert by_id["hit"]["hit"] is True
    assert by_id["hit"]["keyword_hits"] == ["twenty"]
    assert by_id["miss"]["hit"] is False
    assert result["hits"] == 1 and result["total"] == 2 and result["recall"] == 0.5


def test_reports_recall_per_language_and_matches_keywords_across_line_breaks(
    tmp_path, settings, question_dir
):
    """A multilingual set must show WHERE retrieval is weak, not one blended
    number — and a keyword the PDF wrapped mid-phrase (or CJK text with no
    spaces at all) still has to count as present."""
    _ingest(settings, tmp_path, "philosophy", "Contributing to a\nbetter world by creating value.")
    _write_set(
        question_dir,
        "ml.jsonl",
        [
            {
                "id": "en",
                "lang": "en",
                "query": "contributing better world",
                "expected_title": "philosophy",
                "expected_keywords": ["a better world"],
            },
            {"id": "ja", "lang": "ja", "query": "存在しない質問", "expected_title": "nowhere"},
        ],
    )

    result = document_eval.run_eval(
        {"question_set": "ml.jsonl", "workspace": "eval-ws", "k": 3}, settings
    )

    assert result["by_lang"]["en"] == {"hits": 1, "total": 1, "recall": 1.0}
    assert result["by_lang"]["ja"] == {"hits": 0, "total": 1, "recall": 0.0}
    en_row = next(row for row in result["cases"] if row["id"] == "en")
    assert en_row["keyword_hits"] == ["a better world"]


@pytest.mark.parametrize(
    "name", ["../secret.jsonl", "sub/qa.jsonl", "..\\qa.jsonl", "", "missing.jsonl"]
)
def test_unknown_or_escaping_question_set_names_are_rejected(question_dir, settings, name):
    result = document_eval.run_eval({"question_set": name}, settings)

    assert "error" in result


def _content(*cases):
    return "\n".join(json.dumps(c, ensure_ascii=False) for c in cases)


def test_import_accepts_jsonl_and_json_array_and_lists_them_as_imported(settings, question_dir):
    case = {"question": "how many days?", "expected_title": "vacation"}

    jsonl = document_eval.import_set({"name": "mine.jsonl", "content": _content(case)}, settings)
    array = document_eval.import_set(
        {"name": "arr.json", "content": json.dumps([case, {**case, "id": "b"}])}, settings
    )

    assert jsonl == {"ok": True, "name": "mine.jsonl", "count": 1}
    assert array == {"ok": True, "name": "arr.jsonl", "count": 2}
    listed = {s["name"]: s["source"] for s in document_eval.list_question_sets(settings)}
    assert listed == {"mine.jsonl": "imported", "arr.jsonl": "imported"}


@pytest.mark.parametrize(
    "name,content",
    [
        ("../evil.jsonl", '{"query": "q", "expected_title": "t"}'),
        ("bad name.jsonl", '{"query": "q", "expected_title": "t"}'),
        ("empty.jsonl", "   "),
        ("nojson.jsonl", "not json at all"),
        ("noquery.jsonl", '{"expected_title": "t"}'),
        ("notitle.jsonl", '{"query": "q"}'),
        ("badkw.jsonl", '{"query": "q", "expected_title": "t", "expected_keywords": "x"}'),
    ],
)
def test_import_rejects_bad_names_and_bad_cases_without_writing(
    settings, question_dir, name, content
):
    result = document_eval.import_set({"name": name, "content": content}, settings)

    assert "error" in result
    assert document_eval.list_question_sets(settings) == []


def test_import_cannot_shadow_a_built_in_set(settings, question_dir):
    _write_set(question_dir, "core.jsonl", [{"query": "q", "expected_title": "t"}])

    result = document_eval.import_set(
        {"name": "core", "content": '{"query": "q", "expected_title": "t"}'}, settings
    )

    assert "built-in" in result["error"]


def test_imported_set_runs_through_the_same_scorer(tmp_path, settings, question_dir):
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    document_eval.import_set(
        {
            "name": "mine",
            "content": _content(
                {
                    "query": "vacation days",
                    "expected_title": "vacation",
                    "expected_keywords": ["twenty"],
                }
            ),
        },
        settings,
    )

    result = document_eval.run_eval(
        {"question_set": "mine.jsonl", "workspace": "eval-ws"}, settings
    )

    assert result["hits"] == 1 and result["cases"][0]["keyword_hits"] == ["twenty"]


def test_llm_mode_is_off_by_default_and_never_calls_the_answerer(tmp_path, settings, question_dir):
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(question_dir, "q.jsonl", [{"query": "vacation", "expected_title": "vacation"}])
    calls = []

    result = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "eval-ws"},
        settings,
        answer_fn=lambda q, hits: calls.append(q) or "unused",
    )

    assert calls == [] and result["use_llm"] is False and "answer" not in result["cases"][0]


def test_llm_mode_answers_from_retrieved_chunks_and_scores_the_answer(
    tmp_path, settings, question_dir
):
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(
        question_dir,
        "q.jsonl",
        [
            {
                "id": "a",
                "query": "vacation days accrue",
                "expected_title": "vacation",
                "expected_keywords": ["twenty", "unicorn"],
            },
            {"id": "b", "query": "vacation days accrue", "expected_title": "vacation"},
        ],
    )
    seen = []

    def answer(question, hits):
        seen.append([h.title for h in hits])
        if question == "boom":
            raise RuntimeError("provider down")
        return "Staff get Twenty vacation days."

    result = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "eval-ws", "use_llm": True},
        settings,
        answer_fn=answer,
    )

    first = result["cases"][0]
    assert first["answer"] == "Staff get Twenty vacation days."
    assert first["answer_keyword_hits"] == ["twenty"]
    assert seen[0] == ["vacation"]
    assert result["answer_keyword_rate"] == 0.5
    assert result["llm_errors"] == 0 and "llm_avg_ms" in result


def test_an_llm_failure_is_recorded_per_case_and_does_not_abort_the_run(
    tmp_path, settings, question_dir
):
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(
        question_dir,
        "q.jsonl",
        [{"query": "vacation", "expected_title": "vacation", "expected_keywords": ["twenty"]}],
    )

    def down(question, hits):
        raise RuntimeError("provider down")

    result = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "eval-ws", "use_llm": True},
        settings,
        answer_fn=down,
    )

    assert result["llm_errors"] == 1
    assert "provider down" in result["cases"][0]["answer_error"]
    assert result["hits"] == 1


def test_llm_mode_refuses_oversized_sets_before_spending_anything(
    settings, question_dir, monkeypatch
):
    monkeypatch.setattr(document_eval, "MAX_LLM_CASES", 1)
    _write_set(
        question_dir,
        "big.jsonl",
        [{"query": "a", "expected_title": "t"}, {"query": "b", "expected_title": "t"}],
    )

    result = document_eval.run_eval(
        {"question_set": "big.jsonl", "use_llm": True}, settings, answer_fn=lambda q, h: "x"
    )

    assert "limited to 1" in result["error"]


@pytest.mark.parametrize(
    "values,pct,expected",
    [
        ([10, 20, 30, 40, 50, 60, 70, 80, 90, 1000], 50, 50),
        ([10, 20, 30, 40, 50, 60, 70, 80, 90, 1000], 95, 1000),
        (list(range(1, 101)), 95, 95),
        ([7], 95, 7),
        ([], 50, 0.0),
    ],
)
def test_percentiles_are_nearest_rank_and_handle_tiny_inputs(values, pct, expected):
    assert document_eval._percentile(values, pct) == expected


def test_reports_median_p95_and_recall_by_source_format(tmp_path, settings, question_dir):
    """Latency is reported as median + p95 (one cold-start query must not
    define "typical"), and a question's `format` label groups recall by the
    kind of source it had to be answered from."""
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(
        question_dir,
        "fmt.jsonl",
        [
            {"query": "vacation days", "expected_title": "vacation", "format": "text-pdf"},
            {"query": "vacation days", "expected_title": "nowhere", "format": "scanned-pdf"},
            {"query": "vacation days", "expected_title": "vacation"},
        ],
    )

    result = document_eval.run_eval({"question_set": "fmt.jsonl", "workspace": "eval-ws"}, settings)

    assert result["by_format"] == {
        "text-pdf": {"hits": 1, "total": 1, "recall": 1.0},
        "scanned-pdf": {"hits": 0, "total": 1, "recall": 0.0},
    }
    assert result["cases"][2]["format"] == ""
    assert 0 < result["median_latency_ms"] <= result["p95_latency_ms"] <= result["max_latency_ms"]
    assert result["first_query_ms"] > 0


def test_import_keeps_the_format_label(tmp_path, settings, question_dir):
    _ingest(settings, tmp_path, "t", "Some text so the workspace is not empty.")
    document_eval.import_set(
        {
            "name": "f",
            "content": _content({"query": "q", "expected_title": "t", "format": " table-pdf "}),
        },
        settings,
    )

    result = document_eval.run_eval({"question_set": "f.jsonl", "workspace": "eval-ws"}, settings)

    assert result["cases"][0]["format"] == "table-pdf"


def test_llm_mode_reports_median_and_p95_llm_latency(tmp_path, settings, question_dir):
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(question_dir, "q.jsonl", [{"query": "vacation", "expected_title": "vacation"}])

    result = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "eval-ws", "use_llm": True},
        settings,
        answer_fn=lambda q, hits: "ok",
    )

    assert result["llm_median_ms"] >= 0 and result["llm_p95_ms"] >= result["llm_median_ms"]


def test_every_run_is_saved_with_the_pipeline_config_it_was_measured_on(
    tmp_path, settings, question_dir
):
    """A recall number with no record of the model/chunking/index behind it
    can't be compared with a later run — so each run is stored with both."""
    _ingest(settings, tmp_path, "vacation", "Employees accrue twenty vacation days per year.")
    _write_set(question_dir, "q.jsonl", [{"query": "vacation", "expected_title": "vacation"}])

    result = document_eval.run_eval({"question_set": "q.jsonl", "workspace": "eval-ws"}, settings)

    saved = settings.home / "eval_runs" / result["saved_as"]
    stored = json.loads(saved.read_text(encoding="utf-8"))
    assert stored["recall"] == result["recall"] and stored["run_at"] == result["run_at"]
    config = stored["config"]
    assert config["index"]["backend"] == "sqlite" and config["search_k"] == 3
    assert config["chunking"]["max_tokens"] == settings.document_chunk_max_tokens
    assert config["corpus"] == [{"title": "vacation", "chunks": 1}]


def test_saved_runs_never_contain_credentials(tmp_path, question_dir):
    secret = "sk-super-secret-value"
    settings = Settings(
        home=tmp_path / "home",
        document_workspace="w",
        document_index="sqlite",
        embedding_provider="none",
        rerank_provider="none",
        embedding_api_key=secret,
        ocr_api_key=secret,
        api_key=secret,
        qdrant_api_key=secret,
    )
    _ingest(settings, tmp_path, "t", "Some text so the workspace is not empty.")
    _write_set(question_dir, "q.jsonl", [{"query": "x", "expected_title": "t"}])

    result = document_eval.run_eval({"question_set": "q.jsonl", "workspace": "w"}, settings)

    assert secret not in (settings.home / "eval_runs" / result["saved_as"]).read_text(
        encoding="utf-8"
    )


def test_a_run_can_opt_out_of_being_saved_and_a_write_failure_does_not_fail_the_eval(
    tmp_path, settings, question_dir
):
    _ingest(settings, tmp_path, "t", "Some text so the workspace is not empty.")
    _write_set(question_dir, "q.jsonl", [{"query": "x", "expected_title": "t"}])

    unsaved = document_eval.run_eval(
        {"question_set": "q.jsonl", "workspace": "eval-ws", "save": False}, settings
    )
    assert "saved_as" not in unsaved and not (settings.home / "eval_runs").exists()

    (settings.home / "eval_runs").write_text("a file where the history folder should be")
    failed_save = document_eval.run_eval({"question_set": "q.jsonl", "workspace": "eval-ws"}, settings)
    assert failed_save["saved_as"] is None and failed_save["total"] == 1
