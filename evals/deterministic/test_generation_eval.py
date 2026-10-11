"""DETERMINISTIC EVAL — the generation eval's mechanics: the question sample is
deterministic and complete, one turn runs through the real loop and tool, and the
counts mean what their names say. Network-free: the model and the judge are scripts."""

from __future__ import annotations

from types import SimpleNamespace

from evals.helpers import ScriptedClient, response, text_block, tool_block
from tini.config import Settings
from tini.db import connect
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import make_document_index
from tini.ops import generation_eval as ge


def _settings(tmp_path):
    (tmp_path / "home").mkdir(exist_ok=True)
    return Settings(
        home=tmp_path / "home", document_index="sqlite", embedding_provider="none",
        rerank_provider="none", document_workspace="w", max_iterations=4,
    )


def _ingest(settings, tmp_path):
    source = tmp_path / "torque-spec.md"
    source.write_text("# Torque\n\n| Bolt | Dry |\n|---|---|\n| M8 | 25.5 |\n| M10 | 49.0 |\n| M12 | 86.0 |\n| M14 | 135.0 |\n", encoding="utf-8")
    conn = connect(settings.home)
    IngestionPipeline(make_document_index(settings, conn, None)).ingest_path(source, workspace_id="w")
    conn.close()


def _judge(payload: str):
    return (SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response([text_block(payload)]))), "judge-model")


def test_the_sample_is_deterministic_and_keeps_every_unanswerable_question():
    first, second = ge.load_cases(), ge.load_cases()

    assert first == second
    assert len(first) == 54
    assert sum(c["abstain"] for c in first) == 12
    assert {c["source"] for c in first} == {"real", "synthetic-prose", "synthetic-table", "unanswerable"}
    for c in first:
        assert c["workspace"] in {"denso", "denso-xl", "denso-tech"}
        assert c["abstain"] or c["expected_keywords"]
    for lang in ("en", "ja", "vi"):
        assert sum(c["lang"] == lang for c in first) >= 12  # no language is squeezed out


def test_one_turn_runs_the_real_loop_and_tool_and_measures_it(tmp_path):
    settings = _settings(tmp_path)
    _ingest(settings, tmp_path)
    client = ScriptedClient([
        response([tool_block("search_documents", {"query": "M10 torque"})], stop_reason="tool_use"),
        response([text_block("Mô-men xiết bu lông M10 là 49.0 N·m (torque-spec).")]),
    ])
    case = {"id": "q", "source": "synthetic-table", "lang": "vi", "workspace": "w", "abstain": False,
            "query": "Mô-men xiết bu lông M10 là bao nhiêu?", "expected_title": "torque-spec",
            "expected_keywords": ["M10", "49.0"]}

    row = ge.run_case(settings, case, judge=_judge('{"abstained": false, "supported": "yes", "unsupported": []}'), client=client)

    assert row["tool_ok"] and row["searches"] == ["M10 torque"]
    assert row["evidence_has_answer"] and row["reply_has_answer"] and row["cited"]
    assert row["language_ok"]  # a Vietnamese question answered in Vietnamese
    assert row["judge"] == {"abstained": False, "supported": "yes", "unsupported": []}


def test_a_reply_in_the_wrong_language_or_without_the_answer_is_counted_as_such(tmp_path):
    settings = _settings(tmp_path)
    _ingest(settings, tmp_path)
    client = ScriptedClient([
        response([tool_block("search_documents", {"query": "M10"})], stop_reason="tool_use"),
        response([text_block("The torque is about fifty.")]),
    ])
    case = {"id": "q", "source": "real", "lang": "vi", "workspace": "w", "abstain": False,
            "query": "Mô-men xiết bu lông M10 là bao nhiêu?", "expected_title": "torque-spec",
            "expected_keywords": ["M10", "49.0"]}

    row = ge.run_case(settings, case, judge=None, client=client)

    assert row["evidence_has_answer"] is True and row["reply_has_answer"] is False  # a generation miss
    assert row["language_ok"] is False and row["cited"] is False and "judge" not in row


def test_the_language_line_can_be_left_out_so_its_worth_can_be_measured(tmp_path):
    from tini.runtime.session import Session

    session = Session(_settings(tmp_path), memory=None)

    with_line = session.build_context("Quy định về an toàn lao động là gì?")
    without = session.build_context("Quy định về an toàn lao động là gì?", reply_language=False)

    assert "in Vietnamese" in with_line and "in Vietnamese" not in without
    assert without.startswith("Right now it is")


def test_summary_separates_retrieval_misses_from_generation_misses_and_counts_inventions():
    def row(**kw):
        base = {"id": "x", "source": "real", "lang": "en", "abstain": False, "language_ok": True, "tool_ok": True,
                "evidence_has_answer": True, "reply_has_answer": True, "cited": True}
        return {**base, **kw}

    rows = [
        row(),  # answered, cited
        row(reply_has_answer=False),  # the chunks had it, the reply does not: a generation miss
        row(evidence_has_answer=False, reply_has_answer=False),  # retrieval never gave it
        row(abstain=True, judge={"abstained": True, "supported": "yes", "unsupported": []}),
        row(abstain=True, judge={"abstained": False, "supported": "no", "unsupported": ["a made-up value"]}),
        row(abstain=True, judge={"abstained": True, "supported": "partly", "unsupported": ["x"]}),
        {"id": "bad", "source": "real", "lang": "en", "abstain": False, "error": "boom"},
    ]

    summary = ge.summarize(rows)
    everything = summary["all"]

    assert summary["errors"] == 1
    assert (everything["answerable"], everything["answered"]) == (3, 1)
    assert everything["retrieval_limited"] == 1
    assert (everything["generation_missed"], everything["generation_of_evidence"]) == (1, 2)
    assert (everything["unanswerable"], everything["declined"], everything["invented"]) == (3, 2, 2)
    assert "answered" in ge.format_summary(summary)


# --- did the reply name the page the answer is on? ---------------------------------

EVIDENCE = (
    "Internal document results for 'x':\n"
    "1. [manual#chunk-4, pp.7-8]\n   The range is 0 .. 16000 kbps.\n   Source: file:///m.pdf\n"
    "2. [manual#chunk-9, p.20]\n   An unrelated paragraph about warranty.\n   Source: file:///m.pdf\n"
    "3. [other#chunk-1, p.3]\n   The range is 0 .. 16000 kbps.\n   Source: file:///o.pdf\n"
    "4. [notes#chunk-0]\n   The range is 0 .. 16000 kbps.\n   Source: file:///n.md"
)


def test_the_expected_pages_are_those_of_the_chunks_of_the_right_document_that_hold_the_answer():
    assert ge.answer_pages(EVIDENCE, ["0 .. 16000"], "manual") == {7, 8}
    assert ge.answer_pages(EVIDENCE, ["0 .. 16000"], "other") == {3}
    assert ge.answer_pages(EVIDENCE, ["0 .. 16000"], "notes") == set()  # no pages: nothing to check
    assert ge.answer_pages(EVIDENCE, ["not in any chunk"], "manual") == set()


def test_pages_named_in_a_reply_are_read_in_every_language_and_style():
    named = ge.cited_pages
    assert named("see [manual, p.7]") == {7}
    assert named("[manual, pp.7-8] and page 12") == {7, 8, 12}
    assert named("trang 43–44") == {43, 44}
    assert named("Pages 5 to 6") == {5, 6} and named("trang 43 đến 44") == {43, 44}
    assert named("7ページ") == {7} and named("ページ9") == {9}
    assert named("the year 1972 and 4 pages of text") == set()
    assert named("p.1-900") == set()  # a range that wide is not a citation


def test_a_pdf_answer_is_checked_for_its_page_and_a_text_answer_is_not(tmp_path):
    from evals.helpers import pdf_with_pages

    settings = _settings(tmp_path)
    (tmp_path / "manual.pdf").write_bytes(pdf_with_pages([
        "Page one covers the warranty terms for injectors sold in the aftermarket channel.",
        "Page two lists the tightening torque for connector bolts as forty nine newton metres.",
    ]))
    conn = connect(settings.home)
    IngestionPipeline(make_document_index(settings, conn, None), chunk_max_tokens=20, chunk_min_tokens=5,
                      chunk_overlap_tokens=0).ingest_path(tmp_path / "manual.pdf", workspace_id="w")
    conn.close()
    case = {"id": "q", "source": "real", "lang": "en", "workspace": "w", "abstain": False,
            "query": "What is the connector bolt torque?", "expected_title": "manual", "expected_keywords": ["forty nine"]}

    def ask(reply):
        client = ScriptedClient([
            response([tool_block("search_documents", {"query": "connector bolt torque"})], stop_reason="tool_use"),
            response([text_block(reply)]),
        ])
        return ge.run_case(settings, case, judge=None, client=client)

    right = ask("It is forty nine newton metres [manual, p.2].")
    wrong = ask("It is forty nine newton metres [manual, p.1].")
    none = ask("It is forty nine newton metres (manual).")

    assert right["pages_expected"] == [2] and right["page_cited"] and right["page_ok"]
    assert wrong["page_cited"] is True and wrong["page_ok"] is False
    assert none["page_cited"] is False and none["page_ok"] is False


def test_the_summary_counts_pages_only_for_answered_pdf_questions():
    def row(**kw):
        base = {"id": "x", "source": "real", "lang": "en", "abstain": False, "language_ok": True, "tool_ok": True,
                "evidence_has_answer": True, "reply_has_answer": True, "cited": True}
        return {**base, **kw}

    rows = [
        row(pages_expected=[3], page_cited=True, page_ok=True),
        row(pages_expected=[3], page_cited=True, page_ok=False),
        row(pages_expected=[3], page_cited=False, page_ok=False),
        row(),  # a text source: no page to check
        row(reply_has_answer=False, pages_expected=[3], page_cited=True, page_ok=True),  # not answered: not counted
    ]

    everything = ge.summarize(rows)["all"]

    assert (everything["page_ok"], everything["page_cited"], everything["page_base"]) == (1, 2, 3)
    assert "page" in ge.format_summary(ge.summarize(rows))
