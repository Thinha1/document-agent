"""OCR cost per document, derived from the token ledger — no network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tini.documents.ocr import OcrError, VisionModelOcrProvider, make_usage_logger
from tini.ops import ocr_cost

MODEL = "deepseek-v4-flash-vision-exp"  # priced (0.15, 0.30) $/M in ocr_cost


def _row(uri="file:///a.pdf", action="created", pages=10, prompt=10_000, completion=4_000, **extra):
    return {
        "workspace": "w",
        "source_uri": uri,
        "action": action,
        "provider": "deepseek",
        "model": MODEL,
        "pages": pages,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        **extra,
    }


def _write(home, rows):
    (home / "ocr_usage.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_an_empty_ledger_reports_zero_not_an_error(tmp_path):
    summary = ocr_cost.summarize(tmp_path)

    assert summary["documents"] == 0 and summary["total_cost"] == 0.0
    assert summary["avg_cost_per_document"] == 0.0 and summary["rate_per_million"] is None


def test_average_cost_per_document_and_per_page(tmp_path):
    # 10k in + 4k out at $0.15/$0.30 per M = $0.0015 + $0.0012 = $0.0027 per row
    _write(tmp_path, [_row("file:///a.pdf"), _row("file:///b.pdf", pages=20, prompt=20_000, completion=8_000)])

    summary = ocr_cost.summarize(tmp_path)

    assert summary["documents"] == 2 and summary["pages"] == 30
    assert summary["total_cost"] == pytest.approx(0.0027 + 0.0054)
    assert summary["avg_cost_per_document"] == pytest.approx((0.0027 + 0.0054) / 2)
    assert summary["avg_cost_per_page"] == pytest.approx(0.0081 / 30)
    assert summary["avg_prompt_tokens_per_page"] == pytest.approx(1000)
    assert summary["rate_per_million"] == {"in": 0.15, "out": 0.30}


def test_re_ocring_an_unchanged_file_is_counted_as_waste_not_as_a_new_document(tmp_path):
    _write(tmp_path, [_row(action="created"), _row(action="skipped", cached_pages=0)])

    summary = ocr_cost.summarize(tmp_path)

    assert summary["documents"] == 1 and summary["runs"] == 2
    assert summary["wasted_cost"] == pytest.approx(0.0027)
    assert summary["avg_cost_per_document"] == pytest.approx(0.0054), "lifetime spend on that document"


def test_pages_served_from_the_cache_are_free_and_counted(tmp_path):
    _write(tmp_path, [_row(action="created"), _row(action="skipped", pages=0, prompt=0, completion=0, cached_pages=10)])

    summary = ocr_cost.summarize(tmp_path)

    assert summary["cached_pages"] == 10
    assert summary["wasted_cost"] == 0.0
    assert summary["avg_cost_per_document"] == pytest.approx(0.0027)


def test_workspace_filter_and_damaged_lines(tmp_path):
    (tmp_path / "ocr_usage.jsonl").write_text(
        json.dumps(_row()) + "\nnot json\n" + json.dumps({**_row("file:///b.pdf"), "workspace": "other"}) + "\n",
        encoding="utf-8",
    )

    assert ocr_cost.summarize(tmp_path, "w")["documents"] == 1
    assert ocr_cost.summarize(tmp_path)["documents"] == 2


def test_an_unlisted_model_falls_back_to_the_provider_estimate(tmp_path):
    _write(tmp_path, [_row(model="some-new-vision-model", prompt=1_000_000, completion=0)])

    # pricing.py's provider-level "deepseek" estimate: $0.435 per M input tokens
    assert ocr_cost.summarize(tmp_path)["total_cost"] == pytest.approx(0.435)


def test_the_logger_appends_one_timestamped_json_line_per_call(tmp_path):
    log = make_usage_logger(tmp_path / "home")

    log({"title": "a", "pages": 3})
    log({"title": "b", "pages": 4})

    lines = (tmp_path / "home" / "ocr_usage.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["title"] for r in rows] == ["a", "b"] and all("ts" in r for r in rows)


# --- the per-page transcription cache -----------------------------------------

class _FakeVision:
    """Stands in for the OpenAI client: counts calls, can be told to fail."""

    def __init__(self, text="transcribed", fail=False):
        self.calls = 0
        self.text = text
        self.fail = fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            usage=SimpleNamespace(prompt_tokens=1080, completion_tokens=400),
        )


def _provider(tmp_path, client, model=MODEL):
    provider = VisionModelOcrProvider("k", model, cache_dir=tmp_path / "cache")
    provider.client = client
    return provider


def test_an_identical_page_is_transcribed_once_and_then_served_from_the_cache(tmp_path):
    client = _FakeVision("page text")
    ocr = _provider(tmp_path, client)

    assert ocr.ocr_image(b"page-1") == "page text"
    assert ocr.ocr_image(b"page-1") == "page text"

    assert client.calls == 1
    assert ocr.usage_snapshot() == {
        "calls": 1, "prompt_tokens": 1080, "completion_tokens": 400, "cache_hits": 1,
    }


def test_the_cache_survives_a_new_provider_instance_but_not_a_different_page_or_model(tmp_path):
    first = _FakeVision("A")
    _provider(tmp_path, first).ocr_image(b"page-1")

    again = _FakeVision("B")
    assert _provider(tmp_path, again).ocr_image(b"page-1") == "A", "same page, new process"
    assert again.calls == 0
    assert _provider(tmp_path, again).ocr_image(b"page-2") == "B", "a different page misses"
    assert _provider(tmp_path, again, model="another-vision-model").ocr_image(b"page-1") == "B", (
        "the same page under a different model must not reuse the old model's text"
    )


def test_a_failed_page_is_not_cached_and_a_blank_page_is(tmp_path):
    broken = _FakeVision(fail=True)
    with pytest.raises(OcrError):
        _provider(tmp_path, broken).ocr_image(b"page-1")
    healthy = _FakeVision("now works")
    assert _provider(tmp_path, healthy).ocr_image(b"page-1") == "now works", "the failure was not cached"

    blank = _FakeVision("")
    ocr = _provider(tmp_path, blank)
    assert ocr.ocr_image(b"blank-page") == ""
    assert ocr.ocr_image(b"blank-page") == ""
    assert blank.calls == 1, "an empty transcription is a real result, cached like any other"


def test_reingesting_a_scanned_file_costs_nothing_the_second_time(tmp_path):
    """The bug this cache exists for: OCR used to run before the index noticed
    a file was unchanged, so an identical re-ingest paid again for nothing."""
    pytest.importorskip("pypdfium2")
    from evals.deterministic.test_document_ocr import _minimal_pdf_with_pages
    from tini.db import connect
    from tini.documents.pipeline import IngestionPipeline
    from tini.documents.store import SqliteDocumentIndex

    source = tmp_path / "scan.pdf"
    source.write_bytes(_minimal_pdf_with_pages(["1", "2", "3"]))
    client = _FakeVision("a long enough transcription of the page")
    ocr = _provider(tmp_path, client)
    rows: list[dict] = []
    (tmp_path / "home").mkdir()
    conn = connect(tmp_path / "home")
    try:
        pipeline = IngestionPipeline(SqliteDocumentIndex(conn), ocr=ocr, usage_log=rows.append)
        (first,) = pipeline.ingest_path(source, workspace_id="w")
        (second,) = pipeline.ingest_path(source, workspace_id="w")
    finally:
        conn.close()

    assert (first.action, second.action) == ("created", "skipped")
    assert client.calls == 3, "three pages OCR'd once; the re-ingest made no API call"
    assert (rows[0]["pages"], rows[0]["cached_pages"]) == (3, 0)
    assert (rows[1]["pages"], rows[1]["cached_pages"], rows[1]["prompt_tokens"]) == (0, 3, 0)
