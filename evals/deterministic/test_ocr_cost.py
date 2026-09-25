"""OCR cost per document, derived from the token ledger — no network."""

from __future__ import annotations

import json

import pytest

from tini.documents.ocr import make_usage_logger
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
    _write(tmp_path, [_row(action="created"), _row(action="skipped")])

    summary = ocr_cost.summarize(tmp_path)

    assert summary["documents"] == 1 and summary["runs"] == 2
    assert summary["wasted_cost"] == pytest.approx(0.0027)
    assert summary["avg_cost_per_document"] == pytest.approx(0.0027), "mean of its runs, not the sum"


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
