"""TINI_OCR_TABLE_FLOW: a second, table-only OCR call, only for a page whose
transcription left a `[[TABLE:n]]` marker — and the delimiter it writes so
chunking.py can tell a table apart from prose (see that module's docstring).

Off by default and not yet measured against the single-prompt flow — see
docs/document-ingestion.md and tini/documents/ocr.py's module docstring. These
tests are network-free: `VisionModelOcrProvider` is exercised against a fake
OpenAI-shaped client (same pattern as evals/deterministic/test_ocr_cost.py),
and the loader-level merge is exercised against a scripted fake OcrProvider
(same pattern as test_document_ocr.py's `_FakeOcrProvider`).
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("pypdfium2")

from tini.documents.loaders import load_document  # noqa: E402
from tini.documents.ocr import OcrError, VisionModelOcrProvider  # noqa: E402


def _minimal_pdf_with_text(text: str) -> bytes:
    """Same hand-built single-page PDF as test_document_ocr.py's fixture of
    the same name — kept local rather than imported so this file has no
    load-order dependency on another test module."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
            b"/MediaBox [0 0 200 200] /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF"
    )
    return bytes(out)


def _minimal_pdf_with_pages(texts: list[str]) -> bytes:
    """Same hand-built object model as `_minimal_pdf_with_text`, generalized
    to N pages sharing one font — see test_document_ocr.py's copy of this
    helper for the full rationale."""
    count = len(texts)
    font_num = 3 + count * 2
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(count))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        font_num: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, text in enumerate(texts):
        page_num = 3 + i * 2
        content_num = 4 + i * 2
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/MediaBox [0 0 200 200] /Contents {content_num} 0 R >>"
        ).encode()
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects[content_num] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    total = max(objects) + 1
    offsets = [0] * total
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {total}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, total):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(total).encode() + b" /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF"
    )
    return bytes(out)

# =====================================================================
# VisionModelOcrProvider: prompt selection, cache separation, extract_tables
# =====================================================================


class _FakeVision:
    """Same shape as test_ocr_cost.py's fixture: records the prompt text it
    was sent and returns a scripted reply."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"][0]["text"]
        self.prompts.append(prompt)
        reply = self._replies[self.calls] if self.calls < len(self._replies) else self._replies[-1]
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


def _provider(tmp_path, replies, table_flow=False):
    provider = VisionModelOcrProvider("k", "m", cache_dir=tmp_path / "cache", table_flow=table_flow)
    provider.client = _FakeVision(replies)
    return provider


def test_table_flow_off_sends_the_legacy_render_as_markdown_prompt(tmp_path):
    ocr = _provider(tmp_path, ["text"], table_flow=False)
    ocr.ocr_image(b"page")
    assert "Render any table as a Markdown table" in ocr.client.prompts[0]
    assert "[[TABLE" not in ocr.client.prompts[0]


def test_table_flow_on_sends_the_skip_table_prompt_instead(tmp_path):
    ocr = _provider(tmp_path, ["text"], table_flow=True)
    ocr.ocr_image(b"page")
    assert "[[TABLE:n]]" in ocr.client.prompts[0]
    assert "Markdown table" not in ocr.client.prompts[0]


def test_toggling_table_flow_does_not_serve_the_others_cached_text(tmp_path):
    """Same image, same model, same cache_dir, different prompt: the caches
    must not cross, or turning the flag on/off would silently serve stale
    output from the other mode."""
    cache_dir = tmp_path / "cache"
    legacy = VisionModelOcrProvider("k", "m", cache_dir=cache_dir, table_flow=False)
    legacy.client = _FakeVision(["legacy reply"])
    assert legacy.ocr_image(b"same-page") == "legacy reply"

    table_mode = VisionModelOcrProvider("k", "m", cache_dir=cache_dir, table_flow=True)
    table_mode.client = _FakeVision(["table-mode reply"])
    assert table_mode.ocr_image(b"same-page") == "table-mode reply"
    assert table_mode.client.calls == 1, "must not have been served from the legacy cache entry"


def test_extract_tables_parses_a_clean_json_reply(tmp_path):
    reply = json.dumps({"tables": [{"caption": "Sales", "columns": ["A", "B"], "rows": [["1", "2"]]}]})
    ocr = _provider(tmp_path, [reply])
    tables = ocr.extract_tables(b"page")
    assert tables == [{"caption": "Sales", "columns": ["A", "B"], "rows": [["1", "2"]]}]


def test_extract_tables_tolerates_a_chatty_reply_around_the_json(tmp_path):
    reply = 'Sure, here it is:\n```json\n{"tables": []}\n```\nLet me know if you need more.'
    ocr = _provider(tmp_path, [reply])
    assert ocr.extract_tables(b"page") == []


def test_extract_tables_raises_on_a_reply_with_no_json(tmp_path):
    ocr = _provider(tmp_path, ["I can't see a table here."])
    with pytest.raises(OcrError):
        ocr.extract_tables(b"page")


def test_extract_tables_raises_when_tables_is_not_a_list(tmp_path):
    ocr = _provider(tmp_path, ['{"tables": "none"}'])
    with pytest.raises(OcrError):
        ocr.extract_tables(b"page")


def test_extract_tables_is_cached_per_image_separately_from_transcription(tmp_path):
    ocr = _provider(tmp_path, ['{"tables": []}'])
    ocr.extract_tables(b"page")
    ocr.extract_tables(b"page")
    assert ocr.client.calls == 1
    assert ocr.usage_snapshot()["table_calls"] == 1
    assert ocr.usage_snapshot()["table_cache_hits"] == 1
    assert ocr.usage_snapshot()["calls"] == 0, "extract_tables must not touch the transcription counter"


def test_a_failed_extraction_is_never_cached_so_a_retry_can_succeed(tmp_path):
    ocr = _provider(tmp_path, ["not json", '{"tables": []}'])
    with pytest.raises(OcrError):
        ocr.extract_tables(b"page")
    assert ocr.extract_tables(b"page") == []
    assert ocr.client.calls == 2


def test_usage_snapshot_folds_both_flows_tokens_into_one_dollar_total(tmp_path):
    ocr = _provider(tmp_path, ["text", '{"tables": []}'], table_flow=True)
    ocr.ocr_image(b"a")
    ocr.extract_tables(b"a")
    usage = ocr.usage_snapshot()
    assert usage["calls"] == 1 and usage["table_calls"] == 1
    assert usage["prompt_tokens"] == 200 and usage["completion_tokens"] == 100


# =====================================================================
# loaders.py: merging the table-flow's structured result back into the page
# =====================================================================


class _TableFlowOcrProvider:
    """Scripted, network-free: `pages[i]` is the transcription for the i-th
    page OCR'd (page order, matches `_FakeOcrProvider`'s convention at
    concurrency=1); `tables[i]` is the i-th extract_tables call's result, in
    the order those calls happen — which is the order pages carrying a marker
    are visited, not the page index, since a page with no marker never calls
    it. Style mirrors test_document_ocr.py's `_FakeOcrProvider`."""

    def __init__(self, pages: list[str], tables: list[list[dict] | Exception], table_flow: bool = True):
        self.table_flow = table_flow
        self.concurrency = 1
        self._pages = pages
        self._tables = tables
        self.ocr_calls: list[bytes] = []
        self.table_calls: list[bytes] = []
        self._lock = threading.Lock()

    def ocr_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        with self._lock:
            index = len(self.ocr_calls)
            self.ocr_calls.append(image_bytes)
        return self._pages[index] if index < len(self._pages) else ""

    def extract_tables(self, image_bytes: bytes, media_type: str = "image/png") -> list[dict]:
        with self._lock:
            index = len(self.table_calls)
            self.table_calls.append(image_bytes)
        result = self._tables[index] if index < len(self._tables) else []
        if isinstance(result, Exception):
            raise result
        return result


SALES_TABLE = [
    {
        "caption": "Sales by region",
        "columns": ["Region", "Amount"],
        "rows": [["Japan", "2,509,604"], ["Europe", "624,329"]],
        "notes": "Figures in millions of yen.",
    }
]


def _load(tmp_path, pdf_bytes, ocr):
    source = tmp_path / "scanned.pdf"
    source.write_bytes(pdf_bytes)
    return load_document(source, 26_214_400, ocr=ocr)


def test_table_flow_off_leaves_the_legacy_single_call_behavior_untouched(tmp_path):
    """A fake that never sets table_flow (the vast majority of OcrProviders,
    including every one written before this feature) must not have
    extract_tables looked for or called."""
    class _Legacy:
        concurrency = 1

        def ocr_image(self, image_bytes, media_type="image/png"):
            return "some transcribed text, no markers here"

    ocr = _Legacy()
    loaded = _load(tmp_path, _minimal_pdf_with_text("x"), ocr)
    assert loaded.metadata["ocr_applied"] is True
    assert "ocr_tables_found" not in loaded.metadata
    assert "ocr_failed_tables" not in loaded.metadata


def test_a_page_with_no_marker_never_triggers_a_table_call(tmp_path):
    ocr = _TableFlowOcrProvider(pages=["plain text, no table here"], tables=[])
    _load(tmp_path, _minimal_pdf_with_text("x"), ocr)
    assert ocr.table_calls == []


def test_a_resolved_marker_becomes_a_tini_table_block(tmp_path):
    ocr = _TableFlowOcrProvider(pages=["Intro.\n[[TABLE:1]]\nOutro."], tables=[SALES_TABLE])
    loaded = _load(tmp_path, _minimal_pdf_with_text("x"), ocr)
    assert len(ocr.table_calls) == 1
    assert "<!--tini:table id=1-->" in loaded.text
    assert "Sales by region" in loaded.text
    assert "2,509,604" in loaded.text
    assert "[[TABLE:1]]" not in loaded.text
    assert loaded.metadata["ocr_tables_found"] == 1
    assert "ocr_failed_tables" not in loaded.metadata


def test_the_resolved_block_chunks_by_row_end_to_end(tmp_path):
    """loaders.py's merge and chunking.py's table split are two ends of the
    same contract; this exercises them together the way IngestionPipeline
    actually runs them, on top of the per-module unit tests on each side."""
    from tini.documents.chunking import chunk_text

    ocr = _TableFlowOcrProvider(pages=["[[TABLE:1]]"], tables=[SALES_TABLE])
    loaded = _load(tmp_path, _minimal_pdf_with_text("x"), ocr)
    chunks = chunk_text(loaded.text, max_tokens=512, min_tokens=20, overlap_tokens=10)
    table_chunks = [c for c in chunks if c.metadata.get("kind") == "table"]
    assert len(table_chunks) == 1
    assert "Japan" in table_chunks[0].text and "Europe" in table_chunks[0].text


def test_an_extraction_failure_leaves_the_marker_as_literal_text_and_is_counted(tmp_path):
    ocr = _TableFlowOcrProvider(pages=["Intro.\n[[TABLE:1]]\nOutro."], tables=[OcrError("boom")])
    loaded = _load(tmp_path, _minimal_pdf_with_text("x"), ocr)
    assert "[[TABLE:1]]" in loaded.text, "not silently dropped"
    assert loaded.metadata["ocr_failed_tables"] == 1
    assert "ocr_tables_found" not in loaded.metadata


def test_fewer_tables_returned_than_markers_counts_only_the_unmatched_ones_failed(tmp_path):
    ocr = _TableFlowOcrProvider(
        pages=["[[TABLE:1]] and also [[TABLE:2]]"],
        tables=[SALES_TABLE],  # only one table for two markers
    )
    loaded = _load(tmp_path, _minimal_pdf_with_text("x"), ocr)
    assert "<!--tini:table id=1-->" in loaded.text
    assert "[[TABLE:2]]" in loaded.text
    assert loaded.metadata["ocr_tables_found"] == 1
    assert loaded.metadata["ocr_failed_tables"] == 1


def test_a_marker_only_appears_on_pages_that_have_one(tmp_path):
    """Multi-page: page 1 has a table, page 2 doesn't — only one extract_tables
    call happens, and it's for the right page."""
    ocr = _TableFlowOcrProvider(
        pages=["Page one.\n[[TABLE:1]]", "Page two, no table at all."],
        tables=[SALES_TABLE],
    )
    loaded = _load(tmp_path, _minimal_pdf_with_pages(["1", "2"]), ocr)
    assert len(ocr.table_calls) == 1
    assert ocr.table_calls[0] == ocr.ocr_calls[0], "extract_tables ran on the SAME image as the marked page"
    assert "Page two, no table at all." in loaded.text


def test_table_flow_true_but_no_extract_tables_method_degrades_to_literal_marker(tmp_path):
    """Defensive: a provider that claims table_flow but doesn't implement
    extract_tables (shouldn't happen given how make_ocr_provider wires the
    real implementation, but must fail closed rather than crash)."""
    class _Half:
        table_flow = True
        concurrency = 1

        def ocr_image(self, image_bytes, media_type="image/png"):
            return "[[TABLE:1]]"

    loaded = _load(tmp_path, _minimal_pdf_with_text("x"), _Half())
    assert "[[TABLE:1]]" in loaded.text
