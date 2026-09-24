"""OCR fallback for scanned/image-only PDF pages — network-free via a fake
OcrProvider double, except the one opt-in test that calls a real vision model.

The heuristic and the rasterize-then-OCR plumbing are exercised here without
ever needing pypdfium2 to render a REAL scan or a network call to succeed —
`pytest.importorskip("pypdfium2")` only guards against the optional [ocr]
extra being absent, same convention test_qdrant_document_index.py uses for
qdrant-client. The one test that hits a real model is gated behind
TINI_TEST_OCR=1, mirroring the Supabase/mem0 opt-in pattern in
test_fact_store_conformance.py — it calls a real vision API and costs real
money, so `make gate` must stay green and free for a contributor with no
accounts.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("pypdfium2")

from tini.documents.loaders import load_document  # noqa: E402
from tini.documents.ocr import OcrError  # noqa: E402


def _minimal_pdf_with_text(text: str) -> bytes:
    """Same hand-built single-page PDF as test_document_ingestion.py's
    encrypted-PDF fixture — valid enough for both pypdf and pypdfium2 to
    read, with a real (if short) text stream so `text/page_count` can be
    pushed below the "looks scanned" threshold deliberately."""
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
    to N pages sharing one font — enough for pypdf and pypdfium2 to read N
    genuinely separate pages, each short enough to trip the "looks scanned"
    heuristic."""
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


class _FakeOcrProvider:
    """Records every image it's asked to transcribe and returns a scripted
    answer per call — no network, no model, deterministic.

    `concurrency` defaults to 1 (matching every test written before
    concurrent OCR existed): with one worker, `_ocr_pdf_pages` never fans
    out, so `calls`/script indexing stay in page order exactly as before.
    Tests that want the concurrent path pass `concurrency>1` explicitly.
    The lock only protects `call_index` bookkeeping — which page a given
    call happens to land on is still scheduler-dependent, so scripted
    per-page responses are only meaningful at concurrency=1."""

    def __init__(
        self,
        script: list[str] | None = None,
        fail_on: set[int] | None = None,
        concurrency: int = 1,
    ):
        self.calls: list[bytes] = []
        self._script = list(script) if script is not None else None
        self._fail_on = fail_on or set()
        self.concurrency = concurrency
        self._lock = threading.Lock()

    def ocr_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        with self._lock:
            call_index = len(self.calls)
            self.calls.append(image_bytes)
        if call_index in self._fail_on:
            raise OcrError("scripted failure")
        if self._script is not None:
            return self._script[call_index] if call_index < len(self._script) else ""
        return "fake transcription"


def test_a_sparse_pdf_without_ocr_configured_is_flagged_not_silent(tmp_path):
    """No OcrProvider passed — this must NOT raise, and the resulting
    document must say WHY it's thin rather than just being thin. Matches the
    real bug this whole feature exists for: a document that "ingested fine"
    but has almost no retrievable content, with nothing to explain it."""
    source = tmp_path / "sparse.pdf"
    source.write_bytes(_minimal_pdf_with_text("x"))

    loaded = load_document(source, 26_214_400, ocr=None)

    assert "ocr_skipped" in loaded.metadata
    assert "ocr_applied" not in loaded.metadata


def test_a_sparse_pdf_with_ocr_configured_uses_the_provider(tmp_path):
    source = tmp_path / "sparse.pdf"
    source.write_bytes(_minimal_pdf_with_text("x"))
    ocr = _FakeOcrProvider(script=["Transcribed page content, much longer than the original."])

    loaded = load_document(source, 26_214_400, ocr=ocr)

    assert loaded.metadata["ocr_applied"] is True
    assert "ocr_skipped" not in loaded.metadata
    assert "Transcribed page content" in loaded.text
    assert len(ocr.calls) == 1


def test_a_normal_text_pdf_never_calls_ocr_even_when_configured(tmp_path):
    """The whole point of the sparse-text heuristic: a well-behaved text PDF
    must not pay for a vision-model call it doesn't need."""
    source = tmp_path / "normal.pdf"
    source.write_bytes(_minimal_pdf_with_text("A perfectly ordinary line of real page text."))
    ocr = _FakeOcrProvider()

    loaded = load_document(source, 26_214_400, ocr=ocr)

    assert "ocr_applied" not in loaded.metadata
    assert ocr.calls == []
    assert "perfectly ordinary" in loaded.text


def test_an_ocr_failure_does_not_raise_and_is_counted(tmp_path):
    """One page's OCR call failing must not take the whole ingest down with
    it — but it also must not vanish. See tini/documents/loaders.py's
    _ocr_pdf_pages docstring: this is the same "honest, not silent" contract
    as ocr_skipped, just for a request that WAS attempted and failed."""
    source = tmp_path / "sparse.pdf"
    source.write_bytes(_minimal_pdf_with_text("x"))
    ocr = _FakeOcrProvider(fail_on={0})

    loaded = load_document(source, 26_214_400, ocr=ocr)

    # A single-page document whose only OCR attempt failed: no text was
    # recovered, so the original sparse pypdf extraction is what's left.
    assert loaded.metadata.get("ocr_applied") is not True
    assert len(ocr.calls) == 1


def test_multi_page_ocr_is_stitched_in_page_order(tmp_path):
    """At concurrency=1 (the fake's default) `_ocr_pdf_pages` never fans out,
    so call order is guaranteed to match page order — this pins that the
    render-then-OCR refactor didn't disturb it."""
    source = tmp_path / "sparse.pdf"
    source.write_bytes(_minimal_pdf_with_pages(["1", "2", "3"]))
    ocr = _FakeOcrProvider(script=["FIRST-PAGE-MARKER", "SECOND-PAGE-MARKER", "THIRD-PAGE-MARKER"])

    loaded = load_document(source, 26_214_400, ocr=ocr)

    assert len(ocr.calls) == 3
    first = loaded.text.index("FIRST-PAGE-MARKER")
    second = loaded.text.index("SECOND-PAGE-MARKER")
    third = loaded.text.index("THIRD-PAGE-MARKER")
    assert first < second < third


def test_ocr_pages_run_concurrently_up_to_the_providers_limit(tmp_path):
    """The point of this feature: OCR is a multi-second network round trip
    per page, and pages are independent, so wall-clock time for N pages at
    concurrency=N should look like ~1 round trip, not N of them."""
    source = tmp_path / "sparse.pdf"
    source.write_bytes(_minimal_pdf_with_pages(["1", "2", "3", "4"]))

    class _SlowOcrProvider:
        def __init__(self, concurrency: int):
            self.concurrency = concurrency
            self.calls = 0
            self._lock = threading.Lock()

        def ocr_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
            with self._lock:
                self.calls += 1
            time.sleep(0.1)
            return "transcribed page content, well past the sparse-page threshold"

    serial = _SlowOcrProvider(concurrency=1)
    start = time.monotonic()
    load_document(source, 26_214_400, ocr=serial)
    serial_elapsed = time.monotonic() - start

    concurrent = _SlowOcrProvider(concurrency=4)
    start = time.monotonic()
    loaded = load_document(source, 26_214_400, ocr=concurrent)
    concurrent_elapsed = time.monotonic() - start

    assert serial.calls == 4
    assert concurrent.calls == 4
    assert loaded.metadata["ocr_applied"] is True
    # 4 pages x 0.1s serial ~= 0.4s; concurrency=4 collapses that toward one
    # round trip. Generous threshold (half of serial) to avoid CI flakiness.
    assert concurrent_elapsed < serial_elapsed / 2


def test_failed_pages_are_still_counted_when_run_concurrently(tmp_path):
    source = tmp_path / "sparse.pdf"
    source.write_bytes(_minimal_pdf_with_pages(["1", "2", "3"]))
    ocr = _FakeOcrProvider(fail_on={1}, concurrency=3)

    loaded = load_document(source, 26_214_400, ocr=ocr)

    assert len(ocr.calls) == 3
    assert loaded.metadata.get("ocr_failed_pages") == 1


TINI_TEST_OCR = os.getenv("TINI_TEST_OCR") == "1"


@pytest.mark.skipif(
    not TINI_TEST_OCR, reason="set TINI_TEST_OCR=1 (plus TINI_OCR_* / DEEPSEEK_API_KEY) to include it"
)
def test_real_ocr_extracts_the_known_scanned_document():
    """Opt-in only, same reasoning as the Supabase/mem0 tests in
    test_fact_store_conformance.py: this calls a real vision model and costs
    real money. Exercises the exact document that motivated this feature —
    examples/denso_mock_docs/supply-chain-doc-sustainability-guidelines-en.pdf
    is ~99% image content (pypdf extracts ~17 chars/page from it) and was
    completely unusable for retrieval before OCR existed."""
    from tini.config import load_settings
    from tini.documents.ocr import make_ocr_provider

    settings = load_settings()
    ocr = make_ocr_provider(settings)
    assert ocr is not None, "TINI_TEST_OCR=1 requires TINI_OCR_PROVIDER/TINI_OCR_MODEL configured too"

    source = Path("examples/denso_mock_docs/supply-chain-doc-sustainability-guidelines-en.pdf")
    loaded = load_document(source, 26_214_400, ocr=ocr)

    assert loaded.metadata["ocr_applied"] is True
    assert len(loaded.text) > 5000, "a near-empty result means OCR silently failed to extract much"
    assert "DENSO" in loaded.text
