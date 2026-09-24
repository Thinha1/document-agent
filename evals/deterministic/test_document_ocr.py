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


class _FakeOcrProvider:
    """Records every image it's asked to transcribe and returns a scripted
    answer per call — no network, no model, deterministic."""

    def __init__(self, script: list[str] | None = None, fail_on: set[int] | None = None):
        self.calls: list[bytes] = []
        self._script = list(script) if script is not None else None
        self._fail_on = fail_on or set()

    def ocr_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
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
