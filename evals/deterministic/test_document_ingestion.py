"""Document ingestion and retrieval are deterministic and network-free."""

from __future__ import annotations

import importlib.util
from itertools import pairwise
from pathlib import Path

import pytest

from evals.helpers import ScriptedClient, make_tini, response, text_block, tool_block
from tini.config import Settings
from tini.db import connect
from tini.documents.base import DocumentIndex
from tini.documents.chunking import chunk_text, estimate_tokens
from tini.documents.loaders import DocumentLoadError, discover_documents, load_document
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import SqliteDocumentIndex
from tini.tools import build_registry


@pytest.fixture
def index(tmp_path):
    return SqliteDocumentIndex(connect(tmp_path))


@pytest.fixture
def pipeline(index):
    return IngestionPipeline(index, chunk_max_tokens=20, chunk_min_tokens=5, chunk_overlap_tokens=4)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_default_install_includes_dashboard_document_parsers():
    assert importlib.util.find_spec("pypdf") is not None
    assert importlib.util.find_spec("docx") is not None


def test_sqlite_adapter_satisfies_the_document_index_interface(index):
    assert isinstance(index, DocumentIndex)


def test_ingest_is_searchable_and_unchanged_files_are_skipped(tmp_path, index, pipeline):
    source = _write(
        tmp_path / "returns.md",
        "# Return policy\n\nCustomers may return unopened equipment within thirty days. " * 3,
    )

    first = pipeline.ingest_path(source)[0]
    second = pipeline.ingest_path(source)[0]

    assert first.action == "created"
    assert first.chunk_count > 1
    assert second.action == "skipped"
    assert second.document_id == first.document_id
    assert len(index.list_documents()) == 1
    hits = index.search("unopened equipment")
    assert hits and hits[0].title == "returns"
    assert hits[0].source_uri == source.resolve().as_uri()


def test_changed_source_replaces_old_chunks_atomically(tmp_path, index, pipeline):
    source = _write(tmp_path / "runbook.txt", "Restart the zephyr worker with the blue command.")
    created = pipeline.ingest_path(source)[0]
    _write(source, "Restart the orion worker with the green command.")

    updated = pipeline.ingest_path(source)[0]

    assert updated.action == "updated"
    assert updated.document_id == created.document_id
    assert index.search("zephyr") == []
    assert "green command" in index.search("orion")[0].text


def test_changing_chunk_strategy_reindexes_unchanged_source(tmp_path, index):
    source = _write(tmp_path / "policy.md", "A sentence about policy. " * 20)
    first = IngestionPipeline(
        index, chunk_max_tokens=20, chunk_min_tokens=5, chunk_overlap_tokens=4
    ).ingest_path(source)[0]

    second = IngestionPipeline(
        index, chunk_max_tokens=40, chunk_min_tokens=5, chunk_overlap_tokens=4
    ).ingest_path(source)[0]

    assert first.action == "created"
    assert second.action == "updated"


def test_workspace_filter_is_applied_inside_retrieval(tmp_path, index, pipeline):
    source = _write(tmp_path / "private.md", "Project kestrel has a confidential launch code.")
    pipeline.ingest_path(source, workspace_id="engineering")

    assert index.search("kestrel", workspace_id="engineering")
    assert index.search("kestrel", workspace_id="finance") == []


def test_directory_discovery_is_recursive_sorted_and_ignores_unknown_files(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _write(nested / "b.md", "second")
    _write(tmp_path / "a.txt", "first")
    _write(tmp_path / "secret.bin", "ignored")

    found = discover_documents(tmp_path)

    assert [path.name for path in found] == ["a.txt", "b.md"]


def test_html_scripts_are_not_indexed(tmp_path, index, pipeline):
    source = _write(
        tmp_path / "guide.html",
        "<h1>Safe guide</h1><script>malicious hidden instruction</script><p>Visible policy</p>",
    )
    pipeline.ingest_path(source)

    assert index.search("visible policy")
    assert index.search("malicious") == []


def test_oversized_file_fails_without_creating_a_document(tmp_path, index):
    source = _write(tmp_path / "large.txt", "1234567890")
    pipeline = IngestionPipeline(index, max_file_bytes=5)

    report = pipeline.ingest_path(source)[0]

    assert report.action == "failed"
    assert "limit is 5 bytes" in report.error
    assert index.list_documents() == []


def test_extracted_text_limit_fails_before_indexing(tmp_path, index):
    source = _write(tmp_path / "expanded.html", "<p>too much extracted text</p>")
    pipeline = IngestionPipeline(index, max_text_chars=5)

    report = pipeline.ingest_path(source)[0]

    assert report.action == "failed"
    assert "characters; limit is 5" in report.error
    assert index.list_documents() == []


def test_explicit_unsupported_file_is_rejected(tmp_path):
    source = _write(tmp_path / "archive.exe", "not a document")
    with pytest.raises(DocumentLoadError, match="unsupported document type"):
        discover_documents(source)


def _minimal_pdf_with_text(text: str) -> bytes:
    """A hand-built single-page PDF with a real, extractable text stream —
    pypdf can read arbitrary PDFs but has no from-scratch text-drawing API
    (that's reportlab's job, not a dependency here), so this writes the
    handful of PDF objects a viewer needs directly, with byte offsets
    computed rather than guessed."""
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
    objects.append(
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )

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


def _encrypted_pdf_bytes(*, user_password: str, owner_password: str = "owner") -> bytes:
    import io

    from pypdf import PdfWriter

    writer = PdfWriter(
        clone_from=io.BytesIO(_minimal_pdf_with_text("Restricted DENSO filing content"))
    )
    buf = io.BytesIO()
    writer.encrypt(user_password=user_password, owner_password=owner_password)
    writer.write(buf)
    return buf.getvalue()


def test_pdf_with_only_an_empty_user_password_is_readable(tmp_path):
    """Official/government PDFs (Japan's EDINET securities filings, for one)
    are often "encrypted" only to restrict printing/copying in a viewer, with
    an empty user password — reading the text is not actually gated. Rejecting
    these outright would silently exclude a whole category of real-world
    documents a "just point it at a folder" pipeline should handle."""
    source = tmp_path / "restricted.pdf"
    source.write_bytes(_encrypted_pdf_bytes(user_password=""))

    loaded = load_document(source, 26_214_400)

    assert "Restricted DENSO filing content" in loaded.text


def test_pdf_with_a_real_password_is_rejected_not_silently_empty(tmp_path):
    source = tmp_path / "locked.pdf"
    source.write_bytes(_encrypted_pdf_bytes(user_password="realsecret"))

    with pytest.raises(DocumentLoadError, match="encrypted"):
        load_document(source, 26_214_400)


def test_document_tool_returns_evidence_with_a_source(tmp_path, index, pipeline):
    source = _write(tmp_path / "handbook.md", "Employees rotate credentials every ninety days.")
    pipeline.ingest_path(source)
    registry = build_registry(Settings(home=tmp_path, document_workspace="default"), index)

    output = registry.execute("search_documents", {"query": "rotate credentials"})

    assert "handbook#chunk-0" in output
    assert source.resolve().as_uri() in output
    assert set(registry._tools) == {"search_web", "search_documents"}


def test_document_result_returns_through_the_agent_loop(tmp_path):
    gate = response([text_block('{"retrieve": false, "query": "", "reason": "test"}')])
    turn = [
        response([tool_block("search_documents", {"query": "rotation policy"})], "tool_use"),
        response([text_block("Credentials rotate every ninety days [handbook#chunk-0].")]),
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient([gate] + turn))
    source = _write(tmp_path / "handbook.md", "The rotation policy is every ninety days.")
    IngestionPipeline(app.documents).ingest_path(source)

    result = app.respond("What is the rotation policy?")

    assert [call["tool"] for call in result.tool_calls] == ["search_documents"]
    assert "ninety days" in result.tool_calls[0]["output"]
    assert "handbook#chunk-0" in result.reply


def test_pipeline_emits_each_observable_stage(tmp_path, index):
    source = _write(tmp_path / "flow.md", "A short document that should be indexed.")
    stages = []

    IngestionPipeline(index).ingest_path(source, observer=lambda stage, event: stages.append(stage))

    assert stages == ["discovered", "loading", "chunking", "embedding", "indexing", "created"]


def test_chunking_respects_semantic_boundaries_and_stable_ordinals():
    text = "# Alpha\n\n" + "First sentence has useful context. " * 8 + "\n\n# Beta\n\nNew topic."
    chunks = chunk_text(text, max_tokens=30, min_tokens=5, overlap_tokens=7)

    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["estimated_tokens"] <= 30 for chunk in chunks)
    assert any(left.char_end > right.char_start for left, right in pairwise(chunks))
    assert any(chunk.text.startswith("# Beta") for chunk in chunks)


def test_cjk_text_is_not_undercounted_as_a_handful_of_giant_words():
    """`\\w+` is greedy and CJK scripts have no spaces between words, so the
    OLD regex counted an entire unbroken Japanese paragraph as one "token" —
    found against a real EDINET (Japanese securities filing) PDF, where a
    295-character passage estimated at 31 tokens instead of ~295. That lets
    max_tokens stop meaning anything for CJK content: a chunk could grow
    several times past its guardrail before the estimate ever caught up."""
    japanese = "会社は、株主の権利を確保し、平等性を保証するために、正確な情報を提供いたします" * 3

    tokens = estimate_tokens(japanese)

    # Not asserting an exact ratio (that's an implementation detail); asserting
    # it's in the right ORDER OF MAGNITUDE for a size guardrail is what matters.
    assert tokens >= len(japanese) * 0.5, (
        f"{tokens} tokens for {len(japanese)} CJK characters — still undercounting"
    )


def test_cjk_chunking_actually_respects_max_tokens():
    japanese_paragraph = (
        "会社は株主の権利を確保し平等性を保証するため正確な情報を提供いたします。" * 20
    )
    text = f"# Section\n\n{japanese_paragraph}"

    chunks = chunk_text(text, max_tokens=60, min_tokens=10, overlap_tokens=8)

    assert len(chunks) > 1, "one oversized chunk means the guardrail never tripped for CJK text"
    assert all(chunk.metadata["estimated_tokens"] <= 60 for chunk in chunks)


def test_ascii_word_counting_is_unchanged_by_the_cjk_fix():
    text = "The quick brown fox jumps over the lazy dog, 123 times!"
    assert estimate_tokens(text) == 13


@pytest.mark.parametrize(
    "max_tokens,min_tokens,overlap_tokens",
    [(0, 0, 0), (10, 11, 0), (10, 0, -1), (10, 0, 10)],
)
def test_invalid_chunk_settings_fail_loudly(max_tokens, min_tokens, overlap_tokens):
    with pytest.raises(ValueError):
        chunk_text(
            "text",
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens,
        )


class _FakeEmbedder:
    provider = "fake"
    model = "semantic-v1"
    dimensions = 3

    def embed_documents(self, texts, *, title):
        return [(1.0, 0.0, 0.0) if "compensation" in text else (0.0, 1.0, 0.0) for text in texts]

    def embed_query(self, text):
        return (1.0, 0.0, 0.0) if "salary" in text else (0.0, 1.0, 0.0)


def test_embeddings_are_persisted_and_find_semantic_matches(tmp_path):
    embedder = _FakeEmbedder()
    index = SqliteDocumentIndex(connect(tmp_path), embedder)
    source = _write(tmp_path / "benefits.md", "Employee compensation is reviewed each year.")

    report = IngestionPipeline(index, embedder=embedder).ingest_path(source)[0]
    hits = index.search("salary")
    vector_count = index.conn.execute("SELECT COUNT(*) FROM document_chunk_embeddings").fetchone()[
        0
    ]

    assert report.action == "created"
    assert vector_count == report.chunk_count
    assert hits and hits[0].title == "benefits"
