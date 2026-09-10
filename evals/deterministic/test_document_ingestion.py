"""Document ingestion and retrieval are deterministic and network-free."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents.base import DocumentIndex
from tini.documents.chunking import chunk_text
from tini.documents.loaders import DocumentLoadError, discover_documents
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import SqliteDocumentIndex
from tini.tools import build_registry


@pytest.fixture
def index(tmp_path):
    return SqliteDocumentIndex(connect(tmp_path))


@pytest.fixture
def pipeline(index):
    return IngestionPipeline(index, chunk_size=90, chunk_overlap=15)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


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


def test_explicit_unsupported_file_is_rejected(tmp_path):
    source = _write(tmp_path / "archive.exe", "not a document")
    with pytest.raises(DocumentLoadError, match="unsupported document type"):
        discover_documents(source)


def test_document_tool_returns_evidence_with_a_source(tmp_path, index, pipeline):
    source = _write(tmp_path / "handbook.md", "Employees rotate credentials every ninety days.")
    pipeline.ingest_path(source)
    registry = build_registry(Settings(home=tmp_path), index)

    output = registry.execute("search_documents", {"query": "rotate credentials"})

    assert "handbook#chunk-0" in output
    assert source.resolve().as_uri() in output
    assert set(registry._tools) == {"search_web", "search_documents"}


def test_chunking_has_bounded_overlap_and_stable_ordinals():
    text = "Alpha paragraph has useful context. " * 20
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(left.char_end > right.char_start for left, right in pairwise(chunks))


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (10, -1), (10, 10)])
def test_invalid_chunk_settings_fail_loudly(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=chunk_size, overlap=overlap)
