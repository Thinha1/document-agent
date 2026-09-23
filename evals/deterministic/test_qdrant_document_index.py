"""QdrantDocumentIndex held to the same DocumentIndex contract as SQLite.

Runs entirely against qdrant-client's in-process local mode (`location=":memory:"`)
— no Docker, no network, same "deterministic evals stay network-free" rule as
everywhere else in this suite (see test_fact_store_conformance.py for the
sibling suite this one is modeled on: one backend silently missing behavior
the interface promises is exactly the bug class both files exist to catch).
"""

from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from tini.config import Settings  # noqa: E402
from tini.documents.base import DocumentIndex  # noqa: E402
from tini.documents.pipeline import IngestionPipeline  # noqa: E402
from tini.documents.qdrant_store import QdrantDocumentIndex  # noqa: E402


class _FakeEmbedder:
    provider = "fake"
    model = "semantic-v1"
    dimensions = 3

    def embed_documents(self, texts, *, title):
        return [(1.0, 0.0, 0.0) if "compensation" in text else (0.0, 1.0, 0.0) for text in texts]

    def embed_query(self, text):
        return (1.0, 0.0, 0.0) if "salary" in text else (0.0, 1.0, 0.0)


@pytest.fixture
def index():
    client = qdrant_client.QdrantClient(location=":memory:")
    return QdrantDocumentIndex(Settings(), _FakeEmbedder(), client=client)


@pytest.fixture
def pipeline(index):
    return IngestionPipeline(index, chunk_max_tokens=20, chunk_min_tokens=5, chunk_overlap_tokens=4)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_qdrant_adapter_satisfies_the_document_index_interface(index):
    assert isinstance(index, DocumentIndex)


def test_construction_without_an_embedder_fails_loudly():
    """Qdrant has no lexical fallback (see the module docstring) — an index
    that quietly accepted embedder=None would ingest chunks no query can ever
    find, the exact silent-wrong-answer failure this codebase keeps naming."""
    with pytest.raises(ValueError, match="requires embeddings"):
        QdrantDocumentIndex(Settings(), None)


def test_ingest_is_searchable_and_unchanged_files_are_skipped(tmp_path, index, pipeline):
    source = _write(tmp_path / "benefits.md", "Employee compensation is reviewed each year. " * 3)

    first = pipeline.ingest_path(source)[0]
    second = pipeline.ingest_path(source)[0]

    assert first.action == "created"
    assert first.chunk_count >= 1
    assert second.action == "skipped"
    assert second.document_id == first.document_id
    assert len(index.list_documents()) == 1
    hits = index.search("salary")
    assert hits and hits[0].title == "benefits"
    assert hits[0].source_uri == source.resolve().as_uri()


def test_changed_source_replaces_old_chunks_without_orphans(tmp_path, index, pipeline):
    source = _write(tmp_path / "runbook.txt", "Employee compensation review process alpha beta.")
    created = pipeline.ingest_path(source)[0]
    _write(source, "Employee compensation review process alpha beta gamma delta epsilon zeta.")

    updated = pipeline.ingest_path(source)[0]

    assert updated.action == "updated"
    assert updated.document_id == created.document_id
    # A leftover point from the prior version — a different content_hash, so a
    # different point id — would inflate this past what was just written.
    assert index._count(updated.document_id) == updated.chunk_count
    assert index.list_documents()[0].chunk_count == updated.chunk_count


def test_workspace_filter_is_applied_inside_retrieval(tmp_path, index, pipeline):
    source = _write(tmp_path / "private.md", "Project kestrel salary details are confidential.")
    pipeline.ingest_path(source, workspace_id="engineering")

    assert index.search("salary", workspace_id="engineering")
    assert index.search("salary", workspace_id="finance") == []


def test_delete_document_removes_all_its_chunks(tmp_path, index, pipeline):
    source = _write(tmp_path / "handbook.md", "Salary compensation details for new hires. " * 3)
    report = pipeline.ingest_path(source)[0]

    assert index.delete_document(report.document_id) is True
    assert index.search("salary") == []
    assert index.list_documents() == []


def test_deleting_an_unknown_document_returns_false(index):
    assert index.delete_document("does-not-exist") is False


def test_a_chunk_missing_an_embedding_is_refused_not_silently_dropped(tmp_path, index):
    from dataclasses import replace

    from tini.documents.chunking import chunk_text
    from tini.documents.loaders import load_document
    from tini.documents.models import DocumentSource

    source = _write(tmp_path / "note.txt", "A short note about salary.")
    loaded = load_document(source, 26_214_400)
    chunks = chunk_text(loaded.text, 512, 5, 4)
    unembedded = [replace(c, embedding=None) for c in chunks]

    with pytest.raises(ValueError, match="no embedding"):
        index.upsert_document(
            DocumentSource(
                workspace_id="default",
                source_uri=loaded.source_uri,
                title=loaded.title,
                media_type=loaded.media_type,
                size_bytes=loaded.size_bytes,
                content_hash=loaded.content_hash,
            ),
            unembedded,
        )


def test_list_documents_reports_chunk_counts_newest_first(tmp_path, index, pipeline):
    older = _write(tmp_path / "older.md", "Compensation policy alpha. " * 3)
    pipeline.ingest_path(older)
    newer = _write(tmp_path / "newer.md", "Compensation policy beta. " * 3)
    pipeline.ingest_path(newer)

    docs = index.list_documents()

    assert [d.title for d in docs] == ["newer", "older"]
    assert all(d.chunk_count >= 1 and d.status == "ready" for d in docs)
