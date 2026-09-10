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


def test_document_tool_returns_evidence_with_a_source(tmp_path, index, pipeline):
    source = _write(tmp_path / "handbook.md", "Employees rotate credentials every ninety days.")
    pipeline.ingest_path(source)
    registry = build_registry(Settings(home=tmp_path), index)

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
