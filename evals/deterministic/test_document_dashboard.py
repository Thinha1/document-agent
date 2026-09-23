"""Knowledge Base dashboard operations without a browser or network."""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.ops import document_api


@pytest.fixture
def settings(tmp_path):
    return Settings(
        home=tmp_path / "home",
        document_workspace="engineering",
        document_chunk_max_tokens=80,
        document_chunk_min_tokens=10,
        document_chunk_overlap_tokens=8,
        # Pinned rather than left to fall through to os.getenv: this suite
        # must stay green regardless of a contributor's own .env (e.g.
        # TINI_DOCUMENT_INDEX=qdrant / TINI_EMBEDDING_PROVIDER=... configured
        # for real local use, which this repo's own .env now has — see
        # docs/document-ingestion.md). Assertions below are sqlite/no-vector
        # specific (raw document_chunks queries, embedded==0).
        document_index="sqlite",
        embedding_provider="none",
    )


def test_upload_streams_real_stages_and_appears_in_library(settings):
    events = []
    result = document_api.ingest_upload(
        "runbooks/worker.md",
        "engineering",
        b"# Worker recovery\n\nRestart the orion worker with the green command.",
        lambda stage, event: events.append(stage),
        settings,
    )

    assert result["reports"][0]["action"] == "created"
    assert events == [
        "uploaded",
        "discovered",
        "loading",
        "chunking",
        "embedding",
        "indexing",
        "created",
    ]
    library = document_api.documents_payload(settings)
    assert library["items"][0]["title"] == "worker"
    assert library["chunk_count"] == 1


def test_dashboard_can_search_preview_and_remove_a_document(settings):
    document_api.ingest_upload(
        "handbook.md",
        "engineering",
        b"Credentials rotate every ninety days.",
        lambda stage, event: None,
        settings,
    )
    document = document_api.documents_payload(settings)["items"][0]

    hits = document_api.search(
        {"query": "rotate credentials", "workspace": "engineering"}, settings
    )["hits"]
    preview = document_api.chunks(document["document_id"], "engineering", settings=settings)
    removed = document_api.action(
        {"action": "delete", "document_id": document["document_id"], "workspace": "engineering"},
        settings,
    )

    assert hits[0]["title"] == "handbook"
    assert preview["chunks"][0]["text"] == "Credentials rotate every ninety days."
    assert preview["chunks"][0]["embedded"] == 0
    assert removed == {"ok": True}
    assert document_api.documents_payload(settings)["items"] == []


def test_reindex_reads_only_the_source_recorded_for_the_document(settings):
    document_api.ingest_upload(
        "policy.md",
        "engineering",
        b"The old zephyr policy applies.",
        lambda stage, event: None,
        settings,
    )
    document = document_api.documents_payload(settings)["items"][0]
    source = settings.home / "documents" / "engineering" / "policy.md"
    source.write_text("The new kestrel policy applies.", encoding="utf-8")

    result = document_api.action(
        {"action": "reindex", "document_id": document["document_id"], "workspace": "engineering"},
        settings,
    )

    assert result["reports"][0]["action"] == "updated"
    assert document_api.search({"query": "kestrel", "workspace": "engineering"}, settings)["hits"]


@pytest.mark.parametrize("name", ["../secret.md", "/absolute.md", "folder/../../secret.md"])
def test_upload_rejects_paths_that_escape_the_managed_folder(settings, name):
    with pytest.raises(ValueError, match="safe relative path"):
        document_api.ingest_upload(name, "engineering", b"secret", lambda *args: None, settings)


@pytest.mark.parametrize("workspace", ["", "../finance", "contains spaces", "x" * 65])
def test_workspace_names_are_constrained(settings, workspace):
    if workspace == "":
        workspace = "../bad"
    with pytest.raises(ValueError, match="workspace"):
        document_api.ingest_upload("safe.md", workspace, b"text", lambda *args: None, settings)


def test_oversized_upload_is_rejected_before_writing(settings):
    settings.document_max_file_bytes = 3

    result = document_api.ingest_upload(
        "large.md", "engineering", b"four", lambda *args: None, settings
    )

    assert "limit is 3 bytes" in result["error"]
    assert not (settings.home / "documents").exists()


def test_managed_upload_path_stays_under_tini_home(settings):
    document_api.ingest_upload(
        "nested/guide.md", "engineering", b"safe text", lambda *args: None, settings
    )
    target = settings.home / "documents" / "engineering" / "nested" / "guide.md"
    assert target.read_text(encoding="utf-8") == "safe text"
