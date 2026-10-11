"""The document catalog: classification, locking of manual edits, and the
guarantee that editing or re-indexing costs (almost) nothing. Network-free."""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents import catalog
from tini.documents.embeddings import CachingEmbedder
from tini.ops import document_api

# --- classification -----------------------------------------------------------

def test_a_document_is_named_by_its_own_opening_not_by_what_it_mentions():
    # The Integrated Report's cover says so; a later sentence merely MENTIONS the
    # securities report. Picking the most confident match got this wrong.
    text = "統合報告書2025 2025年3月期 代表取締役副社長 ... 詳細は有価証券報告書をご覧ください。"

    result = catalog.classify("annual-report-doc-2025-viewing-ja", text)

    assert (result["doc_type"], result["subtype"]) == ("report", "thường niên tích hợp")
    assert result["period"] == "FY2024", "a fiscal year ended March 2025 is FY2024"
    assert "統合報告書" in result["evidence"]


def test_a_governance_report_that_mentions_the_integrated_report_stays_governance():
    text = "1 Corporate Governance Report CORPORATE GOVERNANCE DENSO CORPORATION Last Update: June 13, 2025 ... see the Integrated Report"

    result = catalog.classify("management-doc-corporate-governance-2025-en", text)

    assert (result["doc_type"], result["period"]) == ("governance", "2025")


def test_a_title_broken_across_lines_still_matches_and_full_width_digits_are_read():
    text = "DENSO TEN AMERICA Limited EXTERNAL PROVIDER QUALITY ASSURANCE\nMANUAL T-QA029 Rev. I April 2018"
    filing = "【提出書類】 有価証券報告書 【事業年度】 第100期(自 ２０２２年４月１日 至 ２０２３年３月３１日)"

    manual = catalog.classify("denso-ten_qa_manual", text)
    report = catalog.classify("edinet", filing)

    assert (manual["doc_type"], manual["issuer"], manual["period"]) == ("supplier", "DENSO TEN America", "2018")
    assert (report["doc_type"], report["period"]) == ("report", "FY2022")


def test_readme_is_excluded_and_an_unrecognised_document_is_flagged_not_guessed():
    assert catalog.classify("README", "how to use this folder")["status"] == "excluded"

    unknown = catalog.classify("meeting-notes", "Notes from the Tuesday sync about lunch.")

    assert unknown["doc_type"] == "other" and unknown["confidence"] < catalog.REVIEW_BELOW


def test_the_profile_reads_language_scan_state_and_table_density():
    narrative = catalog.profile_document("これは日本語の文章です。" * 20, {"page_count": 4}, "application/pdf", 10)
    scanned = catalog.profile_document("x", {"ocr_applied": True}, "application/pdf", 10)
    table = catalog.profile_document("\n".join(f"売上 {i} 1,000 2,000 3,000" for i in range(20)), {}, "", 0)
    vietnamese = catalog.profile_document("Quy định về an toàn lao động và sức khoẻ của nhân viên", {}, "", 0)
    english = catalog.profile_document("Employees must report any suspected violation promptly.", {}, "", 0)

    assert narrative["language"] == "ja" and narrative["pages"] == 4 and not narrative["scanned"]
    assert scanned["scanned"] is True
    assert table["table_heavy"] is True and narrative["table_heavy"] is False
    assert vietnamese["language"] == "vi" and english["language"] == "en"


# --- storage, locking ---------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    (tmp_path / "home").mkdir()
    connection = connect(tmp_path / "home")
    yield connection
    connection.close()


def _entry(text="Code of Conduct for associates", content_hash="h1", title="conduct"):
    return {"workspace": "w", "source_uri": "file:///conduct.pdf", "title": title,
            "content_hash": content_hash, "text": text, "profile": {"pages": 3}}


def test_recording_a_new_document_flags_low_confidence_for_review(conn):
    sure = catalog.record_ingest(conn, _entry())
    unsure = catalog.record_ingest(conn, {**_entry("nothing recognisable"), "source_uri": "file:///x.pdf", "title": "x"})

    assert (sure["doc_type"], sure["needs_review"]) == ("policy", False)
    assert (unsure["doc_type"], unsure["needs_review"]) == ("other", True)


def test_an_edit_is_validated_locked_and_marks_the_entry_reviewed(conn):
    catalog.record_ingest(conn, {**_entry("nothing recognisable"), "title": "x"})

    edited = catalog.update(conn, "w", "file:///conduct.pdf", {"doc_type": "policy", "period": "2024"})

    assert (edited["doc_type"], edited["period"], edited["needs_review"]) == ("policy", "2024", False)
    assert set(edited["locked"]) == {"doc_type", "period"}
    with pytest.raises(catalog.CatalogError, match="doc_type"):
        catalog.update(conn, "w", "file:///conduct.pdf", {"doc_type": "invoice"})
    with pytest.raises(catalog.CatalogError, match="status"):
        catalog.update(conn, "w", "file:///conduct.pdf", {"status": "maybe"})
    with pytest.raises(catalog.CatalogError, match="Cannot edit"):
        catalog.update(conn, "w", "file:///conduct.pdf", {"confidence": 1})
    with pytest.raises(catalog.CatalogError, match="no catalog entry"):
        catalog.update(conn, "w", "file:///missing.pdf", {"period": "1"})


def test_unchanged_content_is_left_alone_and_a_changed_file_never_overwrites_a_locked_field(conn):
    catalog.record_ingest(conn, _entry("nothing recognisable", title="x"))
    catalog.update(conn, "w", "file:///conduct.pdf", {"doc_type": "supplier", "summary": "my note"})

    same = catalog.record_ingest(conn, _entry("Code of Conduct for associates", title="x"))
    assert same["doc_type"] == "supplier", "same hash: not even re-classified"

    changed = catalog.record_ingest(conn, _entry("Code of Conduct, revised. 2025年3月期", content_hash="h2", title="x"))
    assert changed["doc_type"] == "supplier", "the edited field survives a changed file"
    assert changed["summary"] == "my note"
    assert changed["period"] == "FY2024", "an unlocked field is refreshed"
    assert changed["needs_review"] is True, "an auto field changed under the person: ask them"


def test_deleting_removes_the_entry(conn):
    catalog.record_ingest(conn, _entry())

    catalog.delete(conn, "w", "file:///conduct.pdf")

    assert catalog.get(conn, "w", "file:///conduct.pdf") is None


# --- through the dashboard API -------------------------------------------------

class _Counting:
    provider, model, dimensions = "fake", "fake-model", 3

    def __init__(self):
        self.embedded = 0

    def embed_documents(self, texts, *, title):
        self.embedded += len(texts)
        return [(float(len(t)), 1.0, 0.5) for t in texts]

    def embed_query(self, text):
        return (1.0, 1.0, 1.0)


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = Settings(
        home=tmp_path / "home", document_workspace="w", document_index="sqlite",
        embedding_provider="none", rerank_provider="none", document_storage="local",
    )
    counting = _Counting()
    cache = CachingEmbedder(counting, tmp_path / "home" / "embed_cache.sqlite")
    monkeypatch.setattr(document_api, "make_embedding_provider", lambda _s: cache)
    return settings, counting


def _upload(settings, name="conduct.md", body=b"# Code of Conduct\n\nAssociates must report violations.\n\n" * 3):
    return document_api.ingest_upload(name, "w", body, lambda *a: None, settings)


def test_uploading_catalogues_the_document_and_the_payload_carries_it(api):
    settings, _ = api

    _upload(settings)
    payload = document_api.documents_payload(settings)

    (uri, entry), = payload["catalog"].items()
    assert entry["doc_type"] == "policy" and uri.startswith("file://")
    assert set(payload["catalog_options"]["types"]) == set(catalog.DOC_TYPES)


def test_editing_the_catalog_reingests_nothing(api, monkeypatch):
    settings, counting = api
    _upload(settings)
    (item,) = document_api.documents_payload(settings)["items"]
    embedded_before = counting.embedded
    monkeypatch.setattr(document_api, "IngestionPipeline", lambda *a, **k: pytest.fail("edit must not re-ingest"))

    result = document_api.catalog_update(
        {"document_id": item["document_id"], "workspace": "w", "fields": {"period": "FY2024", "status": "superseded"}},
        settings,
    )

    assert result["entry"]["period"] == "FY2024" and result["entry"]["status"] == "superseded"
    assert counting.embedded == embedded_before


def test_reindexing_after_a_manual_edit_costs_no_embeddings_and_keeps_the_edit(api):
    settings, counting = api
    _upload(settings)
    (item,) = document_api.documents_payload(settings)["items"]
    document_api.catalog_update(
        {"document_id": item["document_id"], "workspace": "w", "fields": {"doc_type": "supplier", "summary": "checked"}},
        settings,
    )
    embedded_before = counting.embedded

    result = document_api.action({"action": "reindex", "document_id": item["document_id"], "workspace": "w"}, settings)

    assert result["reports"][0]["action"] == "skipped"
    assert counting.embedded == embedded_before, "unchanged file: every vector came from the cache"
    (entry,) = document_api.documents_payload(settings)["catalog"].values()
    assert (entry["doc_type"], entry["summary"]) == ("supplier", "checked")


def test_a_lightly_edited_file_embeds_only_the_changed_chunks(api):
    settings, counting = api
    paragraphs = [f"# Section {n}\n\n" + f"Associates must follow rule number {n} carefully. " * 12 for n in range(8)]
    _upload(settings, body="\n\n".join(paragraphs).encode())
    first = counting.embedded
    paragraphs[4] += "\n\nA sentence added by hand."

    _upload(settings, body="\n\n".join(paragraphs).encode())

    assert 0 < counting.embedded - first < first, "only the edited region was embedded again"


def test_refresh_catalogues_documents_indexed_before_the_catalog_existed(api, tmp_path):
    settings, counting = api
    _upload(settings)
    conn = connect(settings.home)
    try:
        conn.execute("DELETE FROM document_catalog")
        conn.commit()
    finally:
        conn.close()
    embedded_before = counting.embedded

    summary = document_api.catalog_refresh({"workspace": "w"}, settings)

    assert summary["documents"] == 1 and summary["catalogued"] == 1
    assert counting.embedded == embedded_before, "cataloguing embeds nothing"
    assert len(document_api.documents_payload(settings)["catalog"]) == 1


def test_removing_a_document_removes_its_catalog_entry(api):
    settings, _ = api
    _upload(settings)
    (item,) = document_api.documents_payload(settings)["items"]

    document_api.action({"action": "delete", "document_id": item["document_id"], "workspace": "w"}, settings)

    assert document_api.documents_payload(settings)["catalog"] == {}
