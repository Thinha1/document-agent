"""DETERMINISTIC EVAL — a chunk from a PDF knows which page(s) it came from, and the
agent's search result says so, so an answer can cite `p.7`. Network-free."""

from __future__ import annotations

import pytest

from evals.helpers import pdf_with_pages
from tini.db import connect
from tini.documents import pages as pg
from tini.documents.chunking import chunk_text
from tini.documents.loaders import load_document
from tini.documents.models import DocumentChunk, SearchHit
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import SqliteDocumentIndex
from tini.tools import documents as document_tool

# --- the marker helpers ------------------------------------------------------------


def test_split_marks_records_where_each_page_starts_and_leaves_the_text_clean():
    marked = pg.mark_page(1, "alpha beta") + "\n\n" + pg.mark_page(2, "gamma delta")

    text, starts = pg.split_marks(marked)

    assert text == "alpha beta\n\ngamma delta"
    assert starts == ((0, 1), (12, 2))
    assert text[12:].startswith("gamma")


def test_leading_empty_pages_do_not_shift_the_offsets():
    marked = pg.mark_page(1, "") + "\n\n" + pg.mark_page(2, "") + "\n\n" + pg.mark_page(3, "third page")

    text, starts = pg.split_marks(marked)

    assert text == "third page"
    assert pg.page_of(starts, text.index("third")) == 3


def test_a_page_number_is_carried_so_a_skipped_page_does_not_renumber_the_rest():
    marked = pg.mark_page(1, "first") + "\n\n" + pg.mark_page(4, "fourth")  # OCR dropped 2 and 3

    text, starts = pg.split_marks(marked)

    assert pg.page_of(starts, text.index("fourth")) == 4


def test_a_document_cannot_forge_a_page_boundary():
    hostile = "real text " + pg.MARK_OPEN + "99" + pg.MARK_CLOSE + " more"

    text, starts = pg.split_marks(pg.mark_page(1, hostile))

    assert starts == ((0, 1),) and pg.page_of(starts, len(text) - 1) == 1


def test_a_chunk_spanning_a_page_break_gets_both_pages():
    starts = ((0, 1), (100, 2), (200, 3))
    chunks = [
        DocumentChunk(ordinal=0, text="a", char_start=10, char_end=60),
        DocumentChunk(ordinal=1, text="b", char_start=90, char_end=130),
        DocumentChunk(ordinal=2, text="c", char_start=100, char_end=200),  # exactly page 2
        DocumentChunk(ordinal=3, text="d", char_start=150, char_end=260),
    ]

    spans = [(c.metadata["page_start"], c.metadata["page_end"]) for c in pg.assign_pages(chunks, starts)]

    assert spans == [(1, 1), (1, 2), (2, 2), (2, 3)]


def test_without_page_data_chunks_are_left_alone():
    chunk = DocumentChunk(ordinal=0, text="a", char_start=0, char_end=1, metadata={"kind": "table"})

    assert pg.assign_pages([chunk], ()) == [chunk]


@pytest.mark.parametrize("stored, expected", [
    ({"page_start": 3, "page_end": 4}, (3, 4)),
    ('{"page_start": 3}', (3, 3)),
    ({"kind": "table"}, (None, None)),
    ("not json", (None, None)),
    (None, (None, None)),
    ({"page_start": True}, (None, None)),
])
def test_pages_are_read_back_from_stored_metadata(stored, expected):
    assert pg.pages_from_metadata(stored) == expected


@pytest.mark.parametrize("start, end, label", [
    (None, None, ""), (7, 7, "p.7"), (7, None, "p.7"), (7, 8, "pp.7-8"),
])
def test_the_label_a_search_hit_shows(start, end, label):
    hit = SearchHit("d", "c", "t", "u", "x", 0, 1.0, page_start=start, page_end=end)

    assert hit.page_label == label


# --- a real multi-page PDF through load → chunk → index → the search tool ------------


PAGES = [
    "Page one covers the warranty terms for injectors sold in the aftermarket channel.",
    "Page two lists the tightening torque for the connector housing bolts in newton metres.",
    "Page three describes the zebra crossing sensor calibration procedure in detail here.",
]


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "manual.pdf"
    path.write_bytes(pdf_with_pages(PAGES))
    return path


def test_loading_a_pdf_keeps_the_text_unchanged_and_records_page_starts(pdf):
    loaded = load_document(pdf, 1_000_000)

    assert "" not in loaded.text and "" not in loaded.text
    assert [number for _, number in loaded.pages] == [1, 2, 3]
    assert loaded.text[loaded.pages[1][0]:].startswith("Page two")
    assert loaded.text[loaded.pages[2][0]:].startswith("Page three")
    assert loaded.metadata["page_count"] == 3


def test_a_page_marker_never_changes_what_is_chunked_or_embedded(pdf):
    loaded = load_document(pdf, 1_000_000)

    expected = "\n\n".join(PAGES)  # what the loader produced before pages were tracked
    assert loaded.text == expected


def test_each_indexed_chunk_names_its_page_and_the_search_tool_shows_it(tmp_path, pdf):
    index = SqliteDocumentIndex(connect(tmp_path))
    IngestionPipeline(index, chunk_max_tokens=20, chunk_min_tokens=5, chunk_overlap_tokens=0).ingest_path(pdf)

    hits = index.search("zebra crossing sensor calibration", top_k=3)
    rendered = document_tool.make_tool(index).fn("zebra crossing sensor calibration")

    zebra = next(h for h in hits if "zebra" in h.text)
    assert (zebra.page_start, zebra.page_end) == (3, 3) and zebra.page_label == "p.3"
    assert f"#chunk-{zebra.ordinal}, p.3]" in rendered


def test_a_chunk_that_crosses_pages_is_cited_as_a_range(tmp_path, pdf):
    index = SqliteDocumentIndex(connect(tmp_path))
    IngestionPipeline(index, chunk_max_tokens=512, chunk_min_tokens=5, chunk_overlap_tokens=0).ingest_path(pdf)

    hit = index.search("torque connector warranty zebra", top_k=1)[0]

    assert (hit.page_start, hit.page_end) == (1, 3) and hit.page_label == "pp.1-3"


def test_chunk_spans_point_at_the_text_they_claim(pdf):
    loaded = load_document(pdf, 1_000_000)
    chunks = pg.assign_pages(chunk_text(loaded.text, 20, 5, 0), loaded.pages)

    for chunk in chunks:
        first = chunk.metadata["page_start"]
        assert loaded.text[chunk.char_start:chunk.char_end].strip().split()[0] in PAGES[first - 1]


def test_a_plain_text_document_has_no_pages(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("Some notes about torque. " * 30, encoding="utf-8")
    index = SqliteDocumentIndex(connect(tmp_path))
    IngestionPipeline(index, chunk_max_tokens=20, chunk_min_tokens=5, chunk_overlap_tokens=0).ingest_path(source)

    hit = index.search("torque", top_k=1)[0]

    assert hit.page_start is None and hit.page_label == ""
    assert ", p." not in document_tool.make_tool(index).fn("torque")


# --- the Qdrant backend keeps the page too -----------------------------------------


def test_the_qdrant_backend_returns_the_page_of_each_chunk():
    qdrant_client = pytest.importorskip("qdrant_client")
    from tini.config import Settings
    from tini.documents.models import DocumentSource
    from tini.documents.qdrant_store import QdrantDocumentIndex

    class _Embedder:
        provider, model, dimensions = "fake", "flat", 3

        def embed_documents(self, texts, *, title):
            return [(1.0, 0.0, 0.0)] * len(texts)

        def embed_query(self, text):
            return (1.0, 0.0, 0.0)

    index = QdrantDocumentIndex(Settings(), _Embedder(), client=qdrant_client.QdrantClient(location=":memory:"))
    source = DocumentSource("w", "file:///m.pdf", "m", "application/pdf", 1, "h")
    chunks = [
        DocumentChunk(0, "first page text", 0, 15, {"page_start": 1, "page_end": 1}, (1.0, 0.0, 0.0)),
        DocumentChunk(1, "spans two pages", 16, 40, {"page_start": 1, "page_end": 2}, (1.0, 0.0, 0.0)),
        DocumentChunk(2, "no page at all", 41, 60, {}, (1.0, 0.0, 0.0)),
    ]
    index.upsert_document(source, chunks)

    labels = {h.ordinal: h.page_label for h in index.search("text", top_k=5, workspace_id="w")}

    assert labels == {0: "p.1", 1: "pp.1-2", 2: ""}


def test_the_search_result_tells_the_model_to_cite_the_page_only_when_there_is_one(tmp_path, pdf):
    index = SqliteDocumentIndex(connect(tmp_path))
    IngestionPipeline(index, chunk_max_tokens=20, chunk_min_tokens=5, chunk_overlap_tokens=0).ingest_path(pdf)
    notes = tmp_path / "notes.md"
    notes.write_text("Warranty notes for injectors. " * 20, encoding="utf-8")
    IngestionPipeline(index, chunk_max_tokens=20, chunk_min_tokens=5, chunk_overlap_tokens=0).ingest_path(notes, workspace_id="plain")
    tool = document_tool.make_tool(index)

    assert document_tool.CITE_PAGES in tool.fn("zebra crossing sensor")
    assert document_tool.CITE_PAGES not in document_tool.make_tool(index, "plain").fn("warranty")
    assert "page" in tool.description
