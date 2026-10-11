"""Table-aware chunking: a `<!--tini:table-->` block splits by row, not by the
paragraph/sentence/token path the rest of chunking.py uses.

The failure this guards against is the one demonstrated in
docs/document-ingestion.md: fed a wide table as plain Markdown, `chunk_text`
cuts it at a raw token offset when it exceeds `max_tokens` — no header, split
mid-cell. These tests build the same shape of table `loaders.py`'s
`_render_table_block` produces (the two are each other's only producer/
consumer — evals/deterministic/test_document_ocr.py covers that side) and
check the chunker never does that to it, while every document without the
marker chunks exactly as it always has.
"""

from __future__ import annotations

from tini.documents.chunking import chunk_text, estimate_tokens


def _table_block(table_id: int, caption: str, columns: list[str], rows: list[list[str]], notes: str = "") -> str:
    """Same shape as loaders.py's _render_table_block, built independently
    here so a bug shared by both sides wouldn't hide behind reusing one
    function to build fixtures and check them."""
    row = lambda cells: "| " + " | ".join(cells) + " |"  # noqa: E731
    lines = [f"<!--tini:table id={table_id}-->"]
    if caption:
        lines.append(f"**{caption}**")
    lines.append(row(columns))
    lines.append(row(["---"] * len(columns)))
    lines.extend(row(r) for r in rows)
    if notes:
        lines.append(f"_{notes}_")
    lines.append("<!--/tini:table-->")
    return "\n".join(lines)


def _rows(n: int) -> list[list[str]]:
    return [[f"Region-{i}", f"{1_000_000 + i * 37211:,}", f"{100 + i % 7}.{i % 9}"] for i in range(n)]


COLUMNS = ["Region", "Amount (millions of yen)", "Change (%)"]


# --- a document with no table marker is untouched -----------------------------


def test_plain_text_chunks_exactly_as_before_no_table_path_taken():
    text = "First paragraph.\n\n" + ("word " * 300) + "\n\nLast paragraph."
    assert chunk_text(text, max_tokens=100, min_tokens=20, overlap_tokens=10) == chunk_text(
        text, max_tokens=100, min_tokens=20, overlap_tokens=10
    )
    chunks = chunk_text(text, max_tokens=100, min_tokens=20, overlap_tokens=10)
    assert all("kind" not in c.metadata for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


# --- a small table survives whole, tagged ------------------------------------


def test_a_small_table_stays_one_chunk_with_the_kind_and_id_recorded():
    block = _table_block(1, "FY2023 sales by region", COLUMNS, _rows(3))
    chunks = chunk_text(block, max_tokens=512, min_tokens=120, overlap_tokens=48)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["kind"] == "table"
    assert chunk.metadata["table_id"] == "1"
    assert chunk.metadata["row_groups_total"] == 1
    for column in COLUMNS:
        assert column in chunk.text
    assert "Region-0" in chunk.text and "Region-2" in chunk.text
    assert "FY2023 sales by region" in chunk.text


def test_a_table_surrounded_by_prose_leaves_the_prose_untouched():
    before = "Here is this quarter's regional breakdown:"
    after = "As shown above, growth was broad-based."
    block = _table_block(1, "", COLUMNS, _rows(2))
    text = f"{before}\n{block}\n{after}"
    chunks = chunk_text(text, max_tokens=512, min_tokens=20, overlap_tokens=10)
    kinds = [c.metadata.get("kind") for c in chunks]
    assert "table" in kinds
    prose = [c for c in chunks if c.metadata.get("kind") != "table"]
    assert any(before in c.text for c in prose)
    assert any(after in c.text for c in prose)
    # ordinals are contiguous across the whole document, table included
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


# --- the failure mode from docs/document-ingestion.md, now fixed --------------


def test_a_wide_table_splits_by_row_with_the_header_repeated_every_time():
    block = _table_block(1, "Sales results by region", COLUMNS, _rows(70))
    assert estimate_tokens(block) > 512, "fixture must actually exceed the guardrail"

    chunks = chunk_text(block, max_tokens=512, min_tokens=120, overlap_tokens=48)

    assert len(chunks) > 1, "a table this wide must not fit in one chunk"
    for chunk in chunks:
        assert chunk.metadata["kind"] == "table"
        assert chunk.metadata["estimated_tokens"] <= 512
        for column in COLUMNS:
            assert column in chunk.text, "every chunk must carry the header, not just the first"
        assert "Sales results by region" in chunk.text
        # never a bare/partial row: every row line present is a complete row
        for line in chunk.text.splitlines():
            if line.startswith("| Region-"):
                assert line.count("|") == 4, f"row split mid-cell: {line!r}"


def test_row_groups_cover_every_row_exactly_once_in_order():
    rows = _rows(40)
    block = _table_block(1, "", COLUMNS, rows)
    chunks = chunk_text(block, max_tokens=200, min_tokens=20, overlap_tokens=10)
    assert len(chunks) > 1
    seen = []
    for chunk in chunks:
        for line in chunk.text.splitlines():
            if line.startswith("| Region-"):
                seen.append(line)
    expected = [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows]
    assert seen == expected, "no row skipped, duplicated, or reordered across chunks"


def test_row_groups_are_numbered_and_notes_land_only_on_the_last_chunk():
    block = _table_block(1, "", COLUMNS, _rows(40), notes="Totals exclude intercompany sales.")
    chunks = chunk_text(block, max_tokens=200, min_tokens=20, overlap_tokens=10)
    assert [c.metadata["row_group"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["row_groups_total"] == len(chunks) for c in chunks)
    assert "Totals exclude intercompany sales." not in chunks[0].text
    assert "Totals exclude intercompany sales." in chunks[-1].text


def test_a_single_row_wider_than_max_tokens_is_kept_whole_not_cut():
    huge_row = ["Region-X", "note: " + ("x" * 3000), "1.0"]
    block = _table_block(1, "", COLUMNS, [huge_row])
    chunks = chunk_text(block, max_tokens=100, min_tokens=10, overlap_tokens=10)
    assert len(chunks) == 1
    assert "x" * 3000 in chunks[0].text, "the oversized row must not be truncated or split"


def test_a_table_with_only_a_header_still_produces_one_searchable_chunk():
    block = _table_block(1, "Empty this quarter", COLUMNS, [])
    chunks = chunk_text(block, max_tokens=512, min_tokens=120, overlap_tokens=48)
    assert len(chunks) == 1
    assert "Empty this quarter" in chunks[0].text
    for column in COLUMNS:
        assert column in chunks[0].text


def test_two_tables_in_sequence_each_get_their_own_chunks():
    first = _table_block(1, "Region", COLUMNS, _rows(2))
    second = _table_block(2, "Customer", ["Customer", "Amount", "Share (%)"], _rows(2))
    chunks = chunk_text(f"{first}\n\n{second}", max_tokens=512, min_tokens=20, overlap_tokens=10)
    ids = [c.metadata["table_id"] for c in chunks if c.metadata.get("kind") == "table"]
    assert ids == ["1", "2"]


def test_char_start_and_end_span_the_whole_source_table_for_every_row_group():
    """A table chunk's text repeats the header, so it is not a literal
    substring of the document for anything past the first row-group — both
    fields deliberately point at the table's whole span instead of a false
    per-chunk slice."""
    block = _table_block(1, "", COLUMNS, _rows(40))
    text = f"prefix\n{block}\nsuffix"
    chunks = [c for c in chunk_text(text, max_tokens=150, min_tokens=10, overlap_tokens=10) if c.metadata.get("kind") == "table"]
    assert len(chunks) > 1
    starts = {c.char_start for c in chunks}
    ends = {c.char_end for c in chunks}
    assert len(starts) == 1 and len(ends) == 1
    assert text[next(iter(starts)) : next(iter(ends))] == block
