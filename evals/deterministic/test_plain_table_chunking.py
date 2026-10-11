"""DETERMINISTIC EVAL — a table written as Markdown pipes or aligned plain columns
is chunked by row with its header on every chunk, and nothing else changes.

The failure this fixes was measured (docs/document-ingestion.md, "Table retrieval"):
the prose chunker cut a torque table by size, so a later chunk was bare numbers
(`M36 | 100.3 | 78.1 ...`) and a cross-encoder could not tell it from any other
table. Network-free.
"""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents.chunking import chunk_text
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import SqliteDocumentIndex

MD_ROWS = [f"| M{d} | {d * 1.5:.1f} | {d * 1.1:.1f} |" for d in range(5, 45)]
MD_TABLE = "\n".join(["| Bolt | Dry (N·m) | Lubricated (N·m) |", "|---|---|---|", *MD_ROWS])
MD_DOC = f"# Torque chart (T-1)\n\nTightening torque by bolt size.\n\n{MD_TABLE}\n\nTolerance 10 percent.\n\nClosing paragraph."

TEXT_ROWS = [f"{t:<5} {t * 1.3:8.3f}  {t * 0.9:8.3f}  {t * 0.5:8.3f}" for t in range(-40, 160, 5)]
TEXT_TABLE = "\n".join(["Temp.   Resistance (kΩ)", "(°C)    TS-591    TS-881    TS-442", *TEXT_ROWS])
TEXT_DOC = f"# Thermistor table (T-2)\n\nNominal resistance.\n\n{TEXT_TABLE}\n\nTolerance 2 percent at 25 C."


def _table_chunks(text, **kw):
    return [c for c in chunk_text(text, 200, 40, 20, plain_tables=True, **kw) if c.metadata.get("kind") == "table"]


def test_every_chunk_of_a_markdown_table_repeats_the_header_and_no_row_is_cut():
    chunks = _table_chunks(MD_DOC)

    assert len(chunks) > 1
    for chunk in chunks:
        lines = chunk.text.splitlines()
        assert lines[0] == "# Torque chart (T-1)"  # the nearest heading says what the table is
        assert "| Bolt | Dry (N·m) | Lubricated (N·m) |" in lines and "|---|---|---|" in lines
    seen = [line for c in chunks for line in c.text.splitlines() if line in MD_ROWS]
    assert seen == MD_ROWS  # every row exactly once, whole, in order


def test_aligned_text_tables_keep_both_header_lines_and_the_footnote():
    chunks = _table_chunks(TEXT_DOC)

    assert len(chunks) > 1
    for chunk in chunks:
        lines = chunk.text.splitlines()
        assert "Temp.   Resistance (kΩ)" in lines  # the two-cell group header above the table
        assert "(°C)    TS-591    TS-881    TS-442" in lines
    assert chunks[-1].text.splitlines()[-1] == "Tolerance 2 percent at 25 C."
    assert [line for c in chunks for line in c.text.splitlines() if line in TEXT_ROWS] == TEXT_ROWS


def test_prose_around_the_table_is_still_chunked_and_offsets_stay_inside_the_document():
    chunks = chunk_text(MD_DOC, 200, 40, 20, plain_tables=True)

    assert any("Closing paragraph." in c.text and c.metadata.get("kind") != "table" for c in chunks)
    assert any("Tightening torque by bolt size." in c.text for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert 0 <= chunk.char_start <= chunk.char_end <= len(MD_DOC)


def test_it_is_off_unless_asked_for():
    plain = chunk_text(MD_DOC, 200, 40, 20)

    assert all(c.metadata.get("kind") != "table" for c in plain)
    assert plain == chunk_text(MD_DOC, 200, 40, 20, plain_tables=False)


@pytest.mark.parametrize(
    "text",
    [
        # only three rows: a short list, not a table
        "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n| 7 | 8 | 9 |",
        # aligned words with no numbers: a list of names, not a table of values
        "Name  Role  Team\n" + "\n".join(f"person{i}  engineer  platform" for i in range(8)),
        # the first aligned line already holds a number: no header to repeat
        "\n".join(f"{i:<4} {i * 2:<6} {i * 3:<6}" for i in range(10)),
        # columns that do not line up
        "Header  One  Two\n1  2  3\n4  5\n6  7  8  9\n1  2\n3  4  5  6\n7  8",
        # ordinary prose, even with a couple of double spaces
        "This is a paragraph.  It has two spaces after each sentence.  Nothing here is a table.\n" * 8,
    ],
)
def test_things_that_are_not_tables_are_left_to_the_prose_chunker(text):
    assert _table_chunks(text) == []
    assert chunk_text(text, 200, 40, 20, plain_tables=True) == chunk_text(text, 200, 40, 20)


def test_a_markdown_table_whose_header_is_numeric_is_still_a_table():
    rows = [f"| {name} | {n} | {n + 1} |" for n, name in enumerate("ABCDEF")]
    table = "\n".join(["| Item | 2023 | 2024 |", "|---|---|---|", *rows])

    chunks = _table_chunks(f"Yearly figures\n\n{table}")

    assert chunks and "| Item | 2023 | 2024 |" in chunks[0].text


def test_a_single_row_wider_than_the_budget_is_kept_whole():
    wide = "| " + " | ".join(str(n) for n in range(300)) + " |"
    table = f"| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |\n{wide}\n| 4 | 5 | 6 |\n| 7 | 8 | 9 |\n| 1 | 1 | 1 |"

    chunks = _table_chunks(table)

    assert sum(wide in c.text for c in chunks) == 1
    assert all("| a | b | c |" in c.text for c in chunks)


def test_the_pipeline_only_uses_it_when_told_to_and_the_setting_defaults_on(tmp_path, monkeypatch):
    monkeypatch.delenv("TINI_TABLE_CHUNKING", raising=False)
    assert Settings().document_table_chunking is True
    monkeypatch.setenv("TINI_TABLE_CHUNKING", "0")
    assert Settings().document_table_chunking is False

    source = tmp_path / "torque.md"
    source.write_text(MD_DOC, encoding="utf-8")
    later_rows = {}
    for enabled in (False, True):
        (tmp_path / f"h{enabled}").mkdir()
        conn = connect(tmp_path / f"h{enabled}")
        index = SqliteDocumentIndex(conn)
        IngestionPipeline(
            index, chunk_max_tokens=200, chunk_min_tokens=40, chunk_overlap_tokens=20, table_chunking=enabled
        ).ingest_path(source, workspace_id="w")
        texts = [r[0] for r in conn.execute("SELECT text FROM document_chunks ORDER BY ordinal")]
        later_rows[enabled] = [t for t in texts if "| M44 |" in t]
        conn.close()

    assert "| Bolt | Dry (N·m) | Lubricated (N·m) |" not in later_rows[False][0]  # the old failure
    assert "| Bolt | Dry (N·m) | Lubricated (N·m) |" in later_rows[True][0]
