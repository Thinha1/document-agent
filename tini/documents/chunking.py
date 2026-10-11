"""Structure-aware chunking with token guardrails and semantic-unit overlap.

A table gets its own path, not because prose chunking is wrong for it, but
because it doesn't apply: `_semantic_units` below groups by heading/paragraph/
sentence, and a row of numbers has none of those. Fed to it anyway (a Markdown
table is just more prose to this module by default), a table bigger than
`max_tokens` falls through every boundary it looks for and gets cut on a raw
token offset instead — no header, sometimes mid-cell. `chunk_text` here looks
for a `<!--tini:table id=N-->...<!--/tini:table-->` block first (written only
by loaders.py's OCR table flow, TINI_OCR_TABLE_FLOW=1 — see tini/documents/
ocr.py) and hands each one to `_chunk_table_block`, which splits by ROW and
repeats the caption/header on every piece, instead of running it through the
paragraph/sentence splitter at all. A document with no such block (everything
today, and every document unless that flag is on) takes exactly the path this
module always has — see `_split_around_tables`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

from tini.documents.models import DocumentChunk

# CJK scripts (CJK Unified Ideographs, Hiragana, Katakana, Hangul, fullwidth
# forms) carry no spaces between words, so plain `\w+` — which greedily
# consumes an entire unbroken run as ONE match — collapses a whole Japanese
# paragraph into a handful of "tokens" and lets max_tokens stop meaning
# anything for it. Matched a real EDINET (Japanese securities filing) PDF:
# 295 characters of continuous Japanese prose estimated at 31 tokens instead
# of the ~295 a subword tokenizer would count, letting a chunk grow ~8x past
# its guardrail. Counting each CJK character as its own token is a
# deliberate overestimate (real tokenizers run closer to 0.5-1 token/char
# for CJK) — safe direction for a size guardrail, and English/Latin text
# (the `[^\W\d_...]+` word-run branch) is completely unaffected.
_CJK = "一-鿿぀-ヿ゠-ヿ가-힯＀-￯"
_TOKEN_RE = re.compile(rf"[{_CJK}]|[^\W\d_{_CJK}]+|\d+|[^\w\s]", re.UNICODE)
_BLOCK_RE = re.compile(r"\S.*?(?=[ \t]*\n[ \t]*\n|\s*\Z)", re.DOTALL)
_SENTENCE_RE = re.compile(r".*?(?:[.!?。！？]+(?:\s+|$)|\Z)", re.DOTALL)
_HEADING_RE = re.compile(r"^(?:#{1,6}\s+|[A-ZÀ-Ỹ\d][^\n]{0,100}\n[-=]{3,}\s*$)", re.MULTILINE)

# Written only by loaders.py's _render_table_block (TINI_OCR_TABLE_FLOW=1); the
# two functions are each other's only producer/consumer, kept in sync by
# evals/deterministic/test_table_chunking.py's round-trip tests.
_TABLE_BLOCK_RE = re.compile(r"<!--tini:table id=(\d+)-->\n(.*?)\n<!--/tini:table-->", re.DOTALL)


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    tokens: int
    starts_section: bool = False


def estimate_tokens(text: str) -> int:
    """Return a dependency-free token estimate suitable for chunk guardrails."""
    return len(_TOKEN_RE.findall(text))


def chunk_text(
    text: str,
    max_tokens: int = 512,
    min_tokens: int = 120,
    overlap_tokens: int = 48,
    plain_tables: bool = False,
) -> list[DocumentChunk]:
    """Split at headings, paragraphs, and sentences instead of character offsets
    — except a `<!--tini:table-->` block, which splits by row instead (see the
    module docstring). ``max_tokens`` is a safety ceiling for embedding inputs,
    not a fixed target. Chunks vary naturally with document structure. Overlap
    copies complete trailing semantic units so the next chunk does not begin
    mid-sentence; a table chunk instead repeats its own caption/header, which
    serves the same purpose (self-contained without the previous chunk) without
    literally overlapping row data.

    ``plain_tables`` extends the same row-wise treatment to a table written as
    Markdown pipes or as aligned plain columns (what a text extractor returns):
    header on every chunk, no row cut in half. Off unless asked for — see
    `_plain_table_from_run` for exactly what counts as a table.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if min_tokens < 0 or min_tokens > max_tokens:
        raise ValueError("min_tokens must be between zero and max_tokens")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")

    chunks: list[DocumentChunk] = []
    for kind, payload, offset in _split_around_tables(text):
        if kind == "table":
            table_id, body, start, end = payload
            chunks.extend(_chunk_table_block(table_id, body, start, end, max_tokens))
        elif payload.strip():
            pieces = _split_plain_tables(payload) if plain_tables else [("prose", payload, 0)]
            for piece_kind, piece, inner in pieces:
                if piece_kind == "table":
                    chunks.extend(_chunk_plain_table(piece, offset + inner, max_tokens))
                elif piece.strip():
                    sub = _chunk_prose(piece, max_tokens, min_tokens, overlap_tokens)
                    chunks.extend(
                        replace(
                            chunk,
                            char_start=chunk.char_start + offset + inner,
                            char_end=chunk.char_end + offset + inner,
                        )
                        for chunk in sub
                    )
    return [replace(chunk, ordinal=i) for i, chunk in enumerate(chunks)]


def _split_around_tables(text: str):
    """[("prose", segment_text, offset), ("table", (id, body, start, end), None), ...]
    in document order. No `<!--tini:table-->` block (every document today,
    and every document unless TINI_OCR_TABLE_FLOW=1 produced one) yields
    exactly one ("prose", text, 0) — chunk_text's behavior for those is
    byte-for-byte what it always was."""
    segments: list[tuple[str, object, int | None]] = []
    pos = 0
    for match in _TABLE_BLOCK_RE.finditer(text):
        if match.start() > pos:
            segments.append(("prose", text[pos : match.start()], pos))
        segments.append(("table", (match.group(1), match.group(2), match.start(), match.end()), None))
        pos = match.end()
    if pos < len(text):
        segments.append(("prose", text[pos:], pos))
    return segments or [("prose", text, 0)]


def _chunk_prose(
    text: str,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> list[DocumentChunk]:
    """The chunker this module has always had: heading/paragraph/sentence,
    then raw tokens as a last resort. `chunk_text` runs this on the prose
    between (or around) any table blocks; a document with none runs it on
    the whole text, unchanged from before table-awareness existed."""
    units = _semantic_units(text, max_tokens)
    if not units:
        return []

    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0
    for unit in units:
        section_break = unit.starts_section and current_tokens >= min_tokens
        capacity_break = bool(current) and current_tokens + unit.tokens > max_tokens
        if section_break or capacity_break:
            groups.append(current)
            current = [] if section_break else _overlap_tail(current, overlap_tokens)
            current_tokens = sum(item.tokens for item in current)
            while current and current_tokens + unit.tokens > max_tokens:
                removed = current.pop(0)
                current_tokens -= removed.tokens
        current.append(unit)
        current_tokens += unit.tokens
    if current:
        groups.append(current)

    chunks: list[DocumentChunk] = []
    for group in groups:
        start, end = group[0].start, group[-1].end
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        clean = raw.strip()
        if clean:
            chunks.append(
                DocumentChunk(
                    ordinal=len(chunks),
                    text=clean,
                    char_start=start + leading,
                    char_end=start + trailing,
                    metadata={"estimated_tokens": estimate_tokens(clean)},
                )
            )
    return chunks


def _semantic_units(text: str, max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    for match in _BLOCK_RE.finditer(text):
        start, end = match.span()
        block = match.group()
        tokens = estimate_tokens(block)
        starts_section = bool(_HEADING_RE.match(block.strip()))
        if tokens <= max_tokens:
            units.append(_Unit(start, end, tokens, starts_section))
            continue
        units.extend(_split_large_block(text, start, end, max_tokens, starts_section))
    return units


def _split_large_block(
    text: str, start: int, end: int, max_tokens: int, starts_section: bool
) -> list[_Unit]:
    units: list[_Unit] = []
    block = text[start:end]
    for sentence in _SENTENCE_RE.finditer(block):
        sentence_text = sentence.group()
        if not sentence_text.strip():
            continue
        sentence_start = start + sentence.start()
        sentence_end = start + sentence.end()
        tokens = estimate_tokens(sentence_text)
        if tokens <= max_tokens:
            units.append(_Unit(sentence_start, sentence_end, tokens, starts_section and not units))
        else:
            units.extend(
                _split_oversized_sentence(
                    text, sentence_start, sentence_end, max_tokens, starts_section and not units
                )
            )
    return units


def _split_oversized_sentence(
    text: str, start: int, end: int, max_tokens: int, starts_section: bool
) -> list[_Unit]:
    matches = list(_TOKEN_RE.finditer(text, start, end))
    units: list[_Unit] = []
    for offset in range(0, len(matches), max_tokens):
        batch = matches[offset : offset + max_tokens]
        units.append(
            _Unit(
                batch[0].start(),
                batch[-1].end(),
                len(batch),
                starts_section and not units,
            )
        )
    return units


def _overlap_tail(units: list[_Unit], budget: int) -> list[_Unit]:
    if budget == 0:
        return []
    tail: list[_Unit] = []
    used = 0
    for unit in reversed(units):
        if used + unit.tokens > budget:
            break
        tail.append(unit)
        used += unit.tokens
    return list(reversed(tail))


# --- tables written as plain text (Markdown pipes, or aligned columns) -----------
# `<!--tini:table-->` blocks exist only when the OCR table flow wrote them. A
# Markdown table, or the aligned columns a PDF text extractor returns, arrives as
# ordinary prose, and the prose chunker cuts it by size: rows split in half, and
# every chunk after the first has lost the header that says what its numbers mean
# (a cross-encoder then sees `M36  100.3  78.1 ...` and cannot tell it from any
# other table of numbers). `_plain_table_from_run` recognises such a table strictly
# enough that ordinary prose, lists and code never qualify.
_CELL_SPLIT_RE = re.compile(r"\s{2,}")
_NUMERIC_CELL_RE = re.compile(r"^[-+±]?\d[\d.,]*(?:\s?[–\-~/]\s?[-+]?\d[\d.,]*)?\s?%?$")
_PIPE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_MIN_TABLE_ROWS = 4  # data rows; below this a "table" is just a short list
_MAX_HEADER_LINES = 4  # a two-level header plus separator, with a little slack
_MAX_CAPTION_CHARS = 240  # a caption or footnote is a line, not a paragraph


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        return [cell.strip() for cell in stripped.strip("|").split("|")]
    return _CELL_SPLIT_RE.split(stripped)


def _is_numeric(cell: str) -> bool:
    return bool(_NUMERIC_CELL_RE.match(cell.strip()))


def _is_tabular(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("|"):
        return stripped.count("|") >= 3
    cells = _cells(line)
    return len(cells) >= 3 and all(cells)


def _plain_table_from_run(run: list[str]) -> tuple[list[str], list[str]] | None:
    """(header lines, data lines) if this run of tabular lines is a table."""
    is_pipe = all(line.strip().startswith("|") for line in run)
    if is_pipe:
        header_n = 2 if len(run) > 1 and _PIPE_SEPARATOR_RE.match(run[1].strip()) else 1
    elif any(_is_numeric(cell) for cell in _cells(run[0])):
        return None  # aligned text: the first line must read like a header
    else:
        header_n = 1
    while header_n < min(_MAX_HEADER_LINES, len(run)) and not any(
        _is_numeric(cell) for cell in _cells(run[header_n])
    ):
        header_n += 1
    header, data = run[:header_n], run[header_n:]
    if len(data) < _MIN_TABLE_ROWS:
        return None
    if sum(any(_is_numeric(cell) for cell in _cells(line)) for line in data) < 0.7 * len(data):
        return None  # a list of words, not a table of values
    if not is_pipe:
        modal = Counter(len(_cells(line)) for line in data).most_common(1)[0][1]
        if modal < 0.8 * len(data):
            return None  # columns do not line up
    return header, data


def _plain_caption(lines: list[str], table_start: int) -> list[str]:
    """The nearest Markdown heading above the table and the short prose lines
    between it and the table (what the table is about), at most a few of them."""
    for k in range(table_start - 1, max(-1, table_start - 11), -1):
        if _MD_HEADING_RE.match(lines[k]):
            between = [
                line.strip()
                for line in lines[k + 1 : table_start]
                if line.strip() and len(line) <= _MAX_CAPTION_CHARS
            ]
            return [lines[k].strip(), *between[-3:]]
    return []


def _split_plain_tables(text: str):
    """[("prose", segment, offset), ("table", (caption, header, rows, notes, length), offset)]
    in document order; a text with no such table is one ("prose", text, 0)."""
    lines = text.split("\n")
    starts, position = [], 0
    for line in lines:
        starts.append(position)
        position += len(line) + 1
    pieces: list[tuple[str, object, int]] = []
    cursor = 0  # first character not yet emitted
    cursor_line = 0  # first line not yet emitted
    i = 0
    while i < len(lines):
        if not _is_tabular(lines[i]):
            i += 1
            continue
        j = i
        while j < len(lines) and _is_tabular(lines[j]):
            j += 1
        # a group header such as "Temp.    Resistance (kΩ)" has only two cells, so it is not
        # "tabular" on its own; it belongs to the table when it sits right above it
        first = i
        while (
            first > cursor_line
            and lines[first - 1].strip()
            and not _is_tabular(lines[first - 1])
            and len(_cells(lines[first - 1])) >= 2
            and not any(_is_numeric(cell) for cell in _cells(lines[first - 1]))
            and i - first < 2
        ):
            first -= 1
        table = _plain_table_from_run([line.rstrip() for line in lines[first:j]])
        if table is not None and starts[first] >= cursor:
            caption = _plain_caption(lines, first)
            end_line = j - 1
            notes = ""
            k = j
            while k < len(lines) and not lines[k].strip() and k - j < 1:
                k += 1  # at most one blank line between a table and its footnote
            if k < len(lines) and lines[k].strip() and len(lines[k]) <= _MAX_CAPTION_CHARS and not _is_tabular(lines[k]):
                notes, end_line = lines[k].strip(), k
            end = starts[end_line] + len(lines[end_line])
            if starts[first] > cursor:
                pieces.append(("prose", text[cursor : starts[first]], cursor))
            header, rows = table
            pieces.append(("table", (caption, header, rows, notes, end - starts[first]), starts[first]))
            cursor = end
            cursor_line = end_line + 1
        i = max(j, cursor_line)
    if not pieces:
        return [("prose", text, 0)]
    if cursor < len(text):
        pieces.append(("prose", text[cursor:], cursor))
    return pieces


def _chunk_plain_table(piece, start: int, max_tokens: int) -> list[DocumentChunk]:
    """Row-wise chunks of a plain-text table: the nearest heading and the header
    lines open every chunk, and a row is never cut."""
    caption, header, rows, notes, length = piece
    return _pack_rows(
        [*caption, *header],
        rows,
        notes,
        start,
        start + length,
        max_tokens,
        {"kind": "table", "table_layout": "plain"},
    )


def _split_table_lines(body: str) -> tuple[list[str], list[str], str]:
    """A `<!--tini:table-->` body is always, in order: an optional `**caption**`
    line, the header row, the `---` separator row, then data rows, then an
    optional `_notes_` line — exactly what loaders.py's _render_table_block
    writes (the two are each other's only producer/consumer). Returns
    (preamble lines to repeat on every chunk, data rows, notes line or "")."""
    lines = body.splitlines()
    index = 0
    preamble: list[str] = []
    if index < len(lines) and lines[index].startswith("**") and lines[index].endswith("**"):
        preamble.append(lines[index])
        index += 1
    preamble.extend(lines[index : index + 2])  # header, separator
    index += 2
    data = lines[index:]
    notes = ""
    if data and data[-1].startswith("_") and data[-1].endswith("_"):
        notes = data[-1]
        data = data[:-1]
    return preamble, data, notes


def _chunk_table_block(
    table_id: str, body: str, start: int, end: int, max_tokens: int
) -> list[DocumentChunk]:
    """A table chunks by ROW, not by paragraph/sentence/token — see the module
    docstring for why the general path can't do this. Rows are packed greedily
    under `max_tokens`, counting the caption/header/separator ("preamble")
    against every group's budget because it is repeated at the top of each one,
    the same role overlap plays for prose: a chunk that lands mid-table is
    still self-contained, never a bare row with no header to say what its
    numbers mean.

    A single row wider than `max_tokens` on its own (a pathologically wide
    table) is kept whole rather than split — an oversized chunk is a known,
    visible trade-off (its `estimated_tokens` says so); a row cut mid-cell,
    the failure this function exists to avoid, is not.

    Every chunk from one table shares (start, end): the source table's whole
    span, not a slice of it — a chunk's assembled text repeats the header, so
    it is not a literal substring of the document the way a prose chunk is.
    """
    preamble, rows, notes = _split_table_lines(body)
    return _pack_rows(
        preamble, rows, notes, start, end, max_tokens, {"kind": "table", "table_id": table_id}
    )


def _pack_rows(
    preamble: list[str],
    rows: list[str],
    notes: str,
    start: int,
    end: int,
    max_tokens: int,
    base_metadata: dict,
) -> list[DocumentChunk]:
    """Pack rows greedily under `max_tokens`, the preamble (caption + header)
    counted against every group because it is repeated at the top of each."""
    preamble_tokens = estimate_tokens("\n".join(preamble))

    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = preamble_tokens
    for row in rows:
        row_tokens = estimate_tokens(row)
        if current and current_tokens + row_tokens > max_tokens:
            groups.append(current)
            current = []
            current_tokens = preamble_tokens
        current.append(row)
        current_tokens += row_tokens
    if current or not groups:
        groups.append(current)

    chunks: list[DocumentChunk] = []
    for index, group in enumerate(groups):
        lines = [*preamble, *group]
        if index == len(groups) - 1 and notes:
            lines.append(notes)
        text = "\n".join(lines).strip()
        if not text:
            continue
        chunks.append(
            DocumentChunk(
                ordinal=len(chunks),
                text=text,
                char_start=start,
                char_end=end,
                metadata={
                    **base_metadata,
                    "row_group": index,
                    "row_groups_total": len(groups),
                    "estimated_tokens": estimate_tokens(text),
                },
            )
        )
    return chunks
