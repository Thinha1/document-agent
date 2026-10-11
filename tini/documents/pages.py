"""Which PDF pages a chunk came from, so an answer can cite a page.

A PDF loader joins every page into one string, and chunking works on that string, so
page boundaries would be lost. Instead the loader writes a private-use marker in front
of each page (`mark_page`); once the text is normalised, `split_marks` records where
each page starts and removes the markers, so the text that is chunked, embedded and
searched is exactly what it would have been without them. `assign_pages` then turns a
chunk's character span into `page_start` / `page_end` in its metadata.

Page numbers are the PDF's own 1-based page index (what a viewer's page box shows),
which can differ from a number printed in the page footer. A marker carries its page
number, so a page that OCR skipped or that has no text does not shift the ones after it.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from tini.documents.models import DocumentChunk

MARK_OPEN = ""
MARK_CLOSE = ""
_MARK = re.compile(f"{MARK_OPEN}(\\d+){MARK_CLOSE}")

# (character offset in the final text where the page begins, 1-based page number)
PageStarts = tuple[tuple[int, int], ...]


def mark_page(number: int, text: str) -> str:
    """`text` prefixed with a page marker. The marker characters are removed from the
    text first, so a document cannot forge a page boundary."""
    return f"{MARK_OPEN}{number}{MARK_CLOSE}" + text.replace(MARK_OPEN, "").replace(MARK_CLOSE, "")


def split_marks(text: str) -> tuple[str, PageStarts]:
    """Remove the page markers from `text`, returning it stripped like any other loaded
    text together with where each page begins."""
    cleaned: list[str] = []
    starts: list[tuple[int, int]] = []
    length = 0
    last = 0
    for match in _MARK.finditer(text):
        piece = text[last : match.start()]
        cleaned.append(piece)
        length += len(piece)
        starts.append((length, int(match.group(1))))
        last = match.end()
    cleaned.append(text[last:])
    body = "".join(cleaned)
    leading = len(body) - len(body.lstrip())
    body = body.strip()
    return body, tuple((max(0, offset - leading), number) for offset, number in starts)


def page_of(pages: PageStarts, offset: int) -> int | None:
    """The page whose text contains character `offset`, or None with no page data."""
    if not pages:
        return None
    index = bisect_right([start for start, _ in pages], offset) - 1
    return pages[max(index, 0)][1]


def assign_pages(chunks: Sequence[DocumentChunk], pages: PageStarts) -> list[DocumentChunk]:
    """Chunks with `page_start` and `page_end` added to their metadata. Without page
    data (anything but a PDF) they are returned unchanged."""
    if not pages:
        return list(chunks)
    result = []
    for chunk in chunks:
        first = page_of(pages, chunk.char_start)
        last = page_of(pages, max(chunk.char_start, chunk.char_end - 1))
        result.append(
            replace(chunk, metadata={**chunk.metadata, "page_start": first, "page_end": max(first, last)})
        )
    return result


def pages_from_metadata(metadata: Mapping[str, Any] | str | None) -> tuple[int | None, int | None]:
    """(page_start, page_end) from a chunk's stored metadata (a dict or its JSON text)."""
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata or "{}")
        except ValueError:
            return None, None
    if not isinstance(metadata, Mapping):
        return None, None
    first, last = metadata.get("page_start"), metadata.get("page_end")
    if not isinstance(first, int) or isinstance(first, bool):
        return None, None
    return first, last if isinstance(last, int) and not isinstance(last, bool) else first
