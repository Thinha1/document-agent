"""Deterministic, paragraph-aware text chunking."""

from __future__ import annotations

from tini.documents.models import DocumentChunk


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(start + chunk_size, length)
        end = hard_end if hard_end == length else _preferred_break(text, start, hard_end)
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        clean = raw.strip()
        if clean:
            char_start = start + leading
            chunks.append(
                DocumentChunk(
                    ordinal=len(chunks),
                    text=clean,
                    char_start=char_start,
                    char_end=start + trailing,
                )
            )
        if end >= length:
            break
        next_start = max(0, end - overlap)
        start = end if next_start <= start else next_start
    return chunks


def _preferred_break(text: str, start: int, hard_end: int) -> int:
    floor = start + max(1, int((hard_end - start) * 0.55))
    for separator in ("\n\n", "\n", ". ", " "):
        position = text.rfind(separator, floor, hard_end)
        if position >= floor:
            return position + len(separator)
    return hard_end
