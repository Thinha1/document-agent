"""Structure-aware chunking with token guardrails and semantic-unit overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
) -> list[DocumentChunk]:
    """Split at headings, paragraphs, and sentences instead of character offsets.

    ``max_tokens`` is a safety ceiling for embedding inputs, not a fixed target.
    Chunks vary naturally with document structure. Overlap copies complete trailing
    semantic units so the next chunk does not begin mid-sentence.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if min_tokens < 0 or min_tokens > max_tokens:
        raise ValueError("min_tokens must be between zero and max_tokens")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")

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
