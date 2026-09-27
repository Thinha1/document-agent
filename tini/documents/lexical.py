"""Lexical (keyword) side of hybrid retrieval: text -> sparse BM25-style vector.

Dense embeddings find text that *means* the same as the question; they miss the
things a keyword index gets right — names, figures, clause numbers, a rare term.
Hybrid retrieval runs both and fuses the rankings. Qdrant stores the keyword
side as a sparse vector: {token id: weight}. Its IDF modifier supplies the
inverse-document-frequency half of BM25 on the server; this module supplies the
term-frequency half (saturation + length normalisation) and the tokenizer.

The tokenizer is the part that matters for this corpus. SQLite's FTS5 default
tokenizer splits on whitespace, and Japanese has none: a whole sentence becomes
one token, so a search for 指名委員会 matches nothing. Here a run of CJK
characters becomes overlapping character bigrams (取締役会 -> 取締 締役 役会), so
a query term matches wherever its characters appear together, with no
dictionary and no dependency. Everything else (Latin, Vietnamese, digits) is
split into words. Bigrams over-generate against a real morphological analyser
(MeCab/Sudachi would be more precise) but never miss, and IDF down-weights the
common ones.

Token ids are CRC32, not Python's `hash()`: the id must be identical across
processes and restarts or an index built yesterday matches nothing today.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

# Runs of Han, Hiragana, Katakana (incl. the long-vowel mark), CJK extension A,
# and compatibility ideographs. NFKC first folds half-width katakana and
# full-width Latin, so both spellings of a term land on the same tokens.
_CJK_CHARS = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_CJK_RUN = f"[{_CJK_CHARS}]+"
_TOKEN = re.compile(rf"{_CJK_RUN}|(?:(?![{_CJK_CHARS}])[^\W_])+")
_CJK_START = re.compile(_CJK_RUN)

# BM25 term-frequency saturation. AVG_LENGTH is a stand-in for the corpus'
# average chunk length in tokens (chunks are capped at ~512 model tokens); the
# exact value only moves how hard long chunks are penalised.
K1 = 1.2
B = 0.75
AVG_LENGTH = 300.0


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(unicodedata.normalize("NFKC", text).lower()):
        piece = match.group()
        if _CJK_START.fullmatch(piece):
            if len(piece) == 1:
                tokens.append(piece)
            else:
                tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece)
    return tokens


def _token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def document_vector(text: str, title: str = "") -> tuple[list[int], list[float]]:
    """Sparse vector for a chunk: BM25 term-frequency weight per token id.
    The title is indexed with the text, as the SQLite FTS5 table does."""
    tokens = tokenize(f"{title}\n{text}" if title else text)
    if not tokens:
        return [], []
    length_norm = 1.0 - B + B * len(tokens) / AVG_LENGTH
    weights: dict[int, float] = {}
    for token, count in Counter(tokens).items():
        weight = count * (K1 + 1.0) / (count + K1 * length_norm)
        index = _token_id(token)
        weights[index] = weights.get(index, 0.0) + weight  # CRC32 collisions add up
    indices = sorted(weights)
    return indices, [weights[i] for i in indices]


def query_vector(text: str) -> tuple[list[int], list[float]]:
    """Sparse vector for a question: one unit weight per distinct token. Qdrant
    multiplies it into each chunk's weights and applies IDF, so a rare term
    outranks a common one."""
    indices = sorted({_token_id(token) for token in tokenize(text)})
    return indices, [1.0] * len(indices)
