"""Retrieval QUALITY, not just plumbing — recall@k and MRR over a labeled corpus.

`test_document_ingestion.py` and `test_qdrant_document_index.py` prove each
backend does what `DocumentIndex` promises (skip/replace/workspace-isolation/
interface conformance) against one or two hand-written documents. Neither
tells you whether a change to chunking, fusion weights, or a new backend
actually ranks the RIGHT chunk higher — that needs a fixed corpus with known
answers, measured the same way on every backend so a regression shows up as a
number dropping, not a vibe.

Runs network-free like the rest of `evals/deterministic/`: `_HashEmbedder`
below is a real (if crude) feature-hashing embedding — word + bigram hashing
into a fixed-width vector, the same idea sklearn's `HashingVectorizer` and
Vowpal Wabbit use — not a hardcoded two-cluster fake. That makes cosine
similarity mean something for genuinely different documents, so recall/MRR
here test the retrieval MECHANISM (chunking, indexing, fusion, backend
parity) rather than a specific embedding model's semantic quality. Real
semantic quality depends on whichever model is configured in production and
is exactly the kind of thing `scripts/shootout.py`'s pattern is for — a
network-free unit test can't and shouldn't claim to measure it.
"""

from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import SqliteDocumentIndex

DIMENSIONS = 64


def _hash_embed(text: str, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    vector = [0.0] * dimensions
    for word in words:
        vector[_bucket(word, dimensions)] += 0.5
    for left, right in pairwise(words):
        vector[_bucket(f"{left} {right}", dimensions)] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return tuple(v / norm for v in vector)


def _bucket(token: str, dimensions: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % dimensions


class _HashEmbedder:
    provider = "hash"
    model = "toy-hash-v1"
    dimensions = DIMENSIONS

    def embed_documents(self, texts, *, title):
        return [_hash_embed(text) for text in texts]

    def embed_query(self, text):
        return _hash_embed(text)


# Four topically-adjacent HR policies (all "employee policy" documents, so a
# retriever that just matches generic vocabulary can't coast — it has to land
# on the specific one). Two headed sections each so chunking has something to
# split on with the small chunk_max_tokens the fixture pipeline uses below.
CORPUS = {
    "vacation_policy.md": """# Vacation Policy

## Annual Leave Accrual
Full-time employees accrue twenty vacation days per year, credited monthly at
a rate of 1.67 days. Unused vacation days roll over up to a maximum of ten
days into the following calendar year.

## Requesting Time Off
Employees submit vacation requests through the HR portal at least two weeks
in advance. Manager approval is required before travel bookings are made.
""",
    "sick_leave_policy.md": """# Sick Leave Policy

## Sick Day Allowance
Employees receive twelve paid sick days annually for personal illness or
medical appointments. Sick days do not roll over and reset every January.

## Reporting an Absence
Notify your manager before the start of the workday when taking a sick day.
A doctor's note is required after three consecutive sick days.
""",
    "expense_policy.md": """# Travel Expense Policy

## Reimbursable Expenses
Airfare, hotel, and ground transportation for approved business trips are
reimbursed in full. Meals are reimbursed up to fifty dollars per day with an
itemized receipt.

## Submitting a Claim
Submit expense claims through the finance portal within thirty days of travel
completion. Reimbursement is issued within two pay cycles of approval.
""",
    "security_policy.md": """# Account Security Policy

## Password Requirements
Passwords must be at least fourteen characters and rotated every ninety
days. Reusing any of your last ten passwords is not permitted.

## Reporting a Compromise
Report a suspected compromised account to the security team immediately
through the incident hotline. Do not attempt to reset credentials yourself.
""",
}

# (query, expected document title) — two per document, phrased around each
# section's own vocabulary rather than the whole document's.
QUERIES = [
    ("How many vacation days do employees accrue each year?", "vacation_policy"),
    ("How far in advance should I submit a time off request?", "vacation_policy"),
    ("How many paid sick days are employees given annually?", "sick_leave_policy"),
    ("When is a doctor's note required after taking sick days?", "sick_leave_policy"),
    ("Are meals reimbursed during an approved business trip?", "expense_policy"),
    ("How long do I have to submit a travel expense claim?", "expense_policy"),
    ("How often must account passwords be rotated?", "security_policy"),
    ("Who do I contact if my account is compromised?", "security_policy"),
]


def _seed(tmp_path, index):
    pipeline = IngestionPipeline(index, chunk_max_tokens=30, chunk_min_tokens=5, chunk_overlap_tokens=4)
    for name, text in CORPUS.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        report = pipeline.ingest_path(path)[0]
        assert report.action == "created", f"{name}: {report.error}"


def _recall_and_mrr(index, k: int) -> tuple[float, float]:
    hits_at_k = 0
    reciprocal_ranks = []
    for query, expected_title in QUERIES:
        titles = [hit.title for hit in index.search(query, top_k=k)]
        if expected_title in titles:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (titles.index(expected_title) + 1))
        else:
            reciprocal_ranks.append(0.0)
    return hits_at_k / len(QUERIES), sum(reciprocal_ranks) / len(QUERIES)


@pytest.fixture
def sqlite_index(tmp_path):
    index = SqliteDocumentIndex(connect(tmp_path), _HashEmbedder())
    _seed(tmp_path, index)
    return index


def test_sqlite_hybrid_clears_a_recall_and_mrr_floor(sqlite_index):
    recall, mrr = _recall_and_mrr(sqlite_index, k=3)
    assert recall >= 0.875, f"recall@3 = {recall} — below floor, a retrieval change regressed ranking"
    assert mrr >= 0.875, f"MRR = {mrr} — below floor, correct chunks are ranking too low"


@pytest.fixture
def qdrant_index(tmp_path):
    qdrant_client = pytest.importorskip("qdrant_client")
    from tini.documents.qdrant_store import QdrantDocumentIndex

    client = qdrant_client.QdrantClient(location=":memory:")
    index = QdrantDocumentIndex(Settings(), _HashEmbedder(), client=client)
    _seed(tmp_path, index)
    return index


def test_qdrant_vector_only_clears_a_recall_and_mrr_floor(qdrant_index):
    """Lower MRR floor than SQLite's, on purpose: this backend has no FTS5 leg
    to fall back on (see qdrant_store.py's module docstring), so a correct
    chunk landing 2nd instead of 1st on a pure-vector toy embedder is the
    documented trade-off showing up as a number, not a bug. Measured 0.875
    recall / 0.646 MRR on this corpus when this test was written — floors
    below leave headroom without hiding an actual regression."""
    recall, mrr = _recall_and_mrr(qdrant_index, k=3)
    assert recall >= 0.875, f"recall@3 = {recall} — below floor, a retrieval change regressed ranking"
    assert mrr >= 0.6, f"MRR = {mrr} — below floor, correct chunks are ranking too low"


def test_both_backends_are_within_reach_of_each_other(sqlite_index, qdrant_index):
    """Not a claim that hybrid and pure-vector rank identically — SqliteDocumentIndex
    fuses FTS5 with vector search and QdrantDocumentIndex doesn't (see that
    module's docstring). This just catches one backend silently collapsing
    to near-random ranking while the other still looks fine, which a per-
    backend floor alone wouldn't necessarily flag if both floors were loose."""
    sqlite_recall, sqlite_mrr = _recall_and_mrr(sqlite_index, k=3)
    qdrant_recall, qdrant_mrr = _recall_and_mrr(qdrant_index, k=3)
    assert abs(sqlite_recall - qdrant_recall) <= 0.375
    assert abs(sqlite_mrr - qdrant_mrr) <= 0.375
