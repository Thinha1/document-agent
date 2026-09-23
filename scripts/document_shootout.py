"""Document-index shootout — same corpus, different backends, real receipts.

    python scripts/document_shootout.py sqlite qdrant
    make document-shootout BACKENDS="sqlite qdrant"

`evals/deterministic/test_retrieval_quality.py` pins a network-free recall@k/
MRR FLOOR using a toy hashing embedder — that's a CI gate, not a benchmark,
and it can't honestly time a real embedding call or a real vector index. This
script measures the same corpus and the same metrics through your REAL,
currently-configured embedding provider (`TINI_EMBEDDING_PROVIDER` in .env)
against each real backend, and adds what the floor eval deliberately skips:
per-query latency.

Each backend gets a fresh isolated home + a unique workspace id, so repeated
runs always re-embed and re-index rather than skipping unchanged content.

Output: a markdown table on stdout plus a timestamped .md + .json report in
.tini/shootout/ — same convention as shootout.py's model comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tini.config import Settings, load_settings  # noqa: E402
from tini.db import connect  # noqa: E402
from tini.documents.embeddings import make_embedding_provider  # noqa: E402
from tini.documents.pipeline import IngestionPipeline  # noqa: E402
from tini.documents.store import make_document_index  # noqa: E402

# Same shape as evals/deterministic/test_retrieval_quality.py's fixture corpus,
# duplicated rather than imported: that file's job is a pinned CI floor with a
# toy embedder; this script's job is a live measurement with a real one, and
# the two must be free to diverge without one breaking the other.
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


def run_one(backend: str, k: int = 3) -> dict:
    home = Path(tempfile.mkdtemp(prefix=f"doc-shootout-{backend}-"))
    workspace = f"shootout-{uuid.uuid4().hex[:8]}"  # fresh every run: never skip as "unchanged"
    settings = Settings(home=home, document_index=backend, document_workspace=workspace)
    embedder = make_embedding_provider(settings)
    if embedder is None:
        raise SystemExit(
            "TINI_EMBEDDING_PROVIDER is not configured — this shootout measures real "
            "embedding + backend latency, so it needs one (see .env.example). "
            "Self-hosted works too: TINI_EMBEDDING_PROVIDER=openai_compatible pointed "
            "at Ollama/LM Studio/etc."
        )
    conn = connect(home)
    index = make_document_index(settings, conn, embedder)
    pipeline = IngestionPipeline(index)

    print(f"\n=== {backend} — ingesting {len(CORPUS)} documents ({embedder.provider}:{embedder.model}) ===")
    t_ingest = time.perf_counter()
    for name, text in CORPUS.items():
        path = home / name
        path.write_text(text, encoding="utf-8")
        report = pipeline.ingest_path(path, workspace_id=workspace)[0]
        if report.action == "failed":
            raise SystemExit(f"{backend}: ingesting {name} failed: {report.error}")
    ingest_wall_s = time.perf_counter() - t_ingest

    latencies_ms, hits_at_k, reciprocal_ranks = [], 0, []
    for query, expected_title in QUERIES:
        t0 = time.perf_counter()
        hits = index.search(query, top_k=k, workspace_id=workspace)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        titles = [hit.title for hit in hits]
        if expected_title in titles:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (titles.index(expected_title) + 1))
        else:
            reciprocal_ranks.append(0.0)
        print(f"  {'HIT ' if expected_title in titles else 'MISS'} {latencies_ms[-1]:6.1f}ms  {query[:60]}")

    conn.close()
    n = len(QUERIES)
    return {
        "backend": backend,
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model,
        "recall_at_k": round(hits_at_k / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "k": k,
        "avg_query_latency_ms": round(sum(latencies_ms) / n, 1),
        "max_query_latency_ms": round(max(latencies_ms), 1),
        "min_query_latency_ms": round(min(latencies_ms), 1),
        "ingest_wall_s": round(ingest_wall_s, 2),
    }


def markdown(results: list[dict]) -> str:
    lines = [
        "| backend | recall@k | MRR | avg query latency | min / max | ingest wall |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['backend']} | {r['recall_at_k']} | {r['mrr']} "
            f"| {r['avg_query_latency_ms']}ms | {r['min_query_latency_ms']}/{r['max_query_latency_ms']}ms "
            f"| {r['ingest_wall_s']}s |"
        )
    lines.append("")
    if results:
        lines.append(f"Embedding: {results[0]['embedding_provider']}:{results[0]['embedding_model']}. ")
    lines.append(
        f"{len(QUERIES)} labeled queries over {len(CORPUS)} documents, k={results[0]['k'] if results else '?'}. "
        "Real embedding calls and real backends — numbers depend on your machine, your "
        "embedding provider's latency, and (for Qdrant) whether it's local Docker or "
        "remote. Re-run with your own setup: `make document-shootout`. For the pinned "
        "network-free floor this compares against, see "
        "evals/deterministic/test_retrieval_quality.py."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare document-index backends: speed and accuracy.")
    parser.add_argument(
        "backends", nargs="*", default=["sqlite", "qdrant"], help="document index backends to compare"
    )
    parser.add_argument("--k", type=int, default=3, help="top_k for search (default 3)")
    args = parser.parse_args()

    results = [run_one(backend, k=args.k) for backend in args.backends]
    table = markdown(results)
    print("\n" + table)

    out_dir = load_settings().home / "shootout"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"document-{stamp}.md").write_text(table + "\n", encoding="utf-8")
    (out_dir / f"document-{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nreport: {out_dir}/document-{stamp}.md")


if __name__ == "__main__":
    main()
