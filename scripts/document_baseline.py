"""Snapshot the document-RAG pipeline's measured quality, and diff it later.

    python scripts/document_baseline.py bge-m3-dense-2026-09-24
    python scripts/document_baseline.py hybrid-rrf --compare evals/baselines/bge-m3-dense-2026-09-24.json

Runs every question set in evals/documents/ against the CURRENT pipeline (the
embedding model, index and chunking in your .env) and writes
`evals/baselines/<label>.json`: per-question results plus the pipeline config
that produced them. Change one thing (a reranker, a different chunk size, a
new embedding model), run this again under a new label with `--compare`, and
the output names which questions got fixed and which regressed — a table of
averages hides both.

A throwaway pass runs first so a local model's cold start (a single query can
take 100x a warm one) doesn't land in the numbers. Latency is reported as
median and p95, not the mean, for the same reason.

Retrieval only: no chat-model calls, so it costs nothing beyond local
embedding. Use the dashboard's "Also answer with the chat model" checkbox for
the end-to-end path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tini.config import load_settings  # noqa: E402
from tini.ops import document_eval  # noqa: E402

BASELINES = REPO / "evals" / "baselines"


def _row(name: str, result: dict) -> str:
    def groups(key: str) -> str:
        parts = [f"{k} {v['hits']}/{v['total']}" for k, v in result.get(key, {}).items()]
        return ", ".join(parts) or "-"

    return (
        f"| {name} | {result['hits']}/{result['total']} ({result['recall']:.1%}) "
        f"| {groups('by_lang')} | {groups('by_format')} "
        f"| {result['median_latency_ms']} / {result['p95_latency_ms']} ms |"
    )


def _flips(before: dict, after: dict) -> tuple[list[str], list[str]]:
    old = {c["id"]: c["hit"] for c in before["cases"]}
    fixed = [c["id"] for c in after["cases"] if c["hit"] and old.get(c["id"]) is False]
    broken = [c["id"] for c in after["cases"] if not c["hit"] and old.get(c["id"]) is True]
    return fixed, broken


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot and compare document-RAG quality.")
    parser.add_argument("label", help="name for this snapshot, e.g. bge-m3-dense-2026-09-24")
    parser.add_argument("--workspace", default="", help="default: TINI_DOCUMENT_WORKSPACE")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--compare", type=Path, help="an earlier snapshot to diff against")
    parser.add_argument(
        "--rerank",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto: follow TINI_RERANK_PROVIDER; on: require it; off: force vector order only",
    )
    args = parser.parse_args()

    settings = load_settings()
    workspace = args.workspace or settings.document_workspace
    rerank = {"auto": None, "on": True, "off": False}[args.rerank]
    sets = [
        s["name"] for s in document_eval.list_question_sets(settings) if s["source"] == "built-in"
    ]
    if not sets:
        raise SystemExit("no built-in question sets found under evals/documents/")

    print(f"warming up on {sets[0]} (not recorded)…", file=sys.stderr)
    document_eval.run_eval(
        {
            "question_set": sets[0],
            "workspace": workspace,
            "k": args.k,
            "save": False,
            "rerank": rerank,
        },
        settings,
    )

    runs = {}
    for name in sets:
        print(f"running {name}…", file=sys.stderr)
        result = document_eval.run_eval(
            {
                "question_set": name,
                "workspace": workspace,
                "k": args.k,
                "save": False,
                "rerank": rerank,
            },
            settings,
        )
        if "error" in result:
            raise SystemExit(f"{name}: {result['error']}")
        runs[name] = result

    snapshot = {
        "label": args.label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
    }
    BASELINES.mkdir(parents=True, exist_ok=True)
    target = BASELINES / f"{args.label}.json"
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    config = next(iter(runs.values()))["config"]
    emb = config["embedding"]
    print(f"\n### {args.label}")
    print(
        f"embedding `{emb['model']}` ({emb['dimensions']}d) · index `{config['index']['backend']}"
        f"/{config['index']['collection']}` · chunk max {config['chunking']['max_tokens']} tokens · "
        f"k={config['search_k']} · rerank "
        f"{config['rerank']['provider'] + ' ×' + str(config['rerank']['candidates']) if config['rerank']['enabled'] else 'off'}"
        f" · {len(config['corpus'])} docs, "
        f"{sum(d['chunks'] for d in config['corpus'])} chunks\n"
    )
    print("| question set | recall@k | by language | by format | latency median / p95 |")
    print("|---|---|---|---|---|")
    for name, result in runs.items():
        print(_row(name, result))

    if args.compare:
        previous = json.loads(args.compare.read_text(encoding="utf-8"))
        print(f"\n### vs {previous['label']}")
        for name, result in runs.items():
            before = previous["runs"].get(name)
            if before is None:
                continue
            fixed, broken = _flips(before, result)
            print(
                f"- {name}: {before['hits']}/{before['total']} -> {result['hits']}/{result['total']}"
                f" | fixed: {', '.join(fixed) or 'none'} | regressed: {', '.join(broken) or 'none'}"
            )
    print(f"\nsaved {target.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
