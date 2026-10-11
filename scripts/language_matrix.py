"""Break a synthetic cross-language snapshot down by (question language x document language).

    python scripts/language_matrix.py evals/baselines/xl-dense-2026-09-26.json [more.json ...]

Works on snapshots of `synthetic_denso_xl.jsonl` (see scripts/synth_corpus.py):
every fact there is asked in en/ja/vi over a document that exists in exactly one
language, so the diagonal is same-language retrieval and everything else is
cross-language, over identical answers. Two hit definitions are shown:

  doc    the expected document is among the top-k results
  answer the expected answer text is inside the returned chunks (stricter: right
         document, wrong part of it, does not count)

With several files the columns line up for comparison.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

LANGS = ("en", "ja", "vi")


def load(path: Path) -> list[dict]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, run in snapshot["runs"].items():
        if not name.startswith("synthetic_"):
            continue
        for case in run["cases"]:
            doc_lang = (case.get("format") or "").split("-")[0]
            answered = bool(case["keyword_total"]) and len(case["keyword_hits"]) == case["keyword_total"]
            rows.append(
                {"query": case["lang"], "doc": doc_lang, "doc_hit": case["hit"], "answer_hit": answered}
            )
    return rows


def rate(rows: list[dict], key: str) -> str:
    if not rows:
        return "-"
    hits = sum(1 for r in rows if r[key])
    return f"{hits}/{len(rows)}"


def summarize(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["same language" if row["query"] == row["doc"] else "cross language"].append(row)
        groups[f"question {row['query']} -> document {row['doc']}"].append(row)
    groups["all"] = rows
    return groups


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1
    tables = {Path(p).stem: summarize(load(Path(p))) for p in paths}
    labels = list(tables)
    order = ["all", "same language", "cross language"] + [
        f"question {q} -> document {d}" for q in LANGS for d in LANGS if q != d
    ]
    for key, title in (("doc_hit", "document in top-k"), ("answer_hit", "answer in returned chunks")):
        print(f"\n{title}")
        print("| slice | " + " | ".join(labels) + " |")
        print("|---|" + "---|" * len(labels))
        for slice_name in order:
            cells = [rate(tables[label].get(slice_name, []), key) for label in labels]
            print(f"| {slice_name} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
