"""Score what the agent does with what it retrieves: answers, faithfulness, citation,
reply language, and declining when the documents do not hold the answer.

    python scripts/generation_eval.py run   --label deepseek-flash --provider deepseek --model deepseek-v4-flash \
        --judge-provider deepseek --judge-model deepseek-v4-pro [--no-language-note] [--limit 12]
    python scripts/generation_eval.py show  evals/baselines/generation-<label>-<date>.json
    python scripts/generation_eval.py compare a.json b.json

It runs the real turn (system prompt, `search_documents`, the loop) against the
already-ingested `denso` workspace (internal, synthetic and technical documents), in a scratch home so
`.tini/` is never touched, and calls a chat model and a judge model: real API calls
that cost real money. See tini/ops/generation_eval.py for what is deterministic and
what is the judge's opinion. Prefer a judge that is not the generator.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tini.config import load_settings  # noqa: E402
from tini.ops import generation_eval as ge  # noqa: E402


def _run(args) -> int:
    base = load_settings()
    home = Path(args.home) if args.home else Path(tempfile.gettempdir()) / "tini-generation-eval"
    home.mkdir(parents=True, exist_ok=True)
    settings = replace(base, home=home)
    if args.provider:
        settings = replace(settings, provider=args.provider, small_model=args.model or "")
    if args.model:
        settings = replace(settings, model=args.model)
    judge = None
    if args.judge_provider:
        judge = replace(settings, provider=args.judge_provider, model=args.judge_model or "", small_model="")
    cases = ge.load_cases(settings=base)
    cases = ge.select_cases(cases, sources=args.sources.split(",") if args.sources else None, limit=args.limit)
    print(f"{len(cases)} questions | generator {settings.provider}/{settings.model} | "
          f"judge {judge.provider + '/' + judge.model if judge else 'none'} | language note {'off' if args.no_language_note else 'on'}",
          flush=True)
    result = ge.run_generation_eval(
        cases, settings, judge_settings=judge, reply_language=not args.no_language_note, workers=args.workers,
        progress=lambda done, total: print(f"  {done}/{total}", end="\r", flush=True),
    )
    result["label"] = args.label
    result["language_note"] = not args.no_language_note
    out = ROOT / "evals" / "baselines" / f"generation-{args.label}-{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + ge.format_summary(result["summary"]))
    print(f"\nsaved {out.relative_to(ROOT)}")
    return 0


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _show(args) -> int:
    print(ge.format_summary(_load(args.file)["summary"]))
    return 0


def _compare(args) -> int:
    a, b = _load(args.a), _load(args.b)
    for label, run in (("A", a), ("B", b)):
        print(f"{label}: {run.get('label')}  (language note {'on' if run.get('language_note', True) else 'off'})")
        print(ge.format_summary(run["summary"]) + "\n")
    ra = {r["id"]: r for r in a["rows"] if "error" not in r}
    rb = {r["id"]: r for r in b["rows"] if "error" not in r}
    for key, label in (("language_ok", "reply language"), ("reply_has_answer", "answered")):
        gained = [i for i in ra if i in rb and rb[i].get(key) and not ra[i].get(key)]
        lost = [i for i in ra if i in rb and ra[i].get(key) and not rb[i].get(key)]
        print(f"{label}: B fixed {len(gained)} {gained} | B broke {len(lost)} {lost}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--provider")
    run.add_argument("--model")
    run.add_argument("--judge-provider")
    run.add_argument("--judge-model")
    run.add_argument("--no-language-note", action="store_true", help="drop the reply-language line, to measure what it is worth")
    run.add_argument("--limit", type=int, help="run an evenly spaced subset of the questions")
    run.add_argument("--sources", help=f"comma list of {', '.join(ge.SOURCE_NAMES)} (default: all)")
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--home", help="scratch TINI_HOME (default: a temp directory)")
    run.set_defaults(func=_run)
    show = sub.add_parser("show")
    show.add_argument("file")
    show.set_defaults(func=_show)
    compare = sub.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.set_defaults(func=_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
