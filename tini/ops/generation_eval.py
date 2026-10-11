"""Generation eval: what the agent DOES with what it retrieved.

Retrieval evals ask "were the right chunks returned". This asks the next thing:
given the real system prompt, the real `search_documents` tool and the real loop,
does the reply

* answer correctly (the expected facts are in it, and were they even retrieved?),
* stay faithful to the evidence (no claim the returned chunks do not support),
* cite the document it used, and the page (a PDF answer) it came from,
* come back in the language of the question,
* and DECLINE when the documents do not hold the answer (a table row that does not
  exist, a year the reports never reach) instead of inventing one.

It runs the same turn `Tini.respond` runs (`Session.build_system`,
`Session.build_context`, `run_loop` with the tool registry) minus memory, the
retrieval gate and the chat log, so a run never touches `.tini/` state and never
learns from its own questions.

What is measured mechanically and what is a model's opinion:

* keywords in the reply, keywords in the evidence, the cited title, the language of
  the reply (`detect_language`), the number of searches: all deterministic;
* faithfulness and "did it decline": a judge model reads the question, the evidence
  the agent actually saw and the reply. That is an opinion, from a model that should
  not be the one that generated the reply, and it can be wrong; `judge` in each row
  keeps its answer so a person can audit the calls that matter.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from tini.config import Settings
from tini.db import connect
from tini.documents.embeddings import make_embedding_provider
from tini.documents.multilingual import detect_language
from tini.documents.store import make_document_index
from tini.loop.agent import run_loop
from tini.loop.models import get_client
from tini.ops.document_eval import _load_cases, _squash
from tini.runtime.session import Session
from tini.tools import build_registry

UNANSWERABLE = Path(__file__).resolve().parents[2] / "evals" / "generation" / "unanswerable.jsonl"
EVIDENCE_LIMIT = 60000  # characters of tool output shown to the judge (a turn can search more than once)

# (question set, workspace, how many, source label). The count is spread evenly over
# the languages in the set; every source is sampled deterministically.
DEFAULT_SOURCES = (
    ("denso_qa_multilingual.jsonl", "denso", 12, "real"),
    ("denso_qa.jsonl", "denso", 6, "real"),
    ("synthetic_denso_xl.jsonl", "denso", 12, "synthetic-prose"),
    ("synthetic_denso_tech.jsonl", "denso", 12, "synthetic-table"),
)

JUDGE_PROMPT = """You grade a document question-answering assistant. You are given the \
question, the EVIDENCE its search tool returned (the only source it is allowed to use) \
and its REPLY.

Answer with JSON only:
{{"abstained": <true if the reply says the documents do not contain the answer / declines \
to give a value, false if it gives an answer>,
 "supported": "<yes if every factual claim in the reply is supported by the EVIDENCE, \
'partly' if some are not, 'no' if the substance is not; a reply that only says it cannot \
find the answer is 'yes'>",
 "unsupported": [<each claim in the reply the EVIDENCE does not support, quoted briefly>]}}

Judge only against the EVIDENCE, not against what you know. A number that appears in the \
evidence for a DIFFERENT row, column, year or item than the one asked about does not \
support an answer about the asked one.

QUESTION:
{question}

EVIDENCE:
{evidence}

REPLY:
{reply}"""


def load_cases(sources=DEFAULT_SOURCES, settings: Settings | None = None) -> list[dict]:
    """A deterministic sample of answerable questions plus every unanswerable one."""
    from tini.config import load_settings

    settings = settings or load_settings()
    cases: list[dict] = []
    for name, workspace, count, source in sources:
        rows = sorted(_load_cases(name, settings), key=lambda c: c.get("id", ""))
        by_lang: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_lang[str(row.get("lang") or "en")].append(row)
        per_lang = max(1, count // len(by_lang))
        for lang in sorted(by_lang):
            group = by_lang[lang]
            step = len(group) / min(per_lang, len(group))
            for i in range(min(per_lang, len(group))):
                row = group[int(i * step)]
                cases.append({**row, "lang": lang, "workspace": workspace, "source": source, "abstain": False})
    for line in UNANSWERABLE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append({**json.loads(line), "source": "unanswerable"})
    return cases


SOURCE_NAMES = ("real", "synthetic-prose", "synthetic-table", "unanswerable")


def select_cases(cases: list[dict], sources=None, limit: int | None = None) -> list[dict]:
    """The questions to run: only the chosen sources, then at most `limit` of them
    spread evenly across the whole list so a small run still touches every source
    (a plain prefix would never reach the unanswerable ones, which come last)."""
    picked = [c for c in cases if not sources or c["source"] in sources]
    if limit and 0 < limit < len(picked):
        step = len(picked) / limit
        picked = [picked[int(i * step)] for i in range(limit)]
    return picked


def _evidence(tool_calls: list[dict]) -> str:
    return "\n\n".join(str(c.get("output", "")) for c in tool_calls if c.get("tool") == "search_documents")


def judge_reply(client, model: str, question: str, evidence: str, reply: str) -> dict | None:
    prompt = JUDGE_PROMPT.format(
        question=question, evidence=evidence[:EVIDENCE_LIMIT] or "(the assistant searched nothing)", reply=reply[:4000]
    )
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model, max_tokens=1500, messages=[{"role": "user", "content": prompt}]
            )
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
            return {
                "abstained": bool(obj.get("abstained")),
                "supported": str(obj.get("supported", "")).lower(),
                "unsupported": [str(x)[:160] for x in (obj.get("unsupported") or [])][:5],
            }
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def run_case(
    settings: Settings,
    case: dict,
    *,
    judge: tuple[Any, str] | None,
    reply_language: bool = True,
    client: Any = None,
) -> dict:
    """One question through the real turn, then the mechanical checks and the judge."""
    settings = replace(settings, document_workspace=case["workspace"])
    conn = connect(settings.home)
    started = time.perf_counter()
    try:
        index = make_document_index(settings, conn, make_embedding_provider(settings))
        tools = build_registry(settings, index)
        session = Session(settings, memory=None)
        context = session.build_context(case["query"], reply_language=reply_language)
        content = f"[Context for this turn, not written by the user]\n{context}\n\n[User message]\n{case['query']}"
        result = run_loop(
            client=client or get_client(settings),
            model=settings.model,
            system=session.build_system(),
            messages=[{"role": "user", "content": content}],
            tools=tools,
            max_iterations=settings.max_iterations,
            max_tokens=settings.max_tokens,
        )
    finally:
        conn.close()
    seconds = time.perf_counter() - started
    reply, calls = result.reply.strip(), result.tool_calls
    evidence = _evidence(calls)
    keywords = case.get("expected_keywords") or []
    title = str(case.get("expected_title") or "")
    row = {
        "id": case.get("id", ""),
        "source": case["source"],
        "lang": case["lang"],
        "format": case.get("format", ""),
        "abstain": bool(case.get("abstain")),
        "query": case["query"],
        "reply": reply,
        "searches": [str(c.get("args", {}).get("query", "")) for c in calls if c.get("tool") == "search_documents"],
        "tool_ok": bool(calls),
        "language_ok": detect_language(reply) == detect_language(case["query"]) if reply else False,
        "seconds": round(seconds, 1),
        "tokens_in": result.usage.get("in", 0),
        "tokens_out": result.usage.get("out", 0),
    }
    if not case.get("abstain"):
        row["evidence_has_answer"] = bool(keywords) and all(_squash(k) in _squash(evidence) for k in keywords)
        row["reply_has_answer"] = bool(keywords) and all(_squash(k) in _squash(reply) for k in keywords)
        row["cited"] = bool(title) and title.lower() in reply.lower()
        expected_pages = answer_pages(evidence, keywords, title)
        if expected_pages:  # a PDF answer: was the page named, and was it the right one?
            named = cited_pages(reply)
            row["pages_expected"] = sorted(expected_pages)
            row["pages_cited"] = sorted(named)
            row["page_cited"] = bool(named)
            row["page_ok"] = bool(named & expected_pages)
    if judge is not None and reply:
        row["judge"] = judge_reply(judge[0], judge[1], case["query"], evidence, reply)
    return row


def run_generation_eval(
    cases: list[dict],
    settings: Settings,
    *,
    judge_settings: Settings | None = None,
    reply_language: bool = True,
    workers: int = 3,
    progress: Callable[[int, int], None] | None = None,
    on_row: Callable[[dict, int, int], None] | None = None,
    stop: threading.Event | None = None,
) -> dict:
    judge = None
    if judge_settings is not None:
        judge = (get_client(judge_settings), judge_settings.model)
    rows: list[dict | None] = [None] * len(cases)
    finished = threading.Lock()  # one row reported at a time, so progress never interleaves

    def work(i: int) -> None:
        if stop is not None and stop.is_set():
            return  # cancelled: the rest of the queue is skipped, not run
        try:
            rows[i] = run_case(settings, cases[i], judge=judge, reply_language=reply_language)
        except Exception as exc:  # one failed question must not lose the run
            rows[i] = {"id": cases[i].get("id", ""), "source": cases[i]["source"], "lang": cases[i]["lang"],
                       "abstain": bool(cases[i].get("abstain")), "query": cases[i]["query"], "error": f"{type(exc).__name__}: {exc}"[:200]}
        with finished:
            done = sum(r is not None for r in rows)
            if progress:
                progress(done, len(cases))
            if on_row:
                on_row(rows[i], done, len(cases))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, range(len(cases))))
    done = [r for r in rows if r is not None]
    return {
        "generator": {"provider": settings.provider, "model": settings.model},
        "judge": {"provider": judge_settings.provider, "model": judge_settings.model} if judge_settings else None,
        "rows": done,
        "summary": summarize(done),
        "cancelled": bool(stop is not None and stop.is_set()),
    }


# One search result as `search_documents` prints it: `1. [title#chunk-3, pp.7-8]`, the
# chunk text, then its `Source:` line. The page part is absent for a source with no pages.
_RESULT_HEAD = re.compile(r"^\d+\. \[(?P<title>.+?)#chunk-\d+(?:, pp?\.(?P<first>\d+)(?:-(?P<last>\d+))?)?\]$", re.MULTILINE)
# A page a reply names: `p.7`, `pp. 7-8`, `page 7`, `pages 7–8`, `trang 7`, `ページ7`.
_TO = r"[-–—~〜]|\bto\b|\bđến\b"  # what joins the two ends of a range
_CITED_PAGE = re.compile(
    rf"(?:\bpp?\.\s*|\bpages?\s+|\btrang\s+|ページ\s*)(\d{{1,4}})(?:\s*(?:{_TO})\s*(\d{{1,4}}))?"
    rf"|(\d{{1,4}})(?:\s*(?:{_TO})\s*(\d{{1,4}}))?\s*ページ",  # Japanese puts the number first: 7ページ
    re.IGNORECASE,
)


def answer_pages(evidence: str, keywords: list[str], title: str) -> set[int]:
    """Pages of the returned chunks that hold the whole answer: the chunks of the expected
    document that contain every expected keyword. Empty if none does, or if those chunks
    come from a source without pages."""
    heads = list(_RESULT_HEAD.finditer(evidence))
    pages: set[int] = set()
    for i, head in enumerate(heads):
        body = evidence[head.end(): heads[i + 1].start() if i + 1 < len(heads) else len(evidence)]
        if title and head["title"].lower() != title.lower():
            continue
        if keywords and not all(_squash(k) in _squash(body) for k in keywords):
            continue
        if head["first"]:
            first = int(head["first"])
            pages.update(range(first, int(head["last"] or first) + 1))
    return pages


def cited_pages(reply: str) -> set[int]:
    """Every page number the reply names (a range names all pages in it)."""
    pages: set[int] = set()
    for match in _CITED_PAGE.finditer(reply):
        first = int(match.group(1) or match.group(3))
        last = int(match.group(2) or match.group(4) or first)
        if first <= last <= first + 30:  # a stray "p.1-900" is not a citation of 900 pages
            pages.update(range(first, last + 1))
    return pages


def _rate(num: int, den: int) -> str:
    return f"{num}/{den}" if den else "-"


def summarize(rows: list[dict]) -> dict:
    """Counts, not percentages: at this size a percentage invites more confidence than
    the sample earns."""
    ok = [r for r in rows if "error" not in r]

    def group(members: list[dict]) -> dict:
        answerable = [r for r in members if not r["abstain"]]
        unanswerable = [r for r in members if r["abstain"]]
        judged = [r for r in members if r.get("judge")]
        has_evidence = [r for r in answerable if r["evidence_has_answer"]]
        return {
            "n": len(members),
            "answerable": len(answerable),
            "answered": sum(r["reply_has_answer"] for r in answerable),
            "retrieval_limited": sum(not r["evidence_has_answer"] for r in answerable),
            "generation_missed": sum(not r["reply_has_answer"] for r in has_evidence),
            "generation_of_evidence": len(has_evidence),
            "cited": sum(r["cited"] for r in answerable if r["reply_has_answer"]),
            "answered_and_cited_base": sum(r["reply_has_answer"] for r in answerable),
            # only answers whose chunk has a page (PDFs), and only ones the reply got right
            "page_base": sum(1 for r in answerable if r["reply_has_answer"] and r.get("pages_expected")),
            "page_cited": sum(bool(r.get("page_cited")) for r in answerable if r["reply_has_answer"]),
            "page_ok": sum(bool(r.get("page_ok")) for r in answerable if r["reply_has_answer"]),
            "language_ok": sum(r["language_ok"] for r in members),
            "searched": sum(r["tool_ok"] for r in members),
            "faithful": sum(r["judge"]["supported"] == "yes" for r in judged),
            "judged": len(judged),
            "declined": sum(bool(r["judge"]["abstained"]) for r in unanswerable if r.get("judge")),
            "unanswerable": len(unanswerable),
            "invented": sum(
                (not r["judge"]["abstained"]) or r["judge"]["supported"] != "yes"
                for r in unanswerable
                if r.get("judge")
            ),
            "errors": 0,
        }

    summary = {"all": group(ok), "errors": len(rows) - len(ok)}
    for key in ("lang", "source"):
        for value in sorted({r[key] for r in ok}):
            summary[f"{key}:{value}"] = group([r for r in ok if r[key] == value])
    return summary


def format_summary(summary: dict) -> str:
    lines = [
        (
            f"{'group':22} {'answered':>9} {'retr.limited':>13} {'gen.missed':>11} {'cited':>7} {'page':>7} "
            f"{'lang ok':>8} {'faithful':>9} {'declined':>9} {'invented':>9}"
        )
    ]
    for name, g in summary.items():
        if not isinstance(g, dict):
            continue
        lines.append(
            f"{name:22} {_rate(g['answered'], g['answerable']):>9} {g['retrieval_limited']:>13} "
            f"{_rate(g['generation_missed'], g['generation_of_evidence']):>11} "
            f"{_rate(g['cited'], g['answered_and_cited_base']):>7} {_rate(g['page_ok'], g['page_base']):>7} "
            f"{_rate(g['language_ok'], g['n']):>8} "
            f"{_rate(g['faithful'], g['judged']):>9} {_rate(g['declined'], g['unanswerable']):>9} "
            f"{_rate(g['invented'], g['unanswerable']):>9}"
        )
    return "\n".join(lines)
