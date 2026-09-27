"""Generate a controlled cross-language evaluation set from documents already in the repo.

    python scripts/synth_corpus.py build     # excerpt -> translate -> write questions
    python scripts/synth_corpus.py ingest    # load the generated documents into a workspace

Why this exists: the real DENSO corpus has no Vietnamese documents, denso.com
blocks automated downloads, and its questions mix languages in ways that cannot
isolate "does retrieval survive a language change". Here every generated
document exists in exactly ONE language, and every fact is asked in all three
(en/ja/vi), so for each fact one question is same-language and two are
cross-language, over the same answer.

What is real and what is not:
- The text is an excerpt of a real DENSO PDF from `examples/denso_mock_docs/`,
  machine-translated by the configured model when the target language differs
  from the source. It is NOT an authentic Vietnamese/English/Japanese DENSO
  publication: a translated excerpt is cleaner than a real PDF (no table
  debris), so absolute scores on it are optimistic.
- The questions are written by the same kind of model from the document text,
  so they can echo its wording. Every answer is checked mechanically: the
  `expected_keywords` string must occur verbatim in the document, and a fact is
  dropped if any of its three questions contains the answer.
- Use it to compare retrieval configurations and query languages against each
  other, not to quote a recall figure.

House style for the questions (checked by `check`, and by a test on the committed
files, so the set stays uniform when it is edited):
- self-contained: names the subject (DENSO, the policy, the report topic) and any
  fiscal period; never "according to the text", "this document", "the current year";
- one sentence, one answer, ending in a question mark; similar length across facts;
- the answer is a verbatim, distinctive string of at least 4 characters, with its
  unit when it is a number, so it cannot match a chunk by accident;
- the en/ja/vi versions ask the same thing at the same level of detail;
- exactly FACTS_PER_DOC facts per document.

`build` drafts documents and questions with a model; the committed questions were
then edited by hand to that style, so `build` refuses to overwrite them without
--force.

Outputs (deterministic names, safe to regenerate):
    examples/denso_synthetic/<name>.txt            the documents
    evals/documents/synthetic_denso_xl.jsonl       the questions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SOURCE_DIR = REPO / "examples" / "denso_mock_docs"
DOC_DIR = REPO / "examples" / "denso_synthetic"
QUESTIONS = REPO / "evals" / "documents" / "synthetic_denso_xl.jsonl"

LANGUAGES = {"en": "English", "ja": "Japanese", "vi": "Vietnamese"}
MIN_ANSWER_CHARS = 4
QUESTION_LENGTH = {"en": (60, 135), "ja": (25, 90), "vi": (55, 150)}  # characters
# Phrases that make a question depend on a document the retriever cannot see.
DEPENDENT_PHRASES = {
    "en": ["according to", "this document", "this policy", "the text", "the document",
           "the passage", "current consolidated", "in this "],
    "ja": ["本文", "この文書", "本方針", "この方針", "当連結会計年度", "この資料", "本書"],
    "vi": ["theo văn bản", "trong tài liệu này", "chính sách này", "năm tài chính hiện tại",
           "văn bản này"],
}
EXCERPT_CHARS = 10_000
SLICE_CHARS = 3_000  # one translation call per slice: keeps a reasoning model's output bounded
FACTS_PER_DOC = 4


@dataclass(frozen=True)
class Plan:
    doc_id: str
    source: str  # file in examples/denso_mock_docs
    start: float  # where the excerpt begins, as a fraction of the extracted text
    language: str  # the language the generated document is written in

    @property
    def name(self) -> str:
        return f"syn-{self.doc_id}-{self.language}"


# Each source excerpt appears in exactly one language. Two Japanese sources give
# two non-overlapping excerpts each, so English and Vietnamese are covered
# without a parallel copy in another language.
PLAN = [
    Plan("compliance-code", "compliance-doc-employee-behavior-guideline-en.pdf", 0.20, "vi"),
    Plan("governance-a", "management-doc-corporate-governance-2025-en.pdf", 0.20, "ja"),
    Plan("governance-b", "management-doc-corporate-governance-2025-en.pdf", 0.60, "en"),
    Plan("qa-manual", "denso-ten_qa_manual.pdf", 0.20, "ja"),
    Plan("human-rights", "humanrights-doc-human-rights-policy-en.pdf", 0.0, "ja"),
    Plan("integrated-report-a", "annual-report-doc-2025-viewing-ja.pdf", 0.20, "en"),
    Plan("integrated-report-b", "annual-report-doc-2025-viewing-ja.pdf", 0.60, "vi"),
    Plan("securities-report-a", "edinet_securities_report_candidate.pdf", 0.20, "en"),
    Plan("securities-report-b", "edinet_securities_report_candidate.pdf", 0.60, "vi"),
]

# --- pure helpers (tested without a model) ------------------------------------


def squash(text: str) -> str:
    """Comparison form for checking an answer occurs in a document: NFKC, no
    whitespace, lower case. The same idea as the eval's own `_squash`."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def clean_text(text: str) -> str:
    """Collapse the layout whitespace PDF extraction leaves behind (and a model
    tends to echo back): runs of spaces become one, and blank-line runs one. A
    translated excerpt once came back as 700,000 characters, almost all spaces."""
    lines = [re.sub(r"[ \t\u3000]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def excerpt(text: str, start: float, chars: int = EXCERPT_CHARS) -> str:
    """A window of whole lines beginning `start` of the way through `text`."""
    text = text.strip()
    if len(text) <= chars:
        return text
    begin = int(len(text) * start)
    if begin:
        newline = text.find("\n", begin)
        begin = newline + 1 if newline != -1 else begin
    end = begin + chars
    cut = text.rfind("\n", begin, end)
    return text[begin : cut if cut > begin else end].strip()


def slices(text: str, limit: int = SLICE_CHARS) -> list[str]:
    """Split on line boundaries into pieces of at most `limit` characters (a
    single longer line is split hard)."""
    pieces: list[str] = []
    current = ""
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit and current:
            pieces.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current.strip():
        pieces.append(current)
    return pieces


def parse_json_object(reply: str):
    """The first JSON value in a model reply that may wrap it in prose or fences."""
    starts = [i for i in (reply.find("["), reply.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON in reply")
    start = min(starts)
    closer = "]" if reply[start] == "[" else "}"
    return json.loads(reply[start : reply.rindex(closer) + 1])


def valid_fact(fact: dict, document: str) -> str | None:
    """None if the fact is usable, else the reason it is not."""
    span = str(fact.get("answer_span") or "").strip()
    if not MIN_ANSWER_CHARS <= len(span) <= 60:
        return "answer_span length"
    if squash(span) not in squash(document):
        return "answer not in document"
    for code in LANGUAGES:
        question = str(fact.get(f"q_{code}") or "").strip()
        if not question:
            return f"missing q_{code}"
        if squash(span) in squash(question):
            return f"q_{code} contains the answer"
    return None


def style_problems(query: str, language: str) -> list[str]:
    """Ways a question breaks the house style (see the module docstring)."""
    problems = []
    low = query.lower()
    for phrase in DEPENDENT_PHRASES[language]:
        if phrase in low:
            problems.append(f"depends on its document ({phrase!r})")
    low_len, high_len = QUESTION_LENGTH[language]
    if not low_len <= len(query) <= high_len:
        problems.append(f"length {len(query)} outside {low_len}-{high_len}")
    if not query.rstrip().endswith(("?", "？")):
        problems.append("does not end with a question mark")
    if query.count("?") + query.count("？") > 1:
        problems.append("asks more than one question")
    return problems


def check_cases(cases: list[dict], documents: dict[str, str]) -> list[str]:
    """Every way the question set breaks its own rules; empty means it is sound."""
    problems: list[str] = []
    per_doc: dict[str, dict[str, set]] = {}
    for case in cases:
        name, language = case["expected_title"], case["lang"]
        if name not in documents:
            problems.append(f"{case['id']}: no document {name!r}")
            continue
        answer = case["expected_keywords"][0]
        if len(answer) < MIN_ANSWER_CHARS:
            problems.append(f"{case['id']}: answer {answer!r} is under {MIN_ANSWER_CHARS} characters")
        if squash(answer) not in squash(documents[name]):
            problems.append(f"{case['id']}: answer {answer!r} is not in {name}")
        if squash(answer) in squash(case["query"]):
            problems.append(f"{case['id']}: the question contains its own answer")
        for problem in style_problems(case["query"], language):
            problems.append(f"{case['id']}: {problem}")
        fact_id = case["id"].rsplit("-", 1)[0]
        per_doc.setdefault(name, {}).setdefault(fact_id, set()).add(language)
    for name, facts in per_doc.items():
        if len(facts) != FACTS_PER_DOC:
            problems.append(f"{name}: {len(facts)} facts, expected {FACTS_PER_DOC}")
        for fact_id, languages in facts.items():
            if languages != set(LANGUAGES):
                problems.append(f"{fact_id}: asked in {sorted(languages)}, expected all of {sorted(LANGUAGES)}")
    for plan in PLAN:
        if plan.name not in per_doc:
            problems.append(f"{plan.name}: no questions")
    return problems


def cases_for(plan: Plan, facts: list[dict]) -> list[dict]:
    rows = []
    for number, fact in enumerate(facts, start=1):
        for code in LANGUAGES:
            rows.append(
                {
                    "id": f"xl-{plan.doc_id}-{number}-{code}",
                    "lang": code,
                    "query": str(fact[f"q_{code}"]).strip(),
                    "expected_title": plan.name,
                    "expected_keywords": [str(fact["answer_span"]).strip()],
                    "format": f"{plan.language}-doc",
                }
            )
    return rows


# --- model-backed steps ---------------------------------------------------------

Chat = Callable[[str], str]


def make_chat(settings) -> Chat:
    from tini.loop.models import get_client

    client = get_client(settings)
    model = settings.small_model or settings.model

    def chat(prompt: str) -> str:
        last: Exception | None = None
        for _attempt in range(2):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=12000,  # a reasoning model thinks before it answers
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if text:
                    return text
                last = ValueError("empty reply")
            except Exception as exc:
                last = exc
        raise RuntimeError(f"model call failed twice: {last}")

    return chat


def translate(chat: Chat, text: str, language: str) -> str:
    out = []
    for piece in slices(text):
        out.append(
            chat(
                f"Translate the following text into {LANGUAGES[language]}. Keep every number, "
                "name, date, heading and line structure. It is an excerpt of a corporate "
                "document: do not summarise, add or omit anything, and do not comment. "
                f"Output only the translation.\n\n{piece}"
            )
        )
    return "\n".join(out)


FACT_PROMPT = """Below is a document written in {language}. Write {n} factual questions, each
answerable from one specific passage of it.

For each question give:
  "answer_span": a short string (2 to 60 characters: a number, a name, a date or a key
                 term) copied EXACTLY, character for character, from the document, that is
                 the answer;
  "q_en", "q_ja", "q_vi": the same question in English, Japanese and Vietnamese.

Rules: a question must NOT contain its answer_span; do not copy long phrases from the
document into the question; pick different parts of the document; skip tables of
figures with no context. Each question must make sense WITHOUT the document: name the
subject (DENSO, the policy or report topic) and any fiscal period, and never write
"according to the text", "this document" or "the current year". One sentence, one
answer, ending in a question mark. The answer_span must be at least 4 characters and
carry its unit when it is a number. Reply with a JSON list only.

Document:
{document}"""


def write_facts(chat: Chat, document: str, language: str, n: int = FACTS_PER_DOC) -> list[dict]:
    reply = chat(FACT_PROMPT.format(language=LANGUAGES[language], n=n + 2, document=document))
    facts = parse_json_object(reply)
    return facts if isinstance(facts, list) else []


def build_one(chat: Chat, plan: Plan, source_text: str) -> tuple[str, list[dict], list[str]]:
    """(document text, cases, notes about facts that were dropped)."""
    from tini.documents.multilingual import detect_language

    part = excerpt(source_text, plan.start)
    native = detect_language(part[:2000])
    document = clean_text(part if native == plan.language else translate(chat, part, plan.language))
    facts, notes = [], []
    for fact in write_facts(chat, document, plan.language):
        reason = valid_fact(fact, document)
        if reason:
            notes.append(f"{plan.name}: dropped a fact ({reason})")
        elif len(facts) < FACTS_PER_DOC:
            facts.append(fact)
    return document, cases_for(plan, facts), notes


def extract_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents() -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(DOC_DIR.glob("*.txt"))}


def cmd_check(args) -> int:
    documents = load_documents()
    cases = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line]
    problems = check_cases(cases, documents)
    for problem in problems:
        print(problem)
    print(f"{len(cases)} questions, {len(documents)} documents: {len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_build(args) -> int:
    existing = [p for p in (QUESTIONS, *DOC_DIR.glob("syn-*.txt")) if p.exists()]
    if existing and not args.force:
        print(
            "refusing to overwrite the generated documents and hand-edited questions "
            f"({len(existing)} files exist). Pass --force to regenerate them all.",
            file=sys.stderr,
        )
        return 2
    from tini.config import load_settings

    settings = load_settings()
    chat = make_chat(settings)
    print(f"model: {settings.small_model or settings.model} ({settings.provider})", file=sys.stderr)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    for plan in PLAN:
        if plan.source not in sources:
            sources[plan.source] = extract_text(SOURCE_DIR / plan.source)

    def run(plan: Plan):
        try:
            return plan, build_one(chat, plan, sources[plan.source]), None
        except Exception as exc:
            return plan, None, exc

    cases: list[dict] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for plan, result, error in pool.map(run, PLAN):
            if error is not None:
                failures += 1
                print(f"FAILED {plan.name}: {error}", file=sys.stderr)
                continue
            document, rows, notes = result
            (DOC_DIR / f"{plan.name}.txt").write_text(document + "\n", encoding="utf-8")
            cases.extend(rows)
            for note in notes:
                print(note, file=sys.stderr)
            print(f"{plan.name}: {len(document)} chars, {len(rows) // 3} facts", file=sys.stderr)
    QUESTIONS.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
    )
    print(f"wrote {len(cases)} questions to {QUESTIONS.relative_to(REPO)}")
    return 1 if failures else 0


def cmd_ingest(args) -> int:
    from tini.config import load_settings
    from tini.db import connect
    from tini.documents.embeddings import make_embedding_provider
    from tini.documents.pipeline import IngestionPipeline
    from tini.documents.store import make_document_index

    settings = load_settings()
    conn = connect(settings.home)
    embedder = make_embedding_provider(settings)
    index = make_document_index(settings, conn, embedder)
    pipeline = IngestionPipeline(
        index,
        embedder=embedder,
        chunk_max_tokens=settings.document_chunk_max_tokens,
        chunk_min_tokens=settings.document_chunk_min_tokens,
        chunk_overlap_tokens=settings.document_chunk_overlap_tokens,
    )
    reports = pipeline.ingest_path(DOC_DIR, workspace_id=args.workspace, recursive=False)
    for report in reports:
        print(f"{report.action:8} {report.chunk_count:4} chunks  {report.title}")
    print(f"workspace {args.workspace!r}: {len(reports)} documents")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="generate documents and questions with the configured model")
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--force", action="store_true", help="overwrite existing documents and questions")
    build.set_defaults(run=cmd_build)
    check = sub.add_parser("check", help="verify the committed questions against the documents")
    check.set_defaults(run=cmd_check)
    ingest = sub.add_parser("ingest", help="load the generated documents into a workspace")
    ingest.add_argument("--workspace", default="denso-xl")
    ingest.set_defaults(run=cmd_ingest)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
