"""Cross-language retrieval: search in the language the documents are written in.

A multilingual embedding puts "When was DENSO founded?" and 「デンソーはいつ設立
されましたか」 near each other, but not near enough to reliably beat an English
paragraph that merely shares the question's words. Measured on the DENSO set
(docs/document-ingestion.md), the questions whose answer sits in a document of
another language are where retrieval loses the most, and a keyword ranking
cannot help there: no word is shared.

The fix is to remove the language gap before searching. `MultilingualIndex`
asks the small model to translate the question into each language the corpus
contains, searches with the original and every translation, and fuses the
rankings by reciprocal rank. The translations also give a keyword ranking
(hybrid search) words that actually occur in the document.

Failure is visible and cheap: if the model cannot translate, the search falls
back to the original question alone and the failure is counted, the same
contract as `RerankingIndex`. A translation that does not arrive within
`timeout` seconds counts as a failure too (an unreachable endpoint was seen to
hold each call for 6 minutes, which would freeze every search), and after a few
failures in a row translation is skipped for a while so a dead model costs one
short wait, not one per search. Cost is one small-model call per distinct
question (repeats are cached in memory).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence

from tini.documents.models import (
    DEFAULT_WORKSPACE,
    DocumentSource,
    DocumentSummary,
    IngestReport,
    SearchHit,
)

LANGUAGE_NAMES = {"ja": "Japanese", "en": "English", "vi": "Vietnamese"}

_KANA_HAN = re.compile("[぀-ヿ㐀-䶿一-鿿豈-﫿]")
# Letters that occur in Vietnamese and not in the other languages here: ă â đ ê ô
# ơ ư and vowels carrying the hook/dot/tilde tone marks. (Plain à á ã è é... are
# shared with other Latin-script languages, so they prove nothing.)
_VI_ONLY = re.compile(
    "[ăâđêôơưĂÂĐÊÔƠƯạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]"
)

RRF_K = 60
CACHE_SIZE = 256
# Translations shared by every index in the process. The dashboard and the eval
# harness build a fresh index per request, so a per-index cache would be empty
# each time and repeat every model call.
SHARED_CACHE: OrderedDict = OrderedDict()
_CACHE_LOCK = threading.Lock()
TRANSLATE_TIMEOUT = 20.0  # seconds to wait for a translation (a reasoning model takes 3-9 s)
BREAKER_FAILURES = 3  # consecutive failures before translation is paused...
BREAKER_COOLDOWN = 60.0  # ...for this many seconds

Translator = Callable[[str, Sequence[str]], dict[str, str]]


def detect_language(text: str) -> str:
    """ja / vi / en from the characters alone (no model, no dependency).
    Vietnamese typed without diacritics reads as English: harmless, it just
    means one extra translation."""
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return "en"
    if sum(1 for c in letters if _KANA_HAN.match(c)) / len(letters) > 0.15:
        return "ja"
    if _VI_ONLY.search(text):
        return "vi"
    return "en"


TRANSLATE_PROMPT = """Translate this search query into {targets}.
Keep company names, product names, numbers, dates and codes exactly as written.
Use the wording a formal corporate document in that language would use.
Reply with JSON only, keyed by language code: {{{example}}}

Query: {query}"""


def make_translator(settings) -> Translator:
    """A translator backed by the configured small model. The client is built on
    first use, so an unused feature never opens a connection."""
    state: dict = {}

    def translate(query: str, languages: Sequence[str]) -> dict[str, str]:
        if "client" not in state:
            from tini.loop.models import get_client

            state["client"] = get_client(settings)
        targets = ", ".join(f"{LANGUAGE_NAMES[c]} ({c})" for c in languages)
        example = ", ".join(f'"{c}": "..."' for c in languages)
        response = state["client"].messages.create(
            model=settings.small_model or settings.model,
            max_tokens=800,  # a reasoning model thinks before the JSON
            messages=[
                {
                    "role": "user",
                    "content": TRANSLATE_PROMPT.format(
                        targets=targets, example=example, query=query
                    ),
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if "{" not in text:
            raise ValueError("translator returned no JSON")
        parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
        return {
            code: str(parsed[code]).strip()
            for code in languages
            if isinstance(parsed.get(code), str) and str(parsed[code]).strip()
        }

    return translate


class MultilingualIndex:
    """A `DocumentIndex` whose `search` also looks for the question's
    translations. Every other operation passes straight through."""

    def __init__(
        self,
        inner,
        translator: Translator,
        languages: Sequence[str],
        *,
        timeout: float = TRANSLATE_TIMEOUT,
        breaker_failures: int = BREAKER_FAILURES,
        cooldown: float = BREAKER_COOLDOWN,
        clock: Callable[[], float] = time.monotonic,
        cache: OrderedDict | None = None,
    ):
        self.inner = inner
        self.translator = translator
        self.languages = [code for code in languages if code in LANGUAGE_NAMES]
        self.timeout = timeout
        self.breaker_failures = max(1, breaker_failures)
        self.cooldown = cooldown
        self._clock = clock
        self._failures_in_a_row = 0
        self._paused_until = 0.0
        self.searches = 0
        self.translated = 0
        self.errors = 0
        self.skipped = 0  # searches that skipped translation while it was paused
        self.last_error = ""
        self.last_queries: list[str] = []
        # Private unless a shared one is passed in (make_document_index does).
        self._cache: OrderedDict[str, dict[str, str]] = OrderedDict() if cache is None else cache

    # --- pass-through -------------------------------------------------------

    def upsert_document(
        self, source: DocumentSource, chunks: Sequence
    ) -> IngestReport:
        return self.inner.upsert_document(source, chunks)

    def list_documents(
        self, workspace_id: str = DEFAULT_WORKSPACE, limit: int = 200
    ) -> list[DocumentSummary]:
        return self.inner.list_documents(workspace_id, limit)

    def delete_document(self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE) -> bool:
        return self.inner.delete_document(document_id, workspace_id)

    def opening_text(
        self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE, chars: int = 3000
    ) -> str:
        return self.inner.opening_text(document_id, workspace_id, chars)

    # --- search -------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 6, workspace_id: str = DEFAULT_WORKSPACE
    ) -> list[SearchHit]:
        limit = max(1, min(int(top_k), 50))
        self.searches += 1
        queries = [query, *self._translations(query).values()]
        self.last_queries = queries
        if len(queries) == 1:
            return self.inner.search(query, top_k=limit, workspace_id=workspace_id)
        pool = min(50, max(limit * 3, 20))
        rankings = [
            self.inner.search(text, top_k=pool, workspace_id=workspace_id) for text in queries
        ]
        return _fuse(rankings, limit)

    def translate(self, query: str) -> dict[str, str]:
        """{language: translation} for the configured languages other than the
        question's own; {} when translation is skipped or fails (fail-open, and
        counted in `errors`/`skipped`)."""
        return self._translations(query)

    def _translations(self, query: str) -> dict[str, str]:
        own = detect_language(query)
        targets = [code for code in self.languages if code != own]
        if not targets or not query.strip():
            return {}
        key = f"{own}|{','.join(targets)}|{query}"
        with _CACHE_LOCK:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        now = self._clock()
        if now < self._paused_until:
            self.skipped += 1
            return {}
        try:
            found = _call_with_timeout(self.translator, self.timeout, query, targets)
        except Exception as exc:
            self.errors += 1
            self._failures_in_a_row += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            if self._failures_in_a_row >= self.breaker_failures:
                self._paused_until = now + self.cooldown
                self._failures_in_a_row = 0
            return {}  # not cached: a transient failure should be retried
        self._failures_in_a_row = 0
        # A "translation" identical to the question adds nothing to search.
        found = {code: text for code, text in found.items() if text and text != query}
        self.translated += 1
        with _CACHE_LOCK:
            self._cache[key] = found
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return found


class EnglishRewriter:
    """Turns a question in one of `source_languages` into English, for the reranker
    to read (see `RerankingIndex`). Reuses `MultilingualIndex`'s translation
    machinery — timeout, circuit breaker, shared cache, fail-open counters — through
    a helper that never searches. A question in another language, or a failed
    translation, comes back unchanged."""

    def __init__(self, translator: Translator, source_languages: Sequence[str], *, cache=None, **options):
        self.source_languages = frozenset(source_languages)
        self.helper = MultilingualIndex(None, translator, ["en"], cache=cache, **options)

    def __call__(self, query: str) -> str:
        if detect_language(query) not in self.source_languages:
            return query
        return self.helper.translate(query).get("en") or query


def _call_with_timeout(fn, timeout: float, *args):
    """Run `fn` and stop waiting after `timeout` seconds. The work runs on a
    daemon thread: a call stuck in a network read is abandoned rather than
    cancelled, and cannot keep the process from exiting."""
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # handed back to the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True, name="tini-translate")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"no answer within {timeout:g}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _fuse(rankings: list[list[SearchHit]], limit: int) -> list[SearchHit]:
    """Reciprocal-rank fusion by chunk: a chunk found by several phrasings of the
    question outranks one found by a single phrasing."""
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:limit]
    return [SearchHit(**{**hits[chunk_id].__dict__, "score": scores[chunk_id]}) for chunk_id in ordered]
