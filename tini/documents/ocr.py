"""OCR fallback for image-only/scanned PDF pages.

Only reached when pypdf's plain text extraction comes back too sparse to be
useful (see the `_MIN_CHARS_PER_PAGE` heuristic in loaders.py) — a normal
text PDF never pays this cost or imports anything here.

CPU-only local OCR (PaddleOCR) was tried first, since a self-hosted default
matches this codebase's local-first ethos. It was abandoned after three
attempts: PP-StructureV3's full pipeline and the lighter PaddleOCR class both
crashed with the same oneDNN CPU-backend error at the very first inference
call, and disabling oneDNN (enable_mkldnn=False) just hung indefinitely
instead. This adapter calls a vision-capable chat model over the same
OpenAI-compatible wire format the embedding providers already use —
deepseek-v4-flash-vision-exp (priced at the same rate as deepseek-v4-flash)
is the configuration this was built and tested against, via the
DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL most setups already have for chat:

    pip install 'tini-agent[ocr]'
    TINI_OCR_PROVIDER=deepseek
    TINI_OCR_MODEL=deepseek-v4-flash-vision-exp
    # TINI_OCR_API_KEY=   # falls back to DEEPSEEK_API_KEY / OPENAI_API_KEY
    # TINI_OCR_BASE_URL=  # falls back to the provider's default endpoint

Any OpenAI-compatible vision endpoint works the same way (TINI_OCR_PROVIDER=
openai, TINI_OCR_BASE_URL pointed at it) — this is not DeepSeek-specific.

By default one call transcribes a page AND renders any table as Markdown —
simple, but the Markdown lands in `chunk_text()` (chunking.py) as more prose:
no boundary the chunker recognizes as "table", so a table wider than one chunk
gets cut at an arbitrary token offset, header lost (see the worked example in
docs/document-ingestion.md). TINI_OCR_TABLE_FLOW=1 splits the work instead: the
transcription call skips tables (leaving a `[[TABLE:n]]` marker) and a second,
table-only call — only made for a page a marker was found on — extracts them as
JSON that chunking.py can chunk by row instead of by token. Off by default:
this is new and not yet measured against the single-call flow.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Protocol

from tini.config import Settings


class OcrError(RuntimeError):
    """An OCR provider rejected a request or returned nothing usable."""


class OcrProvider(Protocol):
    def ocr_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        """Transcribe the text visible in one page image. Raises OcrError on
        a request failure; returns "" (not an error) when the model simply
        found no text — a blank page is not a broken request."""
        ...


# Bump when a prompt below changes, so old cached text is not served as if the
# new prompt had produced it. Each flow/prompt has its own version and its own
# slice of the cache (see _cache_path), so switching TINI_OCR_TABLE_FLOW never
# serves one flow's cached output under the other's key.
PROMPT_VERSION = "1"  # the single legacy prompt (table_flow=False)
TEXT_ONLY_PROMPT_VERSION = "1"  # table_flow=True: transcription, tables skipped
TABLE_PROMPT_VERSION = "1"  # table_flow=True: the table-only extraction call

_LEGACY_PROMPT = (
    "Transcribe every word of visible text in this document page image, in reading "
    "order. Render any table as a Markdown table. Output only the transcription — "
    "no commentary, no preamble. If the page has no legible text, output nothing."
)

# Used when table_flow=True. Deliberately narrower than the legacy prompt: this
# call does ONE job (prose, in reading order) so a wide/merged-header table
# doesn't compete with paragraph transcription for the model's attention within
# a single generation — see docs/document-ingestion.md for why (untested
# hypothesis, called out honestly there). extract_tables() below does the table
# job, and only runs on pages this prompt actually marks.
_TEXT_ONLY_PROMPT = (
    "Transcribe every word of visible text in this document page image, in reading "
    "order, EXCEPT any table. Where a table appears, output a single line "
    "\"[[TABLE:n]]\" in its place (n = 1, 2, 3... counting tables on this page in "
    "reading order) and do not transcribe anything from inside it. Output only the "
    "transcription — no commentary, no preamble. If the page has no legible text, "
    "output nothing."
)

# Used when table_flow=True, only for a page the prompt above found a marker on.
# JSON, not Markdown: a Markdown table is just more prose text to the chunker,
# indistinguishable from a paragraph, which is the whole problem this exists to
# fix (see chunking.py's table-aware split). Columns/rows are kept as plain
# strings — a schema simple enough for a vision model to fill in reliably.
_TABLE_PROMPT = (
    "Every table on this document page image, extracted as JSON: "
    '{"tables": [{"caption": "title or nearby heading, or \\"\\"", '
    '"columns": ["..."], "rows": [["..."]], "notes": "footnotes, or \\"\\""}]}. '
    "List tables in the same reading order they appear on the page. Where a header "
    "cell spans multiple columns, repeat its label in each sub-column instead of "
    "merging them — every row in \"rows\" must have exactly one value per entry in "
    "\"columns\". Keep every number exactly as printed: thousands separators, "
    "decimal marks, %, currency symbols and all. Output JSON only — no commentary, "
    'no code fences. If the page has no table, output {"tables": []}.'
)

# The placeholder _TEXT_ONLY_PROMPT asks for. Exported so loaders.py finds
# markers with the exact pattern this module actually produces, in one place.
TABLE_MARKER_RE = re.compile(r"\[\[TABLE:(\d+)\]\]")


class VisionModelOcrProvider:
    provider = "vision-model"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        concurrency: int = 4,
        provider_name: str = "vision-model",
        cache_dir: Path | None = None,
        table_flow: bool = False,
    ):
        from openai import OpenAI

        self.model = model
        self.provider_name = provider_name
        # Running token totals for THIS instance (one per ingest call), so the
        # loader can attribute cost to a document by diffing before/after. Only
        # tokens are kept, never a price: prices change, tokens don't (same
        # rule as tini/ops/pricing.py's ledger). table_calls/table_cache_hits
        # break the table-only call out from the transcription call above it;
        # prompt_tokens/completion_tokens fold both together, since that is
        # what a $ total needs (tini/ops/ocr_cost.py).
        self._usage_lock = threading.Lock()
        self._usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hits": 0,
            "table_calls": 0,
            "table_cache_hits": 0,
        }
        # Transcriptions are cached per page image, so ingesting the same
        # scanned file again (unchanged re-ingest, a new embedding model, a
        # different chunk size) does not pay for OCR twice. Off when None.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # Read by loaders._ocr_pdf_pages to bound how many pages of the same
        # document are in flight at once.
        self.concurrency = max(1, int(concurrency))
        # See TINI_OCR_TABLE_FLOW in config.py. False (default) keeps the exact
        # legacy prompt/cache, so a caller who hasn't opted in sees zero change.
        self.table_flow = bool(table_flow)

    def _cache_path(self, image_bytes: bytes, kind: str, version: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256()
        for part in (self.model.encode(), kind.encode(), version.encode(), image_bytes):
            key.update(part)
            key.update(b"\0")
        return self.cache_dir / f"{key.hexdigest()}.txt"

    def _read_cache(self, cache_path: Path | None) -> str | None:
        if cache_path is None:
            return None
        try:
            return cache_path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_cache(self, cache_path: Path | None, text: str) -> None:
        # Only a successful response is cached (a failure raises before this is
        # called), and a blank result is a valid one worth caching too. Best
        # effort: a cache that cannot be written just means paying again later.
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, cache_path)
        except OSError:
            pass

    def _call(self, prompt: str, image_bytes: bytes, media_type: str) -> tuple[str, object]:
        """One vision request. Returns (text, usage) or raises OcrError."""
        encoded = base64.b64encode(image_bytes).decode("ascii")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise OcrError(f"OCR request failed: {exc}") from exc
        text = (response.choices[0].message.content or "").strip()
        return text, getattr(response, "usage", None)

    def ocr_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        if self.table_flow:
            prompt, kind, version = _TEXT_ONLY_PROMPT, "text", TEXT_ONLY_PROMPT_VERSION
        else:
            prompt, kind, version = _LEGACY_PROMPT, "legacy", PROMPT_VERSION
        cache_path = self._cache_path(image_bytes, kind, version)
        cached = self._read_cache(cache_path)
        if cached is not None:
            with self._usage_lock:
                self._usage["cache_hits"] += 1
            return cached
        text, usage = self._call(prompt, image_bytes, media_type)
        with self._usage_lock:
            self._usage["calls"] += 1
            self._usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        self._write_cache(cache_path, text)
        return text

    def extract_tables(self, image_bytes: bytes, media_type: str = "image/png") -> list[dict]:
        """Only called (via loaders._ocr_pdf_pages) for a page whose ocr_image
        transcription contains at least one [[TABLE:n]] marker. Raises OcrError
        both on a request failure and on a reply that isn't the expected JSON
        shape — a model that can't follow the schema is a failure here, the
        same "don't pass a broken result off as an empty one" rule OcrError
        already applies to a failed request."""
        cache_path = self._cache_path(image_bytes, "table", TABLE_PROMPT_VERSION)
        cached = self._read_cache(cache_path)
        if cached is not None:
            with self._usage_lock:
                self._usage["table_cache_hits"] += 1
            text = cached
        else:
            text, usage = self._call(_TABLE_PROMPT, image_bytes, media_type)
            with self._usage_lock:
                self._usage["table_calls"] += 1
                self._usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                self._usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            start, end = text.index("{"), text.rindex("}")
            parsed = json.loads(text[start : end + 1])
            tables = parsed["tables"]
            if not isinstance(tables, list):
                raise TypeError("'tables' is not a list")
        except (ValueError, KeyError, TypeError) as exc:
            raise OcrError(f"table extraction returned unusable JSON: {exc}") from exc
        if cached is None:
            self._write_cache(cache_path, text)
        return tables

    def usage_snapshot(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)


def make_ocr_provider(settings: Settings) -> OcrProvider | None:
    provider = settings.ocr_provider.strip().lower()
    if provider in {"", "none"}:
        return None
    if not settings.ocr_model:
        raise ValueError("TINI_OCR_MODEL is required when TINI_OCR_PROVIDER is set")
    if not settings.ocr_api_key:
        raise ValueError("TINI_OCR_API_KEY or the provider API key is required")
    return VisionModelOcrProvider(
        settings.ocr_api_key,
        settings.ocr_model,
        settings.ocr_base_url,
        concurrency=settings.ocr_concurrency,
        table_flow=settings.ocr_table_flow,
        provider_name=provider,
        cache_dir=settings.home / "ocr_cache",
    )


def make_usage_logger(home):
    """Append-only ledger of what each OCR'd document cost, one JSON line per
    run (`<home>/ocr_usage.jsonl`). Tokens only — dollars are derived at read
    time by tini/ops/ocr_cost.py so a corrected price fixes past rows too."""
    from datetime import UTC, datetime

    path = home / "ocr_usage.jsonl"

    def log(entry: dict) -> None:
        row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return log
