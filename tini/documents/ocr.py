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
"""

from __future__ import annotations

import base64
import threading
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


class VisionModelOcrProvider:
    provider = "vision-model"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        concurrency: int = 4,
        provider_name: str = "vision-model",
    ):
        from openai import OpenAI

        self.model = model
        self.provider_name = provider_name
        # Running token totals for THIS instance (one per ingest call), so the
        # loader can attribute cost to a document by diffing before/after. Only
        # tokens are kept, never a price: prices change, tokens don't (same
        # rule as tini/ops/pricing.py's ledger).
        self._usage_lock = threading.Lock()
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # Read by loaders._ocr_pdf_pages to bound how many pages of the same
        # document are in flight at once.
        self.concurrency = max(1, int(concurrency))

    def ocr_image(self, image_bytes: bytes, media_type: str = "image/png") -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe every word of visible text in this document "
                                    "page image, in reading order. Render any table as a "
                                    "Markdown table. Output only the transcription — no "
                                    "commentary, no preamble. If the page has no legible text, "
                                    "output nothing."
                                ),
                            },
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
        usage = getattr(response, "usage", None)
        with self._usage_lock:
            self._usage["calls"] += 1
            self._usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        return (response.choices[0].message.content or "").strip()

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
        provider_name=provider,
    )


def make_usage_logger(home):
    """Append-only ledger of what each OCR'd document cost, one JSON line per
    run (`<home>/ocr_usage.jsonl`). Tokens only — dollars are derived at read
    time by tini/ops/ocr_cost.py so a corrected price fixes past rows too."""
    import json
    from datetime import UTC, datetime

    path = home / "ocr_usage.jsonl"

    def log(entry: dict) -> None:
        row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return log
