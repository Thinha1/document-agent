"""What OCR costs per document, derived from the token ledger.

`tini/documents/ocr.py` appends one row per OCR'd document run to
`<home>/ocr_usage.jsonl` (tokens only). Dollars are computed here, at read time,
so fixing a rate below corrects every past row — the same rule as
tini/ops/pricing.py's usage ledger. Rates are $/million tokens and ESTIMATES.
"""

from __future__ import annotations

import json
from pathlib import Path

from tini.ops.pricing import price_for

# Vision models with no catalog price. deepseek-v4-flash-vision-exp is documented
# (tini/documents/ocr.py) as priced like deepseek-v4-flash; checked first because
# the provider-level "deepseek" estimate in pricing.py is roughly 3x higher.
OCR_MODEL_PRICING = {"deepseek-v4-flash-vision-exp": (0.15, 0.30)}


def _rate(provider: str, model: str) -> tuple[float, float]:
    return OCR_MODEL_PRICING.get(model) or price_for(provider, model)


def _cost(row: dict) -> float:
    rate_in, rate_out = _rate(row.get("provider", ""), row.get("model", ""))
    return row.get("prompt_tokens", 0) / 1e6 * rate_in + row.get("completion_tokens", 0) / 1e6 * rate_out


def summarize(home: Path, workspace: str | None = None) -> dict:
    path = Path(home) / "ocr_usage.jsonl"
    rows: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if workspace is None or row.get("workspace") == workspace:
                rows.append(row)

    per_doc: dict[tuple, list[dict]] = {}
    for row in rows:
        per_doc.setdefault((row.get("workspace"), row.get("source_uri")), []).append(row)

    pages = sum(r.get("pages", 0) for r in rows)
    total_cost = sum(_cost(r) for r in rows)
    rate = _rate(rows[-1].get("provider", ""), rows[-1].get("model", "")) if rows else None
    return {
        "documents": len(per_doc),
        "runs": len(rows),
        "pages": pages,
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        "total_cost": total_cost,
        # Lifetime spend per scanned document: every run that paid counts, so a
        # file re-OCR'd after a cache miss costs what it really cost.
        "avg_cost_per_document": (total_cost / len(per_doc)) if per_doc else 0.0,
        "cached_pages": sum(r.get("cached_pages", 0) for r in rows),
        "avg_cost_per_page": (total_cost / pages) if pages else 0.0,
        "avg_pages_per_document": (pages / len(rows)) if rows else 0.0,
        "avg_prompt_tokens_per_page": (sum(r.get("prompt_tokens", 0) for r in rows) / pages) if pages else 0.0,
        "avg_completion_tokens_per_page": (sum(r.get("completion_tokens", 0) for r in rows) / pages) if pages else 0.0,
        "wasted_cost": sum(_cost(r) for r in rows if r.get("action") == "skipped"),
        "rate_per_million": {"in": rate[0], "out": rate[1]} if rate else None,
        "estimate": True,
    }
