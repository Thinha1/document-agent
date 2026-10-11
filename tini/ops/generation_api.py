"""Dashboard side of the generation eval: what to run, what it will cost, run it while
streaming each question as it finishes, and keep the results.

Nothing here runs by itself: a generation eval calls a chat model (and optionally a
second, judge model) for every question, which costs money and minutes, so the
page shows the plan and an estimate first and only the person clicking Run starts
it. One run at a time; closing the page or pressing Cancel stops the queue.

The run uses the real TINI_HOME read-only in spirit: it reads the persona (SOUL.md)
and the catalog (so a superseded document stays hidden, as it is for the agent) but
uses no memory, no retrieval gate and no chat log, so it neither reads nor writes
what Tini has learned. Keys are never sent to the browser or stored in a result.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from tini.config import Settings, load_settings
from tini.loop.models import PROVIDERS, get_client
from tini.ops import generation_eval as ge

RUNS_DIRNAME = "generation_runs"
MAX_QUESTIONS = 80
ARMS = ("on", "off", "both")
# Measured on DeepSeek (8 questions, so a rough guide, not a price): the generator
# reads about this much per question over its ~2.4 searches, and the judge reads the
# evidence once more. Shown as an estimate, never as a bill.
EST_GENERATOR_TOKENS = (12_000, 1_500)
EST_JUDGE_TOKENS = (10_000, 400)

_RUN_LOCK = threading.Lock()


def _providers() -> list[dict]:
    """Provider ids and whether a key is set, never the key."""
    return [
        {"id": name, "configured": bool(os.getenv(p.key_env, "").strip()), "default_model": p.model}
        for name, p in PROVIDERS.items()
    ]


def info(settings: Settings | None = None) -> dict | None:
    """What the page needs to draw the form; None if the question files are missing,
    so a missing file hides this section instead of breaking the Knowledge page."""
    settings = settings or load_settings()
    try:
        cases = ge.load_cases(settings=settings)
    except (OSError, ValueError):
        return None
    counts = {name: sum(c["source"] == name for c in cases) for name in ge.SOURCE_NAMES}
    return {
        "sources": counts,
        "total": len(cases),
        "generator": {"provider": settings.provider, "model": settings.model},
        "providers": _providers(),
        "max_questions": MAX_QUESTIONS,
    }


def _client_settings(base: Settings, spec: dict | None, *, judge: bool) -> tuple[Settings | None, str]:
    """Settings for a generator/judge choice, or (None, why not). A different provider
    starts from a clean slate so the active provider's key or endpoint never leaks
    into it."""
    if judge and spec is None:
        return None, ""  # no judge asked for
    spec = spec or {}
    provider = str(spec.get("provider") or "").strip()
    model = str(spec.get("model") or "").strip()
    if judge and not (provider and model):
        return None, "Pick the judge's provider and model, or turn the judge off."
    if len(model) > 160 or re.search(r"[\r\n]", model):
        return None, "That model id is not valid."
    if provider and provider != base.provider:
        if provider not in PROVIDERS:
            return None, f"Unknown provider '{provider}'."
        candidate = replace(base, provider=provider, model=model, small_model=model, api_key="", base_url=None)
    else:
        candidate = replace(base, model=model or base.model, small_model=model or base.small_model)
    if judge:
        candidate = replace(candidate, small_model="")
    try:
        get_client(candidate)
    except (Exception, SystemExit):
        return None, f"No API key is configured for provider '{candidate.provider}'."
    return candidate, ""


def _options(payload: dict, settings: Settings) -> dict:
    """Validated run options, or {"error": ...}."""
    sources = payload.get("sources")
    if sources is None:
        sources = list(ge.SOURCE_NAMES)
    if not isinstance(sources, list) or not set(sources) <= set(ge.SOURCE_NAMES) or not sources:
        return {"error": "Pick at least one question source."}
    try:
        limit = int(payload.get("limit") or 0)
    except (TypeError, ValueError):
        return {"error": "The question limit must be a number."}
    if limit < 0:
        return {"error": "The question limit must be positive."}
    arm = str(payload.get("arms") or "on")
    if arm not in ARMS:
        return {"error": f"arms must be one of {', '.join(ARMS)}."}
    cases = ge.select_cases(ge.load_cases(settings=settings), sources=sources, limit=limit or None)
    if not cases:
        return {"error": "No questions match those sources."}
    if len(cases) > MAX_QUESTIONS:
        return {"error": f"At most {MAX_QUESTIONS} questions per run."}
    generator, why = _client_settings(settings, payload.get("generator"), judge=False)
    if generator is None:
        return {"error": why}
    judge, why = _client_settings(settings, payload.get("judge"), judge=True)
    if why:
        return {"error": why}
    return {
        "cases": cases,
        "arms": ["on", "off"] if arm == "both" else [arm],
        "generator": generator,
        "judge": judge,
        "sources": sources,
        "limit": limit,
    }


def plan(payload: dict, settings: Settings | None = None) -> dict:
    """What a run would do and roughly what it would cost, without calling any model."""
    settings = settings or load_settings()
    options = _options(payload, settings)
    if "error" in options:
        return options
    n, arms = len(options["cases"]), len(options["arms"])
    judged = options["judge"] is not None
    gen_in, gen_out = EST_GENERATOR_TOKENS
    j_in, j_out = EST_JUDGE_TOKENS
    return {
        "questions": n,
        "arms": options["arms"],
        "by_source": {s: sum(c["source"] == s for c in options["cases"]) for s in ge.SOURCE_NAMES},
        "generator": {"provider": options["generator"].provider, "model": options["generator"].model},
        "judge": (
            {"provider": options["judge"].provider, "model": options["judge"].model} if judged else None
        ),
        "self_judged": bool(
            judged
            and (options["judge"].provider, options["judge"].model)
            == (options["generator"].provider, options["generator"].model)
        ),
        "estimate": {
            "generator_calls": round(n * arms * 3.4),
            "judge_calls": n * arms if judged else 0,
            "input_tokens": n * arms * (gen_in + (j_in if judged else 0)),
            "output_tokens": n * arms * (gen_out + (j_out if judged else 0)),
            "basis": "rough: measured on 8 DeepSeek questions; real usage is in each result",
        },
    }


def _runs_dir(settings: Settings) -> Path:
    return settings.home / RUNS_DIRNAME


def _compact(result: dict) -> dict:
    """The few numbers worth showing in a run list."""
    arms = {}
    for arm, body in (result.get("arms") or {}).items():
        everything = (body.get("summary") or {}).get("all") or {}
        arms[arm] = {
            "n": everything.get("n", 0),
            "answered": f"{everything.get('answered', 0)}/{everything.get('answerable', 0)}",
            "language_ok": f"{everything.get('language_ok', 0)}/{everything.get('n', 0)}",
            "declined": f"{everything.get('declined', 0)}/{everything.get('unanswerable', 0)}",
        }
    return arms


def list_runs(payload: dict, settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    runs = []
    directory = _runs_dir(settings)
    if directory.is_dir():
        for path in sorted(directory.glob("*.json"), reverse=True)[:30]:
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            runs.append(
                {
                    "name": path.name,
                    "created": result.get("created", ""),
                    "generator": result.get("generator"),
                    "judge": result.get("judge"),
                    "cancelled": bool(result.get("cancelled")),
                    "arms": _compact(result),
                }
            )
    return {"runs": runs}


def load_run(payload: dict, settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    name = str(payload.get("name") or "")
    # a browser-supplied name must resolve to a file directly inside the runs folder
    if not re.fullmatch(r"[\w.\-]+\.json", name):
        return {"error": "invalid run name"}
    path = _runs_dir(settings) / name
    if not path.is_file():
        return {"error": "unknown run"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"error": "that run could not be read"}


def _save(settings: Settings, result: dict) -> str | None:
    try:
        directory = _runs_dir(settings)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        target = directory / f"{stamp}.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        return target.name
    except OSError:
        return None  # a write problem must not lose the run the person just paid for


def run(
    payload: dict,
    emit: Callable[[str, dict], None],
    settings: Settings | None = None,
    stop: threading.Event | None = None,
) -> dict:
    """Run the eval, streaming `start`, one `row` per finished question and an `arm_done`
    per arm; returns the final result (also saved under TINI_HOME/generation_runs)."""
    settings = settings or load_settings()
    stop = stop or threading.Event()
    options = _options(payload, settings)
    if "error" in options:
        return options
    if not _RUN_LOCK.acquire(blocking=False):
        return {"error": "A generation eval is already running. Wait for it or cancel it."}
    try:
        cases, arms = options["cases"], options["arms"]
        generator, judge = options["generator"], options["judge"]
        total = len(cases) * len(arms)
        emit("start", {
            "total": total, "arms": arms, "questions": len(cases),
            "generator": {"provider": generator.provider, "model": generator.model},
            "judge": {"provider": judge.provider, "model": judge.model} if judge else None,
        })
        result: dict = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "generator": {"provider": generator.provider, "model": generator.model},
            "judge": {"provider": judge.provider, "model": judge.model} if judge else None,
            "sources": options["sources"],
            "arms": {},
            "cancelled": False,
        }
        done_before = 0
        for arm in arms:
            def on_row(row: dict, done: int, _total: int, arm=arm, offset=done_before) -> None:
                emit("row", {"arm": arm, "row": row, "done": offset + done, "total": total})

            outcome = ge.run_generation_eval(
                cases, generator, judge_settings=judge, reply_language=(arm == "on"),
                workers=2, on_row=on_row, stop=stop,
            )
            result["arms"][arm] = {"rows": outcome["rows"], "summary": outcome["summary"]}
            done_before += len(cases)
            emit("arm_done", {"arm": arm, "summary": outcome["summary"]})
            if outcome["cancelled"]:
                result["cancelled"] = True
                break
        result["saved_as"] = _save(settings, result)
        return result
    finally:
        _RUN_LOCK.release()
