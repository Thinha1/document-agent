"""DETERMINISTIC EVAL — did retrieval trigger? This is a unit test.

No LLM judges anything here. Each case asserts a binary, checkable outcome:
the right tool fired (or didn't), with the right arguments. 0 or 1.

Two tiers:
  offline  — scripted model, always runs, tests OUR code (loop, tools, wiring)
  live     — real model, runs when the active provider has a key, tests the
             MODEL+PROMPT behavior on evals/dataset.jsonl (the real eval)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evals.helpers import HAS_KEY, ScriptedClient, make_tini, response, text_block, tool_block

RUN_LIVE_EVALS = os.getenv("TINI_RUN_LIVE_EVALS") == "1"

DATASET = [
    json.loads(line)
    for line in (Path(__file__).resolve().parents[1] / "dataset.jsonl").read_text().splitlines()
    if line.strip()
]

# ---------- offline tier: our plumbing is deterministic-testable without any model


def test_search_web_result_returns_to_the_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://searxng:8080")
    monkeypatch.setattr(
        "tini.tools.search._searxng",
        lambda query, base_url, max_results: [
            {
                "title": "Worker runbook",
                "content": "Restart safely",
                "url": "https://docs.internal/worker",
                "engine": "wiki",
            }
        ],
    )
    gate = response([text_block('{"retrieve": false, "query": "", "reason": "test"}')])
    turn = [
        response([tool_block("search_web", {"query": "worker restart"})], "tool_use"),
        response([text_block("Use the worker runbook.")]),
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient([gate] + turn))
    result = app.respond("How do I restart the worker?")

    assert [c["tool"] for c in result.tool_calls] == ["search_web"]
    assert "Worker runbook" in result.tool_calls[0]["output"]
    assert result.reply == "Use the worker runbook."


def test_history_records_tool_use(tmp_path):
    """The next turn's working memory must show which retrieval already ran."""
    gate = response([text_block('{"retrieve": false, "query": "", "reason": "test"}')])
    script = [gate] + [
        response([tool_block("search_web", {"query": "incident policy"})], "tool_use"),
        response([text_block("Done.")]),
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient(script))
    app.respond("Find the incident policy")
    assert "[tools used: search_web" in app.session.history[-1]["content"]


def test_no_tool_turn_ends_loop_in_one_iteration(tmp_path):
    script = [
        response([text_block('{"retrieve": false, "query": "", "reason": "test"}')]),
        response([text_block("Paris.")]),
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient(script))
    result = app.respond("capital of france?")
    assert result.reply == "Paris." and result.iterations == 1 and result.tool_calls == []


def test_iteration_guardrail_stops_runaway_loop(tmp_path):
    gate = response([text_block('{"retrieve": false, "query": "", "reason": "test"}')])
    runaway = [
        response([tool_block("search_web", {"query": "same query"}, f"tu_{i}")], "tool_use")
        for i in range(99)
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient([gate] + runaway), max_iterations=3)
    result = app.respond("loop forever")
    assert result.iterations == 3 and "iteration limit" in result.reply


# ---------- live tier: the actual model eval over the dataset


@pytest.mark.skipif(
    not (HAS_KEY and RUN_LIVE_EVALS),
    reason="set TINI_RUN_LIVE_EVALS=1 and configure an API key to run live evals",
)
@pytest.mark.parametrize("case", DATASET, ids=[c["id"] for c in DATASET])
def test_dataset_case(case, tmp_path):
    app = make_tini(tmp_path / "home")
    if "setup_fact" in case:
        app.memory.facts.add(case["setup_fact"]["subject"], case["setup_fact"]["content"])

    result = app.respond(case["input"])
    fired = [c["tool"] for c in result.tool_calls]

    if case["expect_tool"] is None:
        assert fired == [], f"expected no tools, model called {fired}"
    else:
        assert case["expect_tool"] in fired, f"expected {case['expect_tool']}, model called {fired}"
        args = next(c["args"] for c in result.tool_calls if c["tool"] == case["expect_tool"])
        for key, needle in case.get("expect_in_args", {}).items():
            assert needle.lower() in str(args.get(key, "")).lower(), (
                f"expected '{needle}' in args[{key}], got: {args.get(key)}"
            )
        # multi-tool cases (pokemon-team, worldcup-final): the loop must have
        # actually looped, not satisfied one expectation and stopped
        want = case.get("expect_min_tool_calls", 0)
        assert len(fired) >= want, f"only {len(fired)} tool calls, wanted >= {want}"
