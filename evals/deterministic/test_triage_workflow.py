"""The triage graph workflow + its app.py wiring — scripted, zero network.

SCRIPTED-RESPONSE ORDER, read this before adding cases: with graph_workflows
ON, the FIRST scripted response is the triage classifier. A quick turn then
consumes one quick-reply response. A full turn consumes the retrieval gate
next, THEN the loop's responses — i.e. everything test_tool_trigger.py taught
you, shifted one response later.
"""

from __future__ import annotations

import json

from evals.helpers import ScriptedClient, make_tini, response, text_block
from tini.graph.workflows.triage import classify_message


def last_meta(app) -> dict:
    """The persisted turn meta, exactly as a reopened thread would read it."""
    row = app.conn.execute(
        "SELECT meta FROM chat_log WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return json.loads(row["meta"])


class Boom:
    """A client whose every call explodes — for the fail-open cases."""

    def __init__(self):
        from types import SimpleNamespace

        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        raise RuntimeError("boom")


def classify(reply_text: str):
    return classify_message(
        ScriptedClient([response([text_block(reply_text)])]), "small-model", "hello"
    )


def test_classifier_parses_routes_and_fails_open_on_garbage():
    assert classify('{"route": "quick", "reason": "just a greeting"}') == (
        "quick",
        "just a greeting",
    )
    assert classify('{"route": "full", "reason": "needs retrieval"}')[0] == "full"
    # every malformed shape falls open to full — capability over latency
    assert classify("no json here at all")[0] == "full"
    assert classify('{"route": "sideways"}')[0] == "full"
    assert classify_message(Boom(), "small-model", "hi")[0] == "full"


def test_flag_off_is_byte_for_byte_the_old_world(tmp_path):
    gate = response([text_block('{"retrieve": false, "query": "", "reason": "test"}')])
    app = make_tini(
        tmp_path / "home",
        client=ScriptedClient([gate, response([text_block("Hi!")])]),
        graph_workflows=False,
    )
    events = []
    result = app.respond("hello", observer=lambda k, ev: events.append(k))
    assert result.reply == "Hi!"
    assert not any(k in ("graph_start", "route", "graph_end") for k in events)
    meta = last_meta(app)
    assert meta["graph"] is None


def test_quick_turn_answers_on_the_small_model_and_skips_the_gate(tmp_path):
    script = [
        response([text_block('{"route": "quick", "reason": "just thanks"}')]),  # triage
        response([text_block("You're welcome!")]),  # quick reply
    ]
    app = make_tini(
        tmp_path / "home",
        client=ScriptedClient(script),
        graph_workflows=True,
        small_model="small-model",
    )
    events: list[tuple[str, dict]] = []
    result = app.respond("thanks!", observer=lambda k, ev: events.append((k, ev)))

    assert result.reply == "You're welcome!"
    assert result.iterations == 1 and result.tool_calls == []
    kinds = [k for k, _ in events]
    assert "graph_start" in kinds and "route" in kinds and "graph_end" in kinds
    assert "gate" not in kinds, "quick turns never touch memory retrieval"
    assert kinds.count("llm") == 1, "quick turns use only the small reply model"
    meta = last_meta(app)
    assert meta["graph"]["route"] == "quick"
    assert meta["graph"]["reason"] == "just thanks"
    assert meta["gate"] is None
    assert meta["model"] == "small-model"  # honest per-path model


def test_full_turn_runs_the_real_loop_with_both_funnel_stages(tmp_path):
    script = [
        response([text_block('{"route": "full", "reason": "needs retrieval"}')]),  # triage
        response([text_block('{"retrieve": false, "query": "", "reason": "n"}')]),  # gate
        response([text_block("The runbook says to restart the worker.")]),
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient(script), graph_workflows=True)
    events: list[str] = []
    result = app.respond("How do I restart the worker?", observer=lambda k, ev: events.append(k))

    assert result.reply == "The runbook says to restart the worker."
    assert result.tool_calls == []
    assert "route" in events and "gate" in events  # both funnel stages recorded
    meta = last_meta(app)
    assert meta["graph"]["route"] == "full"
    assert meta["gate"] is not None
    assert meta["model"] == app.settings.model
    assert "full_agent" in meta["graph"]["path"]


def test_broken_classifier_fails_open_to_the_full_loop(tmp_path):
    script = [
        response([text_block("not json — classifier had a bad day")]),  # triage
        response([text_block('{"retrieve": false, "query": "", "reason": "n"}')]),  # gate
        response([text_block("Still here.")]),  # loop
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient(script), graph_workflows=True)
    assert app.respond("hello?").reply == "Still here."


def test_broken_graph_engine_fails_open_to_the_plain_loop(tmp_path, monkeypatch):
    """Layer two: even if graph construction itself explodes, respond() answers."""
    from tini.graph.workflows import triage

    def explode(**kwargs):
        raise RuntimeError("graph machinery on fire")

    monkeypatch.setattr(triage, "build_triage_graph", explode)
    script = [
        response([text_block('{"retrieve": false, "query": "", "reason": "n"}')]),  # gate
        response([text_block("Saved by the loop.")]),
    ]
    app = make_tini(tmp_path / "home", client=ScriptedClient(script), graph_workflows=True)
    events: list[str] = []
    result = app.respond("hello", observer=lambda k, ev: events.append(k))
    assert result.reply == "Saved by the loop."
    assert "graph_end" in events  # the failure is on tape
    meta = last_meta(app)
    assert meta["graph"] is None  # no route happened — honest meta
