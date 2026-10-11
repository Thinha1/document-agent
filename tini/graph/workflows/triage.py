"""The triage graph workflow — every message's front door when the flag is on.

The retrieval gate asked one narrow question with a small model ("does this
message need memory?"). Triage generalizes that idea from one gate to a
structure: a small model classifies the message, then a code router picks the
path:

    START → classify → route: quick → quick_reply (small model) → END
                              full  → full_agent (THE loop)     → END

"thanks!" never wakes the big model. A document question runs the exact same
loop as always — as one node. The user never chooses a mode; this graph
IS the choice. And every seam fails open: a broken classifier, a broken
engine, anything → the plain loop answers, same as if the flag were off.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from tini.graph.engine import END, START, Graph, Node
from tini.graph.nodes import key_router

TRIAGE_PROMPT = """\
You are a triage gate for an internal knowledge assistant. Given the user's message,
decide which brain should answer it.

Reply with ONLY this JSON, nothing else:
{{"route": "quick" or "full", "reason": "<5 words>"}}

quick — greetings, thanks, acknowledgements, pure small talk: nothing to do,
        nothing to look up.
full  — any question that may need retrieval, memory, reasoning, or a tool.
        When unsure, choose full.

User message: {message}"""

QUICK_REPLY_PROMPT = """\
You are Tini, a concise internal knowledge assistant. The user's message needs
no retrieval or memory — reply in one or two short, natural sentences.

User message: {message}"""


def classify_message(client, small_model: str, message: str) -> tuple[str, str]:
    """Returns (route, reason). Fails open to "full": a broken triage must
    cost latency, never capability — mirror of retrieval_gate.should_retrieve."""
    try:
        response = client.messages.create(
            model=small_model,
            max_tokens=600,  # reasoning models think before the JSON
            messages=[{"role": "user", "content": TRIAGE_PROMPT.format(message=message)}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        if "{" not in text:
            return "full", "no JSON — failing open"
        decision = json.loads(text[text.index("{") : text.rindex("}") + 1])
        route = decision.get("route")
        if route not in ("quick", "full"):
            return "full", f"bad route '{route}' — failing open"
        return route, decision.get("reason", "")
    except Exception as exc:
        return "full", f"triage failed open ({type(exc).__name__})"


def build_triage_graph(
    *,
    classify_fn: Callable[[str], tuple[str, str]],
    quick_fn: Callable[[dict], str],
    full_fn: Callable[[dict], object],
) -> Graph:
    """Callables injected so evals script them, the dashboard builds it with
    stubs just to describe(), and app.py binds the real client/loop."""
    g = Graph("triage")

    def classify(state: dict) -> dict:
        route, reason = classify_fn(state["message"])
        notify = state.get("_notify")
        if notify:  # the human-readable "why" — traced, and shown on the turn card
            notify("triage", {"route": route, "reason": reason})
        return {"route": route, "triage_reason": reason}

    g.add_node(Node("classify", classify, kind="llm"))
    g.add_node(Node("quick_reply", lambda s: {"reply": quick_fn(s)}, kind="llm"))
    g.add_node(Node("full_agent", lambda s: {"result": full_fn(s)}, kind="agent"))
    g.add_edge(START, "classify")
    g.add_router(
        "classify",
        key_router("route", default="full"),
        {"quick": "quick_reply", "full": "full_agent"},
    )
    g.add_edge("quick_reply", END)
    g.add_edge("full_agent", END)
    return g


def triage_topology() -> dict:
    """The topology as data, for the dashboard — built with stubs, never run."""
    noop = lambda *a, **k: ""  # noqa: E731
    return build_triage_graph(
        classify_fn=lambda m: ("full", ""), quick_fn=noop, full_fn=noop
    ).describe()
