"""Deterministic coverage for named graph workflow commands."""

from __future__ import annotations

from tini.ops import commands


def test_triage_is_discovered_and_accepts_a_message():
    import importlib
    import inspect

    assert commands.discover() == {"triage": "tini.ops.triage:run_triage"}
    fn = importlib.import_module("tini.ops.triage").run_triage
    assert "message" in inspect.signature(fn).parameters


def test_discovery_requires_workflow_and_binder():
    import pkgutil

    import tini.graph.workflows as pkg

    modules = {m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.name.startswith("_")}
    for name, target in commands.discover().items():
        assert name in modules
        assert target == f"tini.ops.{name}:run_{name}"


def test_a_leading_slash_is_a_command():
    assert commands.parse("/triage summarize the runbook") == ("triage", "summarize the runbook")
    assert commands.parse("  /graphs  ") == ("graphs", "")


def test_ordinary_messages_are_not_commands():
    for text in (
        "hello",
        "look in /tmp/x",
        "what is 3/4",
        "",
        "   ",
        "/ triage",
        "the file is at /etc/hosts",
    ):
        assert commands.parse(text) is None, text


def test_unknown_command_lists_available_workflows():
    msg = commands.unknown_reply("nope")
    assert "/nope" in msg and "/triage" in msg and "/graphs" in msg


def test_triage_without_a_message_explains_itself():
    state = commands.run("triage", lambda *a: None, "")
    assert state is not None
    assert "/triage thanks!" in state["digest"]


def test_running_an_unknown_name_returns_none():
    assert commands.run("definitely-not-a-workflow", lambda *a: None) is None
