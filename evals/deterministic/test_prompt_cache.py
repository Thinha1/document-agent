"""Prompt-cache friendliness: the request must START the same way every turn.

Providers with automatic prefix caching (DeepSeek, OpenAI) reuse the longest
identical beginning of a request. Measured on DeepSeek: with the clock and
retrieved memory inside the system prompt, 0% of 1,652 prompt tokens hit the
cache on the second turn; with them on the last user message, 85% did. These
tests pin the structure that makes that true, network-free.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

from evals.helpers import ScriptedClient, make_tini, response, text_block
from tini.loop.agent import run_loop
from tini.loop.models import _cache_hit_tokens
from tini.runtime.session import Session
from tini.tools.registry import ToolRegistry


class _Capture(ScriptedClient):
    """Records what each model call was actually sent."""

    def __init__(self, script):
        super().__init__(script)
        self.calls: list[dict] = []

    def _create(self, **kwargs):
        self.calls.append(copy.deepcopy({"system": kwargs["system"], "messages": kwargs["messages"]}))
        return super()._create(**kwargs)


def _two_turns(tmp_path):
    client = _Capture([response([text_block("first")]), response([text_block("second")])])
    tini = make_tini(tmp_path / "home", client=client)
    tini.respond("Who chairs the board?")
    tini.respond("And the audit committee?")
    return tini, client


def test_the_system_prompt_is_identical_from_turn_to_turn(tmp_path):
    _, client = _two_turns(tmp_path)

    assert client.calls[0]["system"] == client.calls[1]["system"]
    assert "Right now it is" not in client.calls[0]["system"], "a ticking clock breaks the prefix"


def test_earlier_history_is_a_byte_identical_prefix_of_the_next_request(tmp_path):
    client = _Capture([response([text_block(t)]) for t in ("a1", "a2", "a3")])
    tini = make_tini(tmp_path / "home", client=client)
    for question in ("Who chairs the board?", "And the audit committee?", "Any change since 2021?"):
        tini.respond(question)

    _, second, third = client.calls
    # Turn 2 sent [u1, a1, wrapped-u2]; turn 3 sends [u1, a1, u2, a2, wrapped-u3].
    # Everything turn 2 had BEFORE its own (volatile) message must reappear
    # unchanged at the start of turn 3, or the cache misses the whole history.
    assert third["messages"][: len(second["messages"]) - 1] == second["messages"][:-1]
    assert third["system"] == second["system"]
    assert len(third["messages"]) == 5


def test_volatile_context_rides_on_the_last_user_message_and_is_never_stored(tmp_path):
    tini, client = _two_turns(tmp_path)

    last = client.calls[1]["messages"][-1]["content"]
    assert "Right now it is" in last
    assert last.rstrip().endswith("[User message]\nAnd the audit committee?")
    assert [m["content"] for m in tini.session.history if m["role"] == "user"] == [
        "Who chairs the board?",
        "And the audit committee?",
    ], "history keeps the raw messages, not the per-turn wrapper"


def test_memory_and_skills_are_context_not_system(tmp_path):
    class _Memory:
        def gated_retrieve(self, message, notify=None):
            return "user prefers short answers"

        def matching_skills(self, message):
            return "skill: summarise politely"

    from tini.config import Settings

    settings = Settings(home=tmp_path / "home")
    settings.ensure_home()
    session = Session(settings, memory=_Memory())

    assert "short answers" not in session.build_system()
    context = session.build_context("hello")
    assert "user prefers short answers" in context and "summarise politely" in context


def test_the_loop_accumulates_provider_cache_hits():
    def reply(cache):
        r = response([text_block("ok")])
        r.usage = SimpleNamespace(input_tokens=1000, output_tokens=5, cache_hit_tokens=cache)
        return r

    result = run_loop(
        client=ScriptedClient([reply(900)]),
        model="m",
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        tools=ToolRegistry(),
    )

    assert result.usage["cache"] == 900 and result.usage["in"] == 1000


def test_a_provider_that_reports_no_cache_counts_zero():
    result = run_loop(
        client=ScriptedClient([response([text_block("ok")])]),
        model="m",
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        tools=ToolRegistry(),
    )

    assert result.usage["cache"] == 0


def test_cache_hit_tokens_reads_deepseek_and_openai_shapes():
    deepseek = SimpleNamespace(prompt_cache_hit_tokens=1408, prompt_cache_miss_tokens=247)
    openai = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=512))

    assert _cache_hit_tokens(deepseek) == 1408
    assert _cache_hit_tokens(openai) == 512
    assert _cache_hit_tokens(SimpleNamespace()) == 0
    assert _cache_hit_tokens(None) == 0
