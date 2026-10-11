"""DETERMINISTIC EVAL — the dashboard's generation-eval backend: it never starts a model
call on its own, validates what a browser sends, streams and saves a run, honours
Cancel, and never leaks a key. Network-free: the eval itself is stubbed."""

from __future__ import annotations

import json
import threading

import pytest

from tini.config import Settings
from tini.ops import generation_api as api
from tini.ops import generation_eval as ge


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "home").mkdir()
    return Settings(home=tmp_path / "home", provider="deepseek", model="m-gen", small_model="m-gen", api_key="SECRET-KEY-VALUE")


@pytest.fixture(autouse=True)
def keys_exist(monkeypatch):
    # every provider "has a key" and building a client never touches the network
    monkeypatch.setattr(api, "get_client", lambda settings: object())


def _row(case, **kw):
    base = {"id": case["id"], "source": case["source"], "lang": case["lang"], "abstain": case["abstain"],
            "query": case["query"], "reply": "an answer", "searches": ["q"], "tool_ok": True, "language_ok": True,
            "seconds": 1.0, "tokens_in": 10, "tokens_out": 2}
    if not case["abstain"]:
        base.update(evidence_has_answer=True, reply_has_answer=True, cited=True)
    return {**base, **kw}


def _fake_eval(rows_seen=None, stop_after=None):
    def fake(cases, settings, *, judge_settings=None, reply_language=True, workers=1, on_row=None, stop=None, **kw):
        done_rows = []
        for i, case in enumerate(cases):
            if stop is not None and stop.is_set():
                break
            row = _row(case)
            done_rows.append(row)
            if on_row:
                on_row(row, len(done_rows), len(cases))
            if stop_after is not None and len(done_rows) >= stop_after and stop is not None:
                stop.set()
        if rows_seen is not None:
            rows_seen.append(reply_language)
        return {"rows": done_rows, "summary": ge.summarize(done_rows), "cancelled": bool(stop and stop.is_set())}

    return fake


def test_the_form_info_lists_sources_and_providers_but_never_a_key(settings):
    info = api.info(settings)

    assert info["total"] == 54 and set(info["sources"]) == set(ge.SOURCE_NAMES)
    assert info["generator"] == {"provider": "deepseek", "model": "m-gen"}
    assert {"id", "configured", "default_model"} == set(info["providers"][0])
    assert "SECRET-KEY-VALUE" not in json.dumps(info)


def test_planning_calls_no_model_and_estimates_the_size_of_a_run(settings, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a plan must not run the eval")

    monkeypatch.setattr(ge, "run_generation_eval", boom)

    plan = api.plan({"sources": ["real", "unanswerable"], "limit": 10, "arms": "both",
                     "judge": {"provider": "deepseek", "model": "m-judge"}}, settings)

    assert plan["questions"] == 10 and plan["arms"] == ["on", "off"]
    assert plan["estimate"]["judge_calls"] == 20 and plan["estimate"]["generator_calls"] > 0
    assert plan["judge"] == {"provider": "deepseek", "model": "m-judge"} and plan["self_judged"] is False
    assert "SECRET-KEY-VALUE" not in json.dumps(plan)


def test_a_judge_that_is_the_generator_is_flagged(settings):
    plan = api.plan({"judge": {"provider": "deepseek", "model": "m-gen"}}, settings)

    assert plan["self_judged"] is True


@pytest.mark.parametrize("payload, fragment", [
    ({"sources": ["nope"]}, "source"),
    ({"sources": []}, "source"),
    ({"arms": "sideways"}, "arms"),
    ({"limit": "abc"}, "number"),
    ({"limit": -3}, "positive"),
    ({"generator": {"provider": "not-a-provider", "model": "x"}}, "Unknown provider"),
    ({"judge": {"provider": "deepseek", "model": "bad\nmodel"}}, "not valid"),
    ({"judge": {"provider": "", "model": ""}}, "Pick the judge"),
    ({"judge": {"provider": "deepseek", "model": ""}}, "Pick the judge"),
])
def test_what_a_browser_sends_is_validated(settings, payload, fragment):
    assert fragment in api.plan(payload, settings)["error"]


def test_a_provider_without_a_key_is_refused_before_anything_runs(settings, monkeypatch):
    def no_key(candidate):
        raise SystemExit("no key")  # what get_client does when the key is missing

    monkeypatch.setattr(api, "get_client", no_key)

    assert "No API key" in api.plan({}, settings)["error"]


def test_a_different_provider_starts_clean_so_the_active_key_never_leaks(settings, monkeypatch):
    seen = []
    monkeypatch.setattr(api, "get_client", lambda candidate: seen.append(candidate) or object())

    api.plan({"generator": {"provider": "anthropic", "model": "other"}}, settings)

    assert seen[0].provider == "anthropic" and seen[0].api_key == "" and seen[0].base_url is None


def test_a_run_streams_each_question_saves_and_reports_both_arms(settings, monkeypatch):
    arms_seen, events = [], []
    monkeypatch.setattr(ge, "run_generation_eval", _fake_eval(arms_seen))

    result = api.run({"limit": 6, "arms": "both"}, lambda kind, ev: events.append((kind, ev)), settings)

    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds.count("row") == 12 and kinds.count("arm_done") == 2
    assert arms_seen == [True, False]  # language line on, then off
    assert events[0][1]["total"] == 12 and [e["done"] for k, e in events if k == "row"][-1] == 12
    saved = api.load_run({"name": result["saved_as"]}, settings)
    assert set(saved["arms"]) == {"on", "off"} and saved["cancelled"] is False
    assert "SECRET-KEY-VALUE" not in json.dumps(saved)


def test_cancelling_stops_the_queue_and_keeps_what_was_answered(settings, monkeypatch):
    monkeypatch.setattr(ge, "run_generation_eval", _fake_eval(stop_after=3))
    stop = threading.Event()

    result = api.run({"limit": 10, "arms": "both"}, lambda k, e: None, settings, stop)

    assert result["cancelled"] is True
    assert list(result["arms"]) == ["on"]  # the second arm never started
    assert len(result["arms"]["on"]["rows"]) == 3
    assert api.load_run({"name": result["saved_as"]}, settings)["cancelled"] is True


def test_only_one_run_at_a_time(settings, monkeypatch):
    gate, entered = threading.Event(), threading.Event()

    def slow(cases, *a, **k):
        entered.set()
        gate.wait(5)
        return {"rows": [], "summary": ge.summarize([]), "cancelled": False}

    monkeypatch.setattr(ge, "run_generation_eval", slow)
    first = threading.Thread(target=api.run, args=({"limit": 3}, lambda k, e: None, settings))
    first.start()
    entered.wait(5)

    second = api.run({"limit": 3}, lambda k, e: None, settings)
    gate.set()
    first.join(5)

    assert "already running" in second["error"]


def test_saved_runs_are_listed_newest_first_and_a_name_cannot_escape_the_folder(settings, monkeypatch):
    monkeypatch.setattr(ge, "run_generation_eval", _fake_eval())
    api.run({"limit": 3}, lambda k, e: None, settings)

    listed = api.list_runs({}, settings)["runs"]

    assert len(listed) == 1 and listed[0]["arms"]["on"]["n"] == 3
    for bad in ("../state.db", r"..\x.json", "a/b.json", "", "run.txt"):
        assert api.load_run({"name": bad}, settings)["error"] in {"invalid run name", "unknown run"}
    assert api.load_run({"name": "missing.json"}, settings)["error"] == "unknown run"


# --- over HTTP: the real dashboard handler, the run itself stubbed ----------------


@pytest.fixture
def server(tmp_path, monkeypatch):
    import http.server

    from tini.ops import dashboard

    monkeypatch.setenv("TINI_HOME", str(tmp_path / "home"))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body):
    import urllib.request

    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=10)


def test_the_plan_and_runs_routes_answer_json_and_the_page_data_carries_the_form_info(server, tmp_path):
    plan = json.load(_post(server + "/api/documents/generation/plan", {"limit": 5}))
    runs = json.load(_post(server + "/api/documents/generation/runs", {}))
    bad = json.load(_post(server + "/api/documents/generation/load", {"name": "../x.json"}))

    assert plan["questions"] == 5 and "estimate" in plan
    assert runs == {"runs": []}
    assert bad == {"error": "invalid run name"}


def test_the_stream_route_sends_events_then_a_done_event(server, monkeypatch):
    def fake_run(payload, emit, settings, stop):
        emit("start", {"total": 2})
        emit("row", {"arm": "on", "row": {"id": "a"}, "done": 1, "total": 2})
        return {"saved_as": "x.json", "cancelled": False}

    monkeypatch.setattr(api, "run", fake_run)

    response = _post(server + "/api/documents/generation/stream", {})
    events = [json.loads(line[5:]) for line in response.read().decode().splitlines() if line.startswith("data:")]

    assert response.headers["Content-Type"] == "text/event-stream"
    assert [e["kind"] for e in events] == ["start", "row", "done"]
    assert events[-1]["saved_as"] == "x.json"


def test_closing_the_page_mid_run_stops_the_queue(server, monkeypatch):
    import time

    stopped = threading.Event()

    def fake_run(payload, emit, settings, stop):
        for i in range(200):  # a long run; the page is closed after the first event
            emit("row", {"arm": "on", "row": {"id": str(i)}, "done": i, "total": 200})
            if stop.is_set():
                stopped.set()
                break
            time.sleep(0.02)
        return {"cancelled": stop.is_set()}

    monkeypatch.setattr(api, "run", fake_run)

    response = _post(server + "/api/documents/generation/stream", {})
    response.readline()
    response.close()

    assert stopped.wait(8), "the server never noticed the page went away"
