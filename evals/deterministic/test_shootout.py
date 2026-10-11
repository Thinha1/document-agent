"""DETERMINISTIC EVAL — the shootout's scoring contract, offline.

The shootout's value is its honesty: check_case is the same deterministic
contract as the live eval tier. These pin it (and the report math) without any
network or model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "shootout", Path(__file__).resolve().parents[2] / "scripts" / "shootout.py"
)
shootout = importlib.util.module_from_spec(spec)
sys.modules["shootout"] = shootout
spec.loader.exec_module(shootout)


def calls(*pairs):
    return [{"tool": t, "args": a} for t, a in pairs]


def test_right_tool_and_args_pass():
    case = {"expect_tool": "search_web", "expect_in_args": {"query": "worker"}}
    ok, why = shootout.check_case(case, calls(("search_web", {"query": "worker runbook"})))
    assert ok, why


def test_wrong_tool_fails():
    case = {"expect_tool": "search_web"}
    ok, why = shootout.check_case(case, calls(("other_tool", {})))
    assert not ok and "expected search_web" in why


def test_no_tool_case():
    case = {"expect_tool": None}
    assert shootout.check_case(case, [])[0]
    assert not shootout.check_case(case, calls(("search_web", {})))[0]


def test_multi_tool_minimum_enforced():
    """A multi-source case can require more than one retrieval call."""
    case = {
        "expect_tool": "search_web",
        "expect_in_args": {"query": "policy"},
        "expect_min_tool_calls": 3,
    }
    only_one = calls(("search_web", {"query": "policy"}))
    ok, why = shootout.check_case(case, only_one)
    assert not ok and "wanted >= 3" in why
    full = calls(
        ("search_web", {"query": "policy"}),
        ("search_web", {"query": "retention policy"}),
        ("search_web", {"query": "security policy"}),
    )
    assert shootout.check_case(case, full)[0]


def test_markdown_report_math():
    results = [
        {
            "provider": "kimi",
            "model": "kimi-k3",
            "passed": 6,
            "total": 7,
            "cost_usd": 0.1234,
            "avg_latency_s": 9.5,
            "cases": [{"tokens_in": 100, "tokens_out": 50, "passed": True}],
        }
    ]
    table = shootout.markdown(results)
    assert "| kimi:kimi-k3 | 6/7 | $0.1234 | 9.5s | 100/50 |" in table
    assert "re-run" in table  # the reproducibility line is part of the contract


def test_dataset_has_retrieval_cases():
    ids = [c["id"] for c in shootout.DATASET]
    assert "search-runbook" in ids and "search-policy" in ids
