"""The synthetic technical table set is only worth using if its answers are exact.

Generated with no model, so everything is checkable offline: every answer really is
in its document, the committed files are what the generator produces, each fact is
asked in all three languages, and every table technique is represented.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("synth_tech", _ROOT / "scripts" / "synth_tech.py")
tech = importlib.util.module_from_spec(_spec)
sys.modules["synth_tech"] = tech  # a dataclass looks its module up by name
_spec.loader.exec_module(tech)


def test_the_generator_is_deterministic_and_its_answers_are_in_the_documents():
    docs, cases = tech.build_all()
    again_docs, again_cases = tech.build_all()

    assert (docs, cases) == (again_docs, again_cases)
    assert tech.check(docs, cases) == []


def test_ten_facts_asked_in_three_languages_cover_every_technique():
    _, cases = tech.build_all()

    assert len(cases) == 30
    assert {c["lang"] for c in cases} == {"en", "ja", "vi"}
    assert {c["format"] for c in cases} == {
        "grouped-header", "late-row", "chunk-boundary", "multi-row", "semantic-key", "code-key", "plain-lookup",
    }
    for technique in {c["format"] for c in cases}:
        assert {c["lang"] for c in cases if c["format"] == technique} == {"en", "ja", "vi"}


def test_each_document_is_one_language_and_says_it_is_synthetic():
    docs, _ = tech.build_all()

    assert len(docs) == 8
    for name, text in docs.items():
        lang = Path(name).stem.rsplit("-", 1)[1]
        assert tech.NOTICE[lang] in text.splitlines()[2], name
        assert sum(other in text for other in tech.NOTICE.values()) == 1, name  # no second language


def test_the_documents_the_questions_name_exist_and_each_cross_language_fact_has_a_same_language_twin():
    docs, cases = tech.build_all()
    stems = {Path(n).stem for n in docs}

    for case in cases:
        assert case["expected_title"] in stems
    by_fact = {}
    for case in cases:
        by_fact.setdefault(case["id"].rsplit("-", 1)[0], []).append(case)
    for fact, group in by_fact.items():
        doc_lang = group[0]["expected_title"].rsplit("-", 1)[1]
        assert sorted(c["lang"] for c in group) == ["en", "ja", "vi"], fact
        assert doc_lang in {c["lang"] for c in group}, fact  # one of the three is same-language


def test_the_committed_files_are_what_the_generator_writes():
    docs, cases = tech.build_all()

    for name, text in docs.items():
        assert (tech.DOC_DIR / name).read_text(encoding="utf-8") == text, f"{name} is stale: run scripts/synth_tech.py build"
    committed = [json.loads(line) for line in tech.QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    # the chunk-boundary fact is chosen against the chunker, so it may move when the
    # chunker changes; everything else must match exactly
    stable = lambda rows: [c for c in rows if c["format"] != "chunk-boundary"]  # noqa: E731
    assert stable(committed) == stable(cases), "questions are stale: run scripts/synth_tech.py build"


def test_a_question_never_contains_its_answer_and_two_facts_never_share_a_question():
    _, cases = tech.build_all()

    assert len({c["query"] for c in cases}) == len(cases)
    for case in cases:
        assert case["expected_keywords"][-1] not in case["query"], case["id"]
