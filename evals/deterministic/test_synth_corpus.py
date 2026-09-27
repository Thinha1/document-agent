"""The synthetic cross-language set is only worth using if its answers are real.

The generator asks a model for documents and questions; everything the model
returns is checked mechanically here (network-free): an answer that is not in
the document, or a question that gives the answer away, must be dropped rather
than turned into a test case that measures nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "synth_corpus.py"
_spec = importlib.util.spec_from_file_location("synth_corpus", _SCRIPT)
synth = importlib.util.module_from_spec(_spec)
sys.modules["synth_corpus"] = synth  # a dataclass looks its module up by name
_spec.loader.exec_module(synth)


def _fact(**overrides):
    fact = {
        "answer_span": "1949",
        "q_en": "In which year was the company founded?",
        "q_ja": "会社が設立されたのは何年ですか？",
        "q_vi": "Công ty được thành lập vào năm nào?",
    }
    fact.update(overrides)
    return fact


DOCUMENT = "The company was founded in 1949 as a spin-off.\nIt has 164,572 employees."


def test_a_good_fact_is_accepted():
    assert synth.valid_fact(_fact(), DOCUMENT) is None


def test_an_answer_that_is_not_in_the_document_is_rejected():
    assert synth.valid_fact(_fact(answer_span="1961"), DOCUMENT) == "answer not in document"


def test_the_answer_is_found_despite_spacing_and_width_differences():
    assert synth.valid_fact(_fact(answer_span="164, 572"), DOCUMENT) is None
    assert synth.valid_fact(_fact(answer_span="１９４９"), DOCUMENT) is None


@pytest.mark.parametrize("code", ["en", "ja", "vi"])
def test_a_question_that_contains_the_answer_is_rejected(code):
    fact = _fact(**{f"q_{code}": "Was it 1949 that the company was founded?"})
    assert synth.valid_fact(fact, DOCUMENT) == f"q_{code} contains the answer"


def test_a_missing_language_or_an_absurd_answer_length_is_rejected():
    assert synth.valid_fact(_fact(q_vi=""), DOCUMENT) == "missing q_vi"
    assert synth.valid_fact(_fact(answer_span="x"), DOCUMENT) == "answer_span length"
    assert synth.valid_fact(_fact(answer_span="a" * 61), DOCUMENT) == "answer_span length"


def test_a_fact_becomes_one_case_per_language_over_the_same_answer_and_document():
    plan = synth.PLAN[0]
    rows = synth.cases_for(plan, [_fact()])
    assert [r["lang"] for r in rows] == ["en", "ja", "vi"]
    assert {r["expected_title"] for r in rows} == {plan.name}
    assert {tuple(r["expected_keywords"]) for r in rows} == {("1949",)}
    assert {r["format"] for r in rows} == {f"{plan.language}-doc"}
    assert len({r["id"] for r in rows}) == 3


def test_every_document_is_planned_in_exactly_one_language_and_names_are_unique():
    names = [p.name for p in synth.PLAN]
    assert len(names) == len(set(names))
    assert {p.language for p in synth.PLAN} == {"en", "ja", "vi"}


def test_the_two_excerpts_of_one_source_do_not_overlap():
    by_source: dict[str, list] = {}
    for plan in synth.PLAN:
        by_source.setdefault(plan.source, []).append(plan)
    text = "\n".join(f"line {n} " + "x" * 40 for n in range(2000))
    for plans in by_source.values():
        windows = [synth.excerpt(text, p.start) for p in plans if len(plans) > 1]
        for a in range(len(windows)):
            for b in range(a + 1, len(windows)):
                assert not set(windows[a].splitlines()) & set(windows[b].splitlines())


def test_excerpt_takes_whole_lines_and_short_text_is_kept_whole():
    text = "\n".join(f"line {n:04d}" for n in range(1000))
    part = synth.excerpt(text, 0.5, chars=500)
    assert len(part) <= 500
    assert all(line.startswith("line ") and len(line) == 9 for line in part.splitlines())
    assert synth.excerpt("short text", 0.5) == "short text"


def test_layout_whitespace_is_collapsed_without_touching_the_words():
    messy = "Title   \n\n\n\n" + " " * 5000 + "\nBody\u3000\u3000text  here  \n"
    assert synth.clean_text(messy) == "Title\n\nBody text here"
    assert synth.squash(messy) == synth.squash(synth.clean_text(messy)), "answers survive cleaning"


def test_slices_respect_the_limit_and_lose_nothing():
    text = "\n".join(f"paragraph {n} " + "word " * 30 for n in range(40))
    pieces = synth.slices(text, limit=500)
    assert all(len(p) <= 500 for p in pieces)
    assert "".join(pieces).replace("\n", "") == text.replace("\n", "")
    assert synth.slices("x" * 1200, limit=500) == ["x" * 500, "x" * 500, "x" * 200]


def test_json_is_extracted_from_a_chatty_reply():
    reply = 'Here you go:\n```json\n[{"answer_span": "a"}]\n```\nDone.'
    assert synth.parse_json_object(reply) == [{"answer_span": "a"}]
    with pytest.raises(ValueError):
        synth.parse_json_object("no json here")


def test_build_one_translates_only_when_the_language_differs_and_drops_bad_facts():
    calls: list[str] = []

    def chat(prompt: str) -> str:
        calls.append(prompt)
        if prompt.startswith("Translate"):
            return "Công ty được thành lập năm 1949."
        return json.dumps(
            [
                {
                    "answer_span": "1949",
                    "q_en": "When was it founded?",
                    "q_ja": "いつ設立？",
                    "q_vi": "Thành lập khi nào?",
                },
                {"answer_span": "1999", "q_en": "a", "q_ja": "b", "q_vi": "c"},
            ]
        )

    plan = synth.Plan("t", "unused.pdf", 0.0, "vi")
    document, cases, notes = synth.build_one(chat, plan, "The company was founded in 1949.")

    assert document == "Công ty được thành lập năm 1949."
    assert len(cases) == 3 and cases[0]["expected_keywords"] == ["1949"]
    assert notes == ["syn-t-vi: dropped a fact (answer not in document)"]
    assert sum(c.startswith("Translate") for c in calls) == 1

    calls.clear()
    plan_en = synth.Plan("t", "unused.pdf", 0.0, "en")
    synth.build_one(chat, plan_en, "The company was founded in 1949.")
    assert not any(c.startswith("Translate") for c in calls), "already in the target language"


# --- house style ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "language", "fragment"),
    [
        ("According to the text, what are enterprises called?", "en", "depends on its document"),
        ("What is the title of this policy?", "en", "depends on its document"),
        ("What was the amount in the current consolidated fiscal year?", "en", "depends on its document"),
        ("本文によれば、企業は何と呼ばれていますか？", "ja", "depends on its document"),
        ("Theo văn bản, doanh nghiệp được gọi là gì?", "vi", "depends on its document"),
        ("Which year?", "en", "length"),
        ("In which fiscal year did DENSO select its priority issues under the 2030 policy", "en", "question mark"),
        ("What year was DENSO founded? And who founded it?", "en", "more than one question"),
    ],
)
def test_style_problems_are_caught(query, language, fragment):
    assert any(fragment in p for p in synth.style_problems(query, language))


def test_a_self_contained_question_has_no_style_problems():
    assert synth.style_problems("In what year did DENSO establish its technical training institute?", "en") == []
    assert synth.style_problems("DENSOが技術・技能の強化のために技術研修所を設立したのは何年ですか？", "ja") == []
    assert synth.style_problems("DENSO thành lập cơ sở đào tạo kỹ thuật vào năm nào vậy?", "vi") == []


def _cases(doc="syn-compliance-code-vi", answer="1949", n=4):
    plan = next(p for p in synth.PLAN if p.name == doc)
    facts = [
        {
            "answer_span": f"{answer}{i}" if i else answer,
            "q_en": "In which year did DENSO establish its technical training institute?",
            "q_ja": "DENSOが技術・技能の強化のために技術研修所を設立したのは何年ですか？",
            "q_vi": "DENSO thành lập cơ sở đào tạo kỹ thuật vào năm nào vậy?",
        }
        for i in range(n)
    ]
    return synth.cases_for(plan, facts)


def test_check_reports_every_way_a_set_can_be_unsound():
    docs = {p.name: "1949 19491 19492 19493" for p in synth.PLAN}
    assert not [p for p in synth.check_cases(_cases(), docs) if "compliance" in p and "no questions" not in p]

    assert any("not in" in p for p in synth.check_cases(_cases(answer="2077"), docs))
    assert any("under 4" in p for p in synth.check_cases(_cases(answer="19"), docs))
    assert any("3 facts" in p for p in synth.check_cases(_cases(n=3), docs))
    assert any("no document" in p for p in synth.check_cases(_cases(), {}))
    truncated = [c for c in _cases() if c["lang"] != "vi"]
    assert any("asked in" in p for p in synth.check_cases(truncated, docs))


def test_a_question_that_gives_the_answer_away_is_caught():
    docs = {p.name: "1949 19491 19492 19493" for p in synth.PLAN}
    cases = _cases()
    cases[0]["query"] = "Was the DENSO technical training institute established in 1949?"
    assert any("contains its own answer" in p for p in synth.check_cases(cases, docs))


def test_the_committed_question_set_is_sound_and_uniform():
    """The real files: every answer is in its document, no question depends on
    context it does not carry, every document has exactly the same number of
    facts, and every fact is asked in all three languages."""
    documents = synth.load_documents()
    cases = [json.loads(line) for line in synth.QUESTIONS.read_text(encoding="utf-8").splitlines() if line]
    assert synth.check_cases(cases, documents) == []
    assert len(cases) == len(synth.PLAN) * synth.FACTS_PER_DOC * len(synth.LANGUAGES)


def test_the_three_languages_of_a_fact_are_similar_in_length():
    """A gross mismatch (one language answering a much shorter or looser question)
    is how a set drifts apart. Loose bounds: ja is naturally shorter in characters."""
    cases = [
        json.loads(line) for line in synth.QUESTIONS.read_text(encoding="utf-8").splitlines() if line
    ]
    by_fact: dict[str, dict[str, int]] = {}
    for case in cases:
        by_fact.setdefault(case["id"].rsplit("-", 1)[0], {})[case["lang"]] = len(case["query"])
    for fact, lengths in by_fact.items():
        assert 0.7 <= lengths["vi"] / lengths["en"] <= 1.6, (fact, lengths)
        assert 0.15 <= lengths["ja"] / lengths["en"] <= 0.9, (fact, lengths)


def test_build_refuses_to_overwrite_hand_edited_files_without_force(capsys):
    import argparse

    assert synth.cmd_build(argparse.Namespace(force=False, workers=1)) == 2
    assert "--force" in capsys.readouterr().err
