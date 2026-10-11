"""Cross-language retrieval: the question is also searched in the documents' languages.

Network-free: the inner index and the translator are scripted. What matters is
the contract around the model call: it is skipped when there is nothing to
translate, cached when repeated, and when it fails the search still answers
from the original question and the failure is counted rather than hidden.
"""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.documents.models import SearchHit
from tini.documents.multilingual import (
    MultilingualIndex,
    _fuse,
    detect_language,
    make_translator,
)

# --- language detection ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("デンソーはいつ設立されましたか？", "ja"),
        ("DENSOの使命（ミッション）は何ですか？", "ja"),
        ("When was DENSO founded?", "en"),
        ("Bộ quy tắc ứng xử của DENSO áp dụng cho những ai?", "vi"),
        ("DENSO được thành lập năm nào?", "vi"),
        ("How many employees in FY2025?", "en"),
        ("", "en"),
    ],
)
def test_the_query_language_is_detected_from_its_characters(text, expected):
    assert detect_language(text) == expected


# --- the wrapper --------------------------------------------------------------


def _hit(chunk_id: str, text: str = "") -> SearchHit:
    return SearchHit(
        document_id="d",
        chunk_id=chunk_id,
        title="t",
        source_uri="file:///t",
        text=text or chunk_id,
        ordinal=0,
        score=1.0,
    )


class _Inner:
    """Returns a scripted ranking per query text and records every call."""

    def __init__(self, rankings: dict[str, list[str]]):
        self.rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def search(self, query, top_k=6, workspace_id="default"):
        self.calls.append((query, top_k))
        return [_hit(c) for c in self.rankings.get(query, [])][:top_k]

    def list_documents(self, workspace_id="default", limit=200):
        return ["listed"]

    def delete_document(self, document_id, workspace_id="default"):
        return True

    def opening_text(self, document_id, workspace_id="default", chars=3000):
        return "opening"


class _Translator:
    def __init__(self, table=None, fail=False):
        self.table = table or {}
        self.fail = fail
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, query, languages):
        self.calls.append((query, tuple(languages)))
        if self.fail:
            raise RuntimeError("model unavailable")
        return {code: self.table[query][code] for code in languages if code in self.table[query]}


def test_a_translation_finds_the_chunk_the_original_question_missed():
    inner = _Inner(
        {
            "When was DENSO founded?": ["english-noise", "other"],
            "デンソーはいつ設立されましたか": ["answer", "other"],
        }
    )
    translator = _Translator({"When was DENSO founded?": {"ja": "デンソーはいつ設立されましたか"}})
    index = MultilingualIndex(inner, translator, ["ja", "en"])

    hits = index.search("When was DENSO founded?", top_k=3)

    assert "answer" in [h.chunk_id for h in hits]
    # "other" is found by both phrasings, so it outranks either single-phrasing chunk
    assert hits[0].chunk_id == "other"


def test_only_the_missing_languages_are_requested():
    translator = _Translator({"When was DENSO founded?": {"ja": "設立はいつ", "vi": "thành lập năm nào"}})
    index = MultilingualIndex(_Inner({}), translator, ["ja", "en", "vi"])
    index.search("When was DENSO founded?")
    assert translator.calls == [("When was DENSO founded?", ("ja", "vi"))]


def test_a_question_already_in_the_only_corpus_language_costs_no_model_call():
    translator = _Translator()
    inner = _Inner({"When was DENSO founded?": ["a"]})
    index = MultilingualIndex(inner, translator, ["en"])
    assert [h.chunk_id for h in index.search("When was DENSO founded?")] == ["a"]
    assert translator.calls == []


def test_a_repeated_question_is_translated_once():
    translator = _Translator({"When was DENSO founded?": {"ja": "設立はいつ"}})
    index = MultilingualIndex(_Inner({}), translator, ["ja", "en"])
    for _ in range(3):
        index.search("When was DENSO founded?")
    assert len(translator.calls) == 1


def test_indexes_sharing_a_cache_translate_a_question_once_between_them():
    from collections import OrderedDict

    shared: OrderedDict = OrderedDict()
    translator = _Translator({"When was DENSO founded?": {"ja": "設立はいつ"}})
    for _ in range(3):  # the dashboard builds a new index per request
        MultilingualIndex(_Inner({}), translator, ["ja", "en"], cache=shared).search(
            "When was DENSO founded?"
        )
    assert len(translator.calls) == 1


def test_a_failed_translation_still_answers_and_is_counted():
    inner = _Inner({"When was DENSO founded?": ["a", "b"]})
    translator = _Translator(fail=True)
    index = MultilingualIndex(inner, translator, ["ja", "en"])

    hits = index.search("When was DENSO founded?", top_k=3)

    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert index.errors == 1 and "model unavailable" in index.last_error
    index.search("When was DENSO founded?")
    assert len(translator.calls) == 2, "a failure is not cached: the next search retries"


def test_a_translation_that_never_arrives_is_abandoned_and_counted():
    """The configured model was seen to hold each call for 6 minutes. The search
    must not wait that long: it gives up, answers from the original question,
    and says so."""
    import threading

    release = threading.Event()

    def stuck(query, languages):
        release.wait(5)
        return {}

    inner = _Inner({"When was DENSO founded?": ["a"]})
    index = MultilingualIndex(inner, stuck, ["ja", "en"], timeout=0.05)
    try:
        hits = index.search("When was DENSO founded?")
    finally:
        release.set()
    assert [h.chunk_id for h in hits] == ["a"]
    assert index.errors == 1 and "no answer within" in index.last_error


def test_after_repeated_failures_translation_pauses_then_resumes():
    now = [0.0]
    translator = _Translator(fail=True)
    index = MultilingualIndex(
        _Inner({}),
        translator,
        ["ja", "en"],
        breaker_failures=2,
        cooldown=60.0,
        clock=lambda: now[0],
    )
    for n in range(4):
        index.search(f"question number {n}")
    assert len(translator.calls) == 2, "two failures open the breaker; the rest skip the model"
    assert index.skipped == 2

    now[0] = 61.0
    translator.fail = False
    translator.table = {"another question": {"ja": "別の質問"}}
    index.search("another question")
    assert len(translator.calls) == 3, "after the cooldown it tries again"
    assert index.translated == 1


def test_a_translation_identical_to_the_question_is_not_searched_twice():
    inner = _Inner({"DENSO": ["a"]})
    translator = _Translator({"DENSO": {"ja": "DENSO"}})
    index = MultilingualIndex(inner, translator, ["ja", "en"])
    index.search("DENSO")
    assert [q for q, _ in inner.calls] == ["DENSO"]


def test_each_phrasing_gets_a_wider_pool_than_the_final_top_k():
    inner = _Inner({})
    translator = _Translator({"When was DENSO founded?": {"ja": "設立はいつ"}})
    MultilingualIndex(inner, translator, ["ja", "en"]).search("When was DENSO founded?", top_k=3)
    assert all(top_k >= 20 for _, top_k in inner.calls)


def test_documents_operations_pass_straight_through():
    index = MultilingualIndex(_Inner({}), _Translator(), ["ja"])
    assert index.list_documents() == ["listed"]
    assert index.delete_document("x") is True
    assert index.opening_text("x") == "opening"


def test_unknown_language_codes_are_ignored():
    index = MultilingualIndex(_Inner({}), _Translator(), ["ja", "klingon"])
    assert index.languages == ["ja"]


def test_fusion_ranks_a_chunk_seen_by_several_phrasings_first():
    fused = _fuse([[_hit("a"), _hit("b")], [_hit("c"), _hit("b")]], limit=3)
    assert [h.chunk_id for h in fused] == ["b", "a", "c"]


# --- the model-backed translator ---------------------------------------------


class _Block:
    def __init__(self, text, kind="text"):
        self.text, self.type = text, kind


class _Client:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return type("R", (), {"content": [_Block("thinking…", "thinking"), _Block(self.reply)]})()


def _translator_with(client, monkeypatch):
    from tini.loop import models

    monkeypatch.setattr(models, "get_client", lambda settings: client)
    return make_translator(Settings(small_model="small", model="big"))


def test_the_translator_parses_json_out_of_a_chatty_reply(monkeypatch):
    client = _Client('Sure!\n{"ja": "設立はいつ", "vi": "  "}\n')
    translate = _translator_with(client, monkeypatch)
    assert translate("When?", ["ja", "vi"]) == {"ja": "設立はいつ"}
    assert client.requests[0]["model"] == "small"


def test_the_translator_rejects_a_reply_with_no_json(monkeypatch):
    translate = _translator_with(_Client("I cannot do that"), monkeypatch)
    with pytest.raises(ValueError, match="no JSON"):
        translate("When?", ["ja"])


# --- wiring ---------------------------------------------------------------------


def test_it_is_off_unless_languages_are_configured(monkeypatch):
    monkeypatch.delenv("TINI_QUERY_LANGUAGES", raising=False)
    assert Settings().query_languages == ()
    monkeypatch.setenv("TINI_QUERY_LANGUAGES", "JA, en ,vi")
    assert Settings().query_languages == ("ja", "en", "vi")
