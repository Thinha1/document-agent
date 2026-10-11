"""DETERMINISTIC EVAL — the reranker can read a Vietnamese question in English.

The reranker (a cross-encoder) scores a Vietnamese question against an English
chunk worse than the same question in English (measured: 49 Vietnamese questions,
answers found 40 -> 43 at k=6 and documents 43 -> 47 when it reads the English
translation). Search still uses the question as asked; only the reranker's copy is
translated. Everything here is network-free: the translator is a stub.
"""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents import multilingual
from tini.documents import rerank as rerank_module
from tini.documents.models import SearchHit
from tini.documents.multilingual import EnglishRewriter
from tini.documents.rerank import RerankingIndex
from tini.documents.store import make_document_index

VI = "Sứ mệnh của DENSO trong Triết lý DENSO là gì?"
EN_OF_VI = "What is DENSO's mission in the DENSO Philosophy?"
JA = "指名委員会はどのように取締役候補者を決めますか？"


def _hits(n=3):
    return [SearchHit("d", f"c{i}", "t", "u", f"chunk {i}", i, 1.0 / (i + 1)) for i in range(n)]


class _Inner:
    def __init__(self):
        self.searched = []

    def search(self, query, top_k=6, workspace_id="default"):
        self.searched.append(query)
        return _hits()[:top_k]


class _Reranker:
    def __init__(self):
        self.read = []

    def rerank(self, query, passages):
        self.read.append(query)
        return [float(i) for i in range(len(passages))]


def _translator(calls, answer=EN_OF_VI, fail=False):
    def translate(query, languages):
        calls.append((query, tuple(languages)))
        if fail:
            raise RuntimeError("model down")
        return {"en": answer}

    return translate


def test_search_uses_the_question_as_asked_and_only_the_reranker_reads_the_translation():
    calls, inner, reranker = [], _Inner(), _Reranker()
    rewrite = EnglishRewriter(_translator(calls), ["vi"])
    index = RerankingIndex(inner, reranker, 20, rewrite)

    index.search(VI, top_k=2)

    assert inner.searched == [VI]
    assert reranker.read == [EN_OF_VI]
    assert index.rewritten == 1 and calls == [(VI, ("en",))]


def test_only_the_listed_languages_are_translated():
    calls = []
    rewrite = EnglishRewriter(_translator(calls), ["vi"])

    assert rewrite("What is the mission?") == "What is the mission?"
    assert rewrite(JA) == JA  # Japanese is not listed: measured not to help
    assert calls == []
    assert rewrite(VI) == EN_OF_VI and len(calls) == 1


def test_a_failed_translation_falls_back_to_the_question_and_is_counted():
    calls = []
    rewrite = EnglishRewriter(_translator(calls, fail=True), ["vi"])
    inner, reranker = _Inner(), _Reranker()
    index = RerankingIndex(inner, reranker, 20, rewrite)

    hits = index.search(VI, top_k=2)

    assert len(hits) == 2 and reranker.read == [VI]  # still reranked, with the original
    assert index.rewritten == 0 and rewrite.helper.errors == 1
    assert "model down" in rewrite.helper.last_error


def test_a_rewriter_that_raises_can_never_fail_a_search():
    def broken(query):
        raise ValueError("boom")

    reranker = _Reranker()
    index = RerankingIndex(_Inner(), reranker, 20, broken)

    assert len(index.search(VI, top_k=2)) == 2
    assert reranker.read == [VI]


def test_the_same_question_is_translated_once(monkeypatch):
    monkeypatch.setattr(multilingual, "SHARED_CACHE", multilingual.OrderedDict())
    calls = []
    rewrite = EnglishRewriter(_translator(calls), ["vi"], cache=multilingual.SHARED_CACHE)

    assert rewrite(VI) == rewrite(VI) == EN_OF_VI
    assert len(calls) == 1


def test_it_is_off_by_default_and_parses_a_language_list(monkeypatch):
    monkeypatch.delenv("TINI_RERANK_TRANSLATE", raising=False)
    assert Settings().rerank_translate == ()
    monkeypatch.setenv("TINI_RERANK_TRANSLATE", " VI , ja ")
    assert Settings().rerank_translate == ("vi", "ja")


@pytest.mark.parametrize("languages, expects_rewriter", [((), False), (("vi",), True)])
def test_the_factory_wires_the_rewriter_only_when_asked(tmp_path, monkeypatch, languages, expects_rewriter):
    monkeypatch.setattr(rerank_module, "make_reranker", lambda settings: _Reranker())
    monkeypatch.setattr(multilingual, "make_translator", lambda settings: _translator([]))
    (tmp_path / "home").mkdir()
    settings = Settings(
        home=tmp_path / "home", document_index="sqlite", embedding_provider="none",
        rerank_provider="tei", rerank_translate=languages,
    )

    index = make_document_index(settings, connect(settings.home), None)

    assert isinstance(index, RerankingIndex)
    assert isinstance(index.rewrite_query, EnglishRewriter) is expects_rewriter
