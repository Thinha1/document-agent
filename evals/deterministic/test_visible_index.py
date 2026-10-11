"""DETERMINISTIC EVAL — catalog status hides documents from search; the reply
language line names only what it is sure of. Network-free."""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.db import connect
from tini.documents import catalog
from tini.documents.models import SearchHit
from tini.documents.store import SqliteDocumentIndex, make_document_index
from tini.documents.visibility import VisibleIndex
from tini.runtime.session import Session, reply_language_note


@pytest.fixture
def conn(tmp_path):
    (tmp_path / "home").mkdir()
    connection = connect(tmp_path / "home")
    yield connection
    connection.close()


def _record(conn, uri, status=None, workspace="w"):
    catalog.record_ingest(
        conn,
        {"workspace": workspace, "source_uri": uri, "title": uri, "content_hash": "h",
         "text": "Code of Conduct", "profile": {}},
    )
    if status:
        catalog.update(conn, workspace, uri, {"status": status})


class _Ranked:
    """Returns the same ranking for any query, honouring top_k like a real index."""

    def __init__(self, uris):
        self.hits = [SearchHit("d", f"c{n}", u, u, "text", n, 1.0 / (n + 1)) for n, u in enumerate(uris)]
        self.asked: list[int] = []

    def search(self, query, top_k=6, workspace_id="w"):
        self.asked.append(top_k)
        return self.hits[:top_k]


def test_superseded_and_excluded_are_hidden_but_needs_check_is_not(conn):
    _record(conn, "old", "superseded")
    _record(conn, "readme", "excluded")
    _record(conn, "maybe", "needs_check")
    _record(conn, "fresh")
    index = VisibleIndex(_Ranked(["old", "readme", "maybe", "fresh"]), conn)

    assert [h.source_uri for h in index.search("q", top_k=4, workspace_id="w")] == ["maybe", "fresh"]


def test_nothing_hidden_is_a_plain_passthrough(conn):
    inner = _Ranked(["a", "b", "c"])
    index = VisibleIndex(inner, conn)

    assert [h.source_uri for h in index.search("q", top_k=2, workspace_id="w")] == ["a", "b"]
    assert inner.asked == [2]  # no over-fetch when there is nothing to drop


def test_hidden_chunks_do_not_shorten_the_page(conn):
    _record(conn, "old", "superseded")
    inner = _Ranked(["old"] * 5 + ["x1", "x2", "x3"])
    index = VisibleIndex(inner, conn)

    # top_k=3 -> looks at 9 first, which holds all three visible chunks
    assert [h.source_uri for h in index.search("q", top_k=3, workspace_id="w")] == ["x1", "x2", "x3"]
    # a second, wider look when the first page is still short
    inner2 = _Ranked(["old"] * 8 + ["x1", "x2"])
    assert [h.source_uri for h in VisibleIndex(inner2, conn).search("q", top_k=2, workspace_id="w")] == ["x1", "x2"]
    assert inner2.asked == [6, 50]


def test_the_mark_is_per_workspace_and_reversible(conn):
    _record(conn, "old", "superseded", workspace="a")
    index = VisibleIndex(_Ranked(["old", "b"]), conn)

    assert [h.source_uri for h in index.search("q", workspace_id="b")] == ["old", "b"]
    assert [h.source_uri for h in index.search("q", workspace_id="a")] == ["b"]
    catalog.update(conn, "a", "old", {"status": "current"})
    assert [h.source_uri for h in index.search("q", workspace_id="a")] == ["old", "b"]


def test_the_factory_puts_the_filter_innermost_and_delegates_the_rest(tmp_path, conn):
    settings = Settings(home=tmp_path / "home", document_index="sqlite", embedding_provider="none",
                        rerank_provider="none")
    index = make_document_index(settings, conn, None)

    assert isinstance(index, VisibleIndex) and isinstance(index.inner, SqliteDocumentIndex)
    assert index.list_documents("w") == []  # delegated, not lost


def test_reply_language_names_ja_and_vi_but_not_a_guess():
    assert "in Vietnamese" in reply_language_note("Quy định về an toàn lao động là gì?")
    assert "in Japanese" in reply_language_note("サステナビリティ方針は何ですか？")
    for message in ("What is the safety policy?", "Quelle est la politique de sécurité ?", "hello"):
        note = reply_language_note(message)
        assert "message is in" not in note  # detect_language reads any non-ja/vi text as English
        assert "language of the user's latest message" in note


def test_the_language_line_rides_on_context_not_on_the_cached_system_prompt(tmp_path):
    settings = Settings(home=tmp_path / "home")
    settings.ensure_home()
    session = Session(settings, memory=None)

    assert "language of the user's latest message" in session.build_context("Chính sách là gì?")
    assert "latest message" not in session.build_system()
