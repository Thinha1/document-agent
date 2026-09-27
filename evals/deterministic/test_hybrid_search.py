"""Hybrid retrieval on the Qdrant backend: dense ranking fused with a keyword one.

The failure this guards against is real, not hypothetical: FTS5's default
tokenizer cannot split Japanese, so a keyword search for 指名委員会 found nothing.
The keyword side here must find it, and must keep finding chunks the dense side
ranks badly (a name, a figure, a clause number).

Runs on qdrant-client's in-process local mode; no network.
"""

from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from tini.config import Settings  # noqa: E402
from tini.documents import lexical  # noqa: E402
from tini.documents.models import DocumentChunk, DocumentSource  # noqa: E402
from tini.documents.qdrant_store import QdrantDocumentIndex, migrate_to_hybrid  # noqa: E402

# --- tokenizer ---------------------------------------------------------------


def test_japanese_becomes_overlapping_bigrams_so_a_term_matches_inside_a_sentence():
    assert lexical.tokenize("取締役会") == ["取締", "締役", "役会"]
    sentence = set(lexical.tokenize("指名委員会が候補者を決定します"))
    assert set(lexical.tokenize("指名委員会")) <= sentence


def test_latin_vietnamese_and_numbers_split_into_words():
    assert lexical.tokenize("Founded in 1949, Nhà máy") == ["founded", "in", "1949", "nhà", "máy"]


def test_full_width_and_half_width_spellings_agree():
    assert lexical.tokenize("ＦＹ２０２４") == lexical.tokenize("FY2024")
    assert lexical.tokenize("ｶﾞﾊﾞﾅﾝｽ") == lexical.tokenize("ガバナンス")


def test_a_single_cjk_character_is_kept_rather_than_dropped():
    assert lexical.tokenize("長") == ["長"]
    assert lexical.tokenize("2024年") == ["2024", "年"]


def test_token_ids_are_stable_across_processes():
    """CRC32, not hash(): a per-process id would make yesterday's index match
    nothing today. Pinned value so a change of hash function fails loudly."""
    assert lexical.query_vector("governance")[0] == [854162502]


def test_a_repeated_term_counts_more_but_saturates():
    once = dict(zip(*lexical.document_vector("alpha beta"), strict=True))
    many = dict(zip(*lexical.document_vector("alpha " * 50 + "beta"), strict=True))
    alpha = lexical.query_vector("alpha")[0][0]
    assert many[alpha] > once[alpha]
    assert many[alpha] < 50 * once[alpha] / 4, "term frequency must saturate, not scale linearly"


def test_empty_text_gives_an_empty_vector():
    assert lexical.document_vector("") == ([], [])
    assert lexical.query_vector("   ") == ([], [])


# --- retrieval ---------------------------------------------------------------


class _Embedder:
    """Dense side that knows nothing about the words: every text and query is
    the same vector, so ranking by embedding alone is a coin flip. Only the
    keyword side can put the right chunk first."""

    provider = "fake"
    model = "flat"
    dimensions = 3

    def embed_documents(self, texts, *, title):
        return [(1.0, 0.0, 0.0)] * len(texts)

    def embed_query(self, text):
        return (1.0, 0.0, 0.0)


TEXTS = {
    "nomination": "指名委員会が取締役候補者の選任案を決定します。",
    "audit": "監査委員会は財務報告の適正性を確認します。",
    "founded": "The company was founded in 1949 as a spin-off.",
    "vi": "Chính sách nhân sự áp dụng cho toàn bộ nhân viên.",
    "filler": "This chunk discusses unrelated logistics and warehouse space.",
}


def _source(uri: str, content_hash: str) -> DocumentSource:
    return DocumentSource(
        workspace_id="w",
        source_uri=uri,
        title="doc",
        media_type="text/plain",
        size_bytes=1,
        content_hash=content_hash,
    )


def _index(hybrid: bool, client=None):
    client = client or qdrant_client.QdrantClient(location=":memory:")
    index = QdrantDocumentIndex(Settings(), _Embedder(), client=client)
    index.hybrid = hybrid
    source = _source("file:///doc.txt", "h1")
    chunks = [
        DocumentChunk(
            ordinal=i,
            text=text,
            char_start=0,
            char_end=len(text),
            metadata={},
            embedding=(1.0, 0.0, 0.0),
        )
        for i, text in enumerate(TEXTS.values())
    ]
    index.upsert_document(source, chunks)
    return index


def test_hybrid_is_off_unless_asked_for():
    """It did not beat dense-only on the DENSO set, so it must not silently
    change retrieval for anyone who has not opted in."""
    assert Settings().document_hybrid is False


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("指名委員会", "nomination"),
        ("監査委員会", "audit"),
        ("founded 1949", "founded"),
        ("nhân viên", "vi"),
    ],
)
def test_the_keyword_side_puts_the_right_chunk_first_when_dense_cannot(query, expected):
    hit = _index(hybrid=True).search(query, top_k=1, workspace_id="w")[0]
    assert hit.text == TEXTS[expected]


def test_a_two_character_japanese_term_is_findable():
    """FTS5's `trigram` tokenizer is blind to terms under 3 characters. 議長
    is two."""
    index = _index(hybrid=True)
    text = "取締役会の議長は独立社外取締役が務めます。"
    source = _source("file:///b.txt", "h2")
    index.upsert_document(
        source,
        [DocumentChunk(0, text, 0, len(text), {}, embedding=(1.0, 0.0, 0.0))],
    )
    assert index.search("議長", top_k=1, workspace_id="w")[0].text == text


def test_turning_hybrid_off_falls_back_to_dense_only():
    """With a flat embedder, dense-only ranking cannot single out the chunk, so
    the setting demonstrably does something."""
    off = _index(hybrid=False).search("指名委員会", top_k=1, workspace_id="w")[0]
    on = _index(hybrid=True).search("指名委員会", top_k=1, workspace_id="w")[0]
    assert on.text == TEXTS["nomination"]
    assert off.text != on.text or off.score != on.score


def test_a_query_with_no_tokens_still_answers_from_the_dense_side():
    assert _index(hybrid=True).search("？！", top_k=2, workspace_id="w")


def test_hybrid_respects_the_workspace_filter():
    assert _index(hybrid=True).search("指名委員会", top_k=3, workspace_id="other") == []


def test_reingesting_a_changed_document_replaces_its_keyword_vectors():
    index = _index(hybrid=True)
    text = "新しい規程は経営会議が承認します。"
    source = _source("file:///doc.txt", "h-changed")
    index.upsert_document(
        source, [DocumentChunk(0, text, 0, len(text), {}, embedding=(1.0, 0.0, 0.0))]
    )
    assert index.search("経営会議", top_k=1, workspace_id="w")[0].text == text
    assert all("指名委員会" not in hit.text for hit in index.search("指名委員会", top_k=5, workspace_id="w"))


# --- a collection that predates hybrid search --------------------------------


def _legacy_collection(client, name="old"):
    """Dense-only collection as it exists in production today: an unnamed
    vector, no keyword vector."""
    from qdrant_client import models

    client.create_collection(
        name, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    for i, text in enumerate(TEXTS.values()):
        client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=i + 1,
                    vector=[1.0, 0.0, 0.0],
                    payload={
                        "document_id": "d",
                        "workspace_id": "w",
                        "source_uri": "file:///old.txt",
                        "title": "old",
                        "media_type": "text/plain",
                        "content_hash": "x",
                        "status": "ready",
                        "updated_at": "2026-01-01",
                        "ordinal": i,
                        "text": text,
                    },
                )
            ],
        )


def test_a_legacy_collection_keeps_working_dense_only():
    """Qdrant cannot add a vector to an existing collection, so an old one must
    not be touched or broken: it searches and ingests exactly as before."""
    client = qdrant_client.QdrantClient(location=":memory:")
    _legacy_collection(client)
    index = QdrantDocumentIndex(Settings(qdrant_collection="old"), _Embedder(), client=client)
    assert index.has_lexical is False
    assert index.search("anything", top_k=2, workspace_id="w")
    index.upsert_document(
        _source("file:///new.txt", "h9"),
        [DocumentChunk(0, "fresh text", 0, 10, {}, embedding=(1.0, 0.0, 0.0))],
    )


def test_migrating_copies_points_and_adds_keyword_search_without_re_embedding():
    """The originals of most indexed documents may be unreachable, and
    re-embedding costs time: the copy must work from the stored vectors and
    payload alone, and leave the source collection as it was."""
    client = qdrant_client.QdrantClient(location=":memory:")
    _legacy_collection(client)

    assert migrate_to_hybrid(client, "old", "new") == len(TEXTS)

    assert client.count("old").count == len(TEXTS)
    index = QdrantDocumentIndex(Settings(qdrant_collection="new"), _Embedder(), client=client)
    index.hybrid = True
    assert index.has_lexical is True
    assert index.search("指名委員会", top_k=1, workspace_id="w")[0].text == TEXTS["nomination"]
    assert index.search("監査委員会", top_k=1, workspace_id="w")[0].text == TEXTS["audit"]


def test_migrating_refuses_to_overwrite_an_existing_collection():
    client = qdrant_client.QdrantClient(location=":memory:")
    _legacy_collection(client, "old")
    _legacy_collection(client, "taken")
    with pytest.raises(ValueError, match="already exists"):
        migrate_to_hybrid(client, "old", "taken")
