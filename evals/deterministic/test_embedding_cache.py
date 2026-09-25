"""Chunk embeddings are cached so re-ingesting costs only what changed.

Measured before the cache existed (local GPU, 338-chunk Japanese report): an
unchanged re-ingest embedded all 338 chunks and then reported "skipped"; a
one-paragraph edit embedded all 338 again. With a hosted embedding API that is
money, not just time. Network-free: a counting fake stands in for the provider.
"""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.documents.embeddings import (
    CachingEmbedder,
    OpenAICompatibleEmbeddingProvider,
    make_embedding_provider,
)


class _Counting:
    provider = "fake"
    model = "fake-model"
    dimensions = 3

    def __init__(self):
        self.embedded: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts, *, title):
        self.embedded.extend(texts)
        return [(float(len(text)), 1.0, 0.5) for text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return (1.0, 2.0, 3.0)


@pytest.fixture
def cached(tmp_path):
    inner = _Counting()
    return inner, CachingEmbedder(inner, tmp_path / "home" / "embed_cache.sqlite")


def test_an_unchanged_document_is_not_embedded_again(cached):
    inner, embedder = cached
    first = embedder.embed_documents(["alpha", "beta", "gamma"], title="Doc")
    second = embedder.embed_documents(["alpha", "beta", "gamma"], title="Doc")

    assert inner.embedded == ["alpha", "beta", "gamma"], "the second pass made no provider call"
    assert second == first
    assert (embedder.hits, embedder.misses) == (3, 3)


def test_an_edit_embeds_only_the_new_chunks_and_keeps_order(cached):
    inner, embedder = cached
    embedder.embed_documents(["alpha", "beta", "gamma"], title="Doc")
    inner.embedded.clear()

    vectors = embedder.embed_documents(["alpha", "beta EDITED", "gamma", "delta"], title="Doc")

    assert inner.embedded == ["beta EDITED", "delta"]
    assert [v[0] for v in vectors] == [5.0, 11.0, 5.0, 5.0], "results stay in input order"


def test_the_cache_is_shared_across_instances_and_survives_restarts(tmp_path):
    path = tmp_path / "embed_cache.sqlite"
    first = _Counting()
    CachingEmbedder(first, path).embed_documents(["alpha"], title="Doc")
    second = _Counting()

    CachingEmbedder(second, path).embed_documents(["alpha"], title="Doc")

    assert second.embedded == []


def test_a_different_title_model_or_dimension_is_a_different_entry(tmp_path):
    path = tmp_path / "embed_cache.sqlite"
    inner = _Counting()
    CachingEmbedder(inner, path).embed_documents(["alpha"], title="Doc")

    CachingEmbedder(inner, path).embed_documents(["alpha"], title="Renamed")
    other_model = _Counting()
    other_model.model = "other-model"
    CachingEmbedder(other_model, path).embed_documents(["alpha"], title="Doc")

    assert inner.embedded == ["alpha", "alpha"]
    assert other_model.embedded == ["alpha"], "a new model must not reuse the old model's vectors"


def test_queries_are_never_cached(cached):
    inner, embedder = cached

    embedder.embed_query("what is x")
    embedder.embed_query("what is x")

    assert inner.queries == ["what is x", "what is x"]


def test_a_failing_provider_caches_nothing(tmp_path):
    class Down(_Counting):
        def embed_documents(self, texts, *, title):
            raise RuntimeError("provider is down")

    path = tmp_path / "embed_cache.sqlite"
    with pytest.raises(RuntimeError):
        CachingEmbedder(Down(), path).embed_documents(["alpha"], title="Doc")
    healthy = _Counting()

    CachingEmbedder(healthy, path).embed_documents(["alpha"], title="Doc")

    assert healthy.embedded == ["alpha"]


def _settings(tmp_path, **overrides):
    return Settings(
        home=tmp_path / "home",
        embedding_provider="openai_compatible",
        embedding_model="m",
        embedding_api_key="k",
        embedding_dimensions=4,
        **overrides,
    )


def test_the_factory_wraps_the_provider_unless_the_cache_is_off(tmp_path):
    assert isinstance(make_embedding_provider(_settings(tmp_path, embedding_cache=True)), CachingEmbedder)
    assert isinstance(
        make_embedding_provider(_settings(tmp_path, embedding_cache=False)),
        OpenAICompatibleEmbeddingProvider,
    )
    assert make_embedding_provider(Settings(embedding_provider="none")) is None
