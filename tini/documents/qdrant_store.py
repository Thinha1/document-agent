"""Document index, vector edition — the self-hosted Qdrant upgrade path.

Same `DocumentIndex` interface as `SqliteDocumentIndex` (tini/documents/store.py),
different retrieval: Qdrant does ANN search over a real vector index instead of a
Python cosine-similarity scan over packed floats. Point payload carries what
SqliteDocumentIndex splits across `documents` + `document_chunks` — one
denormalized, chunk-level model, the same shape `SupabaseFactStore`
(tini/memory/semantic/supabase_store.py) uses for its hosted `rag_chunks`
table. That file is the reference for a hosted/self-hosted vector adapter in
this codebase; this one follows it.

    pip install 'tini-agent[qdrant]'
    docker compose up -d qdrant           # see docker-compose.yml
    TINI_DOCUMENT_INDEX=qdrant
    TINI_QDRANT_URL=http://localhost:6333  # default; override for a remote instance
    TINI_EMBEDDING_PROVIDER=...            # required, see the trade-off below

Retrieval is hybrid: a dense vector ranking and a keyword ranking (a sparse
BM25-style vector, tini/documents/lexical.py) are fused server-side with
reciprocal-rank fusion, the same idea SqliteDocumentIndex applies to FTS5. The
keyword side is built here, not by FTS5, because FTS5's tokenizer cannot split
Japanese. `TINI_DOCUMENT_HYBRID=0` turns it back into dense-only. An embedder
is required either way. A chunk that arrives
without an embedding is refused rather than silently dropped: a vector index
that quietly loses a fraction of a document is the "no matching facts" lie
`tini/memory/semantic/base.py` warns about, just at the storage layer instead
of the query layer.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tini.config import Settings
from tini.documents import lexical
from tini.documents.embeddings import EmbeddingProvider
from tini.documents.models import (
    DEFAULT_WORKSPACE,
    DocumentChunk,
    DocumentSource,
    DocumentSummary,
    IngestReport,
    SearchHit,
)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

# Scanned in SCROLL_PAGE-sized pages by list_documents/delete/_find_existing.
# Qdrant has no server-side GROUP BY, so reconstructing per-document summaries
# means walking every chunk payload in the workspace — legible and correct at
# the single-operator, self-hosted scale this backend targets; a workspace
# with millions of chunks would want a real aggregation query instead.
SCROLL_PAGE = 256

# Name of the sparse (keyword) vector on each point. The dense embedding stays
# the collection's unnamed default vector, so collections created before hybrid
# search keep working: they only need this second vector added.
LEXICAL = "lexical"


class QdrantDocumentIndex:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider | None,
        client: QdrantClient | None = None,
    ):
        if embedder is None:
            raise ValueError(
                "TINI_DOCUMENT_INDEX=qdrant requires embeddings — set TINI_EMBEDDING_PROVIDER "
                "(this backend has no lexical fallback; see qdrant_store.py's module docstring)"
            )
        self.embedder = embedder
        self.collection = settings.qdrant_collection
        self.hybrid = settings.document_hybrid
        self.fusion = settings.document_fusion
        self.lexical_weight = settings.document_lexical_weight
        self.has_lexical = False  # set by _ensure_collection
        if client is not None:
            self.client = client
        else:
            from qdrant_client import QdrantClient as _QdrantClient

            self.client = _QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            # Qdrant cannot add a vector to an existing collection, so one built
            # before hybrid search stays dense-only and works exactly as before.
            # `migrate_to_hybrid` copies it into a new collection that has the
            # keyword vector; point TINI_QDRANT_COLLECTION at that one.
            existing = self.client.get_collection(self.collection).config.params.sparse_vectors
            self.has_lexical = bool(existing) and LEXICAL in existing
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dimensions, distance=models.Distance.COSINE
            ),
            sparse_vectors_config=_sparse_config(),
        )
        self.has_lexical = True
        # Payload indexes make workspace/document/source filters real index
        # lookups instead of a full collection scan on every call.
        for field_name in ("workspace_id", "document_id", "source_uri"):
            self.client.create_payload_index(
                self.collection, field_name, field_schema=models.PayloadSchemaType.KEYWORD
            )

    # --- writes ---------------------------------------------------------------

    def upsert_document(
        self, source: DocumentSource, chunks: Sequence[DocumentChunk]
    ) -> IngestReport:
        from qdrant_client import models

        existing = self._find_existing(source.workspace_id, source.source_uri)
        if existing and existing["content_hash"] == source.content_hash:
            count = self._count(existing["document_id"])
            if count:
                return IngestReport(
                    source_uri=source.source_uri,
                    title=source.title,
                    action="skipped",
                    chunk_count=count,
                    document_id=existing["document_id"],
                )

        document_id = existing["document_id"] if existing else uuid.uuid4().hex
        action = "updated" if existing else "created"

        missing = [chunk.ordinal for chunk in chunks if chunk.embedding is None]
        if missing:
            raise ValueError(
                f"{source.source_uri}: chunk(s) {missing} have no embedding — "
                "the Qdrant backend cannot index a chunk it can't search"
            )

        # Delete-then-insert, same as SqliteDocumentIndex's DELETE followed by
        # its executemany INSERT: a shrunk document must not leave orphaned
        # chunks from the old version behind as stale search results.
        if existing:
            self._delete_by_document_id(document_id)

        if chunks:
            metadata = json.dumps(source.metadata, ensure_ascii=False, sort_keys=True)
            updated_at = datetime.now(UTC).isoformat()
            points = [
                models.PointStruct(
                    id=_point_id(document_id, source.content_hash, chunk.ordinal),
                    vector=(
                        {
                            "": list(chunk.embedding),
                            LEXICAL: _sparse(lexical.document_vector(chunk.text, source.title)),
                        }
                        if self.has_lexical
                        else list(chunk.embedding)
                    ),
                    payload={
                        "document_id": document_id,
                        "workspace_id": source.workspace_id,
                        "source_uri": source.source_uri,
                        "title": source.title,
                        "media_type": source.media_type,
                        "content_hash": source.content_hash,
                        "status": "ready",
                        "updated_at": updated_at,
                        "ordinal": chunk.ordinal,
                        "text": chunk.text,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "chunk_metadata": json.dumps(
                            chunk.metadata, ensure_ascii=False, sort_keys=True
                        ),
                        "document_metadata": metadata,
                    },
                )
                for chunk in chunks
            ]
            self.client.upsert(collection_name=self.collection, points=points)

        return IngestReport(
            source_uri=source.source_uri,
            title=source.title,
            action=action,
            chunk_count=len(chunks),
            document_id=document_id,
        )

    def opening_text(
        self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE, chars: int = 3000
    ) -> str:
        from qdrant_client import models

        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                    models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
                    models.FieldCondition(key="ordinal", range=models.Range(lt=4)),
                ]
            ),
            limit=8,
            with_payload=True,
        )
        ordered = sorted(points, key=lambda point: point.payload["ordinal"])
        return "\n".join(point.payload["text"] for point in ordered)[:chars]

    def delete_document(self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE) -> bool:
        from qdrant_client import models

        flt = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id)),
            ]
        )
        if not self.client.count(self.collection, count_filter=flt, exact=True).count:
            return False
        self.client.delete(collection_name=self.collection, points_selector=models.FilterSelector(filter=flt))
        return True

    # --- reads ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 6, workspace_id: str = DEFAULT_WORKSPACE
    ) -> list[SearchHit]:
        from qdrant_client import models

        if not query.strip():
            return []
        limit = max(1, min(int(top_k), 50))
        query_vector = list(self.embedder.embed_query(query))
        in_workspace = models.Filter(
            must=[
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
            ]
        )
        keywords = lexical.query_vector(query) if self.hybrid and self.has_lexical else ([], [])
        if keywords[0]:
            # Each ranking gets a wider pool than `limit` so a chunk one leg
            # puts 1st and the other 30th can still surface after fusion.
            pool = min(200, max(limit * 4, 40))
            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=query_vector, filter=in_workspace, limit=pool),
                    models.Prefetch(
                        query=_sparse(keywords), using=LEXICAL, filter=in_workspace, limit=pool
                    ),
                ],
                query=self._fusion_query(),
                limit=limit,
                with_payload=True,
            ).points
        else:
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=in_workspace,
                limit=limit,
                with_payload=True,
            ).points
        return [
            SearchHit(
                document_id=point.payload["document_id"],
                chunk_id=str(point.id),
                title=point.payload["title"],
                source_uri=point.payload["source_uri"],
                text=point.payload["text"],
                ordinal=point.payload["ordinal"],
                score=float(point.score),
            )
            for point in results
        ]

    def list_documents(
        self, workspace_id: str = DEFAULT_WORKSPACE, limit: int = 200
    ) -> list[DocumentSummary]:
        docs: dict[str, dict] = {}
        for payload in self._scroll_payloads(workspace_id):
            doc_id = payload["document_id"]
            entry = docs.get(doc_id)
            if entry is None:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "title": payload["title"],
                    "source_uri": payload["source_uri"],
                    "media_type": payload["media_type"],
                    "content_hash": payload["content_hash"],
                    "status": payload["status"],
                    "updated_at": payload["updated_at"],
                    "chunk_count": 1,
                }
            else:
                entry["chunk_count"] += 1
        ordered = sorted(docs.values(), key=lambda d: d["updated_at"], reverse=True)
        return [DocumentSummary(**d) for d in ordered[: max(1, min(int(limit), 1000))]]

    def _fusion_query(self):
        from qdrant_client import models

        if self.fusion == "dbsf":
            return models.FusionQuery(fusion=models.Fusion.DBSF)
        return models.RrfQuery(rrf=models.Rrf(weights=[1.0, self.lexical_weight]))

    # --- internals ----------------------------------------------------------

    def _find_existing(self, workspace_id: str, source_uri: str) -> dict | None:
        from qdrant_client import models

        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id", match=models.MatchValue(value=workspace_id)
                    ),
                    models.FieldCondition(
                        key="source_uri", match=models.MatchValue(value=source_uri)
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        payload = points[0].payload
        return {"document_id": payload["document_id"], "content_hash": payload["content_hash"]}

    def _count(self, document_id: str) -> int:
        from qdrant_client import models

        flt = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        )
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def _delete_by_document_id(self, document_id: str) -> None:
        from qdrant_client import models

        flt = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        )
        self.client.delete(collection_name=self.collection, points_selector=models.FilterSelector(filter=flt))

    def _scroll_payloads(self, workspace_id: str):
        from qdrant_client import models

        offset = None
        flt = models.Filter(
            must=[models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))]
        )
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                yield point.payload
            if offset is None:
                return


def _sparse_config():
    from qdrant_client import models

    return {LEXICAL: models.SparseVectorParams(modifier=models.Modifier.IDF)}


def migrate_to_hybrid(client: QdrantClient, source: str, target: str, batch: int = 128) -> int:
    """Copy a dense-only collection into a new one that also carries the keyword
    vector, so hybrid search works on documents indexed before it existed.

    Nothing is re-embedded and no original file is read: each point's dense
    vector and payload are copied as they are, and the keyword vector is built
    from the text already in its payload. That costs no API calls and works for
    documents whose source path this machine can no longer open. The source
    collection is left untouched as a backup. Returns the points copied."""
    from qdrant_client import models

    if client.collection_exists(target):
        raise ValueError(f"target collection {target!r} already exists; refusing to overwrite it")
    params = client.get_collection(source).config.params
    dense = params.vectors
    if not isinstance(dense, models.VectorParams):
        raise TypeError(f"{source!r} has named vectors; only a single dense vector is supported")
    client.create_collection(
        collection_name=target, vectors_config=dense, sparse_vectors_config=_sparse_config()
    )
    for field_name in ("workspace_id", "document_id", "source_uri"):
        client.create_payload_index(target, field_name, field_schema=models.PayloadSchemaType.KEYWORD)
    copied = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if points:
            client.upsert(
                collection_name=target,
                points=[
                    models.PointStruct(
                        id=point.id,
                        vector={
                            "": point.vector,
                            LEXICAL: _sparse(
                                lexical.document_vector(
                                    point.payload.get("text", ""), point.payload.get("title", "")
                                )
                            ),
                        },
                        payload=point.payload,
                    )
                    for point in points
                ],
            )
            copied += len(points)
        if offset is None:
            return copied


def _sparse(vector: tuple[list[int], list[float]]):
    from qdrant_client import models

    return models.SparseVector(indices=vector[0], values=vector[1])


def _point_id(document_id: str, content_hash: str, ordinal: int) -> str:
    """A stable UUID per chunk, same inputs `store.py`'s `_chunk_id` hashes.

    Re-ingesting identical content reproduces the same ids; harmless here
    because upsert_document deletes the old document's points first, so this
    only makes upserts idempotent against a retried call, not a correctness
    requirement on its own."""
    import hashlib

    digest = hashlib.sha256(f"{document_id}:{content_hash}:{ordinal}".encode()).hexdigest()[:32]
    return str(uuid.UUID(hex=digest))
