"""SQLite/FTS5 adapter for the document index seam."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import uuid
from collections.abc import Sequence

from tini.config import Settings
from tini.documents.base import DocumentIndex
from tini.documents.embeddings import EmbeddingProvider
from tini.documents.models import (
    DEFAULT_WORKSPACE,
    DocumentChunk,
    DocumentSource,
    DocumentSummary,
    IngestReport,
    SearchHit,
)
from tini.memory.semantic.store import _fts_query


def make_document_index(
    settings: Settings, conn: sqlite3.Connection, embedder: EmbeddingProvider | None
) -> DocumentIndex:
    """Pick the document index backend, the same seam `Memory._make_fact_store`
    uses for semantic memory: one place both `Tini.__init__` and `tini ingest`/
    `tini documents` go through, so a new backend is wired once, not per call site."""
    backend = settings.document_index
    if backend == "qdrant":
        from tini.documents.qdrant_store import QdrantDocumentIndex

        index: DocumentIndex = QdrantDocumentIndex(settings, embedder)
    elif backend == "sqlite":
        index = SqliteDocumentIndex(conn, embedder)
    else:
        raise ValueError(f"Unsupported document index backend: {backend}")
    from tini.documents.visibility import VisibleIndex

    # Innermost, so translation and reranking only ever see visible chunks.
    index = VisibleIndex(index, conn)
    if settings.query_languages:
        from tini.documents.multilingual import SHARED_CACHE, MultilingualIndex, make_translator

        index = MultilingualIndex(
            index, make_translator(settings), settings.query_languages, cache=SHARED_CACHE
        )
    from tini.documents.rerank import RerankingIndex, make_reranker

    reranker = make_reranker(settings)
    if reranker is not None:
        index = RerankingIndex(index, reranker, settings.rerank_candidates)
    return index


class SqliteDocumentIndex:
    def __init__(self, conn: sqlite3.Connection, embedder: EmbeddingProvider | None = None):
        self.conn = conn
        self.embedder = embedder

    def upsert_document(
        self, source: DocumentSource, chunks: Sequence[DocumentChunk]
    ) -> IngestReport:
        metadata = json.dumps(source.metadata, ensure_ascii=False, sort_keys=True)
        existing = self.conn.execute(
            """SELECT id, content_hash, metadata, status FROM documents
               WHERE workspace_id=? AND source_uri=?""",
            (source.workspace_id, source.source_uri),
        ).fetchone()
        if (
            existing
            and existing["content_hash"] == source.content_hash
            and existing["metadata"] == metadata
            and existing["status"] == "ready"
        ):
            count = self.conn.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id=?", (existing["id"],)
            ).fetchone()[0]
            needs_vectors = bool(chunks and chunks[0].embedding is not None)
            vector_count = 0
            if needs_vectors:
                vector_count = self.conn.execute(
                    """SELECT COUNT(*) FROM document_chunk_embeddings e
                       JOIN document_chunks c ON c.id=e.chunk_id
                       WHERE c.document_id=? AND e.provider=? AND e.model=?""",
                    (
                        existing["id"],
                        chunks[0].embedding_provider,
                        chunks[0].embedding_model,
                    ),
                ).fetchone()[0]
            if count and (not needs_vectors or vector_count == count):
                return IngestReport(
                    source_uri=source.source_uri,
                    title=source.title,
                    action="skipped",
                    chunk_count=count,
                    document_id=existing["id"],
                )

        document_id = existing["id"] if existing else uuid.uuid4().hex
        action = "updated" if existing else "created"
        with self.conn:
            if existing:
                self.conn.execute(
                    """UPDATE documents
                       SET title=?, media_type=?, size_bytes=?, content_hash=?, metadata=?,
                           status='indexing', updated_at=datetime('now')
                       WHERE id=?""",
                    (
                        source.title,
                        source.media_type,
                        source.size_bytes,
                        source.content_hash,
                        metadata,
                        document_id,
                    ),
                )
                self.conn.execute("DELETE FROM document_chunks WHERE document_id=?", (document_id,))
            else:
                self.conn.execute(
                    """INSERT INTO documents
                       (id, workspace_id, source_uri, title, media_type, size_bytes,
                        content_hash, metadata, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'indexing')""",
                    (
                        document_id,
                        source.workspace_id,
                        source.source_uri,
                        source.title,
                        source.media_type,
                        source.size_bytes,
                        source.content_hash,
                        metadata,
                    ),
                )
            chunk_rows = [
                (
                    _chunk_id(document_id, source.content_hash, chunk.ordinal),
                    document_id,
                    source.workspace_id,
                    chunk.ordinal,
                    source.title,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                )
                for chunk in chunks
            ]
            self.conn.executemany(
                """INSERT INTO document_chunks
                   (id, document_id, workspace_id, ordinal, title, text,
                    char_start, char_end, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                chunk_rows,
            )
            embedding_rows = []
            for row, chunk in zip(chunk_rows, chunks, strict=True):
                if chunk.embedding is None:
                    continue
                embedding_rows.append(
                    (
                        row[0],
                        source.workspace_id,
                        chunk.embedding_provider,
                        chunk.embedding_model,
                        len(chunk.embedding),
                        _pack_vector(chunk.embedding),
                    )
                )
            self.conn.executemany(
                """INSERT INTO document_chunk_embeddings
                   (chunk_id, workspace_id, provider, model, dimensions, vector)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                embedding_rows,
            )
            self.conn.execute(
                "UPDATE documents SET status='ready', updated_at=datetime('now') WHERE id=?",
                (document_id,),
            )
        return IngestReport(
            source_uri=source.source_uri,
            title=source.title,
            action=action,
            chunk_count=len(chunks),
            document_id=document_id,
        )

    def search(
        self, query: str, top_k: int = 6, workspace_id: str = DEFAULT_WORKSPACE
    ) -> list[SearchHit]:
        limit = max(1, min(int(top_k), 50))
        lexical = self._lexical_search(query, limit * 4, workspace_id)
        vector = self._vector_search(query, workspace_id) if self.embedder is not None else []
        if not vector:
            return lexical[:limit]
        return _reciprocal_rank_fusion(lexical, vector, limit)

    def _lexical_search(self, query: str, limit: int, workspace_id: str) -> list[SearchHit]:
        fts = _fts_query(query)
        if not fts:
            return []
        rows = self.conn.execute(
            """SELECT c.id AS chunk_id, c.document_id, c.title, d.source_uri,
                      c.text, c.ordinal, bm25(document_chunks_fts) AS rank
               FROM document_chunks_fts
               JOIN document_chunks c ON c.rowid = document_chunks_fts.rowid
               JOIN documents d ON d.id = c.document_id
               WHERE document_chunks_fts MATCH ?
                 AND c.workspace_id=? AND d.status='ready'
               ORDER BY rank LIMIT ?""",
            (fts, workspace_id, limit),
        ).fetchall()
        return [
            SearchHit(
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                title=row["title"],
                source_uri=row["source_uri"],
                text=row["text"],
                ordinal=row["ordinal"],
                score=-float(row["rank"]),
            )
            for row in rows
        ]

    def _vector_search(self, query: str, workspace_id: str) -> list[SearchHit]:
        if self.embedder is None or not query.strip():
            return []
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception:
            # Retrieval remains available through FTS if a remote provider is down.
            return []
        rows = self.conn.execute(
            """SELECT c.id AS chunk_id, c.document_id, c.title, d.source_uri,
                      c.text, c.ordinal, e.vector
               FROM document_chunk_embeddings e
               JOIN document_chunks c ON c.id=e.chunk_id
               JOIN documents d ON d.id=c.document_id
               WHERE e.workspace_id=? AND e.provider=? AND e.model=?
                 AND e.dimensions=? AND d.status='ready'""",
            (
                workspace_id,
                self.embedder.provider,
                self.embedder.model,
                self.embedder.dimensions,
            ),
        ).fetchall()
        scored = [
            (
                _cosine_similarity(query_vector, _unpack_vector(row["vector"])),
                SearchHit(
                    document_id=row["document_id"],
                    chunk_id=row["chunk_id"],
                    title=row["title"],
                    source_uri=row["source_uri"],
                    text=row["text"],
                    ordinal=row["ordinal"],
                    score=0.0,
                ),
            )
            for row in rows
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [SearchHit(**{**hit.__dict__, "score": score}) for score, hit in scored[:200]]

    def list_documents(
        self, workspace_id: str = DEFAULT_WORKSPACE, limit: int = 200
    ) -> list[DocumentSummary]:
        rows = self.conn.execute(
            """SELECT d.id, d.title, d.source_uri, d.media_type, d.content_hash,
                      d.status, d.updated_at, COUNT(c.id) AS chunk_count
               FROM documents d
               LEFT JOIN document_chunks c ON c.document_id=d.id
               WHERE d.workspace_id=?
               GROUP BY d.id
               ORDER BY d.updated_at DESC, d.title
               LIMIT ?""",
            (workspace_id, max(1, min(int(limit), 1000))),
        ).fetchall()
        return [
            DocumentSummary(
                document_id=row["id"],
                title=row["title"],
                source_uri=row["source_uri"],
                media_type=row["media_type"],
                content_hash=row["content_hash"],
                chunk_count=row["chunk_count"],
                status=row["status"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def opening_text(
        self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE, chars: int = 3000
    ) -> str:
        rows = self.conn.execute(
            """SELECT text FROM document_chunks WHERE document_id=? AND workspace_id=?
               ORDER BY ordinal LIMIT 4""",
            (document_id, workspace_id),
        ).fetchall()
        return "\n".join(row["text"] for row in rows)[:chars]

    def delete_document(self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE) -> bool:
        with self.conn:
            row = self.conn.execute(
                "SELECT id FROM documents WHERE id=? AND workspace_id=?",
                (document_id, workspace_id),
            ).fetchone()
            if row is None:
                return False
            self.conn.execute("DELETE FROM document_chunks WHERE document_id=?", (document_id,))
            self.conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        return True


def _chunk_id(document_id: str, content_hash: str, ordinal: int) -> str:
    value = f"{document_id}:{content_hash}:{ordinal}".encode()
    return hashlib.sha256(value).hexdigest()[:32]


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(value: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(value) // 4}f", value)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / (left_norm * right_norm)


def _reciprocal_rank_fusion(
    lexical: Sequence[SearchHit], vector: Sequence[SearchHit], limit: int
) -> list[SearchHit]:
    hits: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for ranking in (lexical, vector):
        for rank, hit in enumerate(ranking, start=1):
            hits[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:limit]
    return [SearchHit(**{**hits[key].__dict__, "score": scores[key]}) for key in ordered]
