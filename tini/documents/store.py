"""SQLite/FTS5 adapter for the document index seam."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Sequence

from tini.documents.models import (
    DEFAULT_WORKSPACE,
    DocumentChunk,
    DocumentSource,
    DocumentSummary,
    IngestReport,
    SearchHit,
)
from tini.memory.semantic.store import _fts_query


class SqliteDocumentIndex:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_document(
        self, source: DocumentSource, chunks: Sequence[DocumentChunk]
    ) -> IngestReport:
        existing = self.conn.execute(
            "SELECT id, content_hash FROM documents WHERE workspace_id=? AND source_uri=?",
            (source.workspace_id, source.source_uri),
        ).fetchone()
        if existing and existing["content_hash"] == source.content_hash:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id=?", (existing["id"],)
            ).fetchone()[0]
            return IngestReport(
                source_uri=source.source_uri,
                title=source.title,
                action="skipped",
                chunk_count=count,
                document_id=existing["id"],
            )

        document_id = existing["id"] if existing else uuid.uuid4().hex
        action = "updated" if existing else "created"
        metadata = json.dumps(source.metadata, ensure_ascii=False, sort_keys=True)
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
            self.conn.executemany(
                """INSERT INTO document_chunks
                   (id, document_id, workspace_id, ordinal, title, text,
                    char_start, char_end, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
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
                ],
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
        fts = _fts_query(query)
        if not fts:
            return []
        limit = max(1, min(int(top_k), 50))
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
