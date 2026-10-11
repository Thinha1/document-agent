"""Keep documents the catalog marks superseded/excluded out of search results.

The catalog (tini/documents/catalog.py) is where a person says "this is the
old edition" or "not evidence". Without this wrapper that mark was only a label:
the chunks stayed in the index and answered like any other. `VisibleIndex`
wraps any `DocumentIndex` (same shape as `RerankingIndex`) and drops hits from
hidden sources at read time, so:

* nothing is re-embedded or deleted — flipping the status back restores the
  document at once, and it works the same under SQLite and Qdrant;
* the catalog stays the single source of truth (no copy in the vector payload to
  drift out of date when someone edits it);
* with nothing hidden it is a pass-through: one small SQLite query per search.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from tini.documents import catalog
from tini.documents.models import (
    DEFAULT_WORKSPACE,
    DocumentChunk,
    DocumentSource,
    DocumentSummary,
    IngestReport,
    SearchHit,
)

_MAX_POOL = 50  # the backends' own ceiling on top_k


class VisibleIndex:
    def __init__(self, inner, conn: sqlite3.Connection):
        self.inner = inner
        self.conn = conn

    def upsert_document(
        self, source: DocumentSource, chunks: Sequence[DocumentChunk]
    ) -> IngestReport:
        return self.inner.upsert_document(source, chunks)

    def list_documents(
        self, workspace_id: str = DEFAULT_WORKSPACE, limit: int = 200
    ) -> list[DocumentSummary]:
        return self.inner.list_documents(workspace_id, limit)

    def delete_document(self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE) -> bool:
        return self.inner.delete_document(document_id, workspace_id)

    def opening_text(
        self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE, chars: int = 3000
    ) -> str:
        return self.inner.opening_text(document_id, workspace_id, chars)

    def search(
        self, query: str, top_k: int = 6, workspace_id: str = DEFAULT_WORKSPACE
    ) -> list[SearchHit]:
        limit = max(1, min(int(top_k), _MAX_POOL))
        hidden = catalog.hidden_uris(self.conn, workspace_id)
        if not hidden:
            return self.inner.search(query, top_k=limit, workspace_id=workspace_id)
        # Hidden chunks take slots in the ranking, so look wider first and, if the
        # page is still short, once more at the ceiling.
        asked = min(_MAX_POOL, limit * 3)
        while True:
            raw = self.inner.search(query, top_k=asked, workspace_id=workspace_id)
            kept = [hit for hit in raw if hit.source_uri not in hidden]
            if len(kept) >= limit or len(raw) < asked or asked >= _MAX_POOL:
                return kept[:limit]
            asked = _MAX_POOL

    def __getattr__(self, name: str):
        # embedder, collection, conn... — whatever the wrapped index exposes.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)
