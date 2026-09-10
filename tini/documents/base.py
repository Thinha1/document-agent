"""The storage seam for document ingestion and retrieval.

An index owns the backend-specific work: SQLite uses FTS5 today; a pgvector
adapter can embed and index the same chunks later. Callers only learn this
small interface and never handle backend query syntax or vector dimensions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tini.documents.models import (
    DEFAULT_WORKSPACE,
    DocumentChunk,
    DocumentSource,
    DocumentSummary,
    IngestReport,
    SearchHit,
)


@runtime_checkable
class DocumentIndex(Protocol):
    def upsert_document(
        self, source: DocumentSource, chunks: Sequence[DocumentChunk]
    ) -> IngestReport:
        """Atomically create or replace a source; skip an unchanged hash."""
        ...

    def search(
        self, query: str, top_k: int = 6, workspace_id: str = DEFAULT_WORKSPACE
    ) -> list[SearchHit]:
        """Return only chunks visible inside ``workspace_id``, best first."""
        ...

    def list_documents(
        self, workspace_id: str = DEFAULT_WORKSPACE, limit: int = 200
    ) -> list[DocumentSummary]:
        """List indexed documents, newest first."""
        ...

    def delete_document(self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE) -> bool:
        """Delete a document and all of its chunks within one workspace."""
        ...
