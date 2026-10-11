"""The storage seam for document ingestion and retrieval.

An index owns backend-specific work: SQLite combines FTS5 with local cosine
search today; a pgvector adapter can replace it later. Callers only learn this
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

    def opening_text(
        self, document_id: str, workspace_id: str = DEFAULT_WORKSPACE, chars: int = 3000
    ) -> str:
        """The first ``chars`` characters of a document as indexed (its first
        few chunks, in order). Lets the catalog classify a document from what
        the index already holds, without re-reading the original file."""
        ...
