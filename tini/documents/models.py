"""Values crossing the document-ingestion and retrieval seams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_WORKSPACE = "default"
IngestAction = Literal["created", "updated", "skipped", "failed"]


@dataclass(frozen=True)
class DocumentSource:
    workspace_id: str
    source_uri: str
    title: str
    media_type: str
    size_bytes: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    ordinal: int
    text: str
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = field(default=None, repr=False)
    embedding_provider: str | None = None
    embedding_model: str | None = None


@dataclass(frozen=True)
class IngestReport:
    source_uri: str
    title: str
    action: IngestAction
    chunk_count: int = 0
    document_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    chunk_id: str
    title: str
    source_uri: str
    text: str
    ordinal: int
    score: float


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    title: str
    source_uri: str
    media_type: str
    content_hash: str
    chunk_count: int
    status: str
    updated_at: str
