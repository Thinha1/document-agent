"""The document ingestion module: discover → load → chunk → index."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tini.documents.base import DocumentIndex
from tini.documents.chunking import chunk_text
from tini.documents.loaders import discover_documents, load_document
from tini.documents.models import DEFAULT_WORKSPACE, DocumentSource, IngestReport

IngestObserver = Callable[[str, dict[str, Any]], None]


class IngestionPipeline:
    def __init__(
        self,
        index: DocumentIndex,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        max_file_bytes: int = 25 * 1024 * 1024,
    ):
        self.index = index
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_bytes = max_file_bytes

    def ingest_path(
        self,
        path: Path,
        *,
        workspace_id: str = DEFAULT_WORKSPACE,
        recursive: bool = True,
        observer: IngestObserver | None = None,
    ) -> list[IngestReport]:
        notify = observer or (lambda stage, event: None)
        files = discover_documents(path, recursive=recursive)
        notify("discovered", {"path": str(path), "files": len(files)})
        reports: list[IngestReport] = []
        for source_path in files:
            try:
                notify("loading", {"path": str(source_path)})
                loaded = load_document(source_path, self.max_file_bytes)
                chunks = chunk_text(loaded.text, self.chunk_size, self.chunk_overlap)
                if not chunks:
                    raise ValueError("document produced no chunks")
                notify("indexing", {"path": str(source_path), "chunks": len(chunks)})
                report = self.index.upsert_document(
                    DocumentSource(
                        workspace_id=workspace_id,
                        source_uri=loaded.source_uri,
                        title=loaded.title,
                        media_type=loaded.media_type,
                        size_bytes=loaded.size_bytes,
                        content_hash=loaded.content_hash,
                        metadata=loaded.metadata,
                    ),
                    chunks,
                )
                notify(report.action, {"path": str(source_path), "chunks": len(chunks)})
            except Exception as exc:
                report = IngestReport(
                    source_uri=str(source_path),
                    title=source_path.stem,
                    action="failed",
                    error=str(exc),
                )
                notify("failed", {"path": str(source_path), "error": str(exc)})
            reports.append(report)
        return reports
