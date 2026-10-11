"""The document ingestion module: discover → load → chunk → index."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from tini.documents.base import DocumentIndex
from tini.documents.catalog import profile_document
from tini.documents.chunking import chunk_text
from tini.documents.embeddings import EmbeddingProvider
from tini.documents.loaders import discover_documents, load_document
from tini.documents.models import DEFAULT_WORKSPACE, DocumentSource, IngestReport
from tini.documents.ocr import OcrProvider
from tini.documents.pages import assign_pages

IngestObserver = Callable[[str, dict[str, Any]], None]


class IngestionPipeline:
    def __init__(
        self,
        index: DocumentIndex,
        *,
        embedder: EmbeddingProvider | None = None,
        ocr: OcrProvider | None = None,
        chunk_max_tokens: int = 512,
        chunk_min_tokens: int = 120,
        chunk_overlap_tokens: int = 48,
        table_chunking: bool = False,
        max_file_bytes: int = 25 * 1024 * 1024,
        max_text_chars: int = 5_000_000,
        usage_log: Callable[[dict[str, Any]], None] | None = None,
        catalog_log: Callable[[dict[str, Any]], None] | None = None,
    ):
        if chunk_max_tokens <= 0:
            raise ValueError("chunk_max_tokens must be positive")
        if chunk_min_tokens < 0 or chunk_min_tokens > chunk_max_tokens:
            raise ValueError("chunk_min_tokens must be between zero and chunk_max_tokens")
        if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_max_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be non-negative and smaller than chunk_max_tokens"
            )
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self.index = index
        self.embedder = embedder or getattr(index, "embedder", None)
        self.ocr = ocr
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_min_tokens = chunk_min_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.table_chunking = table_chunking
        self.max_file_bytes = max_file_bytes
        self.max_text_chars = max_text_chars
        self.usage_log = usage_log
        self.catalog_log = catalog_log

    def ingest_path(
        self,
        path: Path,
        *,
        workspace_id: str = DEFAULT_WORKSPACE,
        recursive: bool = True,
        observer: IngestObserver | None = None,
        source_uri: str | None = None,
    ) -> list[IngestReport]:
        """`source_uri` names where the ORIGINAL really lives when `path` is
        only a local staging copy (e.g. an s3:// object downloaded to parse);
        it replaces the file:// URI a document would otherwise be recorded
        under, so a re-ingest of the same object matches the same document."""
        if not workspace_id.strip():
            raise ValueError("workspace_id must not be empty")
        notify = observer or (lambda stage, event: None)
        files = discover_documents(path, recursive=recursive)
        if source_uri is not None and len(files) != 1:
            raise ValueError("source_uri can only be set when ingesting a single file")
        notify("discovered", {"path": str(path), "files": len(files)})
        reports: list[IngestReport] = []
        for source_path in files:
            try:
                notify("loading", {"path": str(source_path)})
                loaded = load_document(
                    source_path,
                    self.max_file_bytes,
                    max_text_chars=self.max_text_chars,
                    ocr=self.ocr,
                )
                notify("chunking", {"path": str(source_path), "characters": len(loaded.text)})
                chunks = chunk_text(
                    loaded.text,
                    self.chunk_max_tokens,
                    self.chunk_min_tokens,
                    self.chunk_overlap_tokens,
                    plain_tables=self.table_chunking,
                )
                if not chunks:
                    raise ValueError("document produced no chunks")
                chunks = assign_pages(chunks, loaded.pages)
                notify(
                    "embedding",
                    {
                        "path": str(source_path),
                        "chunks": len(chunks),
                        "enabled": self.embedder is not None,
                    },
                )
                if self.embedder is not None:
                    vectors = self.embedder.embed_documents(
                        [chunk.text for chunk in chunks], title=loaded.title
                    )
                    chunks = [
                        replace(
                            chunk,
                            embedding=vector,
                            embedding_provider=self.embedder.provider,
                            embedding_model=self.embedder.model,
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ]
                notify("indexing", {"path": str(source_path), "chunks": len(chunks)})
                report = self.index.upsert_document(
                    DocumentSource(
                        workspace_id=workspace_id,
                        source_uri=source_uri or loaded.source_uri,
                        title=loaded.title,
                        media_type=loaded.media_type,
                        size_bytes=loaded.size_bytes,
                        content_hash=loaded.content_hash,
                        metadata={
                            # ocr_usage is a per-run measurement (token counts
                            # vary between runs), not a property of the file:
                            # stored, it would make an unchanged re-ingest look
                            # changed and defeat the skip-if-identical check.
                            **{k: v for k, v in loaded.metadata.items() if k != "ocr_usage"},
                            "_ingestion": {
                                "chunker": "structure-v1",
                                "max_tokens": self.chunk_max_tokens,
                                "min_tokens": self.chunk_min_tokens,
                                "overlap_tokens": self.chunk_overlap_tokens,
                                "embedding_provider": (
                                    self.embedder.provider if self.embedder else "none"
                                ),
                                "embedding_model": self.embedder.model if self.embedder else "",
                            },
                        },
                    ),
                    chunks,
                )
                if self.usage_log and "ocr_usage" in loaded.metadata:
                    # After the upsert so the row can say whether the OCR was
                    # kept ("created"/"updated") or thrown away ("skipped": the
                    # file was unchanged but was OCR'd again before the index
                    # noticed).
                    self.usage_log(
                        {
                            "workspace": workspace_id,
                            "title": loaded.title,
                            "source_uri": source_uri or loaded.source_uri,
                            "action": report.action,
                            **loaded.metadata["ocr_usage"],
                        }
                    )
                if self.catalog_log:
                    # Profile + rule classification happen here, from text the
                    # pipeline already has: no extra model call, no extra
                    # embedding, so cataloguing a document is free. Unchanged
                    # content is left alone by catalog.record_ingest.
                    self.catalog_log(
                        {
                            "workspace": workspace_id,
                            "source_uri": source_uri or loaded.source_uri,
                            "title": loaded.title,
                            "content_hash": loaded.content_hash,
                            "text": loaded.text[:3000],
                            "profile": profile_document(
                                loaded.text, loaded.metadata, loaded.media_type, loaded.size_bytes
                            ),
                        }
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
