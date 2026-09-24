"""Dashboard-facing document operations, kept out of the HTTP router."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from tini.config import Settings, load_settings
from tini.db import connect
from tini.documents.embeddings import make_embedding_provider
from tini.documents.loaders import SUPPORTED_SUFFIXES
from tini.documents.ocr import make_ocr_provider
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import make_document_index

_WORKSPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def documents_payload(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    settings.ensure_home()
    conn = connect(settings.home)
    try:
        index = make_document_index(settings, conn, make_embedding_provider(settings))
        docs = index.list_documents(settings.document_workspace)
        items = [asdict(item) for item in docs]
        if settings.document_index == "sqlite":
            chunks = conn.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE workspace_id=?",
                (settings.document_workspace,),
            ).fetchone()[0]
            vectors = conn.execute(
                "SELECT COUNT(*) FROM document_chunk_embeddings WHERE workspace_id=?",
                (settings.document_workspace,),
            ).fetchone()[0]
        else:
            # Every other backend is pure-vector (see qdrant_store.py's module
            # docstring): a chunk that exists at all is, by construction, embedded.
            # list_documents' own limit caps this the same way the library view
            # is already capped, so it is an undercount past that many documents.
            chunks = sum(item.chunk_count for item in docs)
            vectors = chunks
    finally:
        conn.close()
    return {
        "workspace": settings.document_workspace,
        "items": items,
        "chunk_count": chunks,
        "vector_count": vectors,
        "supported_extensions": sorted(SUPPORTED_SUFFIXES),
        "max_file_bytes": settings.document_max_file_bytes,
        "chunk_max_tokens": settings.document_chunk_max_tokens,
        "chunk_min_tokens": settings.document_chunk_min_tokens,
        "chunk_overlap_tokens": settings.document_chunk_overlap_tokens,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
    }


def ingest_upload(
    filename: str,
    workspace_id: str,
    raw: bytes,
    emit,
    settings: Settings | None = None,
) -> dict:
    settings = settings or load_settings()
    settings.ensure_home()
    workspace_id = _valid_workspace(workspace_id or settings.document_workspace)
    relative_name = _valid_source_name(filename)
    display_name = relative_name.as_posix()
    if not raw:
        return {"error": "The selected file is empty."}
    if len(raw) > settings.document_max_file_bytes:
        return {
            "error": (
                f"{display_name} is {len(raw)} bytes; the limit is "
                f"{settings.document_max_file_bytes} bytes."
            )
        }

    upload_dir = settings.home / "documents" / workspace_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir.joinpath(*relative_name.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if upload_dir.resolve() not in target.resolve().parents:
        raise ValueError("document path escapes its workspace")
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.upload"
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    emit("uploaded", {"path": str(target), "bytes": len(raw)})

    conn = connect(settings.home)
    try:
        embedder = make_embedding_provider(settings)
        pipeline = IngestionPipeline(
            make_document_index(settings, conn, embedder),
            embedder=embedder,
            ocr=make_ocr_provider(settings),
            chunk_max_tokens=settings.document_chunk_max_tokens,
            chunk_min_tokens=settings.document_chunk_min_tokens,
            chunk_overlap_tokens=settings.document_chunk_overlap_tokens,
            max_file_bytes=settings.document_max_file_bytes,
            max_text_chars=settings.document_max_text_chars,
        )
        reports = pipeline.ingest_path(
            target,
            workspace_id=workspace_id,
            observer=lambda stage, event: emit(stage, event),
        )
    finally:
        conn.close()
    return {"reports": [asdict(report) for report in reports]}


def search(payload: dict, settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    workspace = _valid_workspace(payload.get("workspace") or settings.document_workspace)
    query = str(payload.get("query") or "").strip()
    if not query:
        return {"error": "Enter a retrieval query."}
    top_k = max(1, min(int(payload.get("top_k") or 6), 20))
    settings.ensure_home()
    conn = connect(settings.home)
    try:
        index = make_document_index(settings, conn, make_embedding_provider(settings))
        hits = [asdict(hit) for hit in index.search(query, top_k, workspace)]
    finally:
        conn.close()
    return {"hits": hits, "query": query}


def chunks(
    document_id: str,
    workspace_id: str,
    limit: int = 50,
    settings: Settings | None = None,
) -> dict:
    settings = settings or load_settings()
    if settings.document_index != "sqlite":
        # The inspector reads document_chunks/document_chunk_embeddings
        # directly for per-chunk char offsets and an embedded flag — fields
        # the DocumentIndex protocol doesn't expose generically. Rather than
        # run this query against tables a non-sqlite backend never writes
        # (which would misreport "document not found"), say so plainly.
        return {"error": f"Chunk inspection isn't available on the {settings.document_index} backend."}
    workspace_id = _valid_workspace(workspace_id or settings.document_workspace)
    settings.ensure_home()
    conn = connect(settings.home)
    try:
        document = conn.execute(
            "SELECT id, title, source_uri FROM documents WHERE id=? AND workspace_id=?",
            (document_id, workspace_id),
        ).fetchone()
        if document is None:
            return {"error": "Document not found in this workspace."}
        rows = conn.execute(
            """SELECT c.id, c.ordinal, c.text, c.char_start, c.char_end,
                      CASE WHEN e.chunk_id IS NULL THEN 0 ELSE 1 END AS embedded
               FROM document_chunks c
               LEFT JOIN document_chunk_embeddings e ON e.chunk_id=c.id
               WHERE c.document_id=? AND c.workspace_id=?
               ORDER BY ordinal LIMIT ?""",
            (document_id, workspace_id, max(1, min(int(limit), 200))),
        ).fetchall()
        return {"document": dict(document), "chunks": [dict(row) for row in rows]}
    finally:
        conn.close()


def action(payload: dict, settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    workspace = _valid_workspace(payload.get("workspace") or settings.document_workspace)
    document_id = str(payload.get("document_id") or "").strip()
    if not document_id:
        return {"error": "document_id is required."}
    settings.ensure_home()
    conn = connect(settings.home)
    embedder = make_embedding_provider(settings)
    index = make_document_index(settings, conn, embedder)
    try:
        if payload.get("action") == "delete":
            return {"ok": index.delete_document(document_id, workspace)}
        if payload.get("action") != "reindex":
            return {"error": "Unknown document action."}
        match = next(
            (d for d in index.list_documents(workspace, limit=1000) if d.document_id == document_id),
            None,
        )
        if match is None:
            return {"error": "Document not found in this workspace."}
        source = _file_uri_to_path(match.source_uri)
        reports = IngestionPipeline(
            index,
            embedder=embedder,
            ocr=make_ocr_provider(settings),
            chunk_max_tokens=settings.document_chunk_max_tokens,
            chunk_min_tokens=settings.document_chunk_min_tokens,
            chunk_overlap_tokens=settings.document_chunk_overlap_tokens,
            max_file_bytes=settings.document_max_file_bytes,
            max_text_chars=settings.document_max_text_chars,
        ).ingest_path(source, workspace_id=workspace)
        return {"reports": [asdict(report) for report in reports]}
    finally:
        conn.close()


def _valid_workspace(value: str) -> str:
    value = str(value).strip()
    if not _WORKSPACE.fullmatch(value):
        raise ValueError("workspace must use 1-64 letters, numbers, dots, dashes, or underscores")
    return value


def _valid_source_name(value: str) -> PurePosixPath:
    value = unquote(str(value)).strip().replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("document name must be a safe relative path")
    if len(value) > 240 or any(len(part) > 180 for part in path.parts):
        raise ValueError("document name is too long")
    if any(any(ord(char) < 32 for char in part) for part in path.parts):
        raise ValueError("document name contains control characters")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported document type: {path.suffix or '(none)'}")
    return path


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("Only local file sources can be re-indexed from the dashboard.")
    path = url2pathname(unquote(parsed.path))
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    return Path(path)
