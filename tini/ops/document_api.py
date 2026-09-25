"""Dashboard-facing document operations, kept out of the HTTP router."""

from __future__ import annotations

import mimetypes
import os
import re
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from tini.config import Settings, load_settings
from tini.db import connect
from tini.documents import catalog
from tini.documents.blobstore import S3_SCHEME, BlobStoreError, make_blob_store
from tini.documents.embeddings import make_embedding_provider
from tini.documents.loaders import SUPPORTED_SUFFIXES
from tini.documents.ocr import make_ocr_provider, make_usage_logger
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import make_document_index
from tini.ops import ocr_cost

_WORKSPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def documents_payload(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    settings.ensure_home()
    conn = connect(settings.home)
    try:
        index = make_document_index(settings, conn, make_embedding_provider(settings))
        docs = index.list_documents(settings.document_workspace)
        entries = catalog.list_entries(conn, settings.document_workspace)
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
        "storage": settings.document_storage,
        "ocr_cost": ocr_cost.summarize(settings.home, settings.document_workspace),
        "catalog": entries,
        "catalog_options": {"types": catalog.DOC_TYPES, "statuses": catalog.STATUSES},
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

    try:
        store = make_blob_store(settings)
    except ValueError as exc:
        return {"error": str(exc)}
    source_uri = None
    staging = None
    try:
        if store is None:
            target = _stage(raw, settings.home / "documents" / workspace_id, relative_name)
            emit("uploaded", {"path": str(target), "bytes": len(raw)})
        else:
            # The original goes to object storage; the pipeline still needs a
            # local file to parse, so a throwaway copy is staged and removed.
            content_type = mimetypes.guess_type(display_name)[0] or "application/octet-stream"
            try:
                source_uri = store.put(workspace_id, display_name, raw, content_type)
            except BlobStoreError as exc:
                return {"error": str(exc)}
            staging = tempfile.TemporaryDirectory()
            target = _stage(raw, Path(staging.name), relative_name)
            emit("uploaded", {"path": source_uri, "bytes": len(raw)})
        return _ingest_staged(target, workspace_id, settings, emit, source_uri)
    finally:
        if staging is not None:
            staging.cleanup()


def _stage(raw: bytes, base: Path, relative_name: PurePosixPath) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    target = base.joinpath(*relative_name.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if base.resolve() not in target.resolve().parents:
        raise ValueError("document path escapes its workspace")
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.upload"
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _ingest_staged(target: Path, workspace_id: str, settings: Settings, emit, source_uri) -> dict:
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
            usage_log=make_usage_logger(settings.home),
            catalog_log=catalog.make_recorder(settings.home),
        )
        reports = pipeline.ingest_path(
            target,
            workspace_id=workspace_id,
            observer=lambda stage, event: emit(stage, event),
            source_uri=source_uri,
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


def _find_document(index, workspace: str, document_id: str):
    return next(
        (d for d in index.list_documents(workspace, limit=1000) if d.document_id == document_id),
        None,
    )


def _reindex(index, embedder, settings: Settings, workspace: str, match) -> dict:
    """Re-read a document's original and re-ingest it in place. Cheap by design:
    OCR is cached per page image and chunk vectors per chunk text, so an
    unchanged (or lightly hand-edited) file re-embeds and re-OCRs only what is
    actually new — and catalog edits made by a person are never overwritten."""
    pipeline = IngestionPipeline(
        index,
        embedder=embedder,
        ocr=make_ocr_provider(settings),
        chunk_max_tokens=settings.document_chunk_max_tokens,
        chunk_min_tokens=settings.document_chunk_min_tokens,
        chunk_overlap_tokens=settings.document_chunk_overlap_tokens,
        max_file_bytes=settings.document_max_file_bytes,
        max_text_chars=settings.document_max_text_chars,
        usage_log=make_usage_logger(settings.home),
        catalog_log=catalog.make_recorder(settings.home),
    )
    if urlparse(match.source_uri).scheme == S3_SCHEME:
        # The original lives in object storage: fetch it to a throwaway file to
        # parse, but keep the s3:// URI so this updates the SAME document
        # instead of creating a file:// twin.
        try:
            store = make_blob_store(settings)
            if store is None:
                return {"error": "This document is stored in MinIO but TINI_DOCUMENT_STORAGE is not 'minio'."}
            data = store.get(match.source_uri)
        except (BlobStoreError, ValueError) as exc:
            return {"error": str(exc)}
        name = PurePosixPath(unquote(urlparse(match.source_uri).path)).name
        with tempfile.TemporaryDirectory() as staging:
            target = _stage(data, Path(staging), PurePosixPath(name))
            reports = pipeline.ingest_path(target, workspace_id=workspace, source_uri=match.source_uri)
    else:
        reports = pipeline.ingest_path(_file_uri_to_path(match.source_uri), workspace_id=workspace)
    return {"reports": [asdict(report) for report in reports]}


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
        match = _find_document(index, workspace, document_id)
        if payload.get("action") == "delete":
            deleted = index.delete_document(document_id, workspace)
            if deleted and match is not None:
                catalog.delete(conn, workspace, match.source_uri)
            return {"ok": deleted}
        if payload.get("action") != "reindex":
            return {"error": "Unknown document action."}
        if match is None:
            return {"error": "Document not found in this workspace."}
        return _reindex(index, embedder, settings, workspace, match)
    finally:
        conn.close()


def catalog_update(payload: dict, settings: Settings | None = None) -> dict:
    """A person's edit of a document's catalog entry. Metadata only: nothing is
    re-embedded or re-OCR'd, and the edited fields are locked against later
    automatic re-classification."""
    settings = settings or load_settings()
    workspace = _valid_workspace(payload.get("workspace") or settings.document_workspace)
    document_id = str(payload.get("document_id") or "").strip()
    fields = payload.get("fields")
    if not document_id or not isinstance(fields, dict):
        return {"error": "document_id and fields are required."}
    settings.ensure_home()
    conn = connect(settings.home)
    try:
        index = make_document_index(settings, conn, make_embedding_provider(settings))
        match = _find_document(index, workspace, document_id)
        if match is None:
            return {"error": "Document not found in this workspace."}
        try:
            return {"entry": catalog.update(conn, workspace, match.source_uri, fields)}
        except catalog.CatalogError as exc:
            return {"error": str(exc)}
    finally:
        conn.close()


def catalog_refresh(payload: dict, settings: Settings | None = None) -> dict:
    """Give every indexed document a catalog entry, classified from the text the
    index ALREADY holds — no original file is re-read, nothing is OCR'd or
    embedded, so it costs nothing and works for documents whose original is not
    reachable from here (e.g. a path on another machine, which Re-index cannot
    open). Documents whose content is unchanged since their last entry keep it,
    including any edits made by a person."""
    settings = settings or load_settings()
    workspace = _valid_workspace(payload.get("workspace") or settings.document_workspace)
    settings.ensure_home()
    conn = connect(settings.home)
    try:
        index = make_document_index(settings, conn, make_embedding_provider(settings))
        known = catalog.list_entries(conn, workspace)
        summary = {"documents": 0, "catalogued": 0, "updated": 0, "unchanged": 0}
        for match in index.list_documents(workspace, limit=1000):
            summary["documents"] += 1
            before = known.get(match.source_uri)
            opening = index.opening_text(match.document_id, workspace)
            profile = catalog.profile_document(opening, {}, match.media_type, 0)
            # Estimated from the first chunks only: page count and scan state
            # are unknown here, and a full ingest replaces this profile.
            profile["from_index"] = True
            after = catalog.record_ingest(
                conn,
                {
                    "workspace": workspace,
                    "source_uri": match.source_uri,
                    "title": match.title,
                    "content_hash": match.content_hash,
                    "text": opening,
                    "profile": profile,
                },
            )
            if before is None:
                summary["catalogued"] += 1
            elif after["updated_at"] != before["updated_at"]:
                summary["updated"] += 1
            else:
                summary["unchanged"] += 1
        return summary
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
