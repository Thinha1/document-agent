"""Command-line entry points for indexing and inspecting local documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from tini.config import load_settings
from tini.db import connect
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import SqliteDocumentIndex


def ingest_main(argv: list[str]) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(prog="tini ingest", description="Index local documents")
    parser.add_argument("path", type=Path, help="document or directory to ingest")
    parser.add_argument("--workspace", default=settings.document_workspace)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=settings.document_chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.document_chunk_overlap)
    args = parser.parse_args(argv)

    settings.ensure_home()
    conn = connect(settings.home)
    pipeline = IngestionPipeline(
        SqliteDocumentIndex(conn),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_file_bytes=settings.document_max_file_bytes,
    )
    try:
        reports = pipeline.ingest_path(
            args.path, workspace_id=args.workspace, recursive=not args.no_recursive
        )
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return 1
    finally:
        conn.close()

    if not reports:
        print("No supported documents found.")
        return 1
    for report in reports:
        detail = report.error or f"{report.chunk_count} chunks"
        print(f"{report.action.upper():7} {report.title} — {detail}")
    return 1 if any(report.action == "failed" for report in reports) else 0


def documents_main(argv: list[str]) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(prog="tini documents", description="Inspect indexed documents")
    parser.add_argument("--workspace", default=settings.document_workspace)
    parser.add_argument("--search", default="", help="search indexed chunks")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    settings.ensure_home()
    conn = connect(settings.home)
    index = SqliteDocumentIndex(conn)
    try:
        if args.search:
            for hit in index.search(args.search, args.limit, args.workspace):
                print(f"[{hit.title}#chunk-{hit.ordinal}] {hit.text}\n{hit.source_uri}\n")
        else:
            for document in index.list_documents(args.workspace, args.limit):
                print(
                    f"{document.document_id}  {document.title}  "
                    f"{document.chunk_count} chunks  {document.status}"
                )
    finally:
        conn.close()
    return 0
