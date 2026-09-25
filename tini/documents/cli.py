"""Command-line entry points for indexing and inspecting local documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from tini.config import load_settings
from tini.db import connect
from tini.documents.embeddings import make_embedding_provider
from tini.documents.ocr import make_ocr_provider, make_usage_logger
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import make_document_index


def ingest_main(argv: list[str]) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(prog="tini ingest", description="Index local documents")
    parser.add_argument("path", type=Path, help="document or directory to ingest")
    parser.add_argument("--workspace", default=settings.document_workspace)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--chunk-max-tokens", type=int, default=settings.document_chunk_max_tokens)
    parser.add_argument("--chunk-min-tokens", type=int, default=settings.document_chunk_min_tokens)
    parser.add_argument(
        "--chunk-overlap-tokens", type=int, default=settings.document_chunk_overlap_tokens
    )
    args = parser.parse_args(argv)

    settings.ensure_home()
    conn = connect(settings.home)
    try:
        embedder = make_embedding_provider(settings)
        pipeline = IngestionPipeline(
            make_document_index(settings, conn, embedder),
            embedder=embedder,
            ocr=make_ocr_provider(settings),
            chunk_max_tokens=args.chunk_max_tokens,
            chunk_min_tokens=args.chunk_min_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            max_file_bytes=settings.document_max_file_bytes,
            max_text_chars=settings.document_max_text_chars,
            usage_log=make_usage_logger(settings.home),
        )
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
    index = make_document_index(settings, conn, make_embedding_provider(settings))
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
