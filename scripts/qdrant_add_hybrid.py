"""Copy a dense-only Qdrant collection into one that also supports keyword search.

    python scripts/qdrant_add_hybrid.py tini_documents_bge3 tini_documents_bge3_hybrid

Hybrid retrieval needs a keyword (sparse) vector on every chunk, and Qdrant
cannot add a vector to a collection that already exists. This creates the target
collection, copies each point's dense vector and payload across, and builds the
keyword vector from the text already stored in the payload. Nothing is
re-embedded and no original file is read, so it costs no API calls and works for
documents whose source path this machine can no longer open.

The source collection is not modified: it stays as a backup. Afterwards set
TINI_QDRANT_COLLECTION to the target name (in .env) and restart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tini.config import load_settings  # noqa: E402
from tini.documents.qdrant_store import migrate_to_hybrid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="existing dense-only collection")
    parser.add_argument("target", help="new collection to create (must not exist)")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    settings = load_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    try:
        copied = migrate_to_hybrid(client, args.source, args.target)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"copied {copied} chunks from {args.source} to {args.target}")
    print(f"next: set TINI_QDRANT_COLLECTION={args.target} and restart tini")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
