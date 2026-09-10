"""Internal-document ingestion and retrieval."""

from tini.documents.base import DocumentIndex
from tini.documents.embeddings import EmbeddingProvider, make_embedding_provider
from tini.documents.pipeline import IngestionPipeline
from tini.documents.store import SqliteDocumentIndex

__all__ = [
    "DocumentIndex",
    "EmbeddingProvider",
    "IngestionPipeline",
    "SqliteDocumentIndex",
    "make_embedding_provider",
]
