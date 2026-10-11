"""Where a dashboard-uploaded ORIGINAL file lives, apart from the index.

The index (SQLite/Qdrant) only holds chunks. The source file used to sit in
`TINI_HOME/documents/<workspace>/` and be recorded as a `file://` source_uri;
that path dies with the container's volume and cannot be shared between
machines. With `TINI_DOCUMENT_STORAGE=minio` the original is stored in an
S3-compatible bucket instead and the document's `source_uri` becomes
`s3://<bucket>/<workspace>/<name>` — the URI scheme is the flag that says
where the original is, and "Re-index" dispatches on it.

    pip install 'tini-agent[minio]'
    TINI_DOCUMENT_STORAGE=minio
    TINI_MINIO_ENDPOINT=localhost:9000
    TINI_MINIO_ACCESS_KEY=...
    TINI_MINIO_SECRET_KEY=...
    TINI_MINIO_BUCKET=tini-documents

The bucket is created on first upload. Reads are limited to the configured
bucket: a source_uri that names any other bucket is refused rather than
fetched, since that field comes back out of the index.
"""

from __future__ import annotations

from io import BytesIO
from typing import Protocol
from urllib.parse import quote, unquote, urlparse

from tini.config import Settings

S3_SCHEME = "s3"


class BlobStoreError(RuntimeError):
    """The object store was unreachable, refused the request, or the URI was
    not one this store is allowed to read."""


class BlobStore(Protocol):
    def put(self, workspace_id: str, name: str, data: bytes, content_type: str) -> str:
        """Store the original and return the source_uri that identifies it."""
        ...

    def get(self, uri: str) -> bytes: ...


class MinioBlobStore:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        client=None,
    ):
        if client is None:
            try:
                from minio import Minio
            except ImportError as exc:
                raise BlobStoreError(
                    "MinIO storage requires the minio package: pip install 'tini-agent[minio]'"
                ) from exc
            client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.client = client
        self.bucket = bucket
        self._bucket_ready = False

    def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except Exception as exc:
            raise BlobStoreError(f"MinIO bucket check failed: {type(exc).__name__}: {exc}") from exc
        self._bucket_ready = True

    def put(self, workspace_id: str, name: str, data: bytes, content_type: str) -> str:
        key = f"{workspace_id}/{name}"
        self._ensure_bucket()
        try:
            self.client.put_object(
                self.bucket, key, BytesIO(data), len(data), content_type=content_type
            )
        except Exception as exc:
            raise BlobStoreError(f"MinIO upload failed: {type(exc).__name__}: {exc}") from exc
        return f"{S3_SCHEME}://{self.bucket}/{quote(key)}"

    def get(self, uri: str) -> bytes:
        parsed = urlparse(uri)
        key = unquote(parsed.path.lstrip("/"))
        if parsed.scheme != S3_SCHEME or parsed.netloc != self.bucket:
            raise BlobStoreError(f"refusing to read {uri!r}: not in bucket {self.bucket!r}")
        if not key or any(part in {"", ".", ".."} for part in key.split("/")):
            raise BlobStoreError(f"refusing to read {uri!r}: unsafe object key")
        try:
            response = self.client.get_object(self.bucket, key)
        except Exception as exc:
            raise BlobStoreError(f"MinIO download failed: {type(exc).__name__}: {exc}") from exc
        try:
            return response.read()
        finally:
            response.close()
            release = getattr(response, "release_conn", None)
            if release:
                release()


def make_blob_store(settings: Settings) -> BlobStore | None:
    """None means 'keep originals on local disk' — the default, and what every
    caller did before this module existed."""
    backend = settings.document_storage
    if backend in {"", "local"}:
        return None
    if backend == "minio":
        if not (settings.minio_access_key and settings.minio_secret_key):
            raise ValueError("TINI_MINIO_ACCESS_KEY and TINI_MINIO_SECRET_KEY are required")
        if not settings.minio_bucket:
            raise ValueError("TINI_MINIO_BUCKET must not be empty")
        return MinioBlobStore(
            settings.minio_endpoint,
            settings.minio_access_key,
            settings.minio_secret_key,
            settings.minio_bucket,
            settings.minio_secure,
        )
    raise ValueError(f"Unsupported document storage: {backend}")
