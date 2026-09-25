"""Object storage for uploaded originals — network-free via a fake S3 client.

The MinIO adapter only needs bucket_exists/make_bucket/put_object/get_object,
so a dict-backed double exercises the real adapter code (key building, URI
quoting, the bucket/key guards) without a server or the `minio` package.
"""

from __future__ import annotations

import pytest

from tini.config import Settings
from tini.documents.blobstore import BlobStoreError, MinioBlobStore, make_blob_store
from tini.ops import document_api


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data
        self.closed = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        pass


class _FakeS3:
    def __init__(self):
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], bytes] = {}
        self.content_types: dict[tuple[str, str], str] = {}

    def bucket_exists(self, bucket):
        return bucket in self.buckets

    def make_bucket(self, bucket):
        self.buckets.add(bucket)

    def put_object(self, bucket, key, stream, length, content_type=None):
        data = stream.read()
        assert len(data) == length
        self.objects[(bucket, key)] = data
        self.content_types[(bucket, key)] = content_type

    def get_object(self, bucket, key):
        return _FakeResponse(self.objects[(bucket, key)])


@pytest.fixture
def s3():
    return _FakeS3()


@pytest.fixture
def store(s3):
    return MinioBlobStore("unused:9000", "ak", "sk", "docs", client=s3)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        home=tmp_path / "home",
        document_workspace="engineering",
        document_index="sqlite",
        embedding_provider="none",
        rerank_provider="none",
        document_storage="minio",
        minio_access_key="ak",
        minio_secret_key="sk",
        minio_bucket="docs",
    )


def test_put_creates_the_bucket_once_and_returns_an_s3_uri(store, s3):
    uri = store.put("engineering", "runbooks/worker recovery.md", b"hello", "text/markdown")

    assert uri == "s3://docs/engineering/runbooks/worker%20recovery.md"
    assert s3.buckets == {"docs"}
    assert s3.objects[("docs", "engineering/runbooks/worker recovery.md")] == b"hello"
    assert s3.content_types[("docs", "engineering/runbooks/worker recovery.md")] == "text/markdown"
    store.put("engineering", "b.md", b"x", "text/markdown")
    assert s3.buckets == {"docs"}


def test_get_round_trips_a_stored_uri_including_spaces_and_unicode(store):
    uri = store.put("ws", "レポート 2025.md", "本文".encode(), "text/markdown")

    assert store.get(uri) == "本文".encode()


def test_get_only_reads_the_configured_bucket(store, s3):
    s3.objects[("other", "ws/secret.md")] = b"not yours"

    with pytest.raises(BlobStoreError, match="not in bucket"):
        store.get("s3://other/ws/secret.md")
    with pytest.raises(BlobStoreError, match="not in bucket"):
        store.get("file:///etc/passwd")


@pytest.mark.parametrize("uri", ["s3://docs/", "s3://docs/ws/../x.md", "s3://docs/ws/%2E%2E/x.md"])
def test_get_refuses_unsafe_keys(store, uri):
    with pytest.raises(BlobStoreError, match="unsafe"):
        store.get(uri)


def test_provider_failures_become_blobstore_errors_without_the_secret():
    class Down(_FakeS3):
        def bucket_exists(self, bucket):
            raise ConnectionError("connection refused")

    down = MinioBlobStore("h:9000", "ak", "super-secret-key", "docs", client=Down())

    with pytest.raises(BlobStoreError) as caught:
        down.put("ws", "a.md", b"x", "text/markdown")
    assert "connection refused" in str(caught.value)
    assert "super-secret-key" not in str(caught.value)


def test_factory_is_local_by_default_and_strict_about_minio_config():
    assert make_blob_store(Settings(document_storage="local")) is None
    assert make_blob_store(Settings(document_storage="")) is None
    with pytest.raises(ValueError, match="ACCESS_KEY"):
        make_blob_store(Settings(document_storage="minio", minio_access_key="", minio_secret_key=""))
    with pytest.raises(ValueError, match="Unsupported"):
        make_blob_store(Settings(document_storage="ftp"))


def test_upload_stores_the_original_in_minio_and_records_the_s3_uri(settings, store, s3, monkeypatch):
    monkeypatch.setattr(document_api, "make_blob_store", lambda _settings: store)
    events = []

    result = document_api.ingest_upload(
        "runbooks/worker.md",
        "engineering",
        b"# Worker recovery\n\nRestart the orion worker with the green command.",
        lambda stage, event: events.append(stage),
        settings,
    )

    assert result["reports"][0]["action"] == "created"
    assert ("docs", "engineering/runbooks/worker.md") in s3.objects
    assert not (settings.home / "documents").exists(), "nothing may be kept on local disk"
    (item,) = document_api.documents_payload(settings)["items"]
    assert item["source_uri"] == "s3://docs/engineering/runbooks/worker.md"
    assert "uploaded" in events and "indexing" in events


def test_reindex_reads_the_original_back_from_minio_and_updates_the_same_document(
    settings, store, s3, monkeypatch
):
    monkeypatch.setattr(document_api, "make_blob_store", lambda _settings: store)
    document_api.ingest_upload(
        "worker.md", "engineering", b"# Worker\n\nOld text about the orion worker.", lambda *a: None, settings
    )
    (item,) = document_api.documents_payload(settings)["items"]
    s3.objects[("docs", "engineering/worker.md")] = b"# Worker\n\nNew text about the orion worker."

    result = document_api.action(
        {"action": "reindex", "document_id": item["document_id"], "workspace": "engineering"}, settings
    )

    assert result["reports"][0]["action"] == "updated"
    items = document_api.documents_payload(settings)["items"]
    assert len(items) == 1, "reindex must not create a file:// twin"
    assert items[0]["source_uri"] == "s3://docs/engineering/worker.md"


def test_reindex_of_an_s3_document_says_so_when_storage_is_not_minio(settings, store, monkeypatch):
    monkeypatch.setattr(document_api, "make_blob_store", lambda _settings: store)
    document_api.ingest_upload("a.md", "engineering", b"# A\n\ntext about apples.", lambda *a: None, settings)
    (item,) = document_api.documents_payload(settings)["items"]
    monkeypatch.setattr(document_api, "make_blob_store", lambda _settings: None)

    result = document_api.action(
        {"action": "reindex", "document_id": item["document_id"], "workspace": "engineering"}, settings
    )

    assert "not 'minio'" in result["error"]


def test_a_misconfigured_minio_upload_is_an_error_not_a_crash(tmp_path):
    settings = Settings(
        home=tmp_path / "home",
        document_index="sqlite",
        embedding_provider="none",
        rerank_provider="none",
        document_storage="minio",
        minio_access_key="",
        minio_secret_key="",
    )

    result = document_api.ingest_upload("a.md", "ws", b"# A\n\ntext", lambda *a: None, settings)

    assert "ACCESS_KEY" in result["error"]


def test_local_storage_still_writes_under_tini_home(tmp_path):
    settings = Settings(
        home=tmp_path / "home",
        document_workspace="engineering",
        document_index="sqlite",
        embedding_provider="none",
        rerank_provider="none",
        document_storage="local",
    )

    document_api.ingest_upload("a.md", "engineering", b"# A\n\ntext about apples.", lambda *a: None, settings)

    assert (settings.home / "documents" / "engineering" / "a.md").is_file()
    (item,) = document_api.documents_payload(settings)["items"]
    assert item["source_uri"].startswith("file://")
