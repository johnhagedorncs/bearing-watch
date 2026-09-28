"""Artifact store: local backend, S3 backend against a fake client, URI routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from bearingwatch.store.artifact_store import (
    ArtifactStoreError,
    LocalArtifactStore,
    S3ArtifactStore,
    store_from_uri,
)


@pytest.fixture
def a_run(tmp_path: Path) -> Path:
    run = tmp_path / "source" / "20240101T000000Z-abcdef01"
    run.mkdir(parents=True)
    (run / "model_pca.npz").write_bytes(b"weights")
    (run / "model_meta.json").write_text('{"kind": "pca"}')
    (run / "manifest.json").write_text('{"status": "complete"}')
    return run


# ------------------------------------------------------------------ routing


def test_file_uri_with_a_relative_root() -> None:
    store = store_from_uri("file://runs")
    assert isinstance(store, LocalArtifactStore)
    assert str(store.root) == "runs"


def test_file_uri_with_an_absolute_path() -> None:
    store = store_from_uri("file:///var/lib/bearingwatch")
    assert isinstance(store, LocalArtifactStore)
    assert str(store.root) == "/var/lib/bearingwatch"


def test_bare_path_is_treated_as_local() -> None:
    assert isinstance(store_from_uri("/tmp/runs"), LocalArtifactStore)


def test_s3_uri_splits_bucket_and_prefix() -> None:
    store = store_from_uri("s3://my-bucket/some/prefix")
    assert isinstance(store, S3ArtifactStore)
    assert store.bucket == "my-bucket"
    assert store.prefix == "some/prefix"


def test_s3_uri_without_a_prefix() -> None:
    store = store_from_uri("s3://my-bucket")
    assert store.bucket == "my-bucket"
    assert store.prefix == ""


def test_s3_uri_without_a_bucket_is_rejected() -> None:
    with pytest.raises(ArtifactStoreError, match="needs a bucket"):
        store_from_uri("s3://")


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ArtifactStoreError, match="unsupported artifact store scheme"):
        store_from_uri("gs://bucket/prefix")


def test_empty_uri_is_rejected() -> None:
    with pytest.raises(ArtifactStoreError, match="empty"):
        store_from_uri("")


# -------------------------------------------------------------------- local


def test_local_put_and_get_round_trip(a_run: Path, tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    uri = store.put_run(a_run, a_run.name)
    assert uri.startswith("file://")
    assert store.exists(a_run.name)
    assert store.list_runs() == [a_run.name]

    out = store.get_run(a_run.name, tmp_path / "restored")
    assert (out / "model_pca.npz").read_bytes() == b"weights"
    assert (out / "manifest.json").is_file()


def test_local_put_is_idempotent_when_already_in_place(tmp_path: Path) -> None:
    """Copying a directory onto itself is a good way to destroy it."""
    root = tmp_path / "runs"
    run = root / "20240101T000000Z-abcdef01"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")

    store = LocalArtifactStore(root)
    store.put_run(run, run.name)
    assert (run / "manifest.json").is_file()


def test_local_get_to_the_same_path_is_a_noop(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = root / "20240101T000000Z-abcdef01"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")

    store = LocalArtifactStore(root)
    assert store.get_run(run.name, run) == run
    assert (run / "manifest.json").is_file()


def test_local_put_overwrites_a_previous_upload(a_run: Path, tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    store.put_run(a_run, a_run.name)
    (a_run / "extra.txt").write_text("new")
    store.put_run(a_run, a_run.name)
    assert (store.root / a_run.name / "extra.txt").is_file()


def test_local_get_of_a_missing_run_raises(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(ArtifactStoreError, match="not found"):
        store.get_run("nope", tmp_path / "out")


def test_local_put_of_a_missing_directory_raises(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(ArtifactStoreError, match="not a directory"):
        store.put_run(tmp_path / "nope", "nope")


def test_local_list_of_an_absent_root_is_empty(tmp_path: Path) -> None:
    assert LocalArtifactStore(tmp_path / "absent").list_runs() == []


# ----------------------------------------------------------------------- s3


class FakeS3Client:
    """Minimal in-memory stand-in for the boto3 S3 client surface used here."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_order: list[str] = []

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[key] = Path(filename).read_bytes()
        self.upload_order.append(key)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.objects[key])

    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=None, Delimiter=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if MaxKeys:
            keys = keys[:MaxKeys]
        return {"Contents": [{"Key": k} for k in keys], "KeyCount": len(keys)}

    def get_paginator(self, _name: str):
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix="", Delimiter=None):
                keys = sorted(k for k in client.objects if k.startswith(Prefix))
                if Delimiter:
                    prefixes = sorted(
                        {Prefix + k[len(Prefix) :].split(Delimiter)[0] + Delimiter for k in keys}
                    )
                    yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}
                else:
                    yield {"Contents": [{"Key": k} for k in keys]}

        return Paginator()


def test_s3_put_and_get_round_trip(a_run: Path, tmp_path: Path) -> None:
    client = FakeS3Client()
    store = S3ArtifactStore("bucket", "bearingwatch", client=client)

    uri = store.put_run(a_run, a_run.name)
    assert uri == f"s3://bucket/bearingwatch/{a_run.name}"
    assert store.exists(a_run.name)
    assert store.list_runs() == [a_run.name]

    out = store.get_run(a_run.name, tmp_path / "restored")
    assert (out / "model_pca.npz").read_bytes() == b"weights"


def test_s3_uploads_the_manifest_last(a_run: Path) -> None:
    """Mirrors the on-disk rule: manifest present means the run is complete.

    A partial upload must not be mistakable for a finished run.
    """
    client = FakeS3Client()
    S3ArtifactStore("bucket", "p", client=client).put_run(a_run, a_run.name)
    assert client.upload_order[-1].endswith("manifest.json")


def test_s3_get_of_a_missing_run_raises(tmp_path: Path) -> None:
    store = S3ArtifactStore("bucket", "p", client=FakeS3Client())
    with pytest.raises(ArtifactStoreError, match="not found"):
        store.get_run("nope", tmp_path / "out")


def test_s3_put_of_an_empty_directory_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    store = S3ArtifactStore("bucket", "p", client=FakeS3Client())
    with pytest.raises(ArtifactStoreError, match="empty"):
        store.put_run(empty, "empty")


def test_s3_key_layout_has_no_double_slashes(a_run: Path) -> None:
    client = FakeS3Client()
    S3ArtifactStore("bucket", "", client=client).put_run(a_run, a_run.name)
    assert all("//" not in key for key in client.objects)
