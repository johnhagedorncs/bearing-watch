"""Artifact store abstraction: local filesystem and S3, selected by URI.

Training writes a run folder to local disk. Serving needs that run folder, and
in a cluster the pod that serves is not the pod that trained and shares no
filesystem with it. The store is the seam between the two.

Configured by URI so the deployment decides, not the code:

    file://runs                 local directory (default, and what tests use)
    s3://my-bucket/bearingwatch  S3 prefix

boto3 is a guarded import. The local backend has no AWS dependency at all, so
the whole pipeline, the tests and the container run without credentials or
network. That is deliberate: a project that cannot be run without an AWS account
cannot be reviewed by the person reading it.

AWS specifics are in the README under "Artifact storage on S3": the bucket
layout, the IAM actions actually required, and why the pod gets a role rather
than a key.
"""

from __future__ import annotations

import abc
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "ArtifactStore",
    "ArtifactStoreError",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "store_from_uri",
]


class ArtifactStoreError(RuntimeError):
    """Raised for store configuration and transfer failures."""


class ArtifactStore(abc.ABC):
    """Upload and download whole run folders."""

    scheme: str = ""

    @abc.abstractmethod
    def put_run(self, local_dir: str | Path, run_id: str) -> str:
        """Upload a run folder. Returns its URI."""

    @abc.abstractmethod
    def get_run(self, run_id: str, local_dir: str | Path) -> Path:
        """Download a run folder to ``local_dir``. Returns the local path."""

    @abc.abstractmethod
    def list_runs(self) -> list[str]:
        """Run ids present in the store."""

    @abc.abstractmethod
    def exists(self, run_id: str) -> bool: ...

    def uri_for(self, run_id: str) -> str:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    """Filesystem-backed store. The default, and the one the tests exercise."""

    scheme = "file"

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def put_run(self, local_dir: str | Path, run_id: str) -> str:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise ArtifactStoreError(f"not a directory: {local_dir}")
        target = self.root / run_id
        if local_dir.resolve() == target.resolve():
            # Already in place. Training writes straight into the store root in
            # the common local case, and copying a directory onto itself is a
            # good way to destroy it.
            return self.uri_for(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_dir, target)
        return self.uri_for(run_id)

    def get_run(self, run_id: str, local_dir: str | Path) -> Path:
        source = self.root / run_id
        if not source.is_dir():
            raise ArtifactStoreError(f"run {run_id} not found under {self.root}")
        local_dir = Path(local_dir)
        if source.resolve() == local_dir.resolve():
            return local_dir
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(source, local_dir)
        return local_dir

    def list_runs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def exists(self, run_id: str) -> bool:
        return (self.root / run_id).is_dir()

    def uri_for(self, run_id: str) -> str:
        return f"file://{self.root}/{run_id}"


class S3ArtifactStore(ArtifactStore):
    """S3-backed store. boto3 is imported lazily and only when actually used."""

    scheme = "s3"

    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ArtifactStoreError(
                    "s3:// artifact store requires boto3, which is not installed.\n"
                    "  pip install 'bearingwatch[s3]'\n"
                    "  Or use a local store: serve.artifact_store_uri: file://runs"
                ) from exc
            self._client = boto3.client("s3")
        return self._client

    def _key(self, run_id: str, name: str = "") -> str:
        parts = [p for p in (self.prefix, run_id, name) if p]
        return "/".join(parts)

    def put_run(self, local_dir: str | Path, run_id: str) -> str:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise ArtifactStoreError(f"not a directory: {local_dir}")

        files = sorted(p for p in local_dir.rglob("*") if p.is_file())
        if not files:
            raise ArtifactStoreError(f"{local_dir} is empty; nothing to upload")

        # Upload the manifest LAST, mirroring how a run folder is written on
        # disk. A reader that treats "manifest present" as "run complete" then
        # holds for S3 too, and a partial upload cannot be mistaken for a
        # finished run.
        manifest = local_dir / "manifest.json"
        ordered = [p for p in files if p != manifest] + ([manifest] if manifest.is_file() else [])

        for path in ordered:
            rel = path.relative_to(local_dir).as_posix()
            self.client.upload_file(str(path), self.bucket, self._key(run_id, rel))
        return self.uri_for(run_id)

    def get_run(self, run_id: str, local_dir: str | Path) -> Path:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._key(run_id) + "/"

        paginator = self.client.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                found = True
                rel = obj["Key"][len(prefix) :]
                if not rel:
                    continue
                target = local_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, obj["Key"], str(target))
        if not found:
            raise ArtifactStoreError(f"run {run_id} not found at s3://{self.bucket}/{prefix}")
        return local_dir

    def list_runs(self) -> list[str]:
        prefix = (self.prefix + "/") if self.prefix else ""
        paginator = self.client.get_paginator("list_objects_v2")
        runs: set[str] = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                runs.add(cp["Prefix"][len(prefix) :].strip("/"))
        return sorted(runs)

    def exists(self, run_id: str) -> bool:
        response = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=self._key(run_id) + "/", MaxKeys=1
        )
        return response.get("KeyCount", 0) > 0

    def uri_for(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self._key(run_id)}"


def store_from_uri(uri: str, client: Any = None) -> ArtifactStore:
    """Build the right store for a URI.

    ``file://runs``            -> LocalArtifactStore('runs')
    ``s3://bucket/prefix``     -> S3ArtifactStore('bucket', 'prefix')
    A bare path                -> LocalArtifactStore(path)
    """
    if not uri:
        raise ArtifactStoreError("artifact store URI is empty")

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        if scheme == "":
            return LocalArtifactStore(uri)
        # file://runs parses netloc='runs'; file:///abs/path parses path='/abs/path'
        root = f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        return LocalArtifactStore(root or "runs")

    if scheme == "s3":
        if not parsed.netloc:
            raise ArtifactStoreError(f"s3 URI needs a bucket: {uri}")
        return S3ArtifactStore(bucket=parsed.netloc, prefix=parsed.path.strip("/"), client=client)

    raise ArtifactStoreError(
        f"unsupported artifact store scheme {scheme!r} in {uri!r}. Supported: file://, s3://"
    )
