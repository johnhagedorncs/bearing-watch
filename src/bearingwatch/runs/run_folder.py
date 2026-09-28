"""Run folders: one self-describing directory per training run.

RULE: everything a run produces goes in ONE directory, and inference loads that
directory BY PATH. Never a loose model file.

The alternative -- ``models/best.pt`` next to ``scaler.pkl`` next to
``config.yaml``, each overwritten by whichever run finished last -- fails in a
specific and expensive way. The files are individually valid, so nothing raises.
Load a model trained on one feature set with a scaler fitted on another and you
get numbers. Reasonable-looking numbers. There is no error, no shape mismatch if
the widths happen to agree, and no way to tell from the output that the pair
does not belong together. By the time anyone notices, several weeks of scored
results are suspect and none of them can be traced.

A run folder makes that mistake unrepresentable: the model, the scaler, the
threshold, the config snapshot, the metrics and the manifest are written
together and loaded together, and a partial folder is detectable because the
manifest is written last.

Layout:

    runs/<run_id>/
      manifest.json        provenance; written LAST
      config.snapshot.yaml full config as resolved for this run
      model.pt             (or model_pca.npz / model_kurtosis.json)
      model_meta.json      feature names + train metrics
      scaler.npz           standardiser fitted on the healthy window only
      threshold.json       declared-reference median/MAD and z threshold
      metrics.json         training and detection metrics
      scores.parquet       per-snapshot scores, written by `infer`
      reference.parquet    healthy-window features, for the drift monitor

Run ids are ``<UTC timestamp>-<config hash>``: sortable, and two runs with
identical config on the same data are visibly related.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from .manifest import MANIFEST_NAME, Manifest

__all__ = ["RunFolder", "RunFolderError", "latest_run", "list_runs", "new_run_id"]

RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6,16}$")

CONFIG_SNAPSHOT = "config.snapshot.yaml"
SCALER_FILE = "scaler.npz"
THRESHOLD_FILE = "threshold.json"
METRICS_FILE = "metrics.json"
SCORES_FILE = "scores.parquet"
REFERENCE_FILE = "reference.parquet"


class RunFolderError(RuntimeError):
    """Raised for missing, incomplete, or malformed run folders."""


def new_run_id(cfg: Config, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{cfg.config_hash[:8]}"


@dataclass
class RunFolder:
    """Handle on one run directory."""

    path: Path
    run_id: str

    # ---------------------------------------------------------------- create

    @classmethod
    def create(
        cls,
        root: str | Path,
        cfg: Config,
        run_id: str | None = None,
        command: str = "",
    ) -> tuple[RunFolder, Manifest]:
        """Create a fresh run folder and its in-progress manifest."""
        root = Path(root)
        run_id = run_id or new_run_id(cfg)
        path = root / run_id
        if path.exists():
            raise RunFolderError(
                f"run folder already exists: {path}. Run folders are never overwritten -- "
                "that is the property that makes an old result still explainable."
            )
        path.mkdir(parents=True)

        cfg.dump_yaml(path / CONFIG_SNAPSHOT)

        manifest = Manifest(
            run_id=run_id,
            command=command,
            config_hash=cfg.config_hash,
            feature_cache_key=cfg.feature_cache_key,
            model_kind=cfg.model.kind,
            data_root=str(cfg.data.root),
        )
        manifest.add_output("config_snapshot", path / CONFIG_SNAPSHOT, path)
        return cls(path=path, run_id=run_id), manifest

    @classmethod
    def open(cls, path: str | Path, require_complete: bool = True) -> RunFolder:
        """Open an existing run folder, refusing an incomplete one by default."""
        path = Path(path)
        if not path.is_dir():
            raise RunFolderError(
                f"run folder not found: {path}\n"
                "  Train one first:  bearingwatch train --data <dir>\n"
                "  Or list existing: bearingwatch runs"
            )
        if require_complete:
            manifest_path = path / MANIFEST_NAME
            if not manifest_path.is_file():
                raise RunFolderError(
                    f"{path} has no {MANIFEST_NAME}, so the run that created it did not "
                    "finish. The manifest is written last precisely so this is detectable. "
                    "Refusing to load a model from an incomplete run."
                )
            manifest = Manifest.load(path)
            if manifest.status != "complete":
                raise RunFolderError(
                    f"{path} has manifest status {manifest.status!r}, not 'complete'."
                )
        return cls(path=path, run_id=path.name)

    # ------------------------------------------------------------ accessors

    def file(self, name: str) -> Path:
        return self.path / name

    @property
    def manifest(self) -> Manifest:
        return Manifest.load(self.path)

    @property
    def config(self) -> Config:
        """The config this run was created with, read back from its snapshot.

        Read from the snapshot in the run folder, never from configs/ on disk.
        The files in configs/ describe what the NEXT run would do; the snapshot
        describes what THIS run did, and inference has to follow the snapshot or
        it is not reproducing the run it claims to be serving.
        """
        import yaml

        path = self.file(CONFIG_SNAPSHOT)
        if not path.is_file():
            raise RunFolderError(f"no {CONFIG_SNAPSHOT} in {self.path}")
        with path.open("r", encoding="utf-8") as fh:
            return Config.from_dict(yaml.safe_load(fh) or {})

    # -------------------------------------------------------------- scaler

    def save_scaler(self, scaler: Any, feature_names: list[str]) -> Path:
        """Persist a fitted StandardScaler as plain arrays.

        Deliberately not pickle. A pickled sklearn estimator is tied to the
        exact library version that wrote it, and it executes code on load. Mean
        and scale vectors are the entire content of a StandardScaler; storing
        them as arrays makes the artifact readable, portable and inert.
        """
        path = self.file(SCALER_FILE)
        np.savez(
            path,
            mean=np.asarray(scaler.mean_, dtype=np.float64),
            scale=np.asarray(scaler.scale_, dtype=np.float64),
            feature_names=np.array(feature_names, dtype=object),
        )
        return path

    def load_scaler(self) -> tuple[Any, list[str]]:
        path = self.file(SCALER_FILE)
        if not path.is_file():
            raise RunFolderError(f"no {SCALER_FILE} in {self.path}")
        from sklearn.preprocessing import StandardScaler

        with np.load(path, allow_pickle=True) as data:
            mean = np.asarray(data["mean"], dtype=np.float64)
            scale = np.asarray(data["scale"], dtype=np.float64)
            names = [str(n) for n in data["feature_names"].tolist()]

        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = scale
        scaler.var_ = scale**2
        scaler.n_features_in_ = mean.shape[0]
        scaler.n_samples_seen_ = 0
        return scaler, names

    # ----------------------------------------------------------- json blobs

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.file(name)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        return path

    def read_json(self, name: str) -> dict[str, Any]:
        path = self.file(name)
        if not path.is_file():
            raise RunFolderError(f"no {name} in {self.path}")
        with path.open("r", encoding="utf-8") as fh:
            return dict(json.load(fh))

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        return self.write_json(METRICS_FILE, metrics)

    def read_metrics(self) -> dict[str, Any]:
        return self.read_json(METRICS_FILE)

    # ------------------------------------------------------------- parquet

    def write_table(self, name: str, frame: Any) -> Path:
        path = self.file(name)
        frame.to_parquet(path, compression="snappy")
        return path

    def read_table(self, name: str) -> Any:
        import pandas as pd

        path = self.file(name)
        if not path.is_file():
            raise RunFolderError(f"no {name} in {self.path}")
        return pd.read_parquet(path)

    def describe(self) -> dict[str, Any]:
        """Summary for `bearingwatch runs` and the API's /model endpoint."""
        try:
            manifest = self.manifest
        except FileNotFoundError:
            return {"run_id": self.run_id, "path": str(self.path), "status": "incomplete"}
        return {
            "run_id": manifest.run_id,
            "path": str(self.path),
            "status": manifest.status,
            "created_at": manifest.created_at,
            "model_kind": manifest.model_kind,
            "config_hash": manifest.config_hash,
            "feature_cache_key": manifest.feature_cache_key,
            "data_root": manifest.data_root,
            "git_commit": manifest.code_version.get("git_commit"),
            "n_outputs": len(manifest.outputs),
        }


def list_runs(root: str | Path = "runs") -> list[RunFolder]:
    """Every run folder under ``root``, newest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = [
        RunFolder(path=p, run_id=p.name)
        for p in sorted(root.iterdir(), reverse=True)
        if p.is_dir() and RUN_ID_RE.match(p.name)
    ]
    return found


def latest_run(root: str | Path = "runs", require_complete: bool = True) -> RunFolder:
    """The most recent complete run folder.

    Convenience for local work only. Production points at an explicit run id --
    "whatever ran most recently" is not a model version, and a scheduled job
    that resolves it at runtime changes behaviour without anyone deploying
    anything.
    """
    for folder in list_runs(root):
        if not require_complete:
            return folder
        manifest_path = folder.path / MANIFEST_NAME
        if manifest_path.is_file():
            try:
                if Manifest.load(folder.path).status == "complete":
                    return folder
            except (json.JSONDecodeError, OSError):
                continue
    raise RunFolderError(
        f"no complete run folder under {root}. Train one:  bearingwatch train --data <dir>"
    )
