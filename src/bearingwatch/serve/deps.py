"""Loading a run folder, and scoring against it.

``LoadedRun`` is the one place that turns a run folder on disk into something
that can score. Both the ``infer`` CLI and the FastAPI service go through it, so
there is exactly one implementation of "apply the scaler, then the model, then
the threshold", in that order, exactly once each.

That single implementation matters more than it looks. The double-scaling bug
and the mismatched-scaler bug both come from having two code paths that each do
part of this and disagree about what the other already did.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..detect.reference_threshold import ReferenceThreshold
from ..features.pipeline import extract_snapshot_features, require_unscaled
from ..models.base import AnomalyModel, load_model
from ..monitor.band_split import BandSplitDiscriminant
from ..runs.manifest import Manifest
from ..runs.run_folder import (
    REFERENCE_FILE,
    THRESHOLD_FILE,
    RunFolder,
    RunFolderError,
    latest_run,
)

__all__ = ["RUN_DIR_ENV", "LoadedRun", "get_loaded_run", "load_run", "set_loaded_run"]

RUN_DIR_ENV = "BEARINGWATCH_RUN_DIR"
RUNS_ROOT_ENV = "BEARINGWATCH_RUNS_ROOT"
BAND_SPLIT_FILE = "band_split.json"


@dataclass
class LoadedRun:
    """A run folder loaded into memory and ready to score."""

    run_id: str
    path: Path
    config: Config
    manifest: Manifest
    model: AnomalyModel
    scaler: Any
    feature_names: list[str]
    threshold: ReferenceThreshold
    band_split: BandSplitDiscriminant | None = None
    reference: pd.DataFrame | None = field(default=None, repr=False)

    # ------------------------------------------------------------- scoring

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Score a RAW (unscaled) feature table.

        ``require_unscaled`` runs first. Handing this an already-scaled table is
        the double-scaling bug, and it does not fail loudly anywhere else --
        it just lowers every score and makes a degrading bearing look healthy.
        """
        require_unscaled(frame, where="LoadedRun.score_frame")
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} feature(s) required by run {self.run_id} are missing from the "
                f"input, e.g. {missing[:5]}. The feature config used at scoring time does not "
                "match the one this run was trained with -- load the run's config snapshot "
                "rather than configs/ on disk."
            )
        raw = frame[self.feature_names].to_numpy(dtype=np.float64)
        return self.score_matrix(raw)

    def score_matrix(self, raw: np.ndarray) -> np.ndarray:
        """Score a raw feature matrix in ``feature_names`` order."""
        raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        if raw.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features in the order given by "
                f"/model, got {raw.shape[1]}"
            )
        scaled = self.scaler.transform(raw) if self.config.model.standardise else raw
        return self.model.score(scaled)

    def score_signal(self, signal: np.ndarray, n_channels: int | None = None) -> np.ndarray:
        """Featurise a raw snapshot and score it.

        ``signal`` is (n_points, n_channels) or a flat 1-D array for a
        single-channel snapshot. Shape is checked against the run's config: a
        record of the wrong length produces a different spectrum, which the
        model would happily score.
        """
        from datetime import datetime

        from ..data.ims_parser import Snapshot

        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D (n_points, n_channels) array, got {arr.shape}")

        expected_points = self.config.data.expected_points
        if arr.shape[0] != expected_points:
            raise ValueError(
                f"snapshot has {arr.shape[0]} points but this run expects exactly "
                f"{expected_points} ({expected_points / self.config.data.sample_rate_hz:.1f} s "
                f"at {self.config.data.sample_rate_hz} Hz). A shorter record changes the "
                "spectral resolution and every band energy with it."
            )
        expected_channels = n_channels or self.config.data.expected_channels
        if arr.shape[1] != expected_channels:
            raise ValueError(
                f"snapshot has {arr.shape[1]} channel(s) but this run expects {expected_channels}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("snapshot contains non-finite values")

        snap = Snapshot(timestamp=datetime.now(), data=arr, path=Path("<request>"))
        feats = extract_snapshot_features(snap, self.config)
        raw = np.array([[feats[name] for name in self.feature_names]], dtype=np.float64)
        return self.score_matrix(raw)

    # -------------------------------------------------------------- verdict

    def verdict(self, scores: np.ndarray) -> list[dict[str, Any]]:
        """Per-score z, threshold comparison, and a human-readable verdict."""
        scores = np.asarray(scores, dtype=np.float64).ravel()
        z = self.threshold.z(scores)
        out = []
        for s, zi in zip(scores, z):
            out.append(
                {
                    "score": float(s),
                    "z": float(zi),
                    "threshold_z": float(self.threshold.z_threshold),
                    "score_threshold": float(self.threshold.score_threshold),
                    "anomalous": bool(zi > self.threshold.z_threshold),
                    "verdict": "anomalous" if zi > self.threshold.z_threshold else "normal",
                }
            )
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_path": str(self.path),
            "model_kind": self.model.kind,
            "n_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "config_hash": self.manifest.config_hash,
            "feature_cache_key": self.manifest.feature_cache_key,
            "code_version": self.manifest.code_version,
            "created_at": self.manifest.created_at,
            "data_root": self.manifest.data_root,
            "threshold": self.threshold.to_dict(),
            "standardise": self.config.model.standardise,
            "expected_points": self.config.data.expected_points,
            "expected_channels": self.config.data.expected_channels,
            "sample_rate_hz": self.config.data.sample_rate_hz,
            "band_split_available": self.band_split is not None,
            "has_reference": self.reference is not None,
        }


def load_run(path: str | Path | None = None, runs_root: str | Path | None = None) -> LoadedRun:
    """Load a run folder by path, or the latest complete run under ``runs_root``.

    Resolution order: explicit path, then ``BEARINGWATCH_RUN_DIR``, then the
    newest complete run under ``runs_root``. Production should always set an
    explicit run id -- "whatever trained most recently" is not a model version,
    and resolving it at pod start means behaviour changes without a deployment.
    """
    if path is None:
        env_path = os.environ.get(RUN_DIR_ENV)
        if env_path:
            path = env_path

    if path is None:
        root = runs_root or os.environ.get(RUNS_ROOT_ENV, "runs")
        folder = latest_run(root)
    else:
        folder = RunFolder.open(path)

    cfg = folder.config
    manifest = folder.manifest

    model = load_model(manifest.model_kind or cfg.model.kind, folder.path)

    if cfg.model.standardise:
        scaler, scaler_names = folder.load_scaler()
        feature_names = scaler_names
        if model.feature_names_ and list(model.feature_names_) != list(feature_names):
            raise RunFolderError(
                f"run {folder.run_id}: the saved scaler and the saved model disagree about "
                "feature names. This is exactly the mismatch a run folder exists to make "
                "impossible, so the folder has been tampered with or assembled by hand. "
                "Refusing to serve it."
            )
    else:
        scaler = _IdentityScaler()
        feature_names = list(model.feature_names_)

    threshold = ReferenceThreshold.load(folder.file(THRESHOLD_FILE))

    band_split = None
    bs_path = folder.file(BAND_SPLIT_FILE)
    if bs_path.is_file():
        band_split = BandSplitDiscriminant.load(bs_path)

    reference = None
    ref_path = folder.file(REFERENCE_FILE)
    if ref_path.is_file():
        reference = pd.read_parquet(ref_path)

    return LoadedRun(
        run_id=folder.run_id,
        path=folder.path,
        config=cfg,
        manifest=manifest,
        model=model,
        scaler=scaler,
        feature_names=list(feature_names),
        threshold=threshold,
        band_split=band_split,
        reference=reference,
    )


class _IdentityScaler:
    """Stand-in when standardisation is off, so the scoring path stays uniform."""

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------
# process-wide handle used by the API
# --------------------------------------------------------------------------

_LOADED: LoadedRun | None = None
_LOCK = threading.Lock()


def set_loaded_run(run: LoadedRun | None) -> None:
    global _LOADED
    with _LOCK:
        _LOADED = run


def get_loaded_run() -> LoadedRun:
    """The run loaded at application startup.

    Raises rather than lazily loading. A request that triggers a model load is a
    request that takes seconds and can fail in ways a request handler should not
    be discovering -- that is what ``/ready`` is for.
    """
    with _LOCK:
        if _LOADED is None:
            raise RuntimeError(
                "no model is loaded. Set BEARINGWATCH_RUN_DIR to a run folder before "
                "starting the service, or POST to /reload with a run_dir."
            )
        return _LOADED


def feature_frame_from_vectors(
    vectors: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Build a correctly-flagged unscaled feature frame from raw vectors."""
    from ..data.cache import SCALED_METADATA_KEY

    frame = pd.DataFrame(np.asarray(vectors, dtype=np.float64), columns=list(feature_names))
    frame.attrs["bearingwatch"] = {SCALED_METADATA_KEY: False, "source": "api-request"}
    return frame
