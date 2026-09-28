"""Common interface for every anomaly model.

All three models -- autoencoder, PCA, kurtosis threshold -- implement the same
four methods and are trained on the same healthy window from the same feature
table. That is the only way the comparison in docs/RESULTS.md means anything: if
the baselines were fitted differently, or scored differently, a win would be an
artifact of the harness rather than of the model.

Score convention: HIGHER IS MORE ANOMALOUS, always, for every model. Stated
explicitly because a sign flip in one model's score is invisible in a
comparison table -- it just looks like that model performs badly.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["AnomalyModel", "ModelError"]


class ModelError(RuntimeError):
    """Raised for model state errors: scoring before fitting, shape mismatch."""


class AnomalyModel(abc.ABC):
    """Base class. Subclasses implement _fit, _score, save and load."""

    #: Short stable identifier used in run manifests and the API response.
    kind: str = "base"

    def __init__(self) -> None:
        self.feature_names_: list[str] = []
        self.fitted_: bool = False
        self.train_metrics_: dict[str, float] = {}

    # ------------------------------------------------------------------ api

    def fit(self, x: np.ndarray, feature_names: Sequence[str]) -> AnomalyModel:
        """Fit on the healthy window. ``x`` is (n_samples, n_features), scaled."""
        x = _as_2d(x)
        if x.shape[1] != len(feature_names):
            raise ModelError(
                f"feature_names has {len(feature_names)} entries but x has {x.shape[1]} columns"
            )
        if x.shape[0] < 2:
            raise ModelError(
                f"cannot fit on {x.shape[0]} sample(s); the healthy window is too short. "
                "Check data.healthy_fraction and the number of snapshots available."
            )
        self.feature_names_ = list(feature_names)
        self._fit(x)
        self.fitted_ = True
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Anomaly score per row. Higher is more anomalous."""
        if not self.fitted_:
            raise ModelError(f"{type(self).__name__} is not fitted")
        x = _as_2d(x)
        if x.shape[1] != len(self.feature_names_):
            raise ModelError(
                f"model was fitted on {len(self.feature_names_)} features but was given "
                f"{x.shape[1]}. Feature width drifting between train and score is usually "
                "an ablation switch or a schema_version change on one side only."
            )
        scores = np.asarray(self._score(x), dtype=np.float64).ravel()
        if scores.shape[0] != x.shape[0]:  # pragma: no cover - programming error
            raise ModelError("model returned the wrong number of scores")
        return scores

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_features": len(self.feature_names_),
            "feature_names": list(self.feature_names_),
            "train_metrics": dict(self.train_metrics_),
        }

    # -------------------------------------------------------------- to fill

    @abc.abstractmethod
    def _fit(self, x: np.ndarray) -> None: ...

    @abc.abstractmethod
    def _score(self, x: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def save(self, directory: str | Path) -> list[Path]:
        """Persist into a run folder. Returns every file written."""

    @classmethod
    @abc.abstractmethod
    def load(cls, directory: str | Path) -> AnomalyModel:
        """Load from a run folder."""


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ModelError(f"expected a 2-D feature array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        raise ModelError(
            f"feature array contains {n_bad} non-finite value(s). A NaN reaching the model "
            "produces a NaN score, and a NaN score compares False against every threshold, "
            "so the detector silently stops firing."
        )
    return arr


def build_model(kind: str, cfg: Any) -> AnomalyModel:
    """Construct a model by kind. Imports are local to keep torch optional."""
    if kind == "autoencoder":
        from .autoencoder import AutoencoderModel

        return AutoencoderModel(cfg.autoencoder)
    if kind == "pca":
        from .pca_baseline import PCABaseline

        return PCABaseline(cfg.pca)
    if kind == "kurtosis":
        from .kurtosis_baseline import KurtosisBaseline

        return KurtosisBaseline(cfg.kurtosis)
    raise ModelError(f"unknown model kind {kind!r}")


def load_model(kind: str, directory: str | Path) -> AnomalyModel:
    """Load a model of the given kind from a run folder."""
    if kind == "autoencoder":
        from .autoencoder import AutoencoderModel

        return AutoencoderModel.load(directory)
    if kind == "pca":
        from .pca_baseline import PCABaseline

        return PCABaseline.load(directory)
    if kind == "kurtosis":
        from .kurtosis_baseline import KurtosisBaseline

        return KurtosisBaseline.load(directory)
    raise ModelError(f"unknown model kind {kind!r}")
