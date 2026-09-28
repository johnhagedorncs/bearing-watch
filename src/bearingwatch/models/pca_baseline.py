"""PCA reconstruction error -- the baseline the autoencoder has to beat.

Structurally this is the same idea as the autoencoder: project onto a
lower-dimensional manifold fitted on healthy data, reconstruct, and use the
residual as the anomaly score. The difference is that the manifold is linear and
fitted in closed form.

That matters because it makes the comparison a real question rather than a
formality. A linear autoencoder with MSE loss spans the same subspace as PCA, so
the deep model only has something to offer if the healthy manifold is actually
curved AND there is enough healthy data to identify the curvature. With a few
hundred stationary snapshots and under a hundred features, there very often is
not, and PCA wins on equal footing while training in milliseconds and having no
seed, no learning rate and no early-stopping rule to get wrong.

Reporting that is worth more than winning. A candidate who shows they checked
whether the complicated thing was necessary is making a different claim from one
who shows the complicated thing works.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import PCAConfig
from .base import AnomalyModel, ModelError

__all__ = ["PCABaseline"]

_MODEL = "model_pca.npz"
_META = "model_meta.json"


class PCABaseline(AnomalyModel):
    """Score = mean squared error of the rank-k PCA reconstruction."""

    kind = "pca"

    def __init__(self, cfg: PCAConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PCAConfig()
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int = 0
        self.rank_capped_: bool = False
        self.requested_components_: int = 0

    def _fit(self, x: np.ndarray) -> None:
        from sklearn.decomposition import PCA

        n_samples, n_features = x.shape

        # CAP THE RANK RELATIVE TO THE SAMPLE COUNT.
        #
        # A reconstruction model needs residuals on its training data, because
        # those residuals ARE the reference distribution the threshold is built
        # from. PCA with k components reconstructs n samples exactly once
        # k >= min(n - 1, d). Ask for 95% explained variance on a healthy window
        # of 18 snapshots and 67 features and you get ~17 components, the
        # healthy residuals collapse to near zero, the MAD collapses with them,
        # and every subsequent snapshot scores as a large anomaly.
        #
        # The symptom is not an error. It is a model that appears to detect
        # degradation with a perfect hit rate while actually firing on
        # everything that is not its own training set -- measured here at a
        # 17-23% false-positive rate on data that was still healthy.
        #
        # `max_sample_ratio` requires at least that many samples per retained
        # component, which is also roughly what a stable covariance estimate
        # needs.
        ratio = max(1.0, float(self.cfg.max_sample_ratio))
        rank_cap = max(1, int(n_samples // ratio))
        max_components = max(1, min(n_samples - 1, n_features, rank_cap))

        requested = self.cfg.n_components
        capped = False
        if isinstance(requested, float) and 0.0 < requested < 1.0:
            # Resolve the variance target first, then apply the cap.
            probe = PCA(svd_solver="full", random_state=self.cfg.seed).fit(x)
            cumulative = np.cumsum(probe.explained_variance_ratio_)
            wanted = int(np.searchsorted(cumulative, requested) + 1)
            n_components = min(wanted, max_components)
            capped = n_components < wanted
        else:
            wanted = int(requested)
            if wanted < 1:
                raise ModelError(f"pca.n_components resolved to {wanted}")
            n_components = min(wanted, max_components)
            capped = n_components < wanted

        pca = PCA(n_components=n_components, svd_solver="full", random_state=self.cfg.seed)
        pca.fit(x)
        self.rank_capped_ = capped
        self.requested_components_ = wanted

        self.mean_ = np.asarray(pca.mean_, dtype=np.float64)
        self.components_ = np.asarray(pca.components_, dtype=np.float64)
        self.explained_variance_ratio_ = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        self.n_components_ = int(self.components_.shape[0])

        train_scores = self._score(x)
        self.train_metrics_ = {
            "n_components": float(self.n_components_),
            "requested_components": float(self.requested_components_),
            "rank_capped": float(self.rank_capped_),
            "samples_per_component": float(n_samples / max(1, self.n_components_)),
            "explained_variance": float(self.explained_variance_ratio_.sum()),
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
            "n_train": float(n_samples),
        }

    def _score(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:  # pragma: no cover
            raise ModelError("PCABaseline is not fitted")
        centred = x - self.mean_

        # The errstate suppression is platform-specific, not a way of ignoring
        # bad numbers. numpy built against Apple's Accelerate framework (the
        # default on macOS arm64) raises spurious divide-by-zero, overflow and
        # invalid FPE flags from matmul on perfectly ordinary finite input --
        # reproducible with two random matrices and nothing else. Suppressing
        # the flag and then CHECKING THE RESULT is the honest form: a real
        # non-finite value still raises below, so this cannot hide an actual
        # numerical failure.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            recon = centred @ self.components_.T @ self.components_

        if not np.all(np.isfinite(recon)):
            raise ModelError(
                "PCA reconstruction produced non-finite values. This is a genuine numerical "
                "failure, not the macOS/Accelerate false FPE flag -- check the scaled "
                "features for extreme magnitudes."
            )
        return np.mean((centred - recon) ** 2, axis=1)

    def save(self, directory: str | Path) -> list[Path]:
        if self.mean_ is None or self.components_ is None or self.explained_variance_ratio_ is None:
            raise ModelError("cannot save an unfitted PCABaseline")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / _MODEL
        np.savez(
            model_path,
            mean=self.mean_,
            components=self.components_,
            explained_variance_ratio=self.explained_variance_ratio_,
        )

        meta_path = directory / _META
        payload = {
            "kind": self.kind,
            "config": {
                "n_components": self.cfg.n_components,
                "seed": self.cfg.seed,
                "max_sample_ratio": self.cfg.max_sample_ratio,
            },
            "n_components_fitted": self.n_components_,
            "feature_names": self.feature_names_,
            "train_metrics": self.train_metrics_,
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

        return [model_path, meta_path]

    @classmethod
    def load(cls, directory: str | Path) -> PCABaseline:
        directory = Path(directory)
        model_path = directory / _MODEL
        meta_path = directory / _META
        for path in (model_path, meta_path):
            if not path.is_file():
                raise ModelError(f"missing {path.name} in {directory}")

        with meta_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        model = cls(PCAConfig(**payload["config"]))
        with np.load(model_path) as data:
            model.mean_ = data["mean"]
            model.components_ = data["components"]
            model.explained_variance_ratio_ = data["explained_variance_ratio"]
        model.n_components_ = int(payload.get("n_components_fitted", model.components_.shape[0]))
        model.feature_names_ = list(payload["feature_names"])
        model.train_metrics_ = dict(payload.get("train_metrics", {}))
        model.fitted_ = True
        return model
