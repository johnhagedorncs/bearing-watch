"""Single-feature kurtosis baseline -- the floor every other model must clear.

This model does not learn anything. It reads one column out of the feature
vector, the Pearson kurtosis of the target channel, and reports how far above
the healthy median it sits. Rolling-element defects produce impulsive contact,
impulsive contact raises the fourth moment, and this has been the standard
bearing condition indicator since long before anyone put a neural network near
one.

It is in this repo for one reason: a comparison table without a trivial baseline
does not tell you whether the pipeline is doing anything. If a 67-dimensional
autoencoder cannot beat one scalar statistic that requires no training, the
honest conclusion is that the extra machinery is not earning its place on this
problem -- and that conclusion is the useful output.

Where it is known to fail, stated up front rather than found later: kurtosis is
not monotone over a bearing's life. It rises when a localised defect starts
producing discrete impacts, then FALLS back toward 3 as the defect spreads,
contact becomes more continuous, and the signal turns broadband again. A late,
badly worn bearing can read healthy on kurtosis. Any threshold on this feature
therefore has a detection window rather than a detection point, and this
baseline is scored on the same declared-reference basis as the other two so that
limitation shows up in the comparison instead of being argued about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import KurtosisConfig
from .base import AnomalyModel, ModelError

__all__ = ["KurtosisBaseline"]

_META = "model_meta.json"
_MODEL = "model_kurtosis.json"


class KurtosisBaseline(AnomalyModel):
    """Score = |feature - healthy median| / healthy MAD for one chosen feature."""

    kind = "kurtosis"

    def __init__(self, cfg: KurtosisConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or KurtosisConfig()
        self.feature_index_: int = -1
        self.median_: float = 0.0
        self.mad_: float = 1.0

    def _resolve_index(self) -> int:
        name = self.cfg.feature_name
        if name in self.feature_names_:
            return self.feature_names_.index(name)
        # Fall back to any kurtosis column so the baseline still works when the
        # target channel is not channel 0.
        candidates = [n for n in self.feature_names_ if n.endswith("__kurtosis")]
        if candidates:
            return self.feature_names_.index(candidates[0])
        raise ModelError(
            f"kurtosis baseline needs feature {name!r}, which is not in the feature table "
            f"(and no *__kurtosis column exists either). If features.ablation.drop_kurtosis "
            "is set, this baseline cannot be run."
        )

    def _fit(self, x: np.ndarray) -> None:
        self.feature_index_ = self._resolve_index()
        values = x[:, self.feature_index_]

        self.median_ = float(np.median(values))
        mad = float(np.median(np.abs(values - self.median_)))
        # A degenerate MAD happens on quantised or constant features. Falling
        # back to the standard deviation keeps the score finite; falling back to
        # zero would make every score infinite and the detector would fire on
        # everything, which is a different kind of broken.
        self.mad_ = mad if mad > 1e-12 else max(float(np.std(values)), 1e-12)

        train_scores = self._score(x)
        self.train_metrics_ = {
            "feature_index": float(self.feature_index_),
            "healthy_median": self.median_,
            "healthy_mad": self.mad_,
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
            "n_train": float(x.shape[0]),
        }

    def _score(self, x: np.ndarray) -> np.ndarray:
        """SIGNED deviation from the healthy median, in healthy MADs.

        Signed, not clipped at zero, and that distinction cost a debugging
        session worth recording. The physically motivated score here is
        one-sided -- only a RISE in kurtosis is evidence of a defect -- so the
        obvious implementation is ``max(value - median, 0) / mad``.

        That clipping puts a POINT MASS AT ZERO on exactly half the healthy
        window: every snapshot at or below the healthy median scores exactly
        0.0. The downstream threshold then computes a MAD over a sample that is
        50% ties, and the MAD collapses to roughly half the smallest non-zero
        score -- about 0.006 here rather than the ~1.0 the spread actually
        warrants. Dividing by that produced healthy-window z-scores above 500
        and a 42% false-positive rate on declared-healthy data, which reads as
        "the baseline is terrible" rather than "the scale estimate is degenerate".

        The fix is to put the one-sidedness where it belongs. The model reports
        a signed deviation, which is symmetric on healthy data and gives the MAD
        something real to measure; the DETECTOR is already one-sided
        (``z > threshold``, not ``|z| > threshold``), so a drop below the
        healthy median still never fires.
        """
        values = x[:, self.feature_index_]
        return (values - self.median_) / self.mad_

    def save(self, directory: str | Path) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / _MODEL
        with model_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "feature_index": self.feature_index_,
                    "median": self.median_,
                    "mad": self.mad_,
                },
                fh,
                indent=2,
                sort_keys=True,
            )

        meta_path = directory / _META
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "kind": self.kind,
                    "config": {"feature_name": self.cfg.feature_name},
                    "resolved_feature": (
                        self.feature_names_[self.feature_index_]
                        if 0 <= self.feature_index_ < len(self.feature_names_)
                        else None
                    ),
                    "feature_names": self.feature_names_,
                    "train_metrics": self.train_metrics_,
                },
                fh,
                indent=2,
                sort_keys=True,
            )

        return [model_path, meta_path]

    @classmethod
    def load(cls, directory: str | Path) -> KurtosisBaseline:
        directory = Path(directory)
        model_path = directory / _MODEL
        meta_path = directory / _META
        for path in (model_path, meta_path):
            if not path.is_file():
                raise ModelError(f"missing {path.name} in {directory}")

        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        with model_path.open("r", encoding="utf-8") as fh:
            state = json.load(fh)

        model = cls(KurtosisConfig(**meta["config"]))
        model.feature_names_ = list(meta["feature_names"])
        model.train_metrics_ = dict(meta.get("train_metrics", {}))
        model.feature_index_ = int(state["feature_index"])
        model.median_ = float(state["median"])
        model.mad_ = float(state["mad"])
        model.fitted_ = True
        return model
