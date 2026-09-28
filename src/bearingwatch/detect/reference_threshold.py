"""Robust threshold anchored to a DECLARED STATIONARY REFERENCE PERIOD.

This is the single most consequential design decision in the repo, so it gets
the long comment.

To turn an anomaly score into a verdict you need a location and a scale: a
median and a MAD, then z = (s - median) / (scale * MAD). The question is what
data those statistics come from.

THE WRONG ANSWER, which is the common one: compute them from the series you are
scoring. It is appealing because it needs no configuration, adapts to any
signal, and has no reference period to argue about. It is also, for the
failure mode this project exists to catch, mathematically incapable of working.

Take a pure linear ramp of any slope, x_i = a + b*i for i = 0..n-1. The median
is the midpoint. The absolute deviations |x_i - median| are themselves a
symmetric linear ramp, so for large n the MAD converges to a quarter of the
total range, while the maximum absolute deviation is half the range. Therefore

    max|z| -> (range/2) / (1.4826 * range/4) = 2 / 1.4826 = 1.349

and the slope b cancels completely. A bearing whose score ramps by a factor of
two and one whose score ramps by a factor of a thousand produce the SAME max|z|
of about 1.35. Against any sensible threshold -- 3, 3.5, 4 -- the detector never
fires. Not "fires late": never, at any magnitude. And it still fires correctly
on isolated spikes, so it passes every spike test anyone writes for it and looks
like a working detector. ``max_abs_z_self_referenced`` below computes this, and
``tests/test_reference_threshold.py`` asserts the invariance across three orders
of magnitude of ramp slope.

THE PRECONDITION, stated precisely because it matters operationally: the pinning
is exact when the scored window is DOMINATED BY THE RAMP. A window that is still
half stationary has its MAD set by the stationary half, so the ramp end can
exceed the threshold and the detector appears to work. That is not a reprieve --
it is worse. It means a self-referencing detector fires early in a run, while
the window still straddles the healthy period, and then goes silent as the
window slides fully into the degradation, exactly when the bearing is worst. The
apparent competence at the start is what stops anyone looking.

The companion project ``tsad-harness`` derives this result in general and shows
it holds for the windowed and streaming variants too. This repo is where it is
paid for: bearing degradation IS a ramp, and a self-referencing detector is
blind to exactly the thing being detected.

THE RIGHT ANSWER, implemented here: median and MAD come from a period DECLARED
STATIONARY IN ADVANCE -- the healthy window fixed in configs/data.yaml before
any results were looked at. The reference does not move as the run degrades, so
a growing score produces a growing z, which is the entire point.

``reference: self`` is still implemented, purely so the failure can be
demonstrated in a test rather than asserted in prose. It is never correct in
production and the code says so at the point of use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SELF_REFERENCE_MAX_Z",
    "ReferenceThreshold",
    "ThresholdError",
    "max_abs_z_self_referenced",
    "robust_stats",
]

# 2 / 1.4826. The value max|z| converges to for a self-referenced MAD z-score on
# a monotone ramp, independent of the ramp's magnitude.
SELF_REFERENCE_MAX_Z = 2.0 / 1.4826  # ~= 1.3490


class ThresholdError(RuntimeError):
    """Raised when a threshold is used before being fitted, or fitted on nothing."""


def robust_stats(values: np.ndarray, min_mad: float = 1e-9) -> tuple[float, float]:
    """Median and MAD of ``values``, with a principled fallback for a zero MAD.

    A MAD of zero is not rare and is not always a broken input. It happens
    whenever MORE THAN HALF the reference values are identical, which is the
    normal case for a one-sided score: the kurtosis baseline reports
    ``max(value - healthy_median, 0)``, so by construction half the healthy
    scores are exactly 0.0, the median is 0.0, and the MAD is 0.0 too.

    Clamping straight to ``min_mad`` (1e-9) there is catastrophic rather than
    conservative: it divides a legitimate score of 0.5 by a nanoscale number and
    produces z in the hundreds of millions, so the detector fires on the healthy
    window and the run reports a 40% false-positive rate that looks like a model
    failure rather than a divide-by-almost-zero.

    Fallback order: MAD, then standard deviation (a valid if less robust scale
    estimate, and finite whenever the data is not constant), then ``min_mad``
    for genuinely constant input where nothing can be salvaged.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        raise ThresholdError("cannot compute robust statistics on an empty array")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    # Secondary scale estimate from the 5th-95th percentile range. For gaussian
    # data that range is 3.29 sigma, so dividing by 3.29 puts this on the same
    # footing as the MAD's own sigma estimate and the ratio mad/secondary sits
    # near 0.674.
    p5, p95 = np.percentile(values, [5, 95])
    secondary = float((p95 - p5) / 3.29)

    # DEGENERATE MAD. The MAD is meaningless when more than half the reference
    # values are tied: it stops estimating spread and starts reporting the gap
    # to the nearest distinct value. That happens for real -- any one-sided or
    # clipped score puts a point mass at zero -- and it does not produce a MAD
    # of zero, which is why a `max(mad, min_mad)` floor does not catch it. It
    # produces a MAD that is small, positive, and entirely an artifact of the
    # tie, so every z-score inflates by orders of magnitude and the detector
    # fires all over the healthy window.
    #
    # A ratio this extreme cannot arise from any non-degenerate distribution:
    # for gaussian data it is ~0.674, and even for heavy-tailed data it stays
    # well above 0.01.
    if secondary > min_mad and mad < 0.01 * secondary:
        return median, secondary

    if mad > min_mad:
        return median, mad
    if secondary > min_mad:
        return median, secondary

    std = float(np.std(values))
    if std > min_mad:
        return median, std

    return median, float(min_mad)


def max_abs_z_self_referenced(values: np.ndarray, mad_scale: float = 1.4826) -> float:
    """max|z| when the z-score references the series being scored.

    Exists to be measured, not used. For a monotone ramp this returns
    approximately ``SELF_REFERENCE_MAX_Z`` regardless of the ramp's magnitude.
    """
    median, mad = robust_stats(values)
    return float(np.max(np.abs(values - median)) / (mad_scale * mad))


@dataclass
class ReferenceThreshold:
    """Robust z-score threshold anchored to a declared reference period."""

    median: float = 0.0
    mad: float = 1.0
    z_threshold: float = 3.5
    mad_scale: float = 1.4826
    min_mad: float = 1e-9
    reference: str = "healthy_window"
    n_reference: int = 0
    reference_span: tuple[str, str] | None = None
    fitted: bool = False

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        reference_scores: np.ndarray,
        z_threshold: float = 3.5,
        mad_scale: float = 1.4826,
        min_mad: float = 1e-9,
        reference: str = "healthy_window",
        reference_span: tuple[str, str] | None = None,
    ) -> ReferenceThreshold:
        """Fit on scores from the declared reference period.

        ``reference_scores`` must come from the HEALTHY WINDOW, not from the
        series about to be scored. Passing the full series here is the bug this
        whole module is about.
        """
        scores = np.asarray(reference_scores, dtype=np.float64).ravel()
        if scores.size < 2:
            raise ThresholdError(
                f"reference period has {scores.size} score(s); need at least 2. "
                "Check data.healthy_fraction against the number of snapshots."
            )
        if not np.all(np.isfinite(scores)):
            raise ThresholdError("reference scores contain non-finite values")

        median, mad = robust_stats(scores, min_mad)
        return cls(
            median=median,
            mad=mad,
            z_threshold=z_threshold,
            mad_scale=mad_scale,
            min_mad=min_mad,
            reference=reference,
            n_reference=int(scores.size),
            reference_span=reference_span,
            fitted=True,
        )

    @classmethod
    def fit_self_referenced(
        cls,
        scores: np.ndarray,
        z_threshold: float = 3.5,
        mad_scale: float = 1.4826,
        min_mad: float = 1e-9,
    ) -> ReferenceThreshold:
        """Fit on the series being scored. NEVER correct in production.

        Provided only so the blindness-to-ramps result can be demonstrated by
        running it rather than asserted in a comment. See the module docstring.
        """
        thr = cls.fit(scores, z_threshold, mad_scale, min_mad, reference="self")
        return thr

    # ---------------------------------------------------------------- apply

    def z(self, scores: np.ndarray) -> np.ndarray:
        """Signed robust z-score."""
        if not self.fitted:
            raise ThresholdError("ReferenceThreshold is not fitted")
        scores = np.asarray(scores, dtype=np.float64).ravel()
        return (scores - self.median) / (self.mad_scale * self.mad)

    def exceeds(self, scores: np.ndarray) -> np.ndarray:
        """Boolean mask of samples above the threshold.

        One-sided. Only an INCREASE in reconstruction error is evidence of
        degradation; a score below the healthy median means the snapshot
        reconstructs better than typical healthy data, which is not a fault.
        """
        return self.z(scores) > self.z_threshold

    @property
    def score_threshold(self) -> float:
        """The threshold expressed back in raw score units."""
        return float(self.median + self.z_threshold * self.mad_scale * self.mad)

    # ------------------------------------------------------------------- io

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "mad": self.mad,
            "z_threshold": self.z_threshold,
            "mad_scale": self.mad_scale,
            "min_mad": self.min_mad,
            "reference": self.reference,
            "n_reference": self.n_reference,
            "reference_span": list(self.reference_span) if self.reference_span else None,
            "score_threshold": self.score_threshold,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceThreshold:
        span = payload.get("reference_span")
        return cls(
            median=float(payload["median"]),
            mad=float(payload["mad"]),
            z_threshold=float(payload["z_threshold"]),
            mad_scale=float(payload.get("mad_scale", 1.4826)),
            min_mad=float(payload.get("min_mad", 1e-9)),
            reference=str(payload.get("reference", "healthy_window")),
            n_reference=int(payload.get("n_reference", 0)),
            reference_span=(tuple(span) if span else None),  # type: ignore[arg-type]
            fitted=bool(payload.get("fitted", True)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ReferenceThreshold:
        path = Path(path)
        if not path.is_file():
            raise ThresholdError(f"threshold file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
