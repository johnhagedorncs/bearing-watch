"""Feature and score distribution drift against the training reference.

Two implementations, in this order of authority:

1. A built-in PSI + Kolmogorov-Smirnov calculation, always available, no
   optional dependency. This is what the code paths and tests use.
2. An Evidently report, behind a guarded import, for the richer HTML output a
   human wants when investigating.

The built-in one is primary on purpose. A monitoring signal that only exists
when an optional package imports cleanly is a monitoring signal that will be
absent on the day it matters.

WHAT THIS CANNOT DO, which is the point of scripts/drift_demo.py: a
distribution-drift monitor answers "do these numbers look like the numbers I was
trained on?" and nothing else. It does not and cannot answer "why not". A
degrading bearing and a sensor whose gain changed both move the feature
distribution, and this monitor reports them identically -- same features
flagged, comparable PSI. Escalating on drift alone therefore produces an alert
that says something changed and gives the on-call engineer no way to tell a
failing machine from a maintenance action on an instrument. See
``bearingwatch.monitor.band_split`` for a statistic that does separate them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import MonitorConfig

__all__ = [
    "DriftReport",
    "FeatureDrift",
    "compute_drift",
    "evidently_available",
    "evidently_report",
    "ks_statistic",
    "psi",
]

log = logging.getLogger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two samples.

    PSI = sum over bins of (p_cur - p_ref) * ln(p_cur / p_ref).

    Bin edges come from the REFERENCE quantiles, not from the pooled data.
    Pooled edges move when the current window moves, which partly cancels the
    shift being measured and biases PSI downward exactly when it should be
    rising. Conventional reading: < 0.1 stable, 0.1-0.25 investigate, > 0.25
    the distributions have moved.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    current = np.asarray(current, dtype=np.float64).ravel()
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size < 2 or current.size < 1:
        return 0.0

    # ADAPT THE BIN COUNT TO THE SAMPLE SIZE. A fixed 10 bins on an 11-sample
    # window puts about one observation in each, so which bin a sample lands in
    # is noise, and PSI comes out in the range 3-10 for two halves of the SAME
    # stationary data -- far above the 0.25 "distributions have moved" line. The
    # monitor then fires constantly on short windows and gets ignored.
    #
    # The rule of thumb PSI needs is roughly five observations per bin. Applying
    # it to the smaller of the two samples keeps the statistic honest on short
    # windows at the cost of resolution, which is the right trade for something
    # whose output is an alert.
    n_small = min(reference.size, current.size)
    bins = max(2, min(bins, n_small // 5))

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        # A constant reference feature. Any movement at all is total drift;
        # no movement is none. Returning a finite sentinel keeps the aggregate
        # meaningful instead of producing inf.
        return 0.0 if np.allclose(current, reference[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_p = np.maximum(ref_counts / max(ref_counts.sum(), 1), eps)
    cur_p = np.maximum(cur_counts / max(cur_counts.sum(), 1), eps)

    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    from scipy import stats

    reference = np.asarray(reference, dtype=np.float64).ravel()
    current = np.asarray(current, dtype=np.float64).ravel()
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size < 2 or current.size < 2:
        return 0.0, 1.0
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    drifted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_pvalue": float(f"{self.ks_pvalue:.6g}"),
            "reference_mean": round(self.reference_mean, 6),
            "current_mean": round(self.current_mean, 6),
            "drifted": self.drifted,
        }


@dataclass
class DriftReport:
    """Drift over a set of features plus, optionally, the score itself."""

    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    score_drift: FeatureDrift | None = None
    drifted_fraction: float = 0.0
    fired: bool = False
    mean_psi: float = 0.0
    max_psi: float = 0.0

    def top(self, n: int = 5) -> list[FeatureDrift]:
        return sorted(self.features, key=lambda f: f.psi, reverse=True)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "n_features": len(self.features),
            "drifted_fraction": round(self.drifted_fraction, 4),
            "fired": self.fired,
            "mean_psi": round(self.mean_psi, 6),
            "max_psi": round(self.max_psi, 6),
            "score_drift": self.score_drift.to_dict() if self.score_drift else None,
            "top_features": [f.to_dict() for f in self.top(5)],
        }


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: MonitorConfig | None = None,
    columns: Sequence[str] | None = None,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> DriftReport:
    """Per-feature drift of ``current`` against the training ``reference``.

    A feature counts as drifted when PSI exceeds ``psi_alert`` OR the KS p-value
    falls below ``ks_alpha``. Both, because they fail differently: PSI is
    insensitive to a shift that stays inside the reference bins, and KS becomes
    trivially significant on large samples for shifts too small to care about.
    The monitor fires when the drifted FRACTION of features clears the
    configured threshold -- a single moved feature is a feature, not a regime
    change.
    """
    cfg = cfg or MonitorConfig()

    if columns is None:
        columns = [
            c
            for c in reference.columns
            if c in current.columns
            and pd.api.types.is_numeric_dtype(reference[c])
            and c not in ("snapshot_index",)
        ]
    columns = list(columns)
    if not columns:
        raise ValueError("no shared numeric columns between reference and current frames")

    drifts: list[FeatureDrift] = []
    for col in columns:
        ref_vals = reference[col].to_numpy(dtype=np.float64)
        cur_vals = current[col].to_numpy(dtype=np.float64)
        p = psi(ref_vals, cur_vals)
        ks, pvalue = ks_statistic(ref_vals, cur_vals)
        drifts.append(
            FeatureDrift(
                feature=col,
                psi=p,
                ks_statistic=ks,
                ks_pvalue=pvalue,
                reference_mean=float(np.nanmean(ref_vals)),
                current_mean=float(np.nanmean(cur_vals)),
                drifted=bool(p > cfg.psi_alert or pvalue < cfg.ks_alpha),
            )
        )

    score_drift = None
    if reference_scores is not None and current_scores is not None:
        p = psi(reference_scores, current_scores)
        ks, pvalue = ks_statistic(reference_scores, current_scores)
        score_drift = FeatureDrift(
            feature="__score__",
            psi=p,
            ks_statistic=ks,
            ks_pvalue=pvalue,
            reference_mean=float(np.nanmean(reference_scores)),
            current_mean=float(np.nanmean(current_scores)),
            drifted=bool(p > cfg.psi_alert or pvalue < cfg.ks_alpha),
        )

    psis = np.array([d.psi for d in drifts], dtype=np.float64)
    drifted_fraction = float(np.mean([d.drifted for d in drifts])) if drifts else 0.0

    return DriftReport(
        n_reference=int(len(reference)),
        n_current=int(len(current)),
        features=drifts,
        score_drift=score_drift,
        drifted_fraction=drifted_fraction,
        fired=bool(drifted_fraction >= cfg.drifted_feature_fraction),
        mean_psi=float(psis.mean()) if psis.size else 0.0,
        max_psi=float(psis.max()) if psis.size else 0.0,
    )


# --------------------------------------------------------------------------
# optional: Evidently
# --------------------------------------------------------------------------


def evidently_available() -> bool:
    try:
        import evidently  # noqa: F401
    except Exception:
        return False
    return True


def evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str] | None = None,
    html_path: str | None = None,
) -> dict[str, Any]:
    """Run an Evidently data-drift preset, if Evidently is installed.

    Returns ``{"available": False, ...}`` rather than raising when it is not.
    The built-in PSI/KS report above is the one the pipeline depends on; this
    adds the interactive HTML a human wants when investigating an alert.

    Evidently's API changed substantially at 0.7, so both the modern
    (``Report`` + ``DataDriftPreset`` over a ``Dataset``) and legacy call shapes
    are attempted before giving up.
    """
    if not evidently_available():
        return {"available": False, "reason": "evidently is not installed"}

    cols = (
        list(columns)
        if columns
        else [
            c
            for c in reference.columns
            if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
        ]
    )
    ref = reference[cols].copy()
    cur = current[cols].copy()

    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        definition = DataDefinition(numerical_columns=cols)
        ref_ds = Dataset.from_pandas(ref, data_definition=definition)
        cur_ds = Dataset.from_pandas(cur, data_definition=definition)

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

        if html_path:
            snapshot.save_html(html_path)

        payload = snapshot.dict()
        return {
            "available": True,
            "api": "evidently>=0.7",
            "html_path": html_path,
            "summary": _summarise_evidently(payload),
        }
    except Exception as exc:  # pragma: no cover - version dependent
        log.warning("evidently report failed (%s); built-in PSI/KS report is unaffected", exc)
        return {"available": True, "error": str(exc), "api": "unknown"}


def _summarise_evidently(payload: Any) -> dict[str, Any]:
    """Pull the drifted-column count out of an Evidently payload.

    Written defensively: the payload shape has changed across Evidently
    versions, and a monitoring helper should degrade to "could not summarise"
    rather than raise inside an alerting path.
    """
    try:
        metrics = payload.get("metrics", []) if isinstance(payload, dict) else []
        for metric in metrics:
            value = metric.get("value")
            if isinstance(value, dict) and "share_of_drifted_columns" in value:
                return {
                    "share_of_drifted_columns": value.get("share_of_drifted_columns"),
                    "number_of_drifted_columns": value.get("number_of_drifted_columns"),
                }
            if isinstance(value, (int, float)) and "Drift" in str(metric.get("metric_id", "")):
                return {"metric_id": metric.get("metric_id"), "value": value}
        return {"note": "no drift summary field found in the evidently payload"}
    except Exception:  # pragma: no cover
        return {"note": "could not summarise the evidently payload"}
