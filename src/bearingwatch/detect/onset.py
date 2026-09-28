"""Onset detection with persistence, and lead time reported as a RANGE.

PERSISTENCE. One snapshot over threshold is not an onset. Vibration data has
transients -- a load change, someone walking past the rig, an ADC glitch -- and
firing on a single sample makes the detector useless in exactly the way that
gets alerting turned off. The rule here is k-of-n: ``consecutive`` samples over
threshold within a rolling window of ``window``.

LEAD TIME. This is where most write-ups of this problem overclaim, so it is
worth being blunt.

The obvious thing to report is "the detector fired N hours before failure". For
a STEP change that number is meaningful: there is a real moment when the signal
changed, and the detector either found it or did not.

For a RAMP -- which is what bearing degradation is -- there is no such moment.
The score rises continuously from the healthy level. The apparent onset is
wherever the ramp happens to cross a threshold, and that threshold is set by the
statistics of whatever window was declared healthy. Declare a shorter healthy
window and it usually sits earlier in the run where the bearing is cleanest, the
MAD is smaller, the threshold is tighter, and the onset moves EARLIER. Declare a
longer one and it moves later. The number moves by hours based on a choice made
in a config file, and nothing physical changed.

So a single lead-time figure is not a measurement of the bearing. It is a
measurement of the baseline window, reported as though it were a measurement of
the bearing. This module sweeps the baseline fraction and reports the resulting
spread, so the arbitrariness is visible in the output instead of hidden in a
config default. If the range is wide, that IS the finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .reference_threshold import ReferenceThreshold

__all__ = ["LeadTimeRange", "OnsetResult", "detect_onset", "persistence_mask", "sweep_lead_time"]


def persistence_mask(exceeds: np.ndarray, window: int, consecutive: int) -> np.ndarray:
    """True where ``consecutive`` of the last ``window`` samples exceed threshold.

    Causal: sample i only sees samples <= i. A centred window would let the
    detector use the future, which inflates every lead time reported from it.
    """
    exceeds = np.asarray(exceeds, dtype=bool)
    if window < 1 or consecutive < 1:
        raise ValueError("window and consecutive must both be >= 1")
    if consecutive > window:
        raise ValueError("consecutive cannot exceed window")

    n = exceeds.size
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    counts = pd.Series(exceeds.astype(int)).rolling(window, min_periods=1).sum().to_numpy()
    out = counts >= consecutive
    return np.asarray(out, dtype=bool)


@dataclass
class OnsetResult:
    """Where the detector fired, and what it did in the healthy phase."""

    fired: bool
    onset_index: int | None
    onset_timestamp: Any | None
    n_exceed: int
    n_fired: int
    n_total: int
    max_z: float
    healthy_false_positive_rate: float
    healthy_max_z: float
    degraded_detection_rate: float
    lead_snapshots: int | None = None
    lead_hours: float | None = None
    # Always populated by detect_onset. Not Optional, because making them
    # optional pushes a None check onto every caller for a case that cannot
    # happen, and those checks then hide real indexing mistakes.
    z_scores: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fired_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "onset_index": self.onset_index,
            "onset_timestamp": (
                str(self.onset_timestamp) if self.onset_timestamp is not None else None
            ),
            "n_exceed": self.n_exceed,
            "n_fired": self.n_fired,
            "n_total": self.n_total,
            "max_z": self.max_z,
            "healthy_false_positive_rate": self.healthy_false_positive_rate,
            "healthy_max_z": self.healthy_max_z,
            "degraded_detection_rate": self.degraded_detection_rate,
            "lead_snapshots": self.lead_snapshots,
            "lead_hours": self.lead_hours,
        }


def detect_onset(
    scores: np.ndarray,
    threshold: ReferenceThreshold,
    window: int = 5,
    consecutive: int = 3,
    timestamps: Sequence[Any] | None = None,
    healthy_end_index: int | None = None,
    snapshot_interval_minutes: float = 10.0,
) -> OnsetResult:
    """Run the detector over a scored series.

    ``healthy_end_index`` splits the run for reporting: everything before it is
    the healthy phase, where firing is a false positive, and everything from it
    onward is the degradation phase, where firing is the job. Both are reported
    -- a detector's false-positive rate on declared-healthy data is not optional
    context, it is half the result.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("cannot detect onset on an empty score series")

    z = threshold.z(scores)
    exceeds = z > threshold.z_threshold
    fired_mask = persistence_mask(exceeds, window, consecutive)

    fired_idx = np.flatnonzero(fired_mask)
    onset_index = int(fired_idx[0]) if fired_idx.size else None

    onset_timestamp = None
    if onset_index is not None and timestamps is not None and onset_index < len(timestamps):
        onset_timestamp = timestamps[onset_index]

    if healthy_end_index is None:
        healthy_end_index = 0
    healthy_end_index = int(np.clip(healthy_end_index, 0, n))

    healthy_slice = fired_mask[:healthy_end_index]
    degraded_slice = fired_mask[healthy_end_index:]

    healthy_fp = float(np.mean(healthy_slice)) if healthy_slice.size else 0.0
    healthy_max_z = float(np.max(z[:healthy_end_index])) if healthy_end_index > 0 else float("nan")
    degraded_rate = float(np.mean(degraded_slice)) if degraded_slice.size else 0.0

    lead_snapshots = None
    lead_hours = None
    if onset_index is not None:
        # Lead time to the END OF THE RUN, which for a run-to-failure record is
        # the failure. Not to a labelled fault time -- the IMS sets do not carry
        # one, and inventing a label to measure against would be circular.
        lead_snapshots = int(n - 1 - onset_index)
        lead_hours = round(lead_snapshots * snapshot_interval_minutes / 60.0, 3)

    return OnsetResult(
        fired=onset_index is not None,
        onset_index=onset_index,
        onset_timestamp=onset_timestamp,
        n_exceed=int(exceeds.sum()),
        n_fired=int(fired_mask.sum()),
        n_total=n,
        max_z=float(np.max(z)),
        healthy_false_positive_rate=healthy_fp,
        healthy_max_z=healthy_max_z,
        degraded_detection_rate=degraded_rate,
        lead_snapshots=lead_snapshots,
        lead_hours=lead_hours,
        z_scores=z,
        fired_mask=fired_mask,
    )


@dataclass
class LeadTimeRange:
    """Lead time swept across baseline-window choices."""

    fractions: list[float]
    per_fraction: list[dict[str, Any]]
    min_hours: float | None
    max_hours: float | None
    median_hours: float | None
    spread_hours: float | None
    n_fired: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_fractions": self.fractions,
            "per_fraction": self.per_fraction,
            "lead_hours_min": self.min_hours,
            "lead_hours_max": self.max_hours,
            "lead_hours_median": self.median_hours,
            "lead_hours_spread": self.spread_hours,
            "n_baselines_that_fired": self.n_fired,
            "note": (
                "Lead time is reported as a RANGE. For a ramp-shaped event the apparent "
                "onset is set by the baseline window used to define normal, not by a "
                "physical moment in the bearing's life. A single number here would be a "
                "property of the config, not of the bearing."
            ),
        }

    def summary(self) -> str:
        if self.n_fired == 0:
            return "no baseline choice produced a detection"
        return (
            f"{self.min_hours:.1f}-{self.max_hours:.1f} h "
            f"(median {self.median_hours:.1f} h, spread {self.spread_hours:.1f} h) "
            f"across {self.n_fired}/{len(self.fractions)} baseline windows"
        )


def sweep_lead_time(
    scores: np.ndarray,
    fractions: Sequence[float],
    z_threshold: float = 3.5,
    mad_scale: float = 1.4826,
    min_mad: float = 1e-9,
    window: int = 5,
    consecutive: int = 3,
    snapshot_interval_minutes: float = 10.0,
    max_healthy_fraction: float | None = None,
) -> LeadTimeRange:
    """Re-fit the threshold on each baseline fraction and collect the lead times.

    ``max_healthy_fraction`` guards the sweep: a baseline fraction reaching past
    the true healthy phase puts degraded data into the reference period, which
    inflates the MAD and pushes the onset later. That is a real effect worth
    seeing, but it is a different effect from the one being measured, so
    fractions beyond the declared healthy phase are recorded and skipped.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    per_fraction: list[dict[str, Any]] = []
    lead_values: list[float] = []

    for frac in fractions:
        n_ref = int(round(frac * n))
        if n_ref < 2:
            per_fraction.append(
                {"fraction": frac, "skipped": "reference window shorter than 2 samples"}
            )
            continue
        if max_healthy_fraction is not None and frac > max_healthy_fraction + 1e-9:
            per_fraction.append(
                {
                    "fraction": frac,
                    "skipped": (
                        f"exceeds the declared healthy phase ({max_healthy_fraction:.2f}); "
                        "reference period would contain degraded data"
                    ),
                }
            )
            continue

        thr = ReferenceThreshold.fit(
            scores[:n_ref],
            z_threshold=z_threshold,
            mad_scale=mad_scale,
            min_mad=min_mad,
            reference="healthy_window",
        )
        result = detect_onset(
            scores,
            thr,
            window=window,
            consecutive=consecutive,
            healthy_end_index=n_ref,
            snapshot_interval_minutes=snapshot_interval_minutes,
        )
        entry = {
            "fraction": frac,
            "n_reference": n_ref,
            "fired": result.fired,
            "onset_index": result.onset_index,
            "lead_hours": result.lead_hours,
            "max_z": round(result.max_z, 3),
        }
        per_fraction.append(entry)
        if result.fired and result.lead_hours is not None:
            lead_values.append(result.lead_hours)

    if lead_values:
        arr = np.array(lead_values, dtype=np.float64)
        return LeadTimeRange(
            fractions=list(fractions),
            per_fraction=per_fraction,
            min_hours=float(arr.min()),
            max_hours=float(arr.max()),
            median_hours=float(np.median(arr)),
            spread_hours=float(arr.max() - arr.min()),
            n_fired=len(lead_values),
        )
    return LeadTimeRange(
        fractions=list(fractions),
        per_fraction=per_fraction,
        min_hours=None,
        max_hours=None,
        median_hours=None,
        spread_hours=None,
        n_fired=0,
    )
