"""Persistence, onset, and lead time reported as a range."""

from __future__ import annotations

import numpy as np
import pytest

from bearingwatch.detect.onset import detect_onset, persistence_mask, sweep_lead_time
from bearingwatch.detect.reference_threshold import ReferenceThreshold

# ------------------------------------------------------------- persistence


def test_a_single_exceedance_does_not_fire() -> None:
    """One snapshot over threshold is a transient, not an onset."""
    exceeds = np.zeros(20, dtype=bool)
    exceeds[10] = True
    assert not persistence_mask(exceeds, window=5, consecutive=3).any()


def test_three_of_five_fires() -> None:
    exceeds = np.zeros(20, dtype=bool)
    exceeds[10:13] = True
    fired = persistence_mask(exceeds, window=5, consecutive=3)
    assert fired.any()
    assert int(np.flatnonzero(fired)[0]) == 12


def test_persistence_is_causal() -> None:
    """Sample i may only see samples <= i.

    A centred window lets the detector use the future, which inflates every
    lead time computed from it.
    """
    exceeds = np.zeros(20, dtype=bool)
    exceeds[15:18] = True
    fired = persistence_mask(exceeds, window=5, consecutive=3)
    assert not fired[:15].any(), "fired before any exceedance occurred"


def test_persistence_rejects_impossible_parameters() -> None:
    exceeds = np.zeros(10, dtype=bool)
    with pytest.raises(ValueError):
        persistence_mask(exceeds, window=3, consecutive=5)
    with pytest.raises(ValueError):
        persistence_mask(exceeds, window=0, consecutive=1)


def test_persistence_on_an_empty_series() -> None:
    assert persistence_mask(np.zeros(0, dtype=bool), 5, 3).size == 0


# ------------------------------------------------------------------- onset


def _series(n_healthy: int = 100, n_degraded: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    healthy = 1.0 + rng.normal(0, 0.05, n_healthy)
    degraded = 1.0 + np.linspace(0, 3.0, n_degraded) + rng.normal(0, 0.05, n_degraded)
    return np.concatenate([healthy, degraded]), healthy


def test_detects_onset_in_the_degradation_phase_only() -> None:
    scores, healthy = _series()
    threshold = ReferenceThreshold.fit(healthy, z_threshold=3.5)
    result = detect_onset(scores, threshold, window=5, consecutive=3, healthy_end_index=100)

    assert result.fired is True
    assert result.onset_index is not None and result.onset_index >= 100, (
        "onset must land in the degradation phase"
    )
    assert result.healthy_false_positive_rate == 0.0
    assert result.degraded_detection_rate > 0.5


def test_stays_quiet_on_an_entirely_healthy_run() -> None:
    """The other half of the result. A detector that always fires is not a detector."""
    rng = np.random.default_rng(3)
    scores = 1.0 + rng.normal(0, 0.05, 300)
    threshold = ReferenceThreshold.fit(scores[:100], z_threshold=3.5)
    result = detect_onset(scores, threshold, window=5, consecutive=3, healthy_end_index=100)
    assert result.fired is False
    assert result.onset_index is None
    assert result.healthy_false_positive_rate == 0.0


def test_onset_timestamp_is_reported_when_supplied() -> None:
    import pandas as pd

    scores, healthy = _series()
    timestamps = pd.date_range("2004-02-12", periods=len(scores), freq="10min")
    threshold = ReferenceThreshold.fit(healthy, z_threshold=3.5)
    result = detect_onset(scores, threshold, timestamps=list(timestamps), healthy_end_index=100)
    assert result.onset_timestamp == timestamps[result.onset_index]


def test_lead_time_is_measured_to_the_end_of_the_run() -> None:
    scores, healthy = _series()
    threshold = ReferenceThreshold.fit(healthy, z_threshold=3.5)
    result = detect_onset(scores, threshold, healthy_end_index=100, snapshot_interval_minutes=10)
    expected = (len(scores) - 1 - result.onset_index) * 10 / 60
    assert result.lead_hours == pytest.approx(expected, abs=0.01)


def test_empty_series_raises() -> None:
    threshold = ReferenceThreshold.fit(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="empty"):
        detect_onset(np.array([]), threshold)


def test_result_serialises() -> None:
    import json

    scores, healthy = _series()
    threshold = ReferenceThreshold.fit(healthy)
    result = detect_onset(scores, threshold, healthy_end_index=100)
    json.dumps(result.to_dict())


# --------------------------------------------------------------- lead time


def test_lead_time_is_reported_as_a_range() -> None:
    """A single lead-time number is a property of the baseline window.

    For a ramp there is no physical onset moment, so the sweep across baseline
    choices IS the honest answer.
    """
    scores, _ = _series(n_healthy=120, n_degraded=180)
    lead = sweep_lead_time(
        scores,
        fractions=[0.15, 0.20, 0.25, 0.30, 0.40],
        z_threshold=3.5,
        snapshot_interval_minutes=10,
        max_healthy_fraction=0.40,
    )
    assert lead.n_fired > 1
    assert lead.min_hours is not None and lead.max_hours is not None
    assert lead.max_hours >= lead.min_hours
    assert lead.spread_hours == pytest.approx(lead.max_hours - lead.min_hours)
    assert "RANGE" in lead.to_dict()["note"]


def test_lead_time_actually_moves_with_the_baseline_window() -> None:
    """The claim being made: the onset date is an artifact of the baseline.

    If every baseline gave the same answer, reporting a range would be
    pointless. It does not.
    """
    scores, _ = _series(n_healthy=150, n_degraded=150)
    lead = sweep_lead_time(
        scores,
        fractions=[0.10, 0.20, 0.30, 0.40, 0.50],
        z_threshold=3.5,
        snapshot_interval_minutes=10,
        max_healthy_fraction=0.50,
    )
    onsets = [e["onset_index"] for e in lead.per_fraction if e.get("fired")]
    assert len(set(onsets)) > 1, (
        "every baseline gave the same onset; the range claim needs the spread to be real"
    )


def test_sweep_skips_baselines_that_reach_into_the_degradation() -> None:
    """A reference period containing degraded data is a different measurement."""
    scores, _ = _series(n_healthy=100, n_degraded=100)
    lead = sweep_lead_time(scores, fractions=[0.2, 0.9], z_threshold=3.5, max_healthy_fraction=0.5)
    skipped = [e for e in lead.per_fraction if "skipped" in e]
    assert any(e["fraction"] == 0.9 for e in skipped)


def test_sweep_skips_a_baseline_that_is_too_short() -> None:
    lead = sweep_lead_time(np.arange(10.0), fractions=[0.01], z_threshold=3.5)
    assert lead.per_fraction[0]["skipped"].startswith("reference window shorter")


def test_sweep_reports_no_detection_honestly() -> None:
    rng = np.random.default_rng(9)
    flat = 1.0 + rng.normal(0, 0.05, 300)
    lead = sweep_lead_time(flat, fractions=[0.2, 0.3], z_threshold=3.5)
    assert lead.n_fired == 0
    assert lead.min_hours is None
    assert "no baseline choice produced a detection" in lead.summary()
