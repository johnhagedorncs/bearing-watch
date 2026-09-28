"""The declared reference period, and why a self-referencing MAD cannot work.

The central claim of this repo's detection design, tested by running it: a
robust z-score computed against the series being scored has max|z| pinned near
1.349 on any monotone ramp, INDEPENDENT of the ramp's magnitude. Since bearing
degradation is a ramp, such a detector is blind to exactly what it is for.

The companion project tsad-harness derives the general result. These tests
verify it holds here and that the declared-reference alternative does not have
the defect.
"""

from __future__ import annotations

import numpy as np
import pytest

from bearingwatch.detect.reference_threshold import (
    SELF_REFERENCE_MAX_Z,
    ReferenceThreshold,
    ThresholdError,
    max_abs_z_self_referenced,
    robust_stats,
)


def _ramp(n: int = 400, slope: float = 1.0, base: float = 10.0) -> np.ndarray:
    return base + slope * np.arange(n, dtype=np.float64)


@pytest.mark.parametrize("slope", [1e-3, 1.0, 1e3, 1e6])
def test_self_referenced_max_z_is_pinned_regardless_of_ramp_magnitude(slope: float) -> None:
    """max|z| ~= 1.349 for every slope across nine orders of magnitude."""
    observed = max_abs_z_self_referenced(_ramp(slope=slope))
    assert observed == pytest.approx(SELF_REFERENCE_MAX_Z, rel=0.02), (
        f"slope={slope} gave max|z|={observed:.4f}; the self-referenced statistic is "
        "supposed to be invariant to slope, which is precisely why it cannot detect a ramp"
    )


def test_self_referenced_max_z_never_reaches_a_usable_threshold() -> None:
    """No ramp, at any magnitude, clears a threshold of 3.0 or above."""
    for slope in (1e-6, 1e-2, 1.0, 1e4, 1e9):
        observed = max_abs_z_self_referenced(_ramp(slope=slope))
        assert observed < 3.0, (
            f"slope={slope}: max|z|={observed:.3f}. If this ever exceeded 3 the blindness "
            "claim would be wrong; it does not, at any magnitude."
        )


def test_self_reference_still_fires_on_a_spike() -> None:
    """The failure is specific to ramps, which is why it goes unnoticed.

    A self-referencing detector passes every spike test written for it. That is
    what makes it look like a working detector right up until it is asked to
    find a progressive degradation.
    """
    rng = np.random.default_rng(11)
    values = 10.0 + rng.normal(0, 0.1, 400)
    values[200] = 1000.0
    observed = max_abs_z_self_referenced(values)
    assert observed > 100.0, f"a spike must be detectable; got max|z|={observed:.2f}"
    # ...while the same statistic on a ramp of comparable size stays near 1.35.
    assert max_abs_z_self_referenced(_ramp(slope=2.5)) < 3.0


def test_declared_reference_z_grows_with_ramp_magnitude() -> None:
    """The fix: a fixed reference period makes z scale with the ramp."""
    previous = 0.0
    for slope in (1.0, 10.0, 100.0):
        series = _ramp(slope=slope)
        # Reference = a stationary period, here modelled as the healthy level
        # with noise, NOT the ramp itself.
        reference = 10.0 + np.random.default_rng(0).normal(0, 1.0, 200)
        threshold = ReferenceThreshold.fit(reference, z_threshold=3.5)
        observed = float(np.max(threshold.z(series)))
        assert observed > previous, "z must increase with ramp magnitude"
        previous = observed
    assert previous > 3.5, "the declared-reference detector must actually fire"


def test_declared_reference_detects_a_ramp_the_self_reference_misses() -> None:
    """Side by side, same data, opposite outcomes.

    The scored window here is entirely inside the degradation, which is the
    operational case: a rolling window that has slid past the healthy period.
    That is the precondition for the pinning result.
    """
    rng = np.random.default_rng(1)
    healthy = 1.0 + rng.normal(0, 0.05, 150)
    ramp = 1.0 + np.linspace(0, 4.0, 150) + rng.normal(0, 0.05, 150)

    declared = ReferenceThreshold.fit(healthy, z_threshold=3.5)
    assert declared.exceeds(ramp).any(), "declared reference must detect the ramp"

    self_ref = ReferenceThreshold.fit_self_referenced(ramp, z_threshold=3.5)
    assert not self_ref.exceeds(ramp).any(), (
        "self-reference detected the ramp, which contradicts the pinning result"
    )


def test_self_reference_fires_early_then_goes_silent() -> None:
    """The trap: a half-stationary window fires, a fully-ramped one does not.

    The apparent competence while the window straddles the healthy period is
    what stops anyone investigating before it goes quiet.
    """
    rng = np.random.default_rng(12)
    healthy = 1.0 + rng.normal(0, 0.05, 150)
    ramp = 1.0 + np.linspace(0, 4.0, 150) + rng.normal(0, 0.05, 150)

    straddling = np.concatenate([healthy, ramp])
    fully_degraded = ramp

    assert (
        ReferenceThreshold.fit_self_referenced(straddling, z_threshold=3.5)
        .exceeds(straddling)
        .any()
    ), "the straddling window should fire, which is the misleading part"
    assert (
        not ReferenceThreshold.fit_self_referenced(fully_degraded, z_threshold=3.5)
        .exceeds(fully_degraded)
        .any()
    ), "the fully-degraded window should go silent"


def test_threshold_is_one_sided() -> None:
    """Scores BELOW the healthy median are not anomalies."""
    reference = np.random.default_rng(2).normal(10.0, 1.0, 200)
    threshold = ReferenceThreshold.fit(reference, z_threshold=3.0)
    assert not threshold.exceeds(np.array([-100.0])).any()
    assert threshold.exceeds(np.array([1000.0])).all()


def test_score_threshold_matches_the_z_threshold() -> None:
    reference = np.random.default_rng(3).normal(5.0, 2.0, 300)
    threshold = ReferenceThreshold.fit(reference, z_threshold=3.5)
    at_threshold = threshold.score_threshold
    assert threshold.z(np.array([at_threshold]))[0] == pytest.approx(3.5)


def test_fit_requires_at_least_two_reference_points() -> None:
    with pytest.raises(ThresholdError, match="at least 2"):
        ReferenceThreshold.fit(np.array([1.0]))


def test_fit_rejects_non_finite_reference() -> None:
    with pytest.raises(ThresholdError, match="non-finite"):
        ReferenceThreshold.fit(np.array([1.0, 2.0, np.nan]))


def test_robust_stats_falls_back_when_the_mad_is_set_by_ties() -> None:
    """A point mass at zero must not produce a nanoscale MAD.

    Half the values tied at 0.0 is the normal shape of any one-sided score. The
    naive MAD there is an artifact of the tie, not a scale estimate, and using
    it inflates every z-score by orders of magnitude.
    """
    values = np.concatenate([np.zeros(50), np.linspace(0.01, 3.0, 50)])
    _median, scale = robust_stats(values)
    naive_mad = float(np.median(np.abs(values - np.median(values))))

    assert scale > 100 * naive_mad, "the degenerate MAD should have been rejected"
    assert 0.1 < scale < 10.0, f"fallback scale {scale} is not on the right order"


def test_robust_stats_keeps_the_mad_for_well_behaved_data() -> None:
    """The tie guard must not interfere with ordinary data."""
    values = np.random.default_rng(4).normal(0.0, 1.0, 5000)
    _, scale = robust_stats(values)
    assert scale == pytest.approx(0.674, rel=0.1), "should be the plain MAD of a gaussian"


def test_constant_reference_hits_the_floor_rather_than_dividing_by_zero() -> None:
    _, scale = robust_stats(np.full(100, 3.0), min_mad=1e-9)
    assert scale == pytest.approx(1e-9)


def test_threshold_round_trips_through_disk(tmp_path) -> None:
    reference = np.random.default_rng(5).normal(0.0, 1.0, 200)
    original = ReferenceThreshold.fit(reference, z_threshold=4.0, reference="healthy_window")
    path = original.save(tmp_path / "threshold.json")

    loaded = ReferenceThreshold.load(path)
    assert loaded.median == pytest.approx(original.median)
    assert loaded.mad == pytest.approx(original.mad)
    assert loaded.z_threshold == 4.0
    assert loaded.reference == "healthy_window"

    probe = np.array([0.0, 5.0, 10.0])
    np.testing.assert_allclose(loaded.z(probe), original.z(probe))


def test_unfitted_threshold_refuses_to_score() -> None:
    with pytest.raises(ThresholdError, match="not fitted"):
        ReferenceThreshold().z(np.array([1.0]))
