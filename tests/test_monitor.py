"""Drift monitoring, and the band-split discriminant that drift alone cannot do."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bearingwatch.config import MonitorConfig
from bearingwatch.monitor.band_split import (
    VERDICT_GAIN,
    VERDICT_MECHANICAL,
    VERDICT_NO_CHANGE,
    BandSplitDiscriminant,
    BandSplitError,
    band_columns,
    level_and_shape,
)
from bearingwatch.monitor.drift import compute_drift, evidently_available, ks_statistic, psi

# -------------------------------------------------------------------- psi/ks


def test_psi_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert psi(a, b) < 0.1


def test_psi_rises_with_a_shift() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 5000)
    small = psi(reference, rng.normal(0.3, 1, 5000))
    large = psi(reference, rng.normal(3.0, 1, 5000))
    assert large > small > 0
    assert large > 0.25, "a 3-sigma shift should clear the conventional alert line"


def test_psi_handles_a_constant_reference() -> None:
    constant = np.full(100, 5.0)
    assert psi(constant, np.full(50, 5.0)) == 0.0
    assert psi(constant, np.full(50, 9.0)) == 1.0


def test_psi_handles_degenerate_inputs() -> None:
    assert psi(np.array([1.0]), np.array([1.0])) == 0.0
    assert psi(np.array([]), np.array([1.0, 2.0])) == 0.0


def test_ks_detects_a_shift() -> None:
    rng = np.random.default_rng(2)
    _, p_same = ks_statistic(rng.normal(size=500), rng.normal(size=500))
    _, p_shifted = ks_statistic(rng.normal(size=500), rng.normal(3.0, 1, 500))
    assert p_same > 0.01
    assert p_shifted < 1e-6


# --------------------------------------------------------------------- drift


def _frame(values: np.ndarray, n_cols: int = 6) -> pd.DataFrame:
    return pd.DataFrame({f"f{i}": values[:, i] for i in range(n_cols)})


def test_no_drift_between_two_healthy_windows(feature_frame, base_config) -> None:
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    half = n // 2
    report = compute_drift(feature_frame.iloc[:half], feature_frame.iloc[half:n])
    assert report.fired is False, "two halves of the healthy window must not read as drift"


def test_drift_fires_on_the_degraded_phase(feature_frame, base_config) -> None:
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    report = compute_drift(feature_frame.iloc[:n], feature_frame.tail(15))
    assert report.fired is True
    assert report.drifted_fraction > 0.3
    assert report.max_psi > 0.25


def test_drift_report_includes_score_drift() -> None:
    rng = np.random.default_rng(3)
    reference = _frame(rng.normal(size=(200, 6)))
    current = _frame(rng.normal(size=(60, 6)))
    report = compute_drift(
        reference,
        current,
        reference_scores=rng.normal(0, 1, 200),
        current_scores=rng.normal(5, 1, 60),
    )
    assert report.score_drift is not None
    assert report.score_drift.drifted is True


def test_drift_report_serialises(feature_frame, base_config) -> None:
    import json

    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    report = compute_drift(feature_frame.iloc[:n], feature_frame.tail(15))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["n_features"] > 0
    assert len(payload["top_features"]) <= 5


def test_drift_needs_shared_columns() -> None:
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="no shared numeric columns"):
        compute_drift(a, b)


def test_drift_threshold_is_configurable() -> None:
    """Only some features drift, so the firing fraction is what decides."""
    rng = np.random.default_rng(4)
    reference = _frame(rng.normal(size=(300, 6)))

    current_values = rng.normal(size=(120, 6))
    current_values[:, :2] += 3.0  # two of six features move
    current = _frame(current_values)

    strict = compute_drift(reference, current, MonitorConfig(drifted_feature_fraction=0.1))
    lenient = compute_drift(reference, current, MonitorConfig(drifted_feature_fraction=0.9))

    assert 0.2 <= strict.drifted_fraction <= 0.6, "expected roughly a third of features to move"
    assert strict.fired is True
    assert lenient.fired is False


def test_psi_adapts_its_bin_count_to_short_windows() -> None:
    """A fixed bin count on a short window reports drift between identical data.

    With 10 bins and 11 samples each bin holds about one observation, so bin
    membership is noise: two halves of the same stationary series produced PSI
    values of 3-10 against an alert line of 0.25.
    """
    rng = np.random.default_rng(20)
    values = rng.normal(size=24)
    observed = psi(values[:12], values[12:], bins=10)
    assert observed < 0.5, f"two halves of one stationary sample gave PSI={observed:.2f}"


@pytest.mark.optional
def test_evidently_integration_is_optional(feature_frame, base_config) -> None:
    """Must return a structured answer whether or not evidently is installed."""
    from bearingwatch.monitor.drift import evidently_report

    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    columns = [c for c in feature_frame.columns if c.startswith("ch0__band")][:5]
    result = evidently_report(feature_frame.iloc[:n], feature_frame.tail(15), columns=columns)

    assert result["available"] == evidently_available()
    if not result["available"]:
        assert "reason" in result


# ---------------------------------------------------------------- band split


def test_level_and_shape_decomposition() -> None:
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    level, shape = level_and_shape(values)
    np.testing.assert_allclose(level, [2.0, 5.0])
    np.testing.assert_allclose(shape, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    assert np.allclose(shape.mean(axis=1), 0.0), "shape must be mean-free by construction"


def test_shape_is_invariant_under_an_additive_level_shift() -> None:
    """The algebraic core: a gain change is an additive constant in log space."""
    rng = np.random.default_rng(5)
    values = rng.normal(-8, 0.5, size=(20, 30))
    _, shape_a = level_and_shape(values)
    _, shape_b = level_and_shape(values + 1.234)  # a uniform gain change
    np.testing.assert_allclose(shape_a, shape_b, atol=1e-12)


def test_level_and_shape_needs_at_least_two_bands() -> None:
    with pytest.raises(BandSplitError, match="at least 2"):
        level_and_shape(np.array([[1.0]]))


def test_band_columns_excludes_defect_bands(feature_frame) -> None:
    """Defect bands are already total-normalised, so they carry no level."""
    bands = band_columns(feature_frame.columns)
    assert bands
    assert all("__band" in c for c in bands)
    assert not any(name in c for c in bands for name in ("BPFO", "BPFI", "BSF", "FTF"))


def test_healthy_window_classifies_as_no_change(feature_frame, base_config) -> None:
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    half = n // 2
    discriminant = BandSplitDiscriminant.fit(feature_frame.iloc[:half])
    result = discriminant.classify(feature_frame.iloc[half:n])
    assert result.verdict == VERDICT_NO_CHANGE


def test_a_pure_gain_change_is_classified_as_gain(feature_frame, base_config) -> None:
    """THE result: a broadband rescale moves the level and not the shape."""
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    healthy = feature_frame.iloc[:n]
    bands = band_columns(feature_frame.columns)

    discriminant = BandSplitDiscriminant.fit(healthy, band_names=bands)

    # A gain of g multiplies power by g^2, i.e. adds 2*log10(g) to every log
    # band energy. Applied identically to every band, which is what makes it a
    # gain change rather than a spectral change.
    gain = 4.0
    gained = healthy.copy()
    gained[bands] = gained[bands] + 2 * np.log10(gain)

    result = discriminant.classify(gained)
    assert result.verdict == VERDICT_GAIN
    assert result.level_fired is True
    assert result.shape_fired is False
    assert result.implied_gain == pytest.approx(gain, rel=0.05)


def test_real_degradation_is_classified_as_mechanical(feature_frame, base_config) -> None:
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    discriminant = BandSplitDiscriminant.fit(feature_frame.iloc[:n])
    result = discriminant.classify(feature_frame.tail(12))
    assert result.verdict == VERDICT_MECHANICAL
    assert result.shape_fired is True
    assert result.top_shape_bands


def test_gain_and_degradation_are_distinguished_despite_similar_level_shifts(
    feature_frame, base_config
) -> None:
    """The punchline, as an assertion.

    Both cases move the level. Only degradation moves the shape, so only the
    shape statistic separates them.
    """
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    healthy = feature_frame.iloc[:n]
    bands = band_columns(feature_frame.columns)
    discriminant = BandSplitDiscriminant.fit(healthy, band_names=bands)

    degraded = feature_frame.tail(12)
    degraded_result = discriminant.classify(degraded)

    # Construct a gain change producing a comparable LEVEL shift.
    gained = healthy.copy()
    gained[bands] = gained[bands] + degraded_result.level_shift_log10
    gain_result = discriminant.classify(gained)

    assert gain_result.level_z == pytest.approx(degraded_result.level_z, rel=0.25), (
        "the two cases should look similar on level, which is why drift alone conflates them"
    )
    assert gain_result.shape_z < degraded_result.shape_z / 5, (
        "shape must separate them by a wide margin"
    )
    assert gain_result.verdict == VERDICT_GAIN
    assert degraded_result.verdict == VERDICT_MECHANICAL


def test_missing_band_columns_raise(feature_frame, base_config) -> None:
    """A changed band definition voids the reference."""
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    discriminant = BandSplitDiscriminant.fit(feature_frame.iloc[:n])
    with pytest.raises(BandSplitError, match="missing"):
        discriminant.classify(feature_frame.drop(columns=discriminant.band_names[:2]))


def test_unfitted_discriminant_refuses(feature_frame) -> None:
    with pytest.raises(BandSplitError, match="not fitted"):
        BandSplitDiscriminant().classify(feature_frame)


def test_fit_needs_band_columns() -> None:
    with pytest.raises(BandSplitError, match="at least 2 spectral band"):
        BandSplitDiscriminant.fit(pd.DataFrame({"x": [1.0, 2.0]}))


def test_discriminant_round_trips_through_disk(feature_frame, base_config, tmp_path) -> None:
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    original = BandSplitDiscriminant.fit(feature_frame.iloc[:n])
    path = original.save(tmp_path / "band_split.json")

    loaded = BandSplitDiscriminant.load(path)
    a = original.classify(feature_frame.tail(12))
    b = loaded.classify(feature_frame.tail(12))
    assert a.verdict == b.verdict
    assert a.shape_z == pytest.approx(b.shape_z)
    assert a.level_z == pytest.approx(b.level_z)


def test_result_explains_itself(feature_frame, base_config) -> None:
    n = int(round(base_config.data.healthy_fraction * len(feature_frame)))
    discriminant = BandSplitDiscriminant.fit(feature_frame.iloc[:n])
    text = discriminant.classify(feature_frame.tail(12)).explain()
    assert "MECHANICAL" in text or "GAIN" in text or "no change" in text
