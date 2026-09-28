"""Feature extraction: time domain, spectral bands, and defect frequencies."""

from __future__ import annotations

import numpy as np
import pytest

from bearingwatch.config import BearingGeometry, SpectralConfig
from bearingwatch.features.defect_freqs import (
    band_energy,
    compute_defect_bands,
    defect_frequencies,
)
from bearingwatch.features.pipeline import (
    apply_ablation,
    constant_features,
    extract_snapshot_features,
    feature_columns,
)
from bearingwatch.features.spectral import (
    band_energies,
    compute_spectral,
    log_spaced_edges,
    welch_psd,
)
from bearingwatch.features.time_domain import compute_time_domain

# ---------------------------------------------------------------- time domain


def test_rms_of_a_known_sine() -> None:
    t = np.arange(20000) / 20000.0
    x = 2.0 * np.sin(2 * np.pi * 50 * t)
    assert compute_time_domain(x, ["rms"])["rms"] == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)


def test_kurtosis_is_pearson_not_excess() -> None:
    """Gaussian noise must give ~3.0, not ~0.0.

    Papers use both conventions and the difference silently shifts any threshold
    copied between them, so the convention is asserted rather than assumed.
    """
    x = np.random.default_rng(0).normal(size=200000)
    assert compute_time_domain(x, ["kurtosis"])["kurtosis"] == pytest.approx(3.0, abs=0.1)


def test_kurtosis_rises_with_impulsive_content() -> None:
    rng = np.random.default_rng(1)
    smooth = rng.normal(size=20000)
    impulsive = rng.normal(size=20000)
    impulsive[::500] += 30.0
    k_smooth = compute_time_domain(smooth, ["kurtosis"])["kurtosis"]
    k_impulsive = compute_time_domain(impulsive, ["kurtosis"])["kurtosis"]
    assert k_impulsive > k_smooth + 1.0


def test_crest_factor_of_a_sine_is_root_two() -> None:
    t = np.arange(20000) / 20000.0
    x = np.sin(2 * np.pi * 100 * t)
    assert compute_time_domain(x, ["crest_factor"])["crest_factor"] == pytest.approx(
        np.sqrt(2), rel=1e-2
    )


def test_all_time_domain_stats_are_finite() -> None:
    x = np.random.default_rng(2).normal(size=4096)
    values = compute_time_domain(x)
    assert len(values) == 8
    assert all(np.isfinite(v) for v in values.values())


def test_constant_signal_does_not_divide_by_zero() -> None:
    values = compute_time_domain(np.zeros(1024))
    assert all(np.isfinite(v) for v in values.values())


def test_unknown_stat_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown time-domain stat"):
        compute_time_domain(np.ones(10), ["not_a_stat"])


def test_empty_signal_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty signal"):
        compute_time_domain(np.array([]))


# ------------------------------------------------------------------- spectral


def test_welch_finds_a_known_tone() -> None:
    fs = 20000
    t = np.arange(20480) / fs
    x = np.sin(2 * np.pi * 1234.0 * t)
    freqs, psd = welch_psd(x, fs, SpectralConfig())
    assert freqs[int(np.argmax(psd))] == pytest.approx(1234.0, abs=10.0)


def test_log_spaced_edges_are_monotone_and_span_the_range() -> None:
    edges = log_spaced_edges(10.0, 10000.0, 40)
    assert len(edges) == 41
    assert edges[0] == pytest.approx(10.0)
    assert edges[-1] == pytest.approx(10000.0)
    assert np.all(np.diff(edges) > 0)
    # Log spacing means constant RATIOS, not constant differences.
    ratios = edges[1:] / edges[:-1]
    assert np.allclose(ratios, ratios[0])


def test_log_spaced_edges_reject_bad_ranges() -> None:
    with pytest.raises(ValueError):
        log_spaced_edges(0.0, 100.0, 10)
    with pytest.raises(ValueError):
        log_spaced_edges(100.0, 10.0, 10)


def test_band_energies_never_return_a_constant_floor_for_narrow_bands() -> None:
    """A band narrower than the resolution must estimate, not return zero.

    Returning zero is the silent-failure shape: after a log it becomes a
    constant, and a constant feature is invisible everywhere downstream.
    """
    freqs = np.arange(0, 10000, 100.0)
    psd = np.ones_like(freqs)
    edges = np.array([1000.0, 1000.5, 1001.0])  # both bands far narrower than 100 Hz
    energies = band_energies(freqs, psd, edges, log_energy=False)
    assert np.all(energies > 0), "narrow bands must not collapse to zero"


def test_compute_spectral_returns_the_configured_number_of_bands() -> None:
    cfg = SpectralConfig(n_bands=40)
    x = np.random.default_rng(3).normal(size=20480)
    features, freqs, psd = compute_spectral(x, 20000, cfg)
    assert len(features) == 40
    assert freqs.shape == psd.shape


def test_band_energy_rises_where_energy_was_added() -> None:
    fs = 20000
    t = np.arange(20480) / fs
    rng = np.random.default_rng(4)
    quiet = rng.normal(size=20480)
    loud = quiet + 5.0 * np.sin(2 * np.pi * 3000.0 * t)

    cfg = SpectralConfig()
    f_quiet, _, _ = compute_spectral(quiet, fs, cfg)
    f_loud, _, _ = compute_spectral(loud, fs, cfg)

    target = [k for k in f_quiet if "band" in k]
    rises = [f_loud[k] - f_quiet[k] for k in target]
    assert max(rises) > 1.0, "a strong 3 kHz tone must raise at least one band"


# ------------------------------------------------------------ defect freqs


def test_defect_frequencies_match_the_published_za2115_values() -> None:
    """Sanity check on the shipped geometry.

    These are the values the config comment tells the reader to expect. If a
    geometry edit moves them, the geometry is wrong -- the formulas are not.
    """
    f = defect_frequencies(BearingGeometry())
    assert f.bpfo == pytest.approx(236.4, abs=0.5)
    assert f.bpfi == pytest.approx(296.9, abs=0.5)
    assert f.bsf == pytest.approx(139.9, abs=0.5)
    assert f.ftf == pytest.approx(14.8, abs=0.2)


def test_bpfo_is_exactly_n_times_ftf() -> None:
    """An algebraic identity of the formulas, so it must hold exactly."""
    geom = BearingGeometry()
    f = defect_frequencies(geom)
    assert f.bpfo == pytest.approx(geom.n_rolling_elements * f.ftf, rel=1e-12)


def test_bpfi_exceeds_bpfo() -> None:
    f = defect_frequencies(BearingGeometry())
    assert f.bpfi > f.bpfo


def test_defect_frequencies_scale_linearly_with_shaft_speed() -> None:
    slow = defect_frequencies(BearingGeometry(shaft_rpm=1000))
    fast = defect_frequencies(BearingGeometry(shaft_rpm=2000))
    assert fast.bpfo == pytest.approx(2 * slow.bpfo)
    assert fast.ftf == pytest.approx(2 * slow.ftf)


def test_invalid_geometry_is_rejected() -> None:
    from bearingwatch.config import ConfigError

    with pytest.raises(ConfigError):
        defect_frequencies(BearingGeometry(ball_diameter_in=5.0, pitch_diameter_in=2.815))
    with pytest.raises(ConfigError):
        defect_frequencies(BearingGeometry(n_rolling_elements=1))


def test_band_energy_with_a_single_bin_is_not_zero() -> None:
    """np.trapz of one point is 0.0 -- a clean, plausible, wrong answer.

    This is how every FTF and BSF_h1 feature came out constant for a whole run
    without anything raising.
    """
    freqs = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    psd = np.array([1.0, 1.0, 1.0, 7.0, 1.0])
    energy = band_energy(freqs, psd, centre_hz=15.0, rel_bandwidth=0.02, min_bandwidth_hz=5.0)
    assert energy > 0.0
    assert energy == pytest.approx(7.0 * 5.0)


def test_band_energy_with_no_bin_uses_the_nearest() -> None:
    freqs = np.array([0.0, 100.0, 200.0])
    psd = np.array([1.0, 4.0, 1.0])
    energy = band_energy(freqs, psd, centre_hz=101.0, rel_bandwidth=0.001, min_bandwidth_hz=0.5)
    assert energy > 0.0


def test_defect_bands_are_gain_invariant_when_normalised() -> None:
    """The property the band-split discriminant depends on.

    Multiplying the signal by g scales every PSD value by g^2, which cancels
    when each band is divided by the total.
    """
    fs = 20000
    rng = np.random.default_rng(5)
    x = rng.normal(size=20480)
    cfg = SpectralConfig()

    freqs, psd = welch_psd(x, fs, cfg)
    freqs_g, psd_g = welch_psd(3.0 * x, fs, cfg)

    geom = BearingGeometry()
    a = compute_defect_bands(freqs, psd, geom, normalise_by_total=True, log_energy=True)
    b = compute_defect_bands(freqs_g, psd_g, geom, normalise_by_total=True, log_energy=True)

    for key in a:
        if key == "total_energy":
            continue  # the level is SUPPOSED to move
        assert a[key] == pytest.approx(b[key], abs=1e-6), f"{key} was not gain-invariant"

    # ...and the total energy moved by exactly g^2.
    assert b["total_energy"] - a["total_energy"] == pytest.approx(np.log10(9.0), abs=1e-6)


def test_defect_bands_above_nyquist_keep_the_vector_width() -> None:
    """Feature width must not depend on shaft speed."""
    freqs = np.linspace(0, 1000, 500)
    psd = np.ones_like(freqs)
    geom = BearingGeometry(shaft_rpm=20000)  # pushes harmonics above the range
    bands = compute_defect_bands(freqs, psd, geom, harmonics=3)
    assert len(bands) == 4 * 3 + 1


# -------------------------------------------------------------------- pipeline


def test_snapshot_features_have_channel_prefixes(base_config, snapshot_files) -> None:
    from bearingwatch.data.ims_parser import read_snapshot

    snap = read_snapshot(snapshot_files[0], 20480, 4)
    features = extract_snapshot_features(snap, base_config)
    assert all("__" in name for name in features)
    assert any(name.startswith("ch0__") for name in features)


def test_target_channel_mode_produces_fewer_features_than_all(base_config, snapshot_files) -> None:
    from bearingwatch.data.ims_parser import read_snapshot

    snap = read_snapshot(snapshot_files[0], 20480, 4)
    target = extract_snapshot_features(snap, base_config)
    every = extract_snapshot_features(snap, base_config.copy_with({"features.channels": "all"}))
    assert len(every) > len(target)


def test_ablation_drops_the_named_families(base_config, snapshot_files) -> None:
    from bearingwatch.data.ims_parser import read_snapshot

    snap = read_snapshot(snapshot_files[0], 20480, 4)
    full = extract_snapshot_features(snap, base_config)

    no_kurt = apply_ablation(full, base_config.copy_with({"features.ablation.drop_kurtosis": True}))
    assert not any(k.endswith("__kurtosis") for k in no_kurt)
    assert len(no_kurt) < len(full)

    no_defect = apply_ablation(
        full, base_config.copy_with({"features.ablation.drop_defect_bands": True})
    )
    assert not any("BPFO" in k for k in no_defect)


def test_ablation_preserves_shared_values_exactly(base_config, snapshot_files) -> None:
    """Ablation drops columns; it does not recompute them.

    Recomputing would let an unrelated change sneak into the comparison.
    """
    from bearingwatch.data.ims_parser import read_snapshot

    snap = read_snapshot(snapshot_files[0], 20480, 4)
    full = extract_snapshot_features(snap, base_config)
    ablated = apply_ablation(full, base_config.copy_with({"features.ablation.drop_kurtosis": True}))
    for key, value in ablated.items():
        assert value == full[key]


def test_feature_table_has_no_constant_columns(feature_frame) -> None:
    """A feature that never moves is an extraction bug, not a quiet sensor."""
    columns = feature_columns(feature_frame)
    dead = constant_features(feature_frame, columns)
    assert dead == [], f"constant feature(s) present: {dead}"


def test_feature_table_is_chronological_and_complete(feature_frame, snapshot_files) -> None:
    assert len(feature_frame) == len(snapshot_files)
    assert feature_frame.index.is_monotonic_increasing
    assert "source_file" in feature_frame.columns
    assert feature_frame["snapshot_index"].tolist() == list(range(len(snapshot_files)))


def test_degradation_raises_rms_and_kurtosis(feature_frame, truth_degradation_index) -> None:
    """The synthetic fixture must actually degrade, or nothing else is testable."""
    t = truth_degradation_index
    healthy = feature_frame.iloc[:t]
    degraded = feature_frame.iloc[t:]
    assert degraded["ch0__rms"].median() > healthy["ch0__rms"].median()
    assert degraded["ch0__kurtosis"].max() > healthy["ch0__kurtosis"].max()


def test_healthy_channels_stay_flat(feature_frame, truth_degradation_index) -> None:
    """The unfaulted channels are negative controls and must not move much."""
    t = truth_degradation_index
    for channel in ("ch1", "ch2", "ch3"):
        healthy = feature_frame.iloc[:t][f"{channel}__rms"].median()
        later = feature_frame.iloc[t:][f"{channel}__rms"].median()
        assert abs(later - healthy) / healthy < 0.05, f"{channel} moved; it should not have"
