"""The synthetic generator: format fidelity and the degradation it claims to model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sp_signal

from bearingwatch.data.ims_parser import list_snapshot_files, read_snapshot
from bearingwatch.data.synthetic import (
    SyntheticSpec,
    generate_run,
    generate_snapshot_signal,
    severity_at,
)
from bearingwatch.features.time_domain import compute_time_domain


def test_healthy_phase_severity_is_exactly_zero() -> None:
    """Not "small" -- zero.

    The healthy window is the detector's declared stationary reference period.
    If it contains a little bit of the fault, it is not a reference period.
    """
    spec = SyntheticSpec(n_snapshots=100, healthy_fraction=0.4)
    assert all(severity_at(i, spec) == 0.0 for i in range(40))
    assert severity_at(40, spec) == 0.0  # boundary: the first degraded index
    assert severity_at(99, spec) > 0.5


def test_severity_is_monotone_through_the_degradation_phase() -> None:
    spec = SyntheticSpec(n_snapshots=100, healthy_fraction=0.4)
    values = [severity_at(i, spec) for i in range(40, 100)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_generated_files_parse_with_the_real_parser(tmp_path: Path) -> None:
    """If the writer and the parser disagree, this fails -- which is the point."""
    spec = SyntheticSpec(n_snapshots=6, n_channels=4)
    summary = generate_run(tmp_path, spec)

    files = list_snapshot_files(tmp_path)
    assert len(files) == 6
    assert summary["n_snapshots"] == 6

    for path in files:
        snap = read_snapshot(path, expected_points=20480, expected_channels=4)
        assert snap.n_points == 20480
        assert snap.n_channels == 4


def test_snapshots_are_spaced_at_the_configured_interval(tmp_path: Path) -> None:
    generate_run(tmp_path, SyntheticSpec(n_snapshots=5, interval_minutes=10))
    files = list_snapshot_files(tmp_path)
    from bearingwatch.data.ims_parser import parse_timestamp

    stamps = [parse_timestamp(p.name) for p in files]
    deltas = {(b - a).total_seconds() / 60 for a, b in zip(stamps, stamps[1:])}
    assert deltas == {10.0}


def test_only_the_faulted_channel_degrades() -> None:
    """The other channels are negative controls and must stay flat."""
    spec = SyntheticSpec(n_snapshots=100, healthy_fraction=0.4, faulted_channel=0)
    rng = np.random.default_rng(0)

    healthy = generate_snapshot_signal(0, spec, rng)
    degraded = generate_snapshot_signal(99, spec, rng)

    rms = lambda x: float(np.sqrt(np.mean(x**2)))  # noqa: E731
    assert rms(degraded[:, 0]) > 1.3 * rms(healthy[:, 0]), "faulted channel must degrade"
    for ch in (1, 2, 3):
        ratio = rms(degraded[:, ch]) / rms(healthy[:, ch])
        assert 0.95 < ratio < 1.05, f"control channel {ch} moved"


def test_kurtosis_rises_on_the_faulted_channel() -> None:
    """A localised defect produces impulsive contact, which raises the 4th moment."""
    spec = SyntheticSpec(n_snapshots=100, healthy_fraction=0.4)
    rng = np.random.default_rng(0)
    healthy = generate_snapshot_signal(0, spec, rng)[:, 0]
    degraded = generate_snapshot_signal(99, spec, rng)[:, 0]

    k_healthy = compute_time_domain(healthy, ["kurtosis"])["kurtosis"]
    k_degraded = compute_time_domain(degraded, ["kurtosis"])["kurtosis"]

    assert k_healthy == pytest.approx(3.0, abs=0.3), "healthy phase should be near-gaussian"
    assert k_degraded > k_healthy + 0.5


def test_the_defect_frequency_appears_in_the_envelope_spectrum() -> None:
    """The impulse train should actually sit at the configured defect rate.

    Checked on the envelope, which is how bearing diagnostics is done: the
    impulses modulate a high-frequency resonance, so the repetition rate shows
    in the amplitude envelope rather than directly in the spectrum.
    """
    spec = SyntheticSpec(n_snapshots=100, healthy_fraction=0.4, defect_freq_hz=236.4)
    rng = np.random.default_rng(0)
    degraded = generate_snapshot_signal(99, spec, rng)[:, 0]

    envelope = np.abs(sp_signal.hilbert(degraded))
    envelope = envelope - envelope.mean()
    freqs, psd = sp_signal.welch(envelope, fs=spec.sample_rate_hz, nperseg=8192)

    band = (freqs > 150) & (freqs < 400)
    peak_hz = float(freqs[band][np.argmax(psd[band])])
    assert peak_hz == pytest.approx(236.4, rel=0.05)


def test_generation_is_reproducible_for_a_fixed_seed() -> None:
    spec = SyntheticSpec(n_snapshots=10, seed=42)
    a = generate_snapshot_signal(5, spec, np.random.default_rng(spec.seed))
    b = generate_snapshot_signal(5, spec, np.random.default_rng(spec.seed))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_data() -> None:
    a = generate_snapshot_signal(5, SyntheticSpec(seed=1), np.random.default_rng(1))
    b = generate_snapshot_signal(5, SyntheticSpec(seed=2), np.random.default_rng(2))
    assert not np.allclose(a, b)


def test_summary_reports_the_degradation_boundary(tmp_path: Path) -> None:
    summary = generate_run(tmp_path, SyntheticSpec(n_snapshots=20, healthy_fraction=0.4))
    assert summary["healthy_end_index"] == 8
    assert summary["first_degraded_index"] == 8
    assert "SYNTHETIC" in summary["note"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_snapshots": 1},
        {"healthy_fraction": 0.0},
        {"healthy_fraction": 1.0},
        {"faulted_channel": 9},
        {"resonance_hz": 50000.0},
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SyntheticSpec(**kwargs).validate()
