"""Preflight gates: exists, density, format. Each must FAIL, not warn."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from bearingwatch.config import DataConfig
from bearingwatch.data.ims_parser import write_snapshot
from bearingwatch.data.preflight import PreflightError, run_preflight


def _cfg(root: Path, **kwargs) -> DataConfig:
    defaults = {
        "root": str(root),
        "expected_points": 64,
        "sample_rate_hz": 20000,
        "expected_channels": 4,
        "min_files": 5,
        "snapshot_interval_minutes": 10,
        "target_channel": 0,
    }
    defaults.update(kwargs)
    return DataConfig(**defaults)


def _write_run(
    root: Path, n: int = 10, points: int = 64, channels: int = 4, interval_min: int = 10
) -> None:
    start = datetime(2004, 2, 12, 10, 32, 39)
    rng = np.random.default_rng(0)
    for i in range(n):
        name = (start + timedelta(minutes=interval_min * i)).strftime("%Y.%m.%d.%H.%M.%S")
        write_snapshot(root / name, rng.normal(size=(points, channels)))


def test_passes_on_a_well_formed_run(tmp_path: Path) -> None:
    _write_run(tmp_path)
    report = run_preflight(_cfg(tmp_path))
    assert report.passed is True
    assert report.n_files == 10
    assert report.n_points == 64
    assert report.n_channels == 4
    assert report.median_interval_minutes == pytest.approx(10.0)


def test_missing_directory_fails_with_a_pointer_to_the_generator(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="make_synthetic_data"):
        run_preflight(_cfg(tmp_path / "absent"))


def test_empty_directory_fails(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("nothing here")
    with pytest.raises(PreflightError, match="no IMS snapshot files"):
        run_preflight(_cfg(tmp_path))


def test_too_few_files_fails(tmp_path: Path) -> None:
    _write_run(tmp_path, n=3)
    with pytest.raises(PreflightError, match=r"below data\.min_files"):
        run_preflight(_cfg(tmp_path, min_files=5))


def test_wrong_expected_file_count_fails(tmp_path: Path) -> None:
    """A partial download scores fine and means nothing."""
    _write_run(tmp_path, n=10)
    with pytest.raises(PreflightError, match="expected_files"):
        run_preflight(_cfg(tmp_path, expected_files=984))


def test_correct_expected_file_count_passes(tmp_path: Path) -> None:
    _write_run(tmp_path, n=10)
    assert run_preflight(_cfg(tmp_path, expected_files=10)).passed


def test_sparse_cadence_fails_the_density_gate(tmp_path: Path) -> None:
    """A run whose intervals are mostly gaps is not the run the config claims."""
    _write_run(tmp_path, n=10, interval_min=120)
    with pytest.raises(PreflightError, match=r"exceed 1\.5x the declared cadence"):
        run_preflight(_cfg(tmp_path, snapshot_interval_minutes=10))


def test_density_gate_tolerates_occasional_gaps(tmp_path: Path) -> None:
    """Real runs have some gaps; the gate is a fraction, not an absolute."""
    start = datetime(2004, 2, 12, 10, 32, 39)
    rng = np.random.default_rng(0)
    offsets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 300]  # one big gap out of nine
    for minutes in offsets:
        name = (start + timedelta(minutes=minutes)).strftime("%Y.%m.%d.%H.%M.%S")
        write_snapshot(tmp_path / name, rng.normal(size=(64, 4)))
    report = run_preflight(_cfg(tmp_path, max_interval_gap_fraction=0.25))
    assert report.passed
    assert 0 < (report.gap_fraction or 0) <= 0.25


def test_wrong_point_count_fails_the_format_gate(tmp_path: Path) -> None:
    _write_run(tmp_path, n=10, points=32)
    with pytest.raises(PreflightError, match="expected exactly 64 points"):
        run_preflight(_cfg(tmp_path, expected_points=64))


def test_wrong_channel_count_fails_the_format_gate(tmp_path: Path) -> None:
    _write_run(tmp_path, n=10, channels=8)
    with pytest.raises(PreflightError, match="expected 4 channels"):
        run_preflight(_cfg(tmp_path, expected_channels=4))


def test_format_gate_samples_across_the_run_not_just_the_head(tmp_path: Path) -> None:
    """A download that failed part way through has a perfectly good first file."""
    _write_run(tmp_path, n=10)
    files = sorted(p for p in tmp_path.iterdir() if p.is_file())
    write_snapshot(files[-1], np.zeros((32, 4)) + 0.5)  # corrupt the LAST one

    with pytest.raises(PreflightError, match="expected exactly 64 points"):
        run_preflight(_cfg(tmp_path), sample_files=5)


def test_target_channel_out_of_range_fails(tmp_path: Path) -> None:
    _write_run(tmp_path, n=10, channels=4)
    cfg = _cfg(tmp_path)
    cfg.target_channel = 7  # bypass DataConfig.validate to reach the preflight gate
    with pytest.raises(PreflightError, match="target_channel"):
        run_preflight(cfg)


def test_report_serialises_for_the_manifest(tmp_path: Path) -> None:
    _write_run(tmp_path)
    payload = run_preflight(_cfg(tmp_path)).to_dict()
    assert payload["passed"] is True
    assert payload["n_files"] == 10
    assert "median_interval_minutes" in payload
