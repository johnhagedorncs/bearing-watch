"""IMS on-disk format: parsing, and the shape contract that must abort."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from bearingwatch.data.ims_parser import (
    IMSFormatError,
    fingerprint_files,
    is_snapshot_filename,
    list_snapshot_files,
    parse_timestamp,
    read_snapshot,
    write_snapshot,
)


def test_parse_timestamp_reads_the_ims_filename_format() -> None:
    assert parse_timestamp("2004.02.12.10.32.39") == datetime(2004, 2, 12, 10, 32, 39)


def test_parse_timestamp_rejects_anything_else() -> None:
    for bad in ("readme.txt", "2004-02-12", "20040212103239", "2004.02.12.10.32"):
        with pytest.raises(IMSFormatError):
            parse_timestamp(bad)


def test_is_snapshot_filename() -> None:
    assert is_snapshot_filename("2004.02.12.10.32.39")
    assert not is_snapshot_filename("README.md")
    assert not is_snapshot_filename("2004.02.12.10.32.39.txt")


def test_round_trip_through_the_real_writer_and_parser(tmp_path: Path) -> None:
    """Generated data must parse with the same code that reads real data."""
    data = np.random.default_rng(0).normal(size=(20480, 4))
    path = write_snapshot(tmp_path / "2004.02.12.10.32.39", data)

    snap = read_snapshot(path, expected_points=20480, expected_channels=4)
    assert snap.n_points == 20480
    assert snap.n_channels == 4
    assert snap.timestamp == datetime(2004, 2, 12, 10, 32, 39)
    np.testing.assert_allclose(snap.data, data, atol=1e-5)


def test_short_file_aborts(tmp_path: Path) -> None:
    """A short file is a corrupt file. It must not be padded or accepted."""
    path = write_snapshot(tmp_path / "2004.02.12.10.32.39", np.zeros((20479, 4)) + 0.5)
    with pytest.raises(IMSFormatError, match="expected exactly 20480 points"):
        read_snapshot(path, expected_points=20480, expected_channels=4)


def test_long_file_aborts(tmp_path: Path) -> None:
    path = write_snapshot(tmp_path / "2004.02.12.10.32.39", np.zeros((20481, 4)) + 0.5)
    with pytest.raises(IMSFormatError, match="expected exactly 20480 points"):
        read_snapshot(path, expected_points=20480, expected_channels=4)


def test_wrong_channel_count_aborts(tmp_path: Path) -> None:
    """Set 1 has 8 channels, sets 2 and 3 have 4. Mixing them must fail."""
    path = write_snapshot(tmp_path / "2004.02.12.10.32.39", np.zeros((20480, 8)) + 0.5)
    with pytest.raises(IMSFormatError, match="expected 4 channels, found 8"):
        read_snapshot(path, expected_points=20480, expected_channels=4)


def test_non_finite_values_abort(tmp_path: Path) -> None:
    data = np.zeros((20480, 4)) + 0.5
    data[17, 2] = np.nan
    path = tmp_path / "2004.02.12.10.32.39"
    np.savetxt(path, data, fmt="%.6f", delimiter="\t")
    with pytest.raises(IMSFormatError, match="non-finite"):
        read_snapshot(path, expected_points=20480, expected_channels=4)


def test_ragged_rows_abort(tmp_path: Path) -> None:
    """A row missing a column is corruption, not a value to impute."""
    path = tmp_path / "2004.02.12.10.32.39"
    lines = ["\t".join(["0.1"] * 4) for _ in range(20480)]
    lines[500] = "\t".join(["0.1"] * 3)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(IMSFormatError):
        read_snapshot(path, expected_points=20480, expected_channels=4)


def test_missing_file_aborts(tmp_path: Path) -> None:
    with pytest.raises(IMSFormatError, match="not found"):
        read_snapshot(tmp_path / "2004.02.12.10.32.39")


def test_listing_sorts_chronologically_and_skips_non_snapshots(tmp_path: Path) -> None:
    for name in ("2004.02.12.11.02.39", "2004.02.12.10.32.39", "2004.02.12.10.42.39"):
        write_snapshot(tmp_path / name, np.zeros((16, 2)) + 0.5)
    (tmp_path / "README.md").write_text("not data")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")

    files = list_snapshot_files(tmp_path)
    assert [p.name for p in files] == [
        "2004.02.12.10.32.39",
        "2004.02.12.10.42.39",
        "2004.02.12.11.02.39",
    ]


def test_listing_a_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IMSFormatError, match="not a directory"):
        list_snapshot_files(tmp_path / "nope")


def test_channel_accessor_bounds(tmp_path: Path) -> None:
    path = write_snapshot(tmp_path / "2004.02.12.10.32.39", np.zeros((32, 4)) + 0.5)
    snap = read_snapshot(path, expected_points=32, expected_channels=4)
    assert snap.channel(3).shape == (32,)
    with pytest.raises(IndexError):
        snap.channel(4)


def test_fingerprint_changes_when_the_file_set_changes(snapshot_files) -> None:
    full = fingerprint_files(snapshot_files)
    partial = fingerprint_files(snapshot_files[:-5])
    assert full["n_files"] != partial["n_files"]
    assert full["digest"] != partial["digest"]


def test_fingerprint_of_nothing_is_well_defined() -> None:
    assert fingerprint_files([])["n_files"] == 0


def test_parser_reads_the_synthetic_fixture(snapshot_files) -> None:
    """The session fixture must be readable by the real parser, unmodified."""
    snap = read_snapshot(snapshot_files[0], expected_points=20480, expected_channels=4)
    assert snap.n_points == 20480
    assert snap.n_channels == 4
    assert np.all(np.isfinite(snap.data))
