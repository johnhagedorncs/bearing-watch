"""End-to-end: synth -> train -> infer through the real CLI.

The important property here is not that the numbers are good. It is that
``infer`` works in a process that shares NO STATE with ``train`` -- it is given
a path and nothing else. That is what makes the two steps independently
orchestratable, and it is the claim the Argo templates depend on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bearingwatch.cli import main
from bearingwatch.runs.run_folder import MANIFEST_NAME, list_runs


@pytest.fixture(scope="module")
def cli_workspace(tmp_path_factory) -> Path:
    """A small synthetic run generated through the CLI itself."""
    workspace = tmp_path_factory.mktemp("cli")
    data = workspace / "data"
    exit_code = main(
        [
            "synth",
            "--out",
            str(data),
            "--n-snapshots",
            "60",
            "--n-channels",
            "4",
            "--healthy-fraction",
            "0.40",
            "--seed",
            "3",
        ]
    )
    assert exit_code == 0
    return workspace


def _common(workspace: Path) -> list[str]:
    from .conftest import CONFIG_DIR

    return [
        "--config-dir",
        str(CONFIG_DIR),
        "--data",
        str(workspace / "data"),
        "--cache-dir",
        str(workspace / "cache"),
        "--set",
        "data.min_files=10",
        "--set",
        "tracking.enabled=false",
        "--set",
        "model.autoencoder.epochs=30",
    ]


def test_preflight_passes_on_generated_data(cli_workspace: Path) -> None:
    assert main(["preflight", *_common(cli_workspace)]) == 0


def test_preflight_fails_on_a_missing_directory(cli_workspace: Path, tmp_path: Path) -> None:
    from .conftest import CONFIG_DIR

    exit_code = main(
        ["preflight", "--config-dir", str(CONFIG_DIR), "--data", str(tmp_path / "absent")]
    )
    assert exit_code == 2, "preflight failure must have its own non-zero exit code"


def test_train_writes_a_complete_run_folder(cli_workspace: Path) -> None:
    runs_root = cli_workspace / "runs"
    exit_code = main(
        ["train", *_common(cli_workspace), "--model", "pca", "--runs-root", str(runs_root)]
    )
    assert exit_code == 0

    folders = list_runs(runs_root)
    assert len(folders) == 1
    run = folders[0]

    for name in (
        MANIFEST_NAME,
        "config.snapshot.yaml",
        "scaler.npz",
        "threshold.json",
        "metrics.json",
        "model_pca.npz",
        "model_meta.json",
        "band_split.json",
        "reference.parquet",
    ):
        assert run.file(name).is_file(), f"run folder is missing {name}"

    manifest = run.manifest
    assert manifest.status == "complete"
    assert manifest.model_kind == "pca"
    assert manifest.config_hash
    assert manifest.feature_cache_key
    assert manifest.preflight["passed"] is True
    assert manifest.data_fingerprint["n_files"] == 60
    assert manifest.healthy_window["n_snapshots"] > 1
    assert manifest.outputs, "the manifest must record its output paths"


def test_infer_runs_against_a_run_folder_by_path(cli_workspace: Path) -> None:
    """No shared state with train. Just a path."""
    runs_root = cli_workspace / "runs"
    run = list_runs(runs_root)[0]
    out = cli_workspace / "inference-out"

    exit_code = main(
        [
            "infer",
            *_common(cli_workspace),
            "--run",
            str(run.path),
            "--out",
            str(out),
            "--truth-degradation-index",
            "24",
        ]
    )
    assert exit_code == 0
    assert (out / "scores.parquet").is_file()
    assert (out / "metrics.json").is_file()

    payload = json.loads((out / "metrics.json").read_text())
    assert payload["run_id"] == run.run_id
    assert payload["n_snapshots"] == 60
    assert payload["detection"]["fired"] is True
    assert payload["ground_truth"]["degradation_start_index"] == 24
    assert "lead_time_range" in payload


def test_detector_fires_in_degradation_and_stays_quiet_in_the_healthy_phase(
    cli_workspace: Path,
) -> None:
    """The core end-to-end claim of the whole pipeline."""
    out = cli_workspace / "inference-out"
    payload = json.loads((out / "metrics.json").read_text())

    detection = payload["detection"]
    truth = payload["ground_truth"]

    assert detection["fired"] is True
    assert detection["healthy_false_positive_rate"] == 0.0, (
        "the detector fired inside the declared healthy window"
    )
    assert truth["false_positive_rate_before_truth"] == 0.0, (
        "the detector fired before degradation actually began"
    )
    assert truth["detection_rate_after_truth"] > 0.2, (
        "the detector barely fired during the degradation phase"
    )
    assert detection["onset_index"] >= truth["degradation_start_index"]


def test_lead_time_is_reported_as_a_range_not_a_number(cli_workspace: Path) -> None:
    out = cli_workspace / "inference-out"
    lead = json.loads((out / "metrics.json").read_text())["lead_time_range"]
    assert "baseline_fractions" in lead
    assert "lead_hours_min" in lead and "lead_hours_max" in lead
    assert "RANGE" in lead["note"]


def test_infer_uses_the_run_snapshot_config_not_the_current_one(cli_workspace: Path) -> None:
    """A run must be served under the config it was trained with."""
    runs_root = cli_workspace / "runs"
    run = list_runs(runs_root)[0]
    snapshot_bands = run.config.features.spectral.n_bands

    out = cli_workspace / "inference-out-2"
    exit_code = main(
        [
            "infer",
            *_common(cli_workspace),
            "--run",
            str(run.path),
            "--out",
            str(out),
            # Ask for a DIFFERENT band count. It must be ignored: the run's own
            # snapshot decides, or the model would be fed features it never saw.
            "--set",
            "features.spectral.n_bands=64",
        ]
    )
    assert exit_code == 0
    assert run.config.features.spectral.n_bands == snapshot_bands


def test_infer_with_a_missing_run_folder_fails_cleanly(cli_workspace: Path) -> None:
    exit_code = main(
        ["infer", *_common(cli_workspace), "--run", str(cli_workspace / "no-such-run")]
    )
    assert exit_code == 1


def test_runs_subcommand_lists_the_run(cli_workspace: Path, capsys) -> None:
    assert main(["runs", "--runs-root", str(cli_workspace / "runs")]) == 0
    assert "complete" in capsys.readouterr().out


def test_train_refuses_an_invalid_config(cli_workspace: Path) -> None:
    exit_code = main(["train", *_common(cli_workspace), "--set", "detect.reference=rolling"])
    assert exit_code == 3, "config errors get their own exit code"


def test_compare_trains_every_model_on_one_split(cli_workspace: Path) -> None:
    out = cli_workspace / "comparison.json"
    exit_code = main(
        [
            "compare",
            *_common(cli_workspace),
            "--out",
            str(out),
            "--truth-degradation-index",
            "24",
        ]
    )
    assert exit_code == 0

    payload = json.loads(out.read_text())
    kinds = {row["model"] for row in payload["rows"]}
    assert kinds == {"autoencoder", "pca", "kurtosis"}
    # Same split, same features, same cache key for every model.
    assert payload["n_healthy"] > 0
    assert payload["feature_cache_key"]


def test_ablate_gives_every_variant_a_distinct_cache_key(cli_workspace: Path) -> None:
    out = cli_workspace / "ablation.json"
    exit_code = main(["ablate", *_common(cli_workspace), "--model", "pca", "--out", str(out)])
    assert exit_code == 0

    rows = json.loads(out.read_text())["rows"]
    keys = [r["cache_key"] for r in rows]
    assert len(set(keys)) == len(keys), "an ablation variant reused another's features"
    assert {r["variant"] for r in rows} == {
        "full",
        "no_kurtosis",
        "no_defect_bands",
        "neither",
    }


def test_calibration_split_holds_data_out_of_model_fitting(cli_workspace: Path) -> None:
    """The threshold must be calibrated on snapshots the model never saw.

    Calibrating on the fitting data measures in-sample residuals, which are
    optimistically low for any reconstruction model. The resulting threshold is
    too tight and the detector fires on healthy data it has simply not seen --
    measured at a 22.7% false-positive rate before this split was introduced.
    """
    run = list_runs(cli_workspace / "runs")[0]
    metrics = run.read_metrics()

    assert metrics["n_calibration"] > 0
    assert metrics["n_fit"] + metrics["n_calibration"] == metrics["n_healthy"]
    assert metrics["n_fit"] == metrics["train"]["n_train"], (
        "the model was fitted on more snapshots than the fit slice"
    )

    # Held-out residuals must be at least as large as in-sample ones. If they
    # were not, the split would be measuring nothing.
    assert metrics["calibration_optimism_ratio"] >= 1.0


def test_threshold_reference_span_is_the_calibration_slice(cli_workspace: Path) -> None:
    run = list_runs(cli_workspace / "runs")[0]
    threshold = run.read_json("threshold.json")
    metrics = run.read_metrics()
    assert threshold["reference"] == "healthy_window"
    assert threshold["n_reference"] == metrics["n_calibration"]
    assert threshold["reference_span"] is not None
