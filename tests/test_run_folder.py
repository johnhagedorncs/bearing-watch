"""Run-folder discipline: one self-describing directory, loaded by path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bearingwatch.runs.manifest import MANIFEST_NAME, Manifest
from bearingwatch.runs.run_folder import (
    CONFIG_SNAPSHOT,
    RUN_ID_RE,
    RunFolder,
    RunFolderError,
    latest_run,
    list_runs,
    new_run_id,
)


def test_run_id_is_sortable_and_carries_the_config_hash(base_config) -> None:
    run_id = new_run_id(base_config)
    assert RUN_ID_RE.match(run_id)
    assert run_id.endswith(base_config.config_hash[:8])


def test_create_writes_a_config_snapshot(base_config, tmp_path: Path) -> None:
    run, manifest = RunFolder.create(tmp_path, base_config)
    assert (run.path / CONFIG_SNAPSHOT).is_file()
    assert manifest.config_hash == base_config.config_hash
    assert manifest.status == "running"


def test_run_config_comes_from_the_snapshot_not_from_configs_on_disk(
    base_config, tmp_path: Path
) -> None:
    """Inference must follow the run it is serving, not the current defaults."""
    run, manifest = RunFolder.create(tmp_path, base_config)
    manifest.finalise("complete").save(run.path)

    loaded = RunFolder.open(run.path).config
    assert loaded.config_hash == base_config.config_hash
    assert loaded.features.spectral.n_bands == base_config.features.spectral.n_bands


def test_opening_a_folder_without_a_manifest_refuses(base_config, tmp_path: Path) -> None:
    """The manifest is written LAST, so its absence means the run crashed."""
    run, _ = RunFolder.create(tmp_path, base_config)
    with pytest.raises(RunFolderError, match="did not finish"):
        RunFolder.open(run.path)


def test_opening_a_failed_run_refuses(base_config, tmp_path: Path) -> None:
    run, manifest = RunFolder.create(tmp_path, base_config)
    manifest.finalise("failed").save(run.path)
    with pytest.raises(RunFolderError, match="not 'complete'"):
        RunFolder.open(run.path)


def test_incomplete_folder_can_be_opened_deliberately(base_config, tmp_path: Path) -> None:
    run, _ = RunFolder.create(tmp_path, base_config)
    assert RunFolder.open(run.path, require_complete=False).run_id == run.run_id


def test_run_folders_are_never_overwritten(base_config, tmp_path: Path) -> None:
    """Immutability is what keeps an old result explainable."""
    RunFolder.create(tmp_path, base_config, run_id="20240101T000000Z-abcdef01")
    with pytest.raises(RunFolderError, match="already exists"):
        RunFolder.create(tmp_path, base_config, run_id="20240101T000000Z-abcdef01")


def test_opening_a_missing_folder_gives_a_useful_message(tmp_path: Path) -> None:
    with pytest.raises(RunFolderError, match="bearingwatch train"):
        RunFolder.open(tmp_path / "nope")


def test_scaler_round_trips_without_pickle(base_config, tmp_path: Path) -> None:
    """A pickled estimator is version-locked and executes code on load."""
    from sklearn.preprocessing import StandardScaler

    run, _ = RunFolder.create(tmp_path, base_config)
    data = np.random.default_rng(0).normal(size=(50, 6))
    original = StandardScaler().fit(data)
    names = [f"f{i}" for i in range(6)]

    path = run.save_scaler(original, names)
    assert path.suffix == ".npz", "the scaler artifact must not be a pickle"

    loaded, loaded_names = run.load_scaler()
    assert loaded_names == names
    np.testing.assert_allclose(loaded.transform(data), original.transform(data))


def test_loading_a_missing_scaler_raises(base_config, tmp_path: Path) -> None:
    run, _ = RunFolder.create(tmp_path, base_config)
    with pytest.raises(RunFolderError, match="scaler"):
        run.load_scaler()


def test_metrics_round_trip(base_config, tmp_path: Path) -> None:
    run, _ = RunFolder.create(tmp_path, base_config)
    run.write_metrics({"a": 1, "nested": {"b": 2.5}})
    assert run.read_metrics()["nested"]["b"] == 2.5


def test_manifest_records_provenance(base_config, tmp_path: Path) -> None:
    run, manifest = RunFolder.create(tmp_path, base_config, command="bearingwatch train")
    manifest.finalise("complete").save(run.path)

    loaded = Manifest.load(run.path)
    assert loaded.run_id == run.run_id
    assert loaded.command == "bearingwatch train"
    assert loaded.status == "complete"
    assert loaded.completed_at is not None
    assert "package_version" in loaded.code_version
    assert loaded.environment["python"]


def test_manifest_records_output_paths_relatively(base_config, tmp_path: Path) -> None:
    """Relative paths keep a run folder portable across mounts and copies."""
    run, manifest = RunFolder.create(tmp_path, base_config)
    manifest.add_output("model", run.path / "model.pt", run.path)
    assert manifest.outputs["model"] == "model.pt"


def test_manifest_missing_raises_with_an_explanation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="INCOMPLETE run"):
        Manifest.load(tmp_path)


def test_manifest_ignores_unknown_fields_from_a_future_version(base_config, tmp_path: Path) -> None:
    """Forward compatibility: a newer manifest must still load."""
    run, manifest = RunFolder.create(tmp_path, base_config)
    manifest.finalise("complete").save(run.path)

    path = run.path / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["a_field_from_the_future"] = {"nested": True}
    path.write_text(json.dumps(payload))

    assert Manifest.load(run.path).status == "complete"


def test_list_and_latest_run(base_config, tmp_path: Path) -> None:
    ids = ["20240101T000000Z-aaaaaaaa", "20240102T000000Z-bbbbbbbb"]
    for run_id in ids:
        run, manifest = RunFolder.create(tmp_path, base_config, run_id=run_id)
        manifest.finalise("complete").save(run.path)

    found = list_runs(tmp_path)
    assert [f.run_id for f in found] == list(reversed(ids)), "newest first"
    assert latest_run(tmp_path).run_id == ids[-1]


def test_latest_run_skips_incomplete_folders(base_config, tmp_path: Path) -> None:
    good, manifest = RunFolder.create(tmp_path, base_config, run_id="20240101T000000Z-aaaaaaaa")
    manifest.finalise("complete").save(good.path)
    RunFolder.create(tmp_path, base_config, run_id="20240102T000000Z-bbbbbbbb")  # no manifest

    assert latest_run(tmp_path).run_id == "20240101T000000Z-aaaaaaaa"


def test_latest_run_with_nothing_present(tmp_path: Path) -> None:
    with pytest.raises(RunFolderError, match="no complete run folder"):
        latest_run(tmp_path)


def test_list_runs_ignores_stray_directories(base_config, tmp_path: Path) -> None:
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / ".hidden").mkdir()
    run, manifest = RunFolder.create(tmp_path, base_config, run_id="20240101T000000Z-aaaaaaaa")
    manifest.finalise("complete").save(run.path)
    assert [f.run_id for f in list_runs(tmp_path)] == ["20240101T000000Z-aaaaaaaa"]


def test_describe_summarises_a_run(base_config, tmp_path: Path) -> None:
    run, manifest = RunFolder.create(tmp_path, base_config)
    manifest.finalise("complete").save(run.path)
    described = run.describe()
    assert described["run_id"] == run.run_id
    assert described["status"] == "complete"
    assert described["config_hash"] == base_config.config_hash
