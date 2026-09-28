"""FastAPI service, exercised through TestClient against a real trained run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bearingwatch.serve import app as app_module
from bearingwatch.serve.deps import load_run, set_loaded_run


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, request) -> Path:
    """Train one real run folder (PCA, for speed) and hand back its path."""
    from sklearn.preprocessing import StandardScaler

    from bearingwatch.detect.reference_threshold import ReferenceThreshold
    from bearingwatch.features.pipeline import build_feature_table, feature_columns
    from bearingwatch.models.base import build_model
    from bearingwatch.monitor.band_split import BandSplitDiscriminant, band_columns
    from bearingwatch.runs.run_folder import REFERENCE_FILE, THRESHOLD_FILE, RunFolder
    from bearingwatch.serve.deps import BAND_SPLIT_FILE

    cfg = request.getfixturevalue("base_config").copy_with({"model.kind": "pca"})
    frame, _ = build_feature_table(cfg, cache=request.getfixturevalue("feature_cache"))
    columns = feature_columns(frame)
    n_healthy = int(round(cfg.data.healthy_fraction * len(frame)))
    healthy = frame.iloc[:n_healthy]

    runs_root = tmp_path_factory.mktemp("runs")
    run, manifest = RunFolder.create(runs_root, cfg, command="test")

    scaler = StandardScaler().fit(healthy[columns].to_numpy(dtype=np.float64))
    run.save_scaler(scaler, columns)

    model = build_model("pca", cfg.model)
    x_healthy = scaler.transform(healthy[columns].to_numpy(dtype=np.float64))
    model.fit(x_healthy, columns)
    model.save(run.path)

    ReferenceThreshold.fit(model.score(x_healthy), z_threshold=cfg.detect.z_threshold).save(
        run.file(THRESHOLD_FILE)
    )
    BandSplitDiscriminant.fit(healthy, band_names=band_columns(columns)).save(
        run.file(BAND_SPLIT_FILE)
    )
    run.write_table(REFERENCE_FILE, healthy[columns].copy())
    run.write_metrics({"n_healthy": n_healthy})

    manifest.model_kind = "pca"
    manifest.finalise("complete").save(run.path)
    return run.path


@pytest.fixture
def client(trained_run: Path):
    """A TestClient with the trained run loaded.

    The run is loaded explicitly rather than through the startup environment
    variable so the test does not depend on process-wide state leaking in.
    """
    set_loaded_run(load_run(trained_run))
    with TestClient(app_module.app) as test_client:
        set_loaded_run(load_run(trained_run))  # lifespan may have cleared it
        yield test_client
    set_loaded_run(None)


@pytest.fixture
def feature_vector(client) -> list[float]:
    names = client.get("/model").json()["feature_names"]
    return [0.0] * len(names)


# ------------------------------------------------------------------- ops


def test_health_is_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_health_does_not_depend_on_the_model(trained_run: Path) -> None:
    """A liveness probe that checks the model turns a missing model into a crash loop."""
    set_loaded_run(None)
    with TestClient(app_module.app) as unloaded:
        set_loaded_run(None)
        assert unloaded.get("/health").status_code == 200


def test_ready_reports_the_loaded_run(client, trained_run: Path) -> None:
    payload = client.get("/ready").json()
    assert payload["ready"] is True
    assert payload["run_id"] == trained_run.name


def test_ready_is_false_without_a_model() -> None:
    """Readiness gates traffic; liveness restarts. They cannot be one endpoint."""
    set_loaded_run(None)
    with TestClient(app_module.app) as unloaded:
        set_loaded_run(None)
        payload = unloaded.get("/ready").json()
        assert payload["ready"] is False
        assert payload["reason"]


# ----------------------------------------------------------------- model


def test_model_endpoint_reports_provenance(client, trained_run: Path) -> None:
    payload = client.get("/model").json()
    assert payload["run_id"] == trained_run.name
    assert payload["model_kind"] == "pca"
    assert payload["model_version"]
    assert payload["n_features"] == len(payload["feature_names"])
    assert payload["expected_points"] == 20480
    assert payload["expected_channels"] == 4
    assert "git_commit" in payload["code_version"]


def test_model_endpoint_publishes_the_feature_order(client) -> None:
    """A permuted feature vector is the right length and scores a valid number.

    The server cannot detect it, so the order has to be published.
    """
    names = client.get("/model").json()["feature_names"]
    assert names == sorted(set(names), key=names.index), "no duplicate feature names"
    assert all(isinstance(n, str) and "__" in n for n in names)


# ----------------------------------------------------------------- score


def test_score_accepts_a_feature_vector(client, feature_vector) -> None:
    response = client.post("/score", json={"features": [feature_vector]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_scored"] == 1
    assert len(payload["results"]) == 1
    item = payload["results"][0]
    assert set(item) == {
        "score",
        "z",
        "threshold_z",
        "score_threshold",
        "anomalous",
        "verdict",
    }
    assert item["verdict"] in ("normal", "anomalous")


def test_score_response_carries_the_model_identity(client, feature_vector) -> None:
    """A score is only interpretable alongside the model that produced it."""
    payload = client.post("/score", json={"features": [feature_vector]}).json()
    assert payload["run_id"]
    assert payload["model_kind"] == "pca"
    assert payload["model_version"]
    assert payload["n_features"] == len(feature_vector)


def test_score_echoes_the_snapshot_id(client, feature_vector) -> None:
    payload = client.post(
        "/score", json={"features": [feature_vector], "snapshot_id": "rig-a-001"}
    ).json()
    assert payload["snapshot_id"] == "rig-a-001"


def test_score_accepts_a_batch(client, feature_vector) -> None:
    batch = [feature_vector for _ in range(6)]
    payload = client.post("/score", json={"features": batch}).json()
    assert payload["n_scored"] == 6


def test_score_returns_a_band_split_verdict_for_a_batch(client, feature_frame) -> None:
    """The gain-vs-mechanical discriminant needs a window, not one snapshot."""
    names = client.get("/model").json()["feature_names"]
    rows = feature_frame[names].head(8).to_numpy().tolist()
    payload = client.post("/score", json={"features": rows}).json()
    assert payload["band_split"] is not None
    assert payload["band_split"]["verdict"] in (
        "no_change",
        "gain_change",
        "mechanical_change",
        "ambiguous",
    )


def test_score_accepts_a_raw_snapshot(client) -> None:
    rng = np.random.default_rng(0)
    snapshot = (rng.normal(0, 0.09, size=(20480, 4))).tolist()
    response = client.post("/score", json={"snapshot": snapshot})
    assert response.status_code == 200
    assert response.json()["n_scored"] == 1


def test_score_rejects_a_snapshot_of_the_wrong_length(client) -> None:
    """A short record changes the spectral resolution and every band with it."""
    snapshot = np.zeros((1024, 4)).tolist()
    response = client.post("/score", json={"snapshot": snapshot})
    assert response.status_code == 400
    assert "20480" in response.json()["detail"]


def test_score_rejects_a_snapshot_with_the_wrong_channel_count(client) -> None:
    snapshot = np.zeros((20480, 2)).tolist()
    response = client.post("/score", json={"snapshot": snapshot})
    assert response.status_code == 400
    assert "channel" in response.json()["detail"]


def test_score_rejects_the_wrong_feature_width(client) -> None:
    response = client.post("/score", json={"features": [[1.0, 2.0, 3.0]]})
    assert response.status_code == 400


def test_score_rejects_both_snapshot_and_features(client, feature_vector) -> None:
    """Silently preferring one turns a caller's bug into a wrong number."""
    response = client.post(
        "/score", json={"features": [feature_vector], "snapshot": [[0.0] * 4] * 20480}
    )
    assert response.status_code == 422


def test_score_rejects_neither_supplied(client) -> None:
    assert client.post("/score", json={}).status_code == 422


def test_score_rejects_an_oversized_batch(client, feature_vector) -> None:
    batch = [feature_vector for _ in range(200)]
    response = client.post("/score", json={"features": batch})
    assert response.status_code == 400
    assert "max_batch" in response.json()["detail"]


def test_degraded_snapshots_score_higher_than_healthy_ones(client, feature_frame) -> None:
    """End-to-end through HTTP, not just through the library."""
    names = client.get("/model").json()["feature_names"]
    healthy = feature_frame[names].head(5).to_numpy().tolist()
    degraded = feature_frame[names].tail(5).to_numpy().tolist()

    healthy_scores = [
        r["score"] for r in client.post("/score", json={"features": healthy}).json()["results"]
    ]
    degraded_scores = [
        r["score"] for r in client.post("/score", json={"features": degraded}).json()["results"]
    ]
    assert np.median(degraded_scores) > np.median(healthy_scores)


def test_degraded_snapshots_produce_an_anomalous_verdict(client, feature_frame) -> None:
    names = client.get("/model").json()["feature_names"]
    degraded = feature_frame[names].tail(5).to_numpy().tolist()
    results = client.post("/score", json={"features": degraded}).json()["results"]
    assert any(r["anomalous"] for r in results)


# ---------------------------------------------------------------- reload


def test_reload_swaps_the_run(client, trained_run: Path) -> None:
    response = client.post("/reload", json={"run_dir": str(trained_run)})
    assert response.status_code == 200
    assert response.json()["run_id"] == trained_run.name


def test_reload_rejects_a_bad_path(client, tmp_path: Path) -> None:
    response = client.post("/reload", json={"run_dir": str(tmp_path / "nope")})
    assert response.status_code == 400


def test_endpoints_return_503_when_no_model_is_loaded() -> None:
    set_loaded_run(None)
    with TestClient(app_module.app, raise_server_exceptions=False) as unloaded:
        set_loaded_run(None)
        assert unloaded.get("/model").status_code == 500
