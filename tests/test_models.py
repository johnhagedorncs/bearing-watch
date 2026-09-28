"""Model interface, persistence, and the shared score convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bearingwatch.features.pipeline import feature_columns
from bearingwatch.models.base import ModelError, build_model, load_model

ALL_KINDS = ["autoencoder", "pca", "kurtosis"]


@pytest.fixture
def training_matrix(feature_frame, base_config):
    """Scaled features and the healthy split, shared by the model tests."""
    from sklearn.preprocessing import StandardScaler

    columns = feature_columns(feature_frame)
    raw = feature_frame[columns].to_numpy(dtype=np.float64)
    n_healthy = int(round(base_config.data.healthy_fraction * len(raw)))
    scaler = StandardScaler().fit(raw[:n_healthy])
    return scaler.transform(raw), columns, n_healthy


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_fit_and_score_returns_one_score_per_row(kind, base_config, training_matrix) -> None:
    x, columns, n_healthy = training_matrix
    model = build_model(kind, base_config.model)
    model.fit(x[:n_healthy], columns)
    scores = model.score(x)
    assert scores.shape == (x.shape[0],)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_higher_is_more_anomalous(kind, base_config, training_matrix) -> None:
    """The shared score convention.

    A sign flip in one model is invisible in a comparison table -- it just looks
    like that model performs badly -- so it is asserted for all three.
    """
    x, columns, n_healthy = training_matrix
    model = build_model(kind, base_config.model)
    model.fit(x[:n_healthy], columns)
    scores = model.score(x)
    assert np.median(scores[-20:]) > np.median(scores[:n_healthy]), (
        f"{kind}: degraded snapshots must score HIGHER than healthy ones"
    )


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_save_and_load_round_trip_preserves_scores(
    kind, base_config, training_matrix, tmp_path: Path
) -> None:
    """A reloaded model must score identically. Otherwise a run folder lies."""
    x, columns, n_healthy = training_matrix
    model = build_model(kind, base_config.model)
    model.fit(x[:n_healthy], columns)
    before = model.score(x)

    written = model.save(tmp_path)
    assert written and all(p.is_file() for p in written)

    reloaded = load_model(kind, tmp_path)
    after = reloaded.score(x)

    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-8)
    assert reloaded.feature_names_ == model.feature_names_


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_scoring_before_fitting_raises(kind, base_config) -> None:
    model = build_model(kind, base_config.model)
    with pytest.raises(ModelError, match="not fitted"):
        model.score(np.zeros((2, 3)))


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_feature_width_mismatch_raises(kind, base_config, training_matrix) -> None:
    """Train and score must agree on width, or an ablation silently mismatches."""
    x, columns, n_healthy = training_matrix
    model = build_model(kind, base_config.model)
    model.fit(x[:n_healthy], columns)
    with pytest.raises(ModelError, match="fitted on"):
        model.score(x[:, :-3])


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_non_finite_input_raises_rather_than_producing_nan_scores(
    kind, base_config, training_matrix
) -> None:
    """A NaN score compares False against every threshold, so the detector goes quiet."""
    x, columns, n_healthy = training_matrix
    model = build_model(kind, base_config.model)
    model.fit(x[:n_healthy], columns)

    bad = x[:5].copy()
    bad[2, 1] = np.nan
    with pytest.raises(ModelError, match="non-finite"):
        model.score(bad)


def test_fit_rejects_a_single_sample(base_config) -> None:
    model = build_model("pca", base_config.model)
    with pytest.raises(ModelError, match="too short"):
        model.fit(np.zeros((1, 5)), [f"f{i}" for i in range(5)])


def test_fit_rejects_mismatched_feature_names(base_config) -> None:
    model = build_model("pca", base_config.model)
    with pytest.raises(ModelError, match="feature_names"):
        model.fit(np.zeros((10, 5)), ["a", "b"])


def test_unknown_model_kind_raises(base_config) -> None:
    with pytest.raises(ModelError, match="unknown model kind"):
        build_model("transformer", base_config.model)


def test_pca_reports_its_component_count(base_config, training_matrix) -> None:
    x, columns, n_healthy = training_matrix
    model = build_model("pca", base_config.model)
    model.fit(x[:n_healthy], columns)
    assert model.n_components_ >= 1
    assert 0.0 < model.train_metrics_["explained_variance"] <= 1.0000001


def test_pca_with_an_integer_component_count(base_config, training_matrix) -> None:
    x, columns, n_healthy = training_matrix
    cfg = base_config.copy_with({"model.pca.n_components": 5})
    model = build_model("pca", cfg.model)
    model.fit(x[:n_healthy], columns)
    assert model.n_components_ == 5


def test_autoencoder_early_stops_and_records_it(base_config, training_matrix) -> None:
    x, columns, n_healthy = training_matrix
    cfg = base_config.copy_with({"model.autoencoder.epochs": 500, "model.autoencoder.patience": 3})
    model = build_model("autoencoder", cfg.model)
    model.fit(x[:n_healthy], columns)
    assert model.train_metrics_["epochs_run"] < 500
    assert model.train_metrics_["early_stopped"] == 1.0


def test_autoencoder_is_deterministic_for_a_fixed_seed(base_config, training_matrix) -> None:
    x, columns, n_healthy = training_matrix
    scores = []
    for _ in range(2):
        model = build_model("autoencoder", base_config.model)
        model.fit(x[:n_healthy], columns)
        scores.append(model.score(x))
    np.testing.assert_allclose(scores[0], scores[1], rtol=1e-5, atol=1e-7)


def test_kurtosis_baseline_score_is_signed_not_clipped(base_config, training_matrix) -> None:
    """Clipping at zero puts a point mass on half the healthy window.

    That degenerates the downstream MAD and produced a 42% false-positive rate
    on declared-healthy data before it was fixed. The one-sidedness belongs in
    the detector, which already only fires upward.
    """
    x, columns, n_healthy = training_matrix
    model = build_model("kurtosis", base_config.model)
    model.fit(x[:n_healthy], columns)
    healthy_scores = model.score(x[:n_healthy])

    assert (healthy_scores < 0).any(), "healthy scores must be able to go negative"
    n_exactly_zero = int(np.sum(healthy_scores == 0.0))
    assert n_exactly_zero < 0.1 * len(healthy_scores), (
        "a large point mass at exactly zero is the degenerate shape being avoided"
    )


def test_kurtosis_baseline_needs_a_kurtosis_feature(base_config) -> None:
    model = build_model("kurtosis", base_config.model)
    with pytest.raises(ModelError, match="kurtosis"):
        model.fit(np.zeros((10, 3)), ["a", "b", "c"])


def test_loading_from_an_empty_directory_raises(tmp_path: Path) -> None:
    """Inference loads a run folder, not a loose file. Missing pieces must fail."""
    for kind in ALL_KINDS:
        with pytest.raises(ModelError, match="missing"):
            load_model(kind, tmp_path)


def test_model_metadata_is_serialisable(base_config, training_matrix) -> None:
    import json

    x, columns, n_healthy = training_matrix
    model = build_model("pca", base_config.model)
    model.fit(x[:n_healthy], columns)
    json.dumps(model.metadata())
