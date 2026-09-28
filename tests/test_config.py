"""Configuration loading, validation, overrides, and snapshots."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bearingwatch.config import Config, ConfigError, canonical_json, content_hash

from .conftest import CONFIG_DIR


def test_loads_the_shipped_configs() -> None:
    cfg = Config.load(CONFIG_DIR)
    assert cfg.data.expected_points == 20480
    assert cfg.data.sample_rate_hz == 20000
    assert cfg.features.bearing.n_rolling_elements == 16
    assert cfg.detect.reference == "healthy_window"


def test_shipped_detect_reference_is_the_declared_window() -> None:
    """A regression guard on the most consequential default in the repo."""
    assert Config.load(CONFIG_DIR).detect.reference == "healthy_window"


def test_missing_config_directory_raises() -> None:
    with pytest.raises(ConfigError, match="config directory not found"):
        Config.load("/nonexistent/configs")


def test_directory_without_config_files_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no config files found"):
        Config.load(tmp_path)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    """A typo'd key is otherwise a silent no-op that disagrees with the file."""
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump({"data": {"root": "x", "expcted_points": 20480}})
    )
    with pytest.raises(ConfigError, match="unknown configuration key"):
        Config.load(tmp_path)


def test_dotted_overrides_apply() -> None:
    cfg = Config.load(CONFIG_DIR, {"model.kind": "pca", "detect.z_threshold": 5.0})
    assert cfg.model.kind == "pca"
    assert cfg.detect.z_threshold == 5.0


def test_string_overrides_are_coerced() -> None:
    """CLI values arrive as strings and must become the right types."""
    cfg = Config.load(
        CONFIG_DIR,
        {
            "detect.z_threshold": "4.5",
            "model.standardise": "false",
            "features.spectral.n_bands": "48",
            "serve.run_dir": "null",
        },
    )
    assert cfg.detect.z_threshold == 4.5
    assert cfg.model.standardise is False
    assert cfg.features.spectral.n_bands == 48
    assert cfg.serve.run_dir is None


def test_copy_with_does_not_mutate_the_original() -> None:
    cfg = Config.load(CONFIG_DIR)
    original = cfg.model.kind
    other = cfg.copy_with({"model.kind": "kurtosis"})
    assert other.model.kind == "kurtosis"
    assert cfg.model.kind == original


def test_environment_overrides_apply(monkeypatch) -> None:
    """Containers configure by environment, so that path is first-class."""
    monkeypatch.setenv("BEARINGWATCH_CFG__detect__z_threshold", "6.5")
    monkeypatch.setenv("BEARINGWATCH_CFG__model__kind", "pca")
    cfg = Config.load(CONFIG_DIR)
    assert cfg.detect.z_threshold == 6.5
    assert cfg.model.kind == "pca"


def test_explicit_overrides_beat_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("BEARINGWATCH_CFG__detect__z_threshold", "6.5")
    cfg = Config.load(CONFIG_DIR, {"detect.z_threshold": 2.0})
    assert cfg.detect.z_threshold == 2.0


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "override,message",
    [
        ({"data.healthy_fraction": 1.5}, "healthy_fraction"),
        ({"data.healthy_fraction": 0.0}, "healthy_fraction"),
        ({"data.expected_points": 0}, "expected_points"),
        ({"data.target_channel": 99}, "target_channel"),
        ({"data.sample_rate_hz": 0}, "sample_rate_hz"),
        ({"model.kind": "randomforest"}, "model.kind"),
        ({"detect.reference": "rolling"}, "detect.reference"),
        ({"detect.z_threshold": -1.0}, "z_threshold"),
        ({"features.spectral.n_bands": 8}, "n_bands"),
        ({"features.spectral.n_bands": 128}, "n_bands"),
        ({"features.channels": "some"}, "features.channels"),
        ({"features.bearing.n_rolling_elements": 1}, "n_rolling_elements"),
        ({"features.bearing.ball_diameter_in": 99.0}, "ball_diameter_in"),
        ({"features.bearing.shaft_rpm": 0}, "shaft_rpm"),
    ],
)
def test_invalid_values_are_rejected(override: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Config.load(CONFIG_DIR, override)


def test_persistence_consecutive_cannot_exceed_window() -> None:
    with pytest.raises(ConfigError, match="consecutive cannot exceed"):
        Config.load(CONFIG_DIR, {"detect.persistence.consecutive": 99})


def test_noverlap_must_be_less_than_nperseg() -> None:
    with pytest.raises(ConfigError, match="noverlap must be < nperseg"):
        Config.load(CONFIG_DIR, {"features.spectral.noverlap": 99999})


def test_at_least_one_feature_family_must_be_enabled() -> None:
    with pytest.raises(ConfigError, match="at least one of"):
        Config.load(
            CONFIG_DIR,
            {"features.time_domain.enabled": False, "features.spectral.enabled": False},
        )


# -------------------------------------------------------------------- hashes


def test_canonical_json_is_order_independent() -> None:
    """A hash that depends on dict ordering is not a content hash."""
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_content_hash_is_deterministic_across_calls() -> None:
    payload = {"z": [3, 2, 1], "a": {"nested": True}}
    assert content_hash(payload) == content_hash(payload)


def test_config_hash_changes_with_any_setting() -> None:
    cfg = Config.load(CONFIG_DIR)
    assert cfg.copy_with({"serve.port": 9999}).config_hash != cfg.config_hash


# ------------------------------------------------------------------ snapshot


def test_snapshot_round_trips_exactly(tmp_path: Path) -> None:
    """Every run folder gets a snapshot; it must reload to the same config."""
    cfg = Config.load(CONFIG_DIR, {"model.kind": "pca", "detect.z_threshold": 4.25})
    path = cfg.dump_yaml(tmp_path / "config.snapshot.yaml")

    with path.open() as fh:
        reloaded = Config.from_dict(yaml.safe_load(fh))

    assert reloaded.config_hash == cfg.config_hash
    assert reloaded.feature_cache_key == cfg.feature_cache_key
    assert reloaded.model.kind == "pca"
    assert reloaded.detect.z_threshold == 4.25


def test_to_dict_is_json_serialisable() -> None:
    import json

    json.dumps(Config.load(CONFIG_DIR).to_dict())
