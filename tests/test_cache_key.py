"""Feature-cache key correctness.

The property under test: EVERY parameter that changes feature output must change
the cache key. If one does not, a run silently scores features computed under
different settings -- no error, no crash, just numbers that are not what anyone
thinks they are.

These tests perturb each feature parameter one at a time and assert the key
moves, then assert that a moved key actually triggers a recompute rather than a
cache hit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bearingwatch.config import Config
from bearingwatch.data.cache import SCALED_METADATA_KEY, FeatureCache
from bearingwatch.features.pipeline import build_feature_table

# Every parameter that can move a feature value. Each entry is a dotted config
# path and a value different from the shipped default.
PARAMETERS_THAT_CHANGE_OUTPUT = [
    ("features.schema_version", 99),
    ("features.channels", "all"),
    ("features.spectral.nperseg", 8192),
    ("features.spectral.noverlap", 1024),
    ("features.spectral.window", "hamming"),
    ("features.spectral.detrend", "linear"),
    ("features.spectral.n_bands", 64),
    ("features.spectral.f_min_hz", 20.0),
    ("features.spectral.f_max_hz", 8000.0),
    ("features.spectral.log_energy", False),
    ("features.spectral.log_eps", 1e-15),
    ("features.defect_bands.enabled", False),
    ("features.defect_bands.harmonics", 5),
    ("features.defect_bands.rel_bandwidth", 0.05),
    ("features.defect_bands.min_bandwidth_hz", 10.0),
    ("features.defect_bands.log_energy", False),
    ("features.defect_bands.normalise_by_total", False),
    ("features.time_domain.enabled", False),
    ("features.bearing.n_rolling_elements", 12),
    ("features.bearing.ball_diameter_in", 0.4),
    ("features.bearing.pitch_diameter_in", 3.0),
    ("features.bearing.contact_angle_deg", 20.0),
    ("features.bearing.shaft_rpm", 1800.0),
    ("features.ablation.drop_kurtosis", True),
    ("features.ablation.drop_defect_bands", True),
    # Acquisition parameters change the spectrum just as surely as the Welch
    # settings do, so they belong in the key even though they live under `data`.
    ("data.sample_rate_hz", 25000),
    ("data.expected_points", 40960),
]


@pytest.mark.parametrize("dotted,value", PARAMETERS_THAT_CHANGE_OUTPUT)
def test_changing_any_feature_parameter_changes_the_cache_key(
    base_config: Config, dotted: str, value: object
) -> None:
    """One parameter changed -> a different key. No exceptions."""
    changed = base_config.copy_with({dotted: value})
    assert changed.feature_cache_key != base_config.feature_cache_key, (
        f"changing {dotted} did not change the feature cache key. A run with this "
        "parameter altered would read stale features out of the cache and report "
        "them as a result for the new setting."
    )


def test_cache_key_is_stable_across_identical_configs(base_config: Config) -> None:
    """Same config, same key. Otherwise the cache never hits."""
    other = Config.from_dict(base_config.to_dict())
    assert other.feature_cache_key == base_config.feature_cache_key


def test_cache_key_ignores_parameters_that_cannot_change_features(
    base_config: Config,
) -> None:
    """Serving and model settings must NOT bust the feature cache.

    An over-broad key is not harmless: it makes every model experiment
    recompute every feature, which is slow enough that people disable the cache
    and lose the correctness property along with the speed.
    """
    for dotted, value in [
        ("model.kind", "pca"),
        ("model.autoencoder.epochs", 3),
        ("detect.z_threshold", 9.0),
        ("serve.port", 9999),
        ("tracking.experiment_name", "something-else"),
    ]:
        changed = base_config.copy_with({dotted: value})
        assert changed.feature_cache_key == base_config.feature_cache_key, (
            f"{dotted} changed the FEATURE cache key but cannot change feature values"
        )


def test_config_hash_does_change_for_model_parameters(base_config: Config) -> None:
    """The full config hash is broader than the feature key, and must move."""
    changed = base_config.copy_with({"model.kind": "pca"})
    assert changed.config_hash != base_config.config_hash


def test_changed_key_triggers_recompute_not_a_stale_hit(
    base_config: Config, tmp_path: Path
) -> None:
    """The end-to-end property: a changed parameter must RECOMPUTE.

    Asserting the key moved is not enough on its own -- the cache also has to
    act on it. This runs the real extraction path twice and checks the hit flags
    and the resulting feature widths.
    """
    cache = FeatureCache(tmp_path / "cache")

    first, entry_first = build_feature_table(base_config, cache=cache)
    assert entry_first.hit is False, "first extraction should be a miss"

    again, entry_again = build_feature_table(base_config, cache=cache)
    assert entry_again.hit is True, "identical config should hit the cache"
    assert entry_again.key == entry_first.key
    pd.testing.assert_frame_equal(first, again, check_like=True)

    # Now change one feature parameter. This must NOT hit.
    changed_cfg = base_config.copy_with({"features.spectral.n_bands": 64})
    changed, entry_changed = build_feature_table(changed_cfg, cache=cache)

    assert entry_changed.key != entry_first.key
    assert entry_changed.hit is False, (
        "changing n_bands hit the cache. The run would have been scored on 40-band "
        "features while reporting a 64-band configuration."
    )
    assert entry_changed.path != entry_first.path
    assert len(changed.columns) != len(first.columns), (
        "the recomputed table should actually differ, not just live at a new key"
    )


def test_ablation_variants_all_get_distinct_keys(base_config: Config) -> None:
    """No two ablation variants may share a cache key.

    A collision here means an ablated run reads full-feature values and the
    ablation reports no effect -- which is indistinguishable from the feature
    genuinely not mattering.
    """
    variants = {
        "full": {},
        "no_kurtosis": {"features.ablation.drop_kurtosis": True},
        "no_defect_bands": {"features.ablation.drop_defect_bands": True},
        "neither": {
            "features.ablation.drop_kurtosis": True,
            "features.ablation.drop_defect_bands": True,
        },
    }
    keys = {
        name: (base_config.copy_with(ov) if ov else base_config).feature_cache_key
        for name, ov in variants.items()
    }
    assert len(set(keys.values())) == len(keys), f"cache key collision among variants: {keys}"


def test_data_fingerprint_is_part_of_the_key(base_config: Config, tmp_path: Path) -> None:
    """Two different datasets must not share a feature cache entry."""
    fingerprint_a = {"n_files": 10, "digest": "aaaa"}
    fingerprint_b = {"n_files": 10, "digest": "bbbb"}
    key_a = FeatureCache.compute_key(base_config, fingerprint_a)
    key_b = FeatureCache.compute_key(base_config, fingerprint_b)
    assert key_a != key_b


def test_cached_table_is_written_unscaled(base_config: Config, tmp_path: Path) -> None:
    """Cache entries always carry scaled=False. See test_double_scaling.py."""
    cache = FeatureCache(tmp_path / "cache")
    frame, entry = build_feature_table(base_config, cache=cache)
    meta = FeatureCache.read_metadata(entry.path)
    assert meta[SCALED_METADATA_KEY] is False
    assert frame.attrs["bearingwatch"][SCALED_METADATA_KEY] is False
