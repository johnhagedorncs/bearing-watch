"""Shared fixtures.

The synthetic dataset fixture is session-scoped and small (72 snapshots). It is
generated through the REAL writer and read back through the REAL parser, so any
disagreement between the two fails the suite rather than being discovered later
on data nobody can regenerate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bearingwatch.config import Config
from bearingwatch.data.cache import SCALED_METADATA_KEY, FeatureCache
from bearingwatch.data.ims_parser import list_snapshot_files
from bearingwatch.data.synthetic import SyntheticSpec, generate_run
from bearingwatch.features.pipeline import build_feature_table

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# Small enough to keep the suite fast, long enough to have a healthy window,
# a degradation phase, and room for a 5-sample persistence rule.
N_SNAPSHOTS = 72
HEALTHY_FRACTION = 0.40


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete synthetic run in IMS on-disk format."""
    out = tmp_path_factory.mktemp("ims-synth") / "set-test"
    spec = SyntheticSpec(
        n_snapshots=N_SNAPSHOTS,
        n_channels=4,
        healthy_fraction=HEALTHY_FRACTION,
        seed=7,
    )
    generate_run(out, spec)
    return out


@pytest.fixture(scope="session")
def truth_degradation_index() -> int:
    return int(round(HEALTHY_FRACTION * N_SNAPSHOTS))


@pytest.fixture(scope="session")
def base_config(synthetic_dir: Path) -> Config:
    """Config pointed at the synthetic run, with test-scale thresholds.

    Session-scoped and treated as READ-ONLY. Tests that need a variant call
    ``copy_with``, which returns a new Config rather than mutating this one.
    """
    return Config.load(
        CONFIG_DIR,
        {
            "data.root": str(synthetic_dir),
            "data.min_files": 10,
            "data.expected_channels": 4,
            "data.healthy_fraction": 0.30,
            "tracking.enabled": False,
            # Keep the autoencoder quick; these tests check plumbing and
            # invariants, not convergence.
            "model.autoencoder.epochs": 40,
            "model.autoencoder.patience": 10,
        },
    )


@pytest.fixture(scope="session")
def _feature_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("feature-cache")


@pytest.fixture(scope="session")
def feature_cache(_feature_cache_dir: Path) -> FeatureCache:
    return FeatureCache(_feature_cache_dir)


@pytest.fixture(scope="session")
def feature_frame(base_config: Config, feature_cache: FeatureCache) -> pd.DataFrame:
    """Feature table for the synthetic run. Cached across tests in a session."""
    frame, _ = build_feature_table(base_config, cache=feature_cache)
    return frame


@pytest.fixture(scope="session")
def snapshot_files(synthetic_dir: Path) -> list[Path]:
    return list_snapshot_files(synthetic_dir)


@pytest.fixture
def unscaled_frame() -> pd.DataFrame:
    """A tiny correctly-flagged unscaled feature table."""
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 11.0, 12.0, 13.0]},
    )
    frame.attrs["bearingwatch"] = {SCALED_METADATA_KEY: False}
    return frame


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
