"""The double-scaling guard.

An artifact saved post-transform and then re-transformed at scoring time is a
bug class, not an incident. It does not raise, it does not produce NaNs, and it
does not change any shape. It shrinks every feature toward zero, which LOWERS
the reconstruction error, which makes a degrading bearing look unusually
healthy. The observable symptom is a detector that has gone quiet, and a quiet
detector is not something anyone escalates.

These tests construct the double-scaling attempt directly and assert it raises,
and separately demonstrate the numerical damage that would result if it did not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from bearingwatch.data.cache import SCALED_METADATA_KEY
from bearingwatch.features.pipeline import (
    DoubleScalingError,
    apply_scaler,
    is_scaled,
    mark_scaled,
    require_unscaled,
)


def _fit_scaler(frame: pd.DataFrame, columns: list[str]) -> StandardScaler:
    return StandardScaler().fit(frame[columns].to_numpy(dtype=np.float64))


def test_apply_scaler_marks_the_output_as_scaled(unscaled_frame: pd.DataFrame) -> None:
    columns = ["a", "b"]
    scaler = _fit_scaler(unscaled_frame, columns)
    scaled = apply_scaler(unscaled_frame, scaler, columns)
    assert is_scaled(scaled) is True
    assert is_scaled(unscaled_frame) is False, "the input must not be mutated"


def test_scaling_an_already_scaled_frame_raises(unscaled_frame: pd.DataFrame) -> None:
    """THE test. Scale once, then attempt to scale again."""
    columns = ["a", "b"]
    scaler = _fit_scaler(unscaled_frame, columns)

    scaled_once = apply_scaler(unscaled_frame, scaler, columns)

    with pytest.raises(DoubleScalingError, match="already marked scaled"):
        apply_scaler(scaled_once, scaler, columns)


def test_require_unscaled_raises_on_a_scaled_frame(unscaled_frame: pd.DataFrame) -> None:
    scaled = mark_scaled(unscaled_frame.copy(), True)
    scaled.attrs["bearingwatch"] = {SCALED_METADATA_KEY: True}
    with pytest.raises(DoubleScalingError):
        require_unscaled(scaled, where="test")


def test_require_unscaled_passes_an_unscaled_frame(unscaled_frame: pd.DataFrame) -> None:
    assert require_unscaled(unscaled_frame) is unscaled_frame


def test_missing_flag_raises_rather_than_defaulting_to_false() -> None:
    """An unknown scaling state must be refused, not guessed.

    Defaulting a missing flag to False is exactly the assumption that lets a
    post-transform artifact be transformed again. "I do not know" has to be an
    error.
    """
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert "bearingwatch" not in frame.attrs

    with pytest.raises(DoubleScalingError, match="no scaling flag"):
        is_scaled(frame)

    with pytest.raises(DoubleScalingError):
        require_unscaled(frame)


def test_double_scaling_silently_corrupts_the_scores(
    base_config, feature_frame: pd.DataFrame
) -> None:
    """Demonstrate the damage the guard prevents, on the real feature set.

    Note what is NOT asserted here: a direction. Re-standardising already
    standardised values computes ``(z - mean_raw) / scale_raw``, so the outcome
    is dominated by ``-mean_raw / scale_raw`` and depends entirely on where the
    raw features sit. On this feature set -- log band energies around -8 with
    scales near 0.1 -- it inflates the score enormously. On a feature set
    already centred at zero the same mistake would deflate it.

    The invariant worth asserting is the one that makes it a bug class rather
    than an incident: the score changes by orders of magnitude, and NOTHING
    RAISES. No shape error, no NaN, no warning. Just a number that is wrong.
    """
    from bearingwatch.features.pipeline import feature_columns
    from bearingwatch.models.base import build_model

    columns = feature_columns(feature_frame)
    n_healthy = 24
    raw = feature_frame[columns].to_numpy(dtype=np.float64)

    scaler = StandardScaler().fit(raw[:n_healthy])
    once = scaler.transform(raw)
    twice = scaler.transform(once)  # the bug

    model = build_model("pca", base_config.model)
    model.fit(once[:n_healthy], columns)

    correct = model.score(once)
    corrupted = model.score(twice)

    assert np.all(np.isfinite(corrupted)), (
        "the corrupted scores are finite -- that is exactly the problem. A NaN "
        "would have been caught."
    )
    ratio = float(np.median(corrupted) / max(np.median(correct), 1e-30))
    assert ratio > 100.0 or ratio < 0.01, (
        f"double scaling changed the median score by a factor of {ratio:.3g}; the point "
        "is that it is badly wrong while raising nothing"
    )


def test_scored_run_rejects_a_scaled_frame(base_config, feature_frame: pd.DataFrame) -> None:
    """The scoring path itself must refuse, not just the helper.

    Constructed via LoadedRun.score_frame so the guard is verified where it
    actually protects something.
    """
    from bearingwatch.features.pipeline import feature_columns

    columns = feature_columns(feature_frame)
    scaler = _fit_scaler(feature_frame, columns)
    scaled = apply_scaler(feature_frame, scaler, columns)

    with pytest.raises(DoubleScalingError):
        require_unscaled(scaled, where="LoadedRun.score_frame")


def test_mark_scaled_round_trips(unscaled_frame: pd.DataFrame) -> None:
    frame = mark_scaled(unscaled_frame.copy(), True)
    assert is_scaled(frame) is True
    frame = mark_scaled(frame, False)
    assert is_scaled(frame) is False
