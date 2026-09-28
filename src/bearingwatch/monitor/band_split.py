"""Band-split discriminant: sensor gain change vs mechanical degradation.

THE PROBLEM. A distribution-drift monitor fires when the features stop looking
like the training data. It fires on a degrading bearing. It also fires when
somebody swaps an accelerometer for one with a different sensitivity, changes a
charge-amplifier setting, or re-torques a mount. The monitor reports these
identically -- same features flagged, comparable PSI -- because at the level of
"has this distribution moved" they ARE identical. Drift alone cannot separate a
failing machine from a maintenance action on an instrument, and an alert that
cannot make that distinction sends someone to the wrong place.

THE STRUCTURE THAT SEPARATES THEM. A broadband sensor gain change is
multiplicative: x -> g*x. Power scales as g^2, so every PSD value and every band
energy scales by the same factor g^2. In log10 that is an ADDITIVE CONSTANT
applied identically to every band:

    log10 E_b  ->  log10 E_b + 2*log10(g)     for every band b

Decompose the log band-energy vector into a level and a shape:

    level = mean_b( log10 E_b )
    shape_b = log10 E_b - level

Under a gain change the level moves by exactly 2*log10(g) and the shape does not
move AT ALL -- the constant cancels in the subtraction. Under mechanical
degradation the level also moves, but the shape moves too, because a developing
defect puts energy into specific places: the defect-frequency harmonics and the
structural resonance the impacts excite. It does not scale a 30 Hz band and a
5 kHz band by the same factor.

So: level answers "did the amplitude change", shape answers "did the SPECTRUM
change", and the pair separates the two causes that drift alone conflates.

    level moved, shape did not  ->  broadband gain change (instrumentation)
    shape moved                 ->  mechanical change (the bearing)
    neither                     ->  no change

WHERE THIS DOES NOT WORK. Stated up front, because a discriminant sold without
its failure cases is worse than no discriminant:

1. FREQUENCY-DEPENDENT instrumentation faults break it. A loosening mount, a
   damaged cable, or a sensor with a shifted resonance changes the transfer
   function differently at different frequencies. That IS a shape change, and
   this method reports it as mechanical. It only handles gain changes that are
   genuinely broadband.
2. LATE-STAGE degradation can look like gain. Once a spall spreads across a
   race, the response becomes broadband rather than a discrete impulse train,
   and the shape change flattens out while the level keeps climbing. Near the
   end of a run this can read as a gain change -- exactly backwards. It is
   strongest early, which is fortunately when the answer is worth more.
3. CLIPPING is not multiplicative. An over-ranged ADC distorts the waveform and
   generates harmonics, which is a shape change. Correctly not-a-gain-change,
   but also not a bearing fault, and this method cannot say so.
4. SIMULTANEOUS changes are not decomposed. A sensor swap during a period of
   real degradation gives one level shift and one shape shift, and this reports
   "mechanical" without noticing the instrumentation event underneath.
5. It needs the band features to be LOG energies of the SAME bands as the
   reference. Change ``features.spectral.n_bands`` and the reference is void --
   which is one more reason the cache key covers it.

The honest summary: this turns "something changed" into "the spectrum changed
shape, which a broadband gain change cannot do", which is a much more actionable
statement. It is not a proof of a bearing fault.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BandSplitDiscriminant",
    "BandSplitError",
    "BandSplitResult",
    "band_columns",
    "level_and_shape",
]

VERDICT_NO_CHANGE = "no_change"
VERDICT_GAIN = "gain_change"
VERDICT_MECHANICAL = "mechanical_change"
VERDICT_AMBIGUOUS = "ambiguous"


class BandSplitError(RuntimeError):
    """Raised when the discriminant is misconfigured or has no bands to work with."""


def band_columns(columns: Sequence[str], channel_prefix: str | None = None) -> list[str]:
    """Select the log-spaced spectral band columns from a feature table.

    Matches ``ch<N>__band<NN>_...Hz`` and deliberately EXCLUDES the defect-band
    columns. The defect bands are already normalised by total energy in the
    feature pipeline, so they are gain-invariant by construction -- including
    them would mean measuring the level from features that have had the level
    divided out.
    """
    out = [c for c in columns if "__band" in c and c.endswith("Hz")]
    if channel_prefix:
        out = [c for c in out if c.startswith(channel_prefix)]
    return sorted(out)


def level_and_shape(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split log band energies into level (per-row mean) and shape (residual).

    ``values`` is (n_samples, n_bands) of log10 band energies. Returns
    ``(level, shape)`` with shapes (n_samples,) and (n_samples, n_bands).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 2:
        raise BandSplitError(
            f"need at least 2 spectral bands to split level from shape, got {values.shape[1]}"
        )
    level = values.mean(axis=1)
    shape = values - level[:, None]
    return level, shape


def _robust_scale_1d(values: np.ndarray, median: float, min_mad: float) -> float:
    """MAD, falling back to std, falling back to the floor. See detect/reference_threshold."""
    mad = float(np.median(np.abs(values - median)))
    if mad > min_mad:
        return mad
    std = float(np.std(values))
    return std if std > min_mad else float(min_mad)


def _robust_scale(values: np.ndarray, median: np.ndarray, min_mad: float) -> np.ndarray:
    """Per-column MAD with a std fallback where the MAD degenerates."""
    mad = np.median(np.abs(values - median), axis=0)
    std = np.std(values, axis=0)
    out = np.where(mad > min_mad, mad, std)
    return np.maximum(out, min_mad)


@dataclass
class BandSplitResult:
    """One verdict from the discriminant."""

    verdict: str
    level_z: float
    shape_z: float
    level_shift_log10: float
    implied_gain: float
    level_fired: bool
    shape_fired: bool
    n_bands: int
    top_shape_bands: list[tuple[str, float]] = field(default_factory=list)

    def explain(self) -> str:
        if self.verdict == VERDICT_NO_CHANGE:
            return (
                f"no change (level z={self.level_z:.2f}, shape z={self.shape_z:.2f}; "
                "both below threshold)"
            )
        if self.verdict == VERDICT_GAIN:
            return (
                f"broadband GAIN CHANGE: level moved (z={self.level_z:.2f}) but spectral "
                f"shape did not (z={self.shape_z:.2f}). Implied gain "
                f"{self.implied_gain:.3f}x. A multiplicative gain scales every band "
                "equally, so the shape is invariant -- consistent with an "
                "instrumentation change, not a bearing."
            )
        if self.verdict == VERDICT_MECHANICAL:
            worst = ", ".join(f"{n} (z={z:.1f})" for n, z in self.top_shape_bands[:3])
            return (
                f"MECHANICAL CHANGE: spectral shape moved (z={self.shape_z:.2f}), which a "
                f"broadband gain change cannot produce. Level z={self.level_z:.2f}. "
                f"Bands driving it: {worst}"
            )
        return (
            f"ambiguous (level z={self.level_z:.2f}, shape z={self.shape_z:.2f}); "
            "inconclusive against the configured thresholds"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "level_z": round(self.level_z, 4),
            "shape_z": round(self.shape_z, 4),
            "level_shift_log10": round(self.level_shift_log10, 6),
            "implied_gain": round(self.implied_gain, 6),
            "level_fired": self.level_fired,
            "shape_fired": self.shape_fired,
            "n_bands": self.n_bands,
            "top_shape_bands": [[n, round(z, 4)] for n, z in self.top_shape_bands[:5]],
            "explanation": self.explain(),
        }


@dataclass
class BandSplitDiscriminant:
    """Fitted on healthy-window band energies; classifies later windows."""

    band_names: list[str] = field(default_factory=list)
    level_median: float = 0.0
    level_mad: float = 1.0
    shape_median: np.ndarray | None = None
    shape_mad: np.ndarray | None = None
    level_z_threshold: float = 3.5
    shape_z_threshold: float = 3.5
    mad_scale: float = 1.4826
    min_mad: float = 1e-9
    fitted: bool = False

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        reference: pd.DataFrame,
        band_names: Sequence[str] | None = None,
        level_z_threshold: float = 3.5,
        shape_z_threshold: float = 3.5,
        mad_scale: float = 1.4826,
        min_mad: float = 1e-9,
    ) -> BandSplitDiscriminant:
        """Fit on the DECLARED HEALTHY WINDOW.

        Same discipline as the detection threshold: the reference is a period
        declared stationary in advance, not the window being judged. A
        self-referencing version of this would be blind to a slow gain drift for
        exactly the reason set out in detect/reference_threshold.py.
        """
        names = list(band_names) if band_names else band_columns(reference.columns)
        if len(names) < 2:
            raise BandSplitError(
                "band-split needs at least 2 spectral band columns in the reference table; "
                f"found {len(names)}. Is features.spectral.enabled false?"
            )

        values = reference[names].to_numpy(dtype=np.float64)
        level, shape = level_and_shape(values)

        level_median = float(np.median(level))
        level_mad = _robust_scale_1d(level, level_median, min_mad)

        shape_median = np.median(shape, axis=0)
        shape_mad = _robust_scale(shape, shape_median, min_mad)

        return cls(
            band_names=names,
            level_median=level_median,
            level_mad=level_mad,
            shape_median=shape_median,
            shape_mad=shape_mad,
            level_z_threshold=level_z_threshold,
            shape_z_threshold=shape_z_threshold,
            mad_scale=mad_scale,
            min_mad=min_mad,
            fitted=True,
        )

    # -------------------------------------------------------------- classify

    def classify(self, current: pd.DataFrame) -> BandSplitResult:
        """Classify a window of snapshots against the fitted reference."""
        if not self.fitted or self.shape_median is None or self.shape_mad is None:
            raise BandSplitError("BandSplitDiscriminant is not fitted")

        missing = [n for n in self.band_names if n not in current.columns]
        if missing:
            raise BandSplitError(
                f"{len(missing)} band column(s) missing from the current window, "
                f"e.g. {missing[:3]}. The band definition changed between reference and "
                "current, which voids the reference -- check features.spectral.n_bands."
            )

        values = current[self.band_names].to_numpy(dtype=np.float64)
        level, shape = level_and_shape(values)

        # Median over the window, so one transient snapshot cannot swing the
        # verdict. The question is about the regime, not about one second of
        # signal.
        level_obs = float(np.median(level))
        shape_obs = np.median(shape, axis=0)

        level_z = (level_obs - self.level_median) / (self.mad_scale * self.level_mad)
        shape_z_per_band = (shape_obs - self.shape_median) / (self.mad_scale * self.shape_mad)
        shape_z = float(np.max(np.abs(shape_z_per_band)))

        level_fired = bool(abs(level_z) > self.level_z_threshold)
        shape_fired = bool(shape_z > self.shape_z_threshold)

        level_shift = level_obs - self.level_median
        # level is a mean of log10 POWER, and power scales as g^2, so
        # g = 10 ** (shift / 2).
        implied_gain = float(10.0 ** (level_shift / 2.0))

        if shape_fired:
            verdict = VERDICT_MECHANICAL
        elif level_fired:
            verdict = VERDICT_GAIN
        else:
            verdict = VERDICT_NO_CHANGE

        order = np.argsort(-np.abs(shape_z_per_band))
        top = [(self.band_names[i], float(shape_z_per_band[i])) for i in order[:5]]

        return BandSplitResult(
            verdict=verdict,
            level_z=float(level_z),
            shape_z=shape_z,
            level_shift_log10=float(level_shift),
            implied_gain=implied_gain,
            level_fired=level_fired,
            shape_fired=shape_fired,
            n_bands=len(self.band_names),
            top_shape_bands=top,
        )

    # ------------------------------------------------------------------- io

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_names": list(self.band_names),
            "level_median": self.level_median,
            "level_mad": self.level_mad,
            "shape_median": (self.shape_median.tolist() if self.shape_median is not None else None),
            "shape_mad": self.shape_mad.tolist() if self.shape_mad is not None else None,
            "level_z_threshold": self.level_z_threshold,
            "shape_z_threshold": self.shape_z_threshold,
            "mad_scale": self.mad_scale,
            "min_mad": self.min_mad,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BandSplitDiscriminant:
        return cls(
            band_names=list(payload["band_names"]),
            level_median=float(payload["level_median"]),
            level_mad=float(payload["level_mad"]),
            shape_median=np.asarray(payload["shape_median"], dtype=np.float64),
            shape_mad=np.asarray(payload["shape_mad"], dtype=np.float64),
            level_z_threshold=float(payload.get("level_z_threshold", 3.5)),
            shape_z_threshold=float(payload.get("shape_z_threshold", 3.5)),
            mad_scale=float(payload.get("mad_scale", 1.4826)),
            min_mad=float(payload.get("min_mad", 1e-9)),
            fitted=bool(payload.get("fitted", True)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BandSplitDiscriminant:
        path = Path(path)
        if not path.is_file():
            raise BandSplitError(f"band-split state not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
