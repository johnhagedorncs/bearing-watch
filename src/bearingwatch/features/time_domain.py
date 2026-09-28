"""Time-domain condition indicators.

These are the standard vibration health indicators. Each responds to a different
aspect of the waveform, which is why more than one is worth computing:

* rms             -- overall energy. Rises late; insensitive to early impulsing.
* peak            -- max |x|. Sensitive to single events, and to single glitches.
* peak_to_peak    -- max - min. Same, without assuming a zero mean.
* crest_factor    -- peak / rms. Impulsiveness normalised by energy. Rises early
                     in bearing degradation, then FALLS again as the defect
                     spreads and rms catches up, which is why a crest-factor
                     threshold alone reads a badly worn bearing as healthy.
* kurtosis        -- 4th standardised moment. ~3 for gaussian. Rises with
                     impulsive content. Shares the crest-factor turn-over.
* skewness        -- 3rd standardised moment. Asymmetry between impact and
                     rebound.
* shape_factor    -- rms / mean|x|. Waveform shape independent of amplitude.
* impulse_factor  -- peak / mean|x|. Like crest factor with a different
                     normaliser; less influenced by a few large samples.

Kurtosis is computed as the NON-excess (Pearson) fourth moment, so gaussian
noise gives 3.0 rather than 0.0. Papers disagree on the convention and the
difference silently shifts any threshold copied between them, so it is stated
here and asserted in the tests.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TIME_DOMAIN_STATS", "compute_time_domain", "crest_factor", "rms"]

TIME_DOMAIN_STATS = (
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "shape_factor",
    "impulse_factor",
)

_EPS = 1e-15


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def crest_factor(x: np.ndarray) -> float:
    r = rms(x)
    return float(np.max(np.abs(x)) / r) if r > _EPS else 0.0


def compute_time_domain(
    x: np.ndarray,
    stats: tuple[str, ...] | list[str] = TIME_DOMAIN_STATS,
) -> dict[str, float]:
    """Compute the requested time-domain indicators for a 1-D signal."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot compute time-domain features on an empty signal")

    unknown = sorted(set(stats) - set(TIME_DOMAIN_STATS))
    if unknown:
        raise ValueError(f"unknown time-domain stat(s) {unknown}; known: {list(TIME_DOMAIN_STATS)}")

    abs_x = np.abs(x)
    mean_abs = float(np.mean(abs_x))
    peak_v = float(np.max(abs_x))
    rms_v = float(np.sqrt(np.mean(np.square(x))))
    centred = x - float(np.mean(x))
    std = float(np.std(x))

    out: dict[str, float] = {}
    for name in stats:
        if name == "rms":
            out[name] = rms_v
        elif name == "peak":
            out[name] = peak_v
        elif name == "peak_to_peak":
            out[name] = float(np.max(x) - np.min(x))
        elif name == "crest_factor":
            out[name] = peak_v / rms_v if rms_v > _EPS else 0.0
        elif name == "kurtosis":
            # Pearson (non-excess): 3.0 for gaussian noise.
            out[name] = float(np.mean(centred**4) / (std**4)) if std > _EPS else 0.0
        elif name == "skewness":
            out[name] = float(np.mean(centred**3) / (std**3)) if std > _EPS else 0.0
        elif name == "shape_factor":
            out[name] = rms_v / mean_abs if mean_abs > _EPS else 0.0
        elif name == "impulse_factor":
            out[name] = peak_v / mean_abs if mean_abs > _EPS else 0.0

    return out
