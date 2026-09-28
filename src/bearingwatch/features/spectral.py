"""Welch PSD and its reduction to log-spaced band energies.

The raw Welch PSD of a 20,480-point record at 20 kHz with nperseg=4096 gives
2049 frequency bins. Feeding 2049 correlated, wildly-scaled numbers per channel
into an autoencoder trained on a few hundred healthy snapshots is a good way to
fit noise, so the PSD is reduced to a fixed number of band energies.

Bands are LOG-SPACED, not linear. Bearing-relevant structure spans shaft rate
(~33 Hz) to structural resonances (several kHz). Linear bands put almost every
bin above 1 kHz and resolve nothing at the low end, where the shaft harmonics
and cage frequency live.

Energies are stored as log10. Band energies across a 10 kHz span cover many
decades; on a linear scale the standardiser is dominated by whichever band
happens to be largest, and every other band contributes nothing to the
reconstruction error.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from ..config import SpectralConfig

__all__ = ["band_energies", "compute_spectral", "log_spaced_edges", "welch_psd"]

# numpy 2.0 renamed `trapz` to `trapezoid` and deprecated the old spelling. This
# package supports numpy >= 1.24, which spans both, so the name is resolved once
# here rather than at every call site. Doing it by attribute lookup rather than
# by version comparison means it keeps working if the alias is removed outright.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def welch_psd(
    x: np.ndarray,
    sample_rate_hz: float,
    cfg: SpectralConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density. Returns (frequencies_hz, psd)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = int(min(cfg.nperseg, x.size))
    noverlap = int(min(cfg.noverlap, nperseg - 1)) if nperseg > 1 else 0

    freqs, psd = sp_signal.welch(
        x,
        fs=float(sample_rate_hz),
        window=cfg.window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=cfg.detrend,
        scaling="density",
        average="mean",
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def log_spaced_edges(f_min: float, f_max: float, n_bands: int) -> np.ndarray:
    """``n_bands + 1`` logarithmically spaced band edges."""
    if f_min <= 0 or f_max <= f_min:
        raise ValueError(f"require 0 < f_min < f_max, got f_min={f_min}, f_max={f_max}")
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    return np.logspace(np.log10(f_min), np.log10(f_max), n_bands + 1)


def band_energies(
    freqs: np.ndarray,
    psd: np.ndarray,
    edges: np.ndarray,
    log_energy: bool = True,
    log_eps: float = 1e-18,
) -> np.ndarray:
    """Integrate the PSD within each band defined by ``edges``.

    Bands narrower than the frequency resolution can contain no bins. Rather
    than returning zero (which after a log becomes a large constant that the
    scaler then treats as signal), such a band takes the PSD value at the
    nearest bin times the band width -- a first-order estimate that stays on the
    same scale as its neighbours.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    out = np.empty(len(edges) - 1, dtype=np.float64)

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        n_in = int(mask.sum())
        if n_in >= 2:
            out[i] = _trapezoid(psd[mask], freqs[mask])
        elif n_in == 1:
            out[i] = float(psd[mask][0]) * (hi - lo)
        else:
            centre = 0.5 * (lo + hi)
            nearest = int(np.argmin(np.abs(freqs - centre)))
            out[i] = float(psd[nearest]) * (hi - lo)

    if log_energy:
        return np.log10(out + log_eps)
    return out


def compute_spectral(
    x: np.ndarray,
    sample_rate_hz: float,
    cfg: SpectralConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Band-energy features for one signal.

    Returns ``(features, freqs, psd)``. The PSD is handed back so the defect-band
    extractor can reuse it instead of recomputing the same Welch estimate --
    which also guarantees both feature families describe the same spectrum.
    """
    freqs, psd = welch_psd(x, sample_rate_hz, cfg)

    nyquist = float(sample_rate_hz) / 2.0
    f_max = min(cfg.f_max_hz, nyquist)
    edges = log_spaced_edges(cfg.f_min_hz, f_max, cfg.n_bands)
    energies = band_energies(freqs, psd, edges, cfg.log_energy, cfg.log_eps)

    features = {
        f"band{i:02d}_{edges[i]:.0f}_{edges[i + 1]:.0f}Hz": float(v) for i, v in enumerate(energies)
    }
    return features, freqs, psd
