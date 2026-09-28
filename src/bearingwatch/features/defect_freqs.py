"""Bearing defect frequencies and the energy in bands around them.

A defect on a bearing surface produces an impact each time a rolling element
passes over it. The repetition rate depends only on the geometry and the shaft
speed, so it is predictable in advance -- which is what makes bearing
diagnostics different from generic anomaly detection. Energy appearing at BPFO
and its harmonics is evidence about WHICH surface is failing, not just that
something changed.

Standard formulas, with n rolling elements, ball diameter d, pitch diameter D,
contact angle theta, and shaft rate fr (Hz):

    FTF  = (fr / 2) * (1 - (d/D) cos(theta))
    BPFO = (n / 2) * fr * (1 - (d/D) cos(theta))       = n * FTF
    BPFI = (n / 2) * fr * (1 + (d/D) cos(theta))
    BSF  = (D / (2d)) * fr * (1 - ((d/D) cos(theta))^2)

Sanity check for the ZA-2115 geometry at 2000 RPM: BPFO ~236 Hz, BPFI ~297 Hz,
BSF ~140 Hz, FTF ~15 Hz. If a geometry edit moves these far from those values,
the geometry is wrong -- the formulas are not.

Two caveats worth stating rather than discovering:

1. These are kinematic frequencies assuming pure rolling. Real rolling elements
   slip, so the observed lines sit a percent or two off and smear. That is why
   the bands here have a width instead of being single bins.
2. BSF is the ball-spin rate. A defect on a ball strikes both races per
   revolution, so the fault often shows at 2 * BSF rather than at BSF. Both are
   covered because harmonics are included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import BearingGeometry

__all__ = ["DefectFrequencies", "band_energy", "compute_defect_bands", "defect_frequencies"]

# numpy 2.0 renamed `trapz` to `trapezoid` and deprecated the old spelling. This
# package supports numpy >= 1.24, which spans both, so the name is resolved once
# here rather than at every call site. Doing it by attribute lookup rather than
# by version comparison means it keeps working if the alias is removed outright.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


@dataclass(frozen=True)
class DefectFrequencies:
    """Kinematic defect frequencies in Hz."""

    ftf: float  # fundamental train (cage) frequency
    bpfo: float  # ball pass frequency, outer race
    bpfi: float  # ball pass frequency, inner race
    bsf: float  # ball (roller) spin frequency
    shaft_hz: float

    def as_dict(self) -> dict[str, float]:
        return {"FTF": self.ftf, "BPFO": self.bpfo, "BPFI": self.bpfi, "BSF": self.bsf}


def defect_frequencies(geometry: BearingGeometry) -> DefectFrequencies:
    """Compute defect frequencies from bearing geometry and shaft speed."""
    geometry.validate()

    n = float(geometry.n_rolling_elements)
    d = float(geometry.ball_diameter_in)
    big_d = float(geometry.pitch_diameter_in)
    theta = math.radians(float(geometry.contact_angle_deg))
    fr = float(geometry.shaft_rpm) / 60.0

    ratio = (d / big_d) * math.cos(theta)

    ftf = (fr / 2.0) * (1.0 - ratio)
    bpfo = (n / 2.0) * fr * (1.0 - ratio)
    bpfi = (n / 2.0) * fr * (1.0 + ratio)
    bsf = (big_d / (2.0 * d)) * fr * (1.0 - ratio**2)

    return DefectFrequencies(ftf=ftf, bpfo=bpfo, bpfi=bpfi, bsf=bsf, shaft_hz=fr)


def band_energy(
    freqs: np.ndarray,
    psd: np.ndarray,
    centre_hz: float,
    rel_bandwidth: float = 0.02,
    min_bandwidth_hz: float = 5.0,
) -> float:
    """Integrate a PSD over a band around ``centre_hz``.

    Band half-width is max(rel_bandwidth * centre, min_bandwidth_hz / 2). The
    floor matters: at FTF (~15 Hz) a 2% band is +/-0.3 Hz, narrower than the
    Welch resolution, so without a floor the band would frequently contain zero
    bins and silently return 0.0.
    """
    half = max(rel_bandwidth * centre_hz, min_bandwidth_hz / 2.0)
    lo, hi = centre_hz - half, centre_hz + half
    mask = (freqs >= lo) & (freqs <= hi)
    n_in = int(mask.sum())

    if n_in >= 2:
        # Trapezoidal integration of PSD over the band gives energy in physical
        # units; a bare sum would depend on the frequency resolution.
        return float(_trapezoid(psd[mask], freqs[mask]))

    width = hi - lo
    if n_in == 1:
        # EXACTLY ONE BIN IN THE BAND. np.trapz of a single point returns 0.0 --
        # a clean, plausible zero rather than an error. That is how this went
        # unnoticed the first time: FTF sits at ~15 Hz and BSF at ~140 Hz, and
        # with a 4096-point Welch segment at 20 kHz the bin spacing is 4.9 Hz,
        # so those bands caught one bin and every FTF and BSF feature came out
        # as a constant log10(eps) for the entire run. Constant features carry
        # no information and standardise to zero, so nothing downstream
        # complained. Rectangular approximation instead.
        return float(psd[mask][0]) * width

    # No bin at all: the band is narrower than the resolution. Use the nearest
    # bin rather than returning zero, for the same reason.
    if freqs.size == 0:
        return 0.0
    nearest = int(np.argmin(np.abs(freqs - centre_hz)))
    return float(psd[nearest]) * width


def compute_defect_bands(
    freqs: np.ndarray,
    psd: np.ndarray,
    geometry: BearingGeometry,
    harmonics: int = 3,
    rel_bandwidth: float = 0.02,
    min_bandwidth_hz: float = 5.0,
    log_energy: bool = True,
    normalise_by_total: bool = True,
    log_eps: float = 1e-18,
) -> dict[str, float]:
    """Energy in bands at each defect frequency and its harmonics.

    With ``normalise_by_total`` the energies are divided by the total in-band
    spectral energy first. That normalisation is what makes these features
    invariant to a broadband gain change: multiplying the input signal by g
    scales every PSD value by g^2, which cancels in the ratio. The absolute
    level is not thrown away -- it is kept as its own feature, ``total_energy``
    -- so a gain change is still visible, just separable from a change in
    spectral shape. That separation is the whole basis of the discriminant in
    ``bearingwatch.monitor.band_split``.
    """
    if harmonics < 1:
        raise ValueError("harmonics must be >= 1")

    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    nyquist = float(freqs[-1]) if freqs.size else 0.0

    total = float(_trapezoid(psd, freqs)) if freqs.size > 1 else 0.0

    defects = defect_frequencies(geometry)
    out: dict[str, float] = {}

    for name, f0 in defects.as_dict().items():
        for h in range(1, harmonics + 1):
            centre = f0 * h
            key = f"{name}_h{h}"
            if centre >= nyquist:
                # Above Nyquist there is nothing to measure. Emitting the key
                # with a floor value keeps the feature vector a fixed width
                # regardless of shaft speed, which the scaler and the model
                # both require.
                out[key] = math.log10(log_eps) if log_energy else 0.0
                continue
            energy = band_energy(freqs, psd, centre, rel_bandwidth, min_bandwidth_hz)
            if normalise_by_total and total > 0.0:
                energy = energy / total
            out[key] = math.log10(energy + log_eps) if log_energy else energy

    out["total_energy"] = math.log10(total + log_eps) if log_energy else total
    return out
