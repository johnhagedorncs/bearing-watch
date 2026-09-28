"""Feature pipeline: snapshots in, one feature table out -- plus the scaling guard.

Two things happen here that are worth reading carefully.

THE CACHE. Feature extraction is the slow step, so it is cached under a
content-addressed key (see data/cache.py). Nothing in this module bumps a
version by hand; the key falls out of the config.

THE SCALING FLAG. Standardisation is a transform fitted on training data and
applied at scoring time. The bug class is applying it twice: an artifact is
saved AFTER the transform, and the scoring path -- which correctly transforms
whatever it is given -- transforms it again.

The result is not an error. It is a feature vector in a systematically wrong
region of space, scored by a model that has no way to know. The DIRECTION of the
damage depends on the raw feature means: re-standardising already-standardised
values computes ``(z - mean_raw) / scale_raw``, so it is dominated by
``-mean_raw / scale_raw``. Measured on this repo's own feature set, where the
log band energies sit around -8 with scales near 0.1, that inflates
reconstruction error by roughly SEVEN ORDERS OF MAGNITUDE and the detector fires
on everything including the declared healthy window. For a feature set already
centred near zero the same mistake deflates the score instead, and the detector
goes quiet.

Both failures look like a model problem rather than a plumbing problem, which is
why they cost so much to find: nothing raises, no shape changes, no NaN appears,
and the only symptom is a detector behaving strangely. Do not rely on noticing
it from the numbers -- check the flag.

The fix is to make the state explicit and check it rather than infer it:

* Every feature table carries a ``scaled: bool`` in its metadata.
* Tables written to the feature cache are ALWAYS unscaled.
* ``apply_scaler`` refuses to transform a table already marked scaled.
* ``require_unscaled`` is called on the scoring path before transforming.
* A table with NO flag is an error, not an assumed False -- guessing is exactly
  what caused the problem.

``tests/test_double_scaling.py`` constructs the double-scaling attempt and
asserts it raises.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data.cache import SCALED_METADATA_KEY, CacheEntry, FeatureCache
from ..data.ims_parser import Snapshot, fingerprint_files, list_snapshot_files, read_snapshot
from .defect_freqs import compute_defect_bands, defect_frequencies
from .spectral import compute_spectral, welch_psd
from .time_domain import compute_time_domain

__all__ = [
    "DoubleScalingError",
    "FeatureMetadata",
    "apply_ablation",
    "apply_scaler",
    "build_feature_table",
    "extract_snapshot_features",
    "feature_columns",
    "is_scaled",
    "mark_scaled",
    "require_unscaled",
]

# Index name for the timestamp index on every feature table.
INDEX_NAME = "timestamp"


class DoubleScalingError(RuntimeError):
    """Raised when an already-scaled feature table is handed to a scaler again."""


@dataclass
class FeatureMetadata:
    """Metadata travelling with a feature table."""

    scaled: bool
    cache_key: str
    feature_config_hash: str
    schema_version: int
    n_features: int
    n_rows: int
    source_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            SCALED_METADATA_KEY: self.scaled,
            "cache_key": self.cache_key,
            "feature_config_hash": self.feature_config_hash,
            "schema_version": self.schema_version,
            "n_features": self.n_features,
            "n_rows": self.n_rows,
            "source_root": self.source_root,
        }


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def extract_snapshot_features(snapshot: Snapshot, cfg: Config) -> dict[str, float]:
    """Every feature for one snapshot, keyed ``ch{i}__{feature}``.

    Channel prefixes are kept even in single-channel mode so a feature name
    always says which sensor it came from. Silent channel reordering between
    IMS sets is otherwise indistinguishable from a change in the bearing.
    """
    fcfg = cfg.features
    fs = float(cfg.data.sample_rate_hz)

    if fcfg.channels == "all":
        full_channels: Sequence[int] = range(snapshot.n_channels)
    else:
        full_channels = [cfg.data.target_channel]
    full_set = set(full_channels)

    out: dict[str, float] = {}

    for ch in range(snapshot.n_channels):
        x = snapshot.channel(ch)
        prefix = f"ch{ch}__"

        if ch not in full_set:
            # Context channels: the healthy bearings on the same shaft. They are
            # negative controls -- same load, same speed, same duration -- so a
            # rise seen on every channel at once is a rig or sensor change, not
            # a bearing.
            if fcfg.context_stats:
                for name, value in compute_time_domain(x, fcfg.context_stats).items():
                    out[f"{prefix}{name}"] = value
            continue

        if fcfg.time_domain.enabled:
            for name, value in compute_time_domain(x, fcfg.time_domain.stats).items():
                out[f"{prefix}{name}"] = value

        psd_freqs: np.ndarray | None = None
        psd: np.ndarray | None = None

        if fcfg.spectral.enabled:
            bands, psd_freqs, psd = compute_spectral(x, fs, fcfg.spectral)
            for name, value in bands.items():
                out[f"{prefix}{name}"] = value

        if fcfg.defect_bands.enabled:
            if psd_freqs is None or psd is None:
                psd_freqs, psd = welch_psd(x, fs, fcfg.spectral)
            defect = compute_defect_bands(
                psd_freqs,
                psd,
                fcfg.bearing,
                harmonics=fcfg.defect_bands.harmonics,
                rel_bandwidth=fcfg.defect_bands.rel_bandwidth,
                min_bandwidth_hz=fcfg.defect_bands.min_bandwidth_hz,
                log_energy=fcfg.defect_bands.log_energy,
                normalise_by_total=fcfg.defect_bands.normalise_by_total,
                log_eps=fcfg.spectral.log_eps,
            )
            for name, value in defect.items():
                out[f"{prefix}{name}"] = value

    return apply_ablation(out, cfg)


def apply_ablation(features: dict[str, float], cfg: Config) -> dict[str, float]:
    """Drop feature families per the ablation switches.

    Ablation happens by DROPPING COLUMNS rather than by not computing them, so
    the ablated run and the full run see numerically identical values for every
    column they share. Recomputing would let an unrelated change sneak into the
    comparison.
    """
    ab = cfg.features.ablation
    if not (ab.drop_kurtosis or ab.drop_defect_bands):
        return features

    defect_names = tuple(defect_frequencies(cfg.features.bearing).as_dict())

    def keep(name: str) -> bool:
        base = name.split("__", 1)[-1]
        if ab.drop_kurtosis and base == "kurtosis":
            return False
        return not (ab.drop_defect_bands and base.split("_h")[0] in defect_names)

    return {k: v for k, v in features.items() if keep(k)}


def constant_features(frame: pd.DataFrame, columns: Sequence[str], tol: float = 0.0) -> list[str]:
    """Columns with no variance at all across the given rows.

    Worth checking explicitly and reporting loudly, because a constant feature
    is the signature of an extraction bug rather than of a quiet sensor -- and
    it is invisible downstream. A constant column standardises to zero, adds
    nothing to any reconstruction error, and never appears in a metric. The
    defect-band extractor in this repo produced four of them (every FTF harmonic
    and BSF_h1) for exactly one reason: those bands caught a single Welch bin,
    ``np.trapz`` of a single point is 0.0, and log10(0 + eps) is a constant.
    Nothing raised. The only way to see it was to look for columns that never
    move.
    """
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    return [c for c, s in zip(columns, spread) if s <= tol]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric feature columns, excluding bookkeeping columns."""
    return [
        c
        for c in frame.columns
        if c not in ("source_file", "snapshot_index") and pd.api.types.is_numeric_dtype(frame[c])
    ]


def build_feature_table(
    cfg: Config,
    root: str | Path | None = None,
    cache: FeatureCache | None = None,
    paths: Iterable[Path] | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, CacheEntry]:
    """Extract (or load from cache) the feature table for a whole run.

    Returns ``(frame, cache_entry)``. The frame is indexed by snapshot timestamp,
    carries ``source_file`` and ``snapshot_index`` columns, and is ALWAYS
    unscaled -- ``frame.attrs['bearingwatch'][scaled]`` is False.
    """
    root = Path(root) if root is not None else Path(cfg.data.root)
    file_list = list(paths) if paths is not None else list_snapshot_files(root)
    if not file_list:
        raise FileNotFoundError(f"no snapshot files under {root}")

    fingerprint = fingerprint_files(file_list)
    key = FeatureCache.compute_key(cfg, fingerprint)
    cache = cache or FeatureCache()

    def compute() -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        index: list[Any] = []
        for i, path in enumerate(file_list):
            snap = read_snapshot(
                path,
                expected_points=cfg.data.expected_points,
                expected_channels=cfg.data.expected_channels,
                timestamp_format=cfg.data.filename_timestamp_format,
            )
            feats = extract_snapshot_features(snap, cfg)
            feats["snapshot_index"] = float(i)
            rows.append(feats)
            index.append(snap.timestamp)
            if progress and (i + 1) % 50 == 0:
                print(f"  featurised {i + 1}/{len(file_list)} snapshots", flush=True)

        frame = pd.DataFrame(rows, index=pd.Index(index, name=INDEX_NAME))
        frame["source_file"] = [p.name for p in file_list]
        frame["snapshot_index"] = frame["snapshot_index"].astype(int)
        return frame.sort_index()

    meta = {
        SCALED_METADATA_KEY: False,
        "cache_key": key,
        "feature_config_hash": cfg.feature_cache_key,
        "schema_version": cfg.features.schema_version,
        "source_root": str(root),
        "data_fingerprint": fingerprint,
    }
    frame, entry = cache.get_or_compute(key, compute, metadata=meta)

    attrs = dict(frame.attrs.get("bearingwatch") or {})
    attrs.setdefault(SCALED_METADATA_KEY, False)
    attrs.update({k: v for k, v in meta.items() if k != SCALED_METADATA_KEY})
    frame.attrs["bearingwatch"] = attrs
    return frame, entry


# --------------------------------------------------------------------------
# the scaling guard
# --------------------------------------------------------------------------


def _attrs(frame: pd.DataFrame) -> dict[str, Any]:
    return dict(frame.attrs.get("bearingwatch") or {})


def is_scaled(frame: pd.DataFrame) -> bool:
    """Read the scaled flag. Raises if the flag is absent.

    An absent flag is NOT treated as False. Defaulting to False is exactly the
    assumption that lets a post-transform artifact get transformed a second
    time -- the guard has to be able to say "I do not know", and refusing is the
    only safe answer to that.
    """
    attrs = _attrs(frame)
    if SCALED_METADATA_KEY not in attrs:
        raise DoubleScalingError(
            "feature table carries no scaling flag, so its scaling state is unknown. "
            "Refusing to guess: an already-scaled table transformed a second time lands "
            "in a systematically wrong region of feature space and is scored without any "
            "error being raised. Build the table with build_feature_table() or set "
            f"frame.attrs['bearingwatch']['{SCALED_METADATA_KEY}'] explicitly."
        )
    return bool(attrs[SCALED_METADATA_KEY])


def require_unscaled(frame: pd.DataFrame, where: str = "scoring") -> pd.DataFrame:
    """Assert a feature table has not already been standardised."""
    if is_scaled(frame):
        raise DoubleScalingError(
            f"{where}: feature table is already marked scaled, and is about to be "
            "standardised again. This is double scaling. It does not raise anywhere "
            "downstream -- the model returns a score, the detector returns a verdict, "
            "and both are meaningless. On this feature set it inflates reconstruction "
            "error by about seven orders of magnitude; on a zero-centred one it would "
            "deflate it instead. Feed the scoring path the RAW feature table; the run "
            "folder's scaler is applied exactly once, here."
        )
    return frame


def mark_scaled(frame: pd.DataFrame, scaled: bool = True) -> pd.DataFrame:
    attrs = _attrs(frame)
    attrs[SCALED_METADATA_KEY] = bool(scaled)
    frame.attrs["bearingwatch"] = attrs
    return frame


def apply_scaler(frame: pd.DataFrame, scaler: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Standardise a feature table exactly once, refusing a second pass."""
    require_unscaled(frame, where="apply_scaler")
    values = scaler.transform(frame[list(columns)].to_numpy(dtype=np.float64))
    out = pd.DataFrame(values, index=frame.index, columns=list(columns))
    for extra in ("source_file", "snapshot_index"):
        if extra in frame.columns:
            out[extra] = frame[extra]
    out.attrs["bearingwatch"] = _attrs(frame)
    return mark_scaled(out, True)
