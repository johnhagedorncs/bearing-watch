"""Configuration loading, merging, and content hashing.

Config is assembled from the YAML files in ``configs/`` (one file per concern)
into a single nested structure. Two properties matter more than the field list:

1. Every parameter that changes feature output lives under ``features``, and
   nothing else does. ``feature_cache_key`` hashes exactly that subtree. If a
   parameter that changes the numbers lives outside it, the cache goes stale
   silently -- there is no error, just wrong features scored against a model
   that never saw them.
2. ``config_hash`` covers the whole config and goes into the run manifest, so a
   run folder can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import MISSING as dataclasses_MISSING
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

DEFAULT_CONFIG_DIR = Path("configs")
CONFIG_FILES = ("data.yaml", "features.yaml", "model.yaml", "detect.yaml", "serve.yaml")


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or self-inconsistent."""


# --------------------------------------------------------------------------
# dataclass helpers
# --------------------------------------------------------------------------


def _from_mapping(cls: type[T], data: Mapping[str, Any] | None, path: str = "") -> T:
    """Build a (possibly nested) dataclass from a mapping, rejecting unknown keys.

    Rejecting unknown keys is deliberate. A typo in a YAML key is otherwise a
    silent no-op: the run uses the default, the config file says otherwise, and
    the two disagree forever.
    """
    data = dict(data or {})
    if not is_dataclass(cls):  # pragma: no cover - programming error
        raise TypeError(f"{cls!r} is not a dataclass")

    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        where = path or cls.__name__
        raise ConfigError(
            f"unknown configuration key(s) {unknown} under '{where}'. Known keys: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        # Resolve nested dataclasses through default_factory rather than through
        # the annotation. `from __future__ import annotations` turns every
        # annotation into a string, and typing.get_type_hints() cannot evaluate
        # `str | None` on Python 3.9 -- so the factory is the only reliable
        # runtime handle on the nested type.
        nested = _nested_type(f)
        if nested is not None and isinstance(value, Mapping):
            kwargs[name] = _from_mapping(nested, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def _nested_type(f: Any) -> type | None:
    """Return the nested dataclass type for a field, or None if it is a scalar."""
    factory = f.default_factory
    if factory is dataclasses_MISSING:
        return None
    if isinstance(factory, type) and is_dataclass(factory):
        return factory
    return None


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Used for every content hash in this project. Determinism is the whole point
    -- a hash that depends on dict ordering is not a content hash.
    """
    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), default=str)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def content_hash(obj: Any, length: int = 16) -> str:
    """SHA-256 over the canonical JSON form, truncated for readability."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:length]


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


@dataclass
class DataConfig:
    root: str = "data/raw/set2"
    expected_points: int = 20480
    sample_rate_hz: int = 20000
    filename_timestamp_format: str = "%Y.%m.%d.%H.%M.%S"
    snapshot_interval_minutes: int = 10
    set_name: str = "set2"
    expected_channels: int = 4
    expected_files: int | None = None
    min_files: int = 50
    target_channel: int = 0
    healthy_fraction: float = 0.30
    max_interval_gap_fraction: float = 0.25

    def validate(self) -> None:
        if self.expected_points <= 0:
            raise ConfigError("data.expected_points must be positive")
        if self.expected_channels <= 0:
            raise ConfigError("data.expected_channels must be positive")
        if not 0 < self.healthy_fraction < 1:
            raise ConfigError(
                f"data.healthy_fraction must be in (0, 1), got {self.healthy_fraction}"
            )
        if not 0 <= self.target_channel < self.expected_channels:
            raise ConfigError(
                f"data.target_channel {self.target_channel} outside [0, {self.expected_channels})"
            )
        if self.sample_rate_hz <= 0:
            raise ConfigError("data.sample_rate_hz must be positive")


@dataclass
class TimeDomainConfig:
    enabled: bool = True
    stats: list[str] = field(
        default_factory=lambda: [
            "rms",
            "peak",
            "peak_to_peak",
            "crest_factor",
            "kurtosis",
            "skewness",
            "shape_factor",
            "impulse_factor",
        ]
    )


@dataclass
class SpectralConfig:
    enabled: bool = True
    nperseg: int = 4096
    noverlap: int = 2048
    window: str = "hann"
    detrend: str = "constant"
    n_bands: int = 48
    f_min_hz: float = 10.0
    f_max_hz: float = 10000.0
    log_energy: bool = True
    log_eps: float = 1e-18


@dataclass
class DefectBandConfig:
    enabled: bool = True
    harmonics: int = 3
    rel_bandwidth: float = 0.02
    min_bandwidth_hz: float = 5.0
    log_energy: bool = True
    normalise_by_total: bool = True


@dataclass
class BearingGeometry:
    """Bearing geometry feeding the defect-frequency formulas.

    The values shipped in configs/features.yaml are the commonly cited Rexnord
    ZA-2115 numbers. They carry a VERIFY comment for a reason: geometry tables
    propagate between papers unchecked, and a wrong pitch diameter shifts every
    defect frequency without producing an error anywhere.
    """

    name: str = "Rexnord ZA-2115 (double row) -- VERIFY AGAINST DATASET DOCS"
    n_rolling_elements: int = 16
    ball_diameter_in: float = 0.331
    pitch_diameter_in: float = 2.815
    contact_angle_deg: float = 15.17
    shaft_rpm: float = 2000.0

    def validate(self) -> None:
        if self.n_rolling_elements < 2:
            raise ConfigError("bearing.n_rolling_elements must be >= 2")
        if not 0 < self.ball_diameter_in < self.pitch_diameter_in:
            raise ConfigError(
                "bearing.ball_diameter_in must be positive and smaller than pitch_diameter_in"
            )
        if self.shaft_rpm <= 0:
            raise ConfigError("bearing.shaft_rpm must be positive")


@dataclass
class AblationConfig:
    drop_kurtosis: bool = False
    drop_defect_bands: bool = False


@dataclass
class FeatureConfig:
    schema_version: int = 3
    # "target" -> full feature set on data.target_channel only, plus a couple of
    # context stats from every other channel. "all" -> full feature set on every
    # channel. Set 1 has 8 channels, so "all" there is ~550 features against a
    # few hundred healthy snapshots, which is not a ratio worth defending.
    channels: str = "target"
    context_stats: list[str] = field(default_factory=lambda: ["rms", "kurtosis"])
    time_domain: TimeDomainConfig = field(default_factory=TimeDomainConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    defect_bands: DefectBandConfig = field(default_factory=DefectBandConfig)
    bearing: BearingGeometry = field(default_factory=BearingGeometry)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    CHANNEL_MODES = ("target", "all")

    def validate(self) -> None:
        self.bearing.validate()
        if self.channels not in self.CHANNEL_MODES:
            raise ConfigError(
                f"features.channels must be one of {self.CHANNEL_MODES}, got {self.channels!r}"
            )
        if self.spectral.enabled:
            if self.spectral.noverlap >= self.spectral.nperseg:
                raise ConfigError("features.spectral.noverlap must be < nperseg")
            if not 32 <= self.spectral.n_bands <= 64:
                raise ConfigError(
                    f"features.spectral.n_bands must be in [32, 64]; got {self.spectral.n_bands}"
                )
            if self.spectral.f_min_hz <= 0 or self.spectral.f_max_hz <= self.spectral.f_min_hz:
                raise ConfigError("features.spectral requires 0 < f_min_hz < f_max_hz")
        if not (self.time_domain.enabled or self.spectral.enabled):
            raise ConfigError("at least one of time_domain / spectral must be enabled")


@dataclass
class AutoencoderConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 8
    dropout: float = 0.0
    activation: str = "relu"
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.2
    patience: int = 25
    seed: int = 0


@dataclass
class PCAConfig:
    n_components: float = 0.95
    seed: int = 0
    # Minimum healthy snapshots required per retained component. A PCA that can
    # reconstruct its own training set exactly has no residual distribution to
    # threshold against; see models/pca_baseline.py for the measured effect.
    max_sample_ratio: float = 3.0


@dataclass
class KurtosisConfig:
    feature_name: str = "ch0__kurtosis"


@dataclass
class ModelConfig:
    kind: str = "autoencoder"
    standardise: bool = True
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    kurtosis: KurtosisConfig = field(default_factory=KurtosisConfig)

    KINDS = ("autoencoder", "pca", "kurtosis")

    def validate(self) -> None:
        if self.kind not in self.KINDS:
            raise ConfigError(f"model.kind must be one of {self.KINDS}, got {self.kind!r}")
        if not 0 < self.autoencoder.val_fraction < 1:
            raise ConfigError("model.autoencoder.val_fraction must be in (0, 1)")


@dataclass
class PersistenceConfig:
    window: int = 5
    consecutive: int = 3


@dataclass
class DetectConfig:
    reference: str = "healthy_window"
    # Fraction of the healthy window HELD OUT from model fitting and used only
    # to calibrate the threshold. See detect/reference_threshold.py.
    calibration_fraction: float = 0.25
    z_threshold: float = 3.5
    mad_scale: float = 1.4826
    min_mad: float = 1e-9
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    lead_time_baseline_fractions: list[float] = field(
        default_factory=lambda: [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    )

    REFERENCES = ("healthy_window", "self")

    def validate(self) -> None:
        if self.reference not in self.REFERENCES:
            raise ConfigError(
                f"detect.reference must be one of {self.REFERENCES}, got {self.reference!r}"
            )
        if self.persistence.consecutive > self.persistence.window:
            raise ConfigError(
                "detect.persistence.consecutive cannot exceed detect.persistence.window"
            )
        if self.z_threshold <= 0:
            raise ConfigError("detect.z_threshold must be positive")
        if not 0 <= self.calibration_fraction < 0.9:
            raise ConfigError(
                f"detect.calibration_fraction must be in [0, 0.9), got {self.calibration_fraction}"
            )


@dataclass
class ServeConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    run_dir: str | None = None
    artifact_store_uri: str = "file://runs"
    max_batch: int = 64
    request_timeout_s: int = 30


@dataclass
class TrackingConfig:
    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str = "bearing-watch"


@dataclass
class BandSplitConfig:
    enabled: bool = True
    level_z_threshold: float = 3.5
    shape_z_threshold: float = 3.5


@dataclass
class MonitorConfig:
    evidently_enabled: bool = True
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_alpha: float = 0.01
    drifted_feature_fraction: float = 0.30
    window_size: int = 20
    band_split: BandSplitConfig = field(default_factory=BandSplitConfig)


@dataclass
class Config:
    """The whole configuration for one run."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    # ---------------------------------------------------------------- hashes

    @property
    def feature_cache_key(self) -> str:
        """Content-addressed key over EVERY parameter that changes feature output.

        Deliberately includes the sampling parameters from ``data`` -- sample
        rate and record length change the PSD, so a cache key that ignored them
        would collide across incompatible data.
        """
        payload = {
            "features": _jsonable(self.features),
            "sample_rate_hz": self.data.sample_rate_hz,
            "expected_points": self.data.expected_points,
            "expected_channels": self.data.expected_channels,
        }
        return content_hash(payload)

    @property
    def config_hash(self) -> str:
        return content_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {k: _jsonable(v) for k, v in asdict(self).items()}

    def validate(self) -> None:
        self.data.validate()
        self.features.validate()
        self.model.validate()
        self.detect.validate()

    def copy_with(self, overrides: Mapping[str, Any]) -> Config:
        """Return a copy with dotted-path overrides applied, e.g. ``model.kind``."""
        raw = self.to_dict()
        for dotted, value in overrides.items():
            _set_dotted(raw, dotted, value)
        return Config.from_dict(raw)

    # ---------------------------------------------------------------- build

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Config:
        cfg = _from_mapping(cls, raw)
        cfg.validate()
        return cfg

    @classmethod
    def load(
        cls,
        config_dir: str | Path = DEFAULT_CONFIG_DIR,
        overrides: Mapping[str, Any] | None = None,
    ) -> Config:
        """Load and merge every config file, then apply dotted-path overrides."""
        config_dir = Path(config_dir)
        if not config_dir.is_dir():
            raise ConfigError(f"config directory not found: {config_dir}")

        merged: dict[str, Any] = {}
        found = False
        for name in CONFIG_FILES:
            path = config_dir / name
            if not path.exists():
                continue
            found = True
            with path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"{path} must contain a YAML mapping at the top level")
            _deep_merge(merged, loaded)
        if not found:
            raise ConfigError(
                f"no config files found in {config_dir}; expected some of {CONFIG_FILES}"
            )

        _apply_env_overrides(merged)
        for dotted, value in (overrides or {}).items():
            _set_dotted(merged, dotted, value)
        return cls.from_dict(merged)

    def dump_yaml(self, path: str | Path) -> Path:
        """Write a full config snapshot. Every run folder gets one of these."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=True, default_flow_style=False)
        return path


# --------------------------------------------------------------------------
# merge / override plumbing
# --------------------------------------------------------------------------


def _deep_merge(base: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _set_dotted(target: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = target
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], MutableMapping):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = _coerce_scalar(value)


def _coerce_scalar(value: Any) -> Any:
    """Turn a CLI string into the type it looks like. Non-strings pass through."""
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


ENV_PREFIX = "BEARINGWATCH_CFG__"


def _apply_env_overrides(merged: MutableMapping[str, Any]) -> None:
    """Apply BEARINGWATCH_CFG__data__root=... style environment overrides.

    Containers configure by environment; this keeps that path first-class
    instead of requiring a rebuilt config file per deployment.
    """
    for key, value in sorted(os.environ.items()):
        if not key.startswith(ENV_PREFIX):
            continue
        dotted = key[len(ENV_PREFIX) :].lower().replace("__", ".")
        _set_dotted(merged, dotted, value)
