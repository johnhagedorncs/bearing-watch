"""MLflow experiment tracking, behind a guarded import.

MLflow is a convenience index over runs, not the source of truth. The run folder
is the source of truth: it holds the model, the scaler, the threshold, the
config snapshot and the manifest, and it is complete without MLflow ever having
been installed. If tracking is disabled, unavailable, or the server is down, the
run still completes and still produces a usable model.

That ordering is deliberate. Making the tracking server a hard dependency of
training means an outage in a metadata service stops model production, which is
a bad trade for a nicer results table.

Everything here is a no-op when MLflow is missing or disabled, so callers do not
need conditionals around every log call.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["MLflowTracker", "NullTracker", "build_tracker", "mlflow_available"]

log = logging.getLogger(__name__)


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except Exception:
        return False
    return True


class NullTracker:
    """Does nothing, successfully. Used when tracking is off or unavailable."""

    enabled = False
    run_id: str | None = None

    @contextmanager
    def run(self, run_name: str, tags: Mapping[str, Any] | None = None) -> Iterator[NullTracker]:
        yield self

    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None: ...
    def log_artifacts(self, directory: str | Path, artifact_path: str | None = None) -> None: ...
    def set_tags(self, tags: Mapping[str, Any]) -> None: ...


class MLflowTracker:
    """Thin wrapper over the MLflow fluent API.

    Every method swallows and logs transport errors. A tracking server that
    becomes unreachable mid-run must not take the training run down with it --
    the artifacts are already on disk.
    """

    enabled = True

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "bearing-watch"):
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run_id: str | None = None

    @contextmanager
    def run(self, run_name: str, tags: Mapping[str, Any] | None = None) -> Iterator[MLflowTracker]:
        active = self._mlflow.start_run(run_name=run_name, tags=dict(tags or {}))
        self.run_id = active.info.run_id
        try:
            yield self
        finally:
            self._mlflow.end_run()

    def _safe(self, what: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on server state
            log.warning("mlflow %s failed (continuing, run folder is authoritative): %s", what, exc)

    def log_params(self, params: Mapping[str, Any]) -> None:
        # MLflow truncates long param values, so flatten and clip here rather
        # than discovering a silently truncated config in the UI later.
        flat = {k: _clip(v) for k, v in _flatten(params).items()}
        self._safe("log_params", self._mlflow.log_params, flat)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        # MLflow metrics must be numeric. Booleans are included deliberately --
        # they carry as 0.0/1.0, and flags like `early_stopped` are worth
        # charting. Everything else (strings, lists, None) is DROPPED rather
        # than coerced: a coerced string is a metric that means nothing.
        numeric = {
            k: float(v) for k, v in _flatten(metrics).items() if isinstance(v, (int, float, bool))
        }
        if numeric:
            self._safe("log_metrics", self._mlflow.log_metrics, numeric, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._safe("log_artifact", self._mlflow.log_artifact, str(path), artifact_path)

    def log_artifacts(self, directory: str | Path, artifact_path: str | None = None) -> None:
        self._safe("log_artifacts", self._mlflow.log_artifacts, str(directory), artifact_path)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        self._safe("set_tags", self._mlflow.set_tags, {k: str(v) for k, v in tags.items()})


def build_tracker(
    enabled: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str = "bearing-watch",
) -> NullTracker | MLflowTracker:
    """Return a real tracker if possible, otherwise a working no-op."""
    if not enabled:
        return NullTracker()
    if not mlflow_available():
        log.info("mlflow not installed; skipping experiment tracking (run folder unaffected)")
        return NullTracker()
    try:
        return MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)
    except Exception as exc:  # pragma: no cover - depends on server state
        log.warning("could not initialise mlflow (%s); continuing without tracking", exc)
        return NullTracker()


def _flatten(obj: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, name, sep))
        else:
            out[name] = value
    return out


def _clip(value: Any, limit: int = 480) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
