"""The run manifest: what was run, on what, with what, producing what.

A model file on its own answers none of the questions that get asked about it
six weeks later. Which commit produced this? Which config? Which data? Was the
scaler saved next to it the one it was trained with? Where did the metrics go?

Without a manifest those questions are answered by memory and by filesystem
timestamps, which is to say they are answered wrong. The manifest is written
once, at the end of a successful run, into the run folder, and it records:

* ``run_id`` and creation time
* ``code_version`` -- package version plus git commit when one is available
* ``config_hash`` and a full config snapshot path
* ``feature_cache_key`` -- so a scored run can be traced to the exact feature
  parameters, which is the thing a stale cache would otherwise hide
* ``data_fingerprint`` and the preflight report
* every output path the run wrote
* the environment (python, numpy, torch versions)

The manifest is written LAST. A run folder with a manifest completed; a run
folder without one crashed part way, and the difference is visible from a
directory listing instead of by opening files.
"""

from __future__ import annotations

import contextlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["MANIFEST_NAME", "Manifest", "code_version", "environment_info", "git_commit"]

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Current git commit, or None outside a repository.

    Returns None rather than raising. A run outside a checkout is legitimate --
    the container has no .git -- and the manifest records the absence honestly
    instead of failing the run over provenance metadata.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_dirty(cwd: str | Path | None = None) -> bool | None:
    """Whether the working tree has uncommitted changes.

    A dirty tree means the recorded commit does NOT describe the code that ran,
    so the manifest flags it rather than implying reproducibility it cannot
    offer.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def code_version() -> dict[str, Any]:
    try:
        from bearingwatch import __version__
    except Exception:  # pragma: no cover
        __version__ = "unknown"
    commit = git_commit()
    return {
        "package_version": __version__,
        "git_commit": commit,
        "git_dirty": git_dirty() if commit else None,
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("numpy", "scipy", "pandas", "sklearn", "torch", "mlflow", "evidently"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[name] = None
    return info


@dataclass
class Manifest:
    """Everything needed to explain, reproduce, or distrust a run."""

    run_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    completed_at: str | None = None
    schema_version: int = MANIFEST_SCHEMA_VERSION
    command: str = ""
    status: str = "running"

    code_version: dict[str, Any] = field(default_factory=code_version)
    environment: dict[str, Any] = field(default_factory=environment_info)

    config_hash: str = ""
    feature_cache_key: str = ""
    model_kind: str = ""

    data_root: str = ""
    data_fingerprint: dict[str, Any] = field(default_factory=dict)
    preflight: dict[str, Any] = field(default_factory=dict)

    healthy_window: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    parent_run_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def add_output(self, name: str, path: str | Path, run_dir: str | Path | None = None) -> None:
        """Record an output path, relative to the run folder when possible.

        Relative paths keep a run folder portable: moving or copying it, or
        mounting it at a different path in a container, does not invalidate
        every reference inside its own manifest.
        """
        path = Path(path)
        if run_dir is not None:
            # A path outside the run folder stays absolute rather than failing.
            with contextlib.suppress(ValueError):
                path = path.relative_to(Path(run_dir))
        self.outputs[name] = str(path)

    def finalise(self, status: str = "complete") -> Manifest:
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / MANIFEST_NAME
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, default=str)
        return path

    @classmethod
    def load(cls, run_dir: str | Path) -> Manifest:
        run_dir = Path(run_dir)
        path = run_dir / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {run_dir}. A run folder without a manifest is an "
                "INCOMPLETE run -- the manifest is written last, so its absence means the "
                "run crashed before finishing. Do not serve a model from it."
            )
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})
