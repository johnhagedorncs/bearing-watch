"""Request and response models for the scoring API.

The request accepts EITHER a raw snapshot or a pre-computed feature vector, and
never both. Two paths because they serve different callers: an edge collector
posts the waveform and lets the service featurise it, while a batch job that has
already run the feature pipeline posts vectors and skips the work. Rejecting the
both-supplied case explicitly beats silently preferring one, which turns a
caller's bug into a wrong number instead of an error.

Feature vectors must be in the exact order given by ``GET /model``. A vector in
the wrong order is still the right length, so nothing downstream can catch it --
the model scores it and returns a plausible number. The endpoint documents the
order and the response echoes ``n_features`` so a caller can assert on it.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator

# NOTE: this module uses typing.Optional / typing.Union rather than the `X | Y`
# syntax used everywhere else in the package. `from __future__ import
# annotations` defers annotations for static tools, but pydantic RESOLVES them
# at runtime to build validators, and `list[float] | None` cannot be evaluated
# on Python 3.9. The future import does not help here -- this is the one place
# in the codebase where the older spelling is required rather than preferred.


class ScoreRequest(BaseModel):
    """Score a snapshot or a batch of feature vectors."""

    snapshot: Optional[Union[list[list[float]], list[float]]] = Field(
        default=None,
        description=(
            "Raw snapshot as (n_points, n_channels), or a flat list for one channel. "
            "Must have exactly the point count this run was trained on (see /model)."
        ),
    )
    features: Optional[Union[list[list[float]], list[float]]] = Field(
        default=None,
        description=(
            "Pre-computed feature vector(s), in the exact order returned by GET /model. "
            "Order is not checkable server-side -- a permuted vector scores as a valid number."
        ),
    )
    snapshot_id: Optional[str] = Field(
        default=None, description="Caller-supplied id, echoed back for correlation."
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> ScoreRequest:
        if (self.snapshot is None) == (self.features is None):
            raise ValueError(
                "provide exactly one of 'snapshot' or 'features'. Supplying both is "
                "ambiguous and supplying neither is empty."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"features": [[0.1, 0.2, 0.3]], "snapshot_id": "rig-a-2024-02-12T10:32:39"}
            ]
        }
    }


class ScoreItem(BaseModel):
    score: float = Field(description="Anomaly score. Higher is more anomalous.")
    z: float = Field(description="Robust z against the declared healthy reference period.")
    threshold_z: float
    score_threshold: float
    anomalous: bool
    verdict: str


class ScoreResponse(BaseModel):
    results: list[ScoreItem]
    n_scored: int
    run_id: str = Field(description="Run folder that produced this score.")
    model_kind: str
    model_version: str = Field(
        description="Config hash of the run. Two runs with the same hash used the same settings."
    )
    n_features: int
    snapshot_id: Optional[str] = None
    band_split: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Gain-change vs mechanical-change verdict, when enough snapshots were "
            "supplied in one request to judge a window."
        ),
    )

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness is distinct from health.

    ``/health`` says the process is alive; ``/ready`` says it can actually serve
    a request, which requires a loaded model. Collapsing them into one endpoint
    means a pod with no model still receives traffic and returns 500s.
    """

    ready: bool
    run_id: Optional[str] = None
    reason: Optional[str] = None


class ModelResponse(BaseModel):
    run_id: str
    run_path: str
    model_kind: str
    model_version: str
    n_features: int
    feature_names: list[str] = Field(
        description="Exact order required by the 'features' field of POST /score."
    )
    created_at: str
    code_version: dict[str, Any]
    threshold: dict[str, Any]
    expected_points: int
    expected_channels: int
    sample_rate_hz: int
    data_root: str
    band_split_available: bool

    model_config = {"protected_namespaces": ()}


class ReloadRequest(BaseModel):
    run_dir: str = Field(description="Run folder to load in place of the current one.")


class ErrorResponse(BaseModel):
    detail: str
