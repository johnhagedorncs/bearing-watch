"""FastAPI scoring service.

Endpoints:
    GET  /health   liveness -- the process is up
    GET  /ready    readiness -- a model is loaded and can score
    GET  /model    which run is loaded, and the feature order /score expects
    POST /score    score a snapshot or feature vectors
    POST /reload   swap in a different run folder without restarting

The model loads ONCE at startup, from a run folder, and the identity of that run
is in every response. That is not decoration: a score is only interpretable
alongside the model that produced it, and "which model was serving on the 14th"
is the first question asked about any surprising result. Answering it from a
container image tag is guesswork; answering it from a run_id echoed in the
response is not.

Health and readiness are separate. Liveness restarts a wedged process;
readiness gates traffic. A pod whose model failed to load is alive and must not
receive traffic, and one endpoint cannot express both.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status

from .. import __version__
from ..data.cache import SCALED_METADATA_KEY
from .deps import RUN_DIR_ENV, LoadedRun, get_loaded_run, load_run, set_loaded_run
from .schemas import (
    ErrorResponse,
    HealthResponse,
    ModelResponse,
    ReadyResponse,
    ReloadRequest,
    ScoreItem,
    ScoreRequest,
    ScoreResponse,
)

log = logging.getLogger(__name__)

# Minimum snapshots in one request before the band-split discriminant is worth
# reporting. Below this the window median is noise and the verdict would flap.
MIN_BAND_SPLIT_ROWS = 5


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model at startup.

    A failure here is logged and swallowed rather than raised. The process must
    come up so that /health and /ready can answer -- a container that exits on a
    missing model gives an orchestrator a crash loop and no diagnostic, whereas
    one that reports not-ready is inspectable.
    """
    run_dir = os.environ.get(RUN_DIR_ENV)
    try:
        loaded = load_run(run_dir)
        set_loaded_run(loaded)
        log.info("loaded run %s (%s) from %s", loaded.run_id, loaded.model.kind, loaded.path)
    except Exception as exc:
        set_loaded_run(None)
        log.error(
            "could not load a model at startup (%s). The service is UP but NOT READY; "
            "set %s or POST /reload.",
            exc,
            RUN_DIR_ENV,
        )
    yield
    set_loaded_run(None)


app = FastAPI(
    title="bearing-watch",
    version=__version__,
    description=(
        "Anomaly scoring for run-to-failure bearing vibration data. Scores come from a "
        "model trained only on a declared healthy window; the threshold references that "
        "same declared period rather than the series being scored."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness. Deliberately does NOT touch the model.

    A liveness probe that checks the model restarts the pod when the model is
    missing, which does not fix a missing model and does hide the problem behind
    a crash loop.
    """
    return HealthResponse(status="ok", version=__version__)


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready() -> ReadyResponse:
    """Readiness: is a model loaded and able to score?"""
    try:
        loaded = get_loaded_run()
    except RuntimeError as exc:
        return ReadyResponse(ready=False, reason=str(exc))
    return ReadyResponse(ready=True, run_id=loaded.run_id)


@app.get(
    "/model",
    response_model=ModelResponse,
    tags=["model"],
    responses={503: {"model": ErrorResponse}},
)
def model_info(loaded: LoadedRun = Depends(get_loaded_run)) -> ModelResponse:
    """Metadata for the loaded run, including the feature order /score expects."""
    d = loaded.describe()
    return ModelResponse(
        run_id=d["run_id"],
        run_path=d["run_path"],
        model_kind=d["model_kind"],
        model_version=d["config_hash"],
        n_features=d["n_features"],
        feature_names=d["feature_names"],
        created_at=d["created_at"],
        code_version=d["code_version"],
        threshold=d["threshold"],
        expected_points=d["expected_points"],
        expected_channels=d["expected_channels"],
        sample_rate_hz=d["sample_rate_hz"],
        data_root=d["data_root"],
        band_split_available=d["band_split_available"],
    )


@app.post(
    "/score",
    response_model=ScoreResponse,
    tags=["model"],
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def score(req: ScoreRequest, loaded: LoadedRun = Depends(get_loaded_run)) -> ScoreResponse:
    """Score a raw snapshot or a batch of feature vectors."""
    try:
        if req.snapshot is not None:
            arr = np.asarray(req.snapshot, dtype=np.float64)
            scores = loaded.score_signal(arr)
            raw_matrix = None
        else:
            raw_matrix = np.atleast_2d(np.asarray(req.features, dtype=np.float64))
            if raw_matrix.shape[0] > loaded.config.serve.max_batch:
                raise ValueError(
                    f"batch of {raw_matrix.shape[0]} exceeds serve.max_batch="
                    f"{loaded.config.serve.max_batch}"
                )
            scores = loaded.score_matrix(raw_matrix)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    verdicts = loaded.verdict(scores)

    band_split = None
    if (
        loaded.band_split is not None
        and raw_matrix is not None
        and raw_matrix.shape[0] >= MIN_BAND_SPLIT_ROWS
    ):
        band_split = _band_split_verdict(loaded, raw_matrix)

    return ScoreResponse(
        results=[ScoreItem(**v) for v in verdicts],
        n_scored=len(verdicts),
        run_id=loaded.run_id,
        model_kind=loaded.model.kind,
        model_version=loaded.manifest.config_hash,
        n_features=len(loaded.feature_names),
        snapshot_id=req.snapshot_id,
        band_split=band_split,
    )


def _band_split_verdict(loaded: LoadedRun, raw_matrix: np.ndarray) -> dict[str, Any] | None:
    """Run the gain-vs-mechanical discriminant on a batch, if it applies."""
    import pandas as pd

    if loaded.band_split is None:
        return None
    frame = pd.DataFrame(raw_matrix, columns=loaded.feature_names)
    frame.attrs["bearingwatch"] = {SCALED_METADATA_KEY: False}
    try:
        return loaded.band_split.classify(frame).to_dict()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("band-split classification failed: %s", exc)
        return None


@app.post(
    "/reload",
    response_model=ModelResponse,
    tags=["ops"],
    responses={400: {"model": ErrorResponse}},
)
def reload_model(req: ReloadRequest) -> ModelResponse:
    """Load a different run folder without restarting the process.

    Useful locally and in a demo. In a cluster the deployment is the mechanism:
    a new model is a new pod pointed at a new run folder, so the running model
    is a property of the deployed manifest rather than of whoever last called
    this endpoint.
    """
    try:
        loaded = load_run(req.run_dir)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    set_loaded_run(loaded)
    log.info("reloaded run %s from %s", loaded.run_id, loaded.path)
    return model_info(loaded)
