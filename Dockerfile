# Multi-stage build: a builder that compiles wheels, and a slim runtime that
# gets none of the build toolchain.
#
# The split is not about image size for its own sake. The runtime image has no
# compiler, no build headers, no pytest, no ruff and no mypy, so the attack
# surface of what actually runs in production is the application and its runtime
# dependencies and nothing else. It also runs as a non-root user, because a
# container that runs as root has root on any host path it is handed.

# ---------------------------------------------------------------- builder ---
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what is needed to resolve dependencies first, so the dependency
# layer is cached and does not rebuild on every source edit.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Build a wheel and pre-download every runtime dependency into /wheels. The
# runtime stage installs from there with --no-index, so it needs no network and
# no package index at deploy time.
RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /wheels \
    && python -m pip wheel --wheel-dir /wheels /wheels/bearingwatch-*.whl

# ---------------------------------------------------------------- runtime ---
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="bearing-watch" \
      org.opencontainers.image.description="Run-to-failure bearing anomaly detection API" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Where the service looks for a run folder. Overridden per deployment; the
    # service comes up NOT READY rather than crash-looping if it is unset.
    BEARINGWATCH_RUN_DIR="" \
    BEARINGWATCH_RUNS_ROOT=/app/runs

# curl is here only for the HEALTHCHECK below. Without it the healthcheck has
# nothing to run; with it, that is the one extra binary in the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root, with a fixed uid so a volume mounted from the host has predictable
# ownership rather than depending on whatever uid the daemon assigns.
RUN groupadd --gid 10001 bearing \
    && useradd --uid 10001 --gid bearing --create-home --shell /usr/sbin/nologin bearing

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels bearingwatch \
    && rm -rf /wheels

# Config travels with the image so a container is runnable with no mounts. A
# deployment can still override any value through BEARINGWATCH_CFG__* env vars.
COPY --chown=bearing:bearing configs/ /app/configs/

# Mount point for run folders. Created here so it exists and is writable even
# when nothing is mounted over it.
RUN mkdir -p /app/runs && chown -R bearing:bearing /app

USER bearing

EXPOSE 8000

# Readiness, not liveness. A container whose model failed to load is alive and
# must report itself unhealthy so it does not receive traffic -- exiting instead
# would give the orchestrator a crash loop and no diagnostic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready | grep -q '"ready":true' || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["bearingwatch.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
