# bearing-watch
#
# Quickstart from a clean clone, no downloads:
#
#     make demo
#
# That generates synthetic data in the real IMS format, trains, scores, and
# runs the drift demonstration.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON      ?= python3
VENV        ?= .venv
BIN         := $(VENV)/bin
DATA        ?= data/raw/synthetic
RUNS        ?= runs
MODEL       ?= autoencoder
N_SNAPSHOTS ?= 320
HEALTHY     ?= 0.40
# Index where degradation starts in the synthetic data: n_snapshots * healthy.
TRUTH       := $(shell echo $$(( $(N_SNAPSHOTS) * 40 / 100 )))

.PHONY: help
help:
	@echo "bearing-watch"
	@echo ""
	@echo "  make demo         full pipeline on synthetic data (start here)"
	@echo ""
	@echo "  make venv         create a virtualenv and install the package"
	@echo "  make synth        generate synthetic data in IMS format"
	@echo "  make preflight    run the data gates"
	@echo "  make train        train on the healthy window -> runs/<run_id>/"
	@echo "  make infer        score the run with the newest model"
	@echo "  make compare      autoencoder vs PCA vs kurtosis, one split"
	@echo "  make ablate       do kurtosis and the defect bands earn their place?"
	@echo "  make drift-demo   the gain-change vs degradation demonstration"
	@echo ""
	@echo "  make test         pytest with coverage"
	@echo "  make lint         ruff + mypy"
	@echo "  make fmt          ruff format"
	@echo "  make check        lint + test"
	@echo ""
	@echo "  make serve        run the API locally"
	@echo "  make docker       build the runtime image"
	@echo "  make compose-up   API + MLflow + monitor"
	@echo "  make k8s-build    render the kustomize overlays"
	@echo ""
	@echo "  make clean        remove caches and generated artifacts"

# ---------------------------------------------------------------- environment

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

.PHONY: venv
venv: $(BIN)/python
	$(BIN)/python -m pip install -e ".[dev]"
	@echo "installed. optional extras: pip install -e '.[tracking,monitor,s3]'"

.PHONY: venv-all
venv-all: $(BIN)/python
	$(BIN)/python -m pip install -e ".[dev,tracking,monitor,s3]"

# ------------------------------------------------------------------ pipeline

.PHONY: synth
synth: venv
	$(BIN)/python scripts/make_synthetic_data.py \
		--out $(DATA) --n-snapshots $(N_SNAPSHOTS) --healthy-fraction $(HEALTHY)

.PHONY: preflight
preflight:
	$(BIN)/bearingwatch preflight --data $(DATA)

.PHONY: train
train:
	$(BIN)/bearingwatch train --data $(DATA) --model $(MODEL) --runs-root $(RUNS)

.PHONY: infer
infer:
	$(BIN)/bearingwatch infer --data $(DATA) --runs-root $(RUNS) \
		--truth-degradation-index $(TRUTH)

.PHONY: compare
compare:
	$(BIN)/bearingwatch compare --data $(DATA) \
		--truth-degradation-index $(TRUTH) --out docs/comparison.json

.PHONY: ablate
ablate:
	$(BIN)/bearingwatch ablate --data $(DATA) --model $(MODEL) \
		--out docs/ablation-$(MODEL).json

.PHONY: drift-demo
drift-demo:
	$(BIN)/python scripts/drift_demo.py --data $(DATA) --runs-root $(RUNS) \
		--truth-degradation-index $(TRUTH) --out docs/drift_demo.json

.PHONY: runs
runs:
	$(BIN)/bearingwatch runs --runs-root $(RUNS)

# The one command a reviewer should need.
.PHONY: demo
demo: venv synth preflight train infer compare drift-demo
	@echo ""
	@echo "=============================================================="
	@echo "done. every number above came from SYNTHETIC data."
	@echo "see docs/RESULTS.md, and README.md for what must be regenerated"
	@echo "on the real NASA IMS dataset before any of it is a claim."
	@echo "=============================================================="

# --------------------------------------------------------------------- checks

.PHONY: test
test:
	$(BIN)/pytest --cov=bearingwatch --cov-report=term-missing -q

.PHONY: test-fast
test-fast:
	$(BIN)/pytest -q -x

.PHONY: lint
lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/ruff format --check src tests scripts
	$(BIN)/mypy

.PHONY: fmt
fmt:
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

.PHONY: check
check: lint test

# ------------------------------------------------------------------- serving

.PHONY: serve
serve:
	$(BIN)/bearingwatch serve --runs-root $(RUNS)

.PHONY: docker
docker:
	docker build --target runtime -t bearing-watch:local .

.PHONY: compose-up
compose-up:
	docker compose up --build

.PHONY: compose-down
compose-down:
	docker compose down -v

.PHONY: k8s-build
k8s-build:
	@for overlay in deploy/k8s/overlays/*/; do \
		echo "=== $$overlay ==="; \
		kustomize build --enable-alpha-plugins "$$overlay" > /dev/null && echo "ok"; \
	done

# --------------------------------------------------------------------- clean

.PHONY: clean
clean:
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV) data/raw/synthetic mlruns monitoring
	find runs -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
