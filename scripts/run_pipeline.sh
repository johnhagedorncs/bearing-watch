#!/usr/bin/env bash
#
# The whole pipeline on synthetic data, from a clean clone, with no downloads.
#
#     ./scripts/run_pipeline.sh
#
# Every step is a separate command against a durable artifact, which is the same
# shape as the Argo workflow in deploy/argo/. Nothing here is a wrapper around a
# single long-running script: `infer` genuinely re-loads the run folder from
# disk, in a new process, with no shared state.

set -euo pipefail

DATA="${DATA:-data/raw/synthetic}"
RUNS="${RUNS:-runs}"
MODEL="${MODEL:-autoencoder}"
N_SNAPSHOTS="${N_SNAPSHOTS:-320}"
HEALTHY_FRACTION="${HEALTHY_FRACTION:-0.40}"
PYTHON="${PYTHON:-python3}"

# Index where degradation actually begins. Known only because this data is
# synthetic; the real IMS sets carry no labelled fault time.
TRUTH=$(awk "BEGIN {printf \"%d\", ${N_SNAPSHOTS} * ${HEALTHY_FRACTION}}")

if command -v bearingwatch >/dev/null 2>&1; then
  BW="bearingwatch"
else
  echo "note: 'bearingwatch' is not on PATH; falling back to 'python -m bearingwatch'"
  export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
  BW="${PYTHON} -m bearingwatch"
fi

step() {
  echo
  echo "=============================================================================="
  echo "$1"
  echo "=============================================================================="
}

step "1/6  generate synthetic data in the IMS on-disk format"
if [ -d "${DATA}" ] && [ -n "$(ls -A "${DATA}" 2>/dev/null)" ]; then
  echo "  ${DATA} already populated, skipping generation"
else
  ${PYTHON} scripts/make_synthetic_data.py \
    --out "${DATA}" \
    --n-snapshots "${N_SNAPSHOTS}" \
    --healthy-fraction "${HEALTHY_FRACTION}"
fi

step "2/6  preflight: does the data exist, at the density claimed, in the format expected"
${BW} preflight --data "${DATA}"

step "3/6  train on the declared healthy window only"
${BW} train --data "${DATA}" --model "${MODEL}" --runs-root "${RUNS}"

step "4/6  infer: load the run folder by path and score the whole run"
${BW} infer \
  --data "${DATA}" \
  --runs-root "${RUNS}" \
  --truth-degradation-index "${TRUTH}" \
  --fail-if-quiet

step "5/6  compare: autoencoder vs PCA vs a kurtosis threshold, one identical split"
${BW} compare \
  --data "${DATA}" \
  --truth-degradation-index "${TRUTH}" \
  --out docs/comparison.json

step "6/6  drift demonstration: a sensor gain change is indistinguishable from wear"
${PYTHON} scripts/drift_demo.py \
  --data "${DATA}" \
  --runs-root "${RUNS}" \
  --truth-degradation-index "${TRUTH}" \
  --out docs/drift_demo.json

echo
echo "=============================================================================="
echo "done."
echo
echo "  run folders     ${RUNS}/"
echo "  comparison      docs/comparison.json"
echo "  drift demo      docs/drift_demo.json"
echo
echo "EVERY NUMBER ABOVE CAME FROM SYNTHETIC DATA. It demonstrates that the"
echo "pipeline runs and that the detector fires in the degradation phase and not"
echo "in the healthy phase. It is not a result about real bearings. See"
echo "scripts/download_ims.md to obtain the real dataset and docs/RESULTS.md for"
echo "what has to be regenerated."
echo "=============================================================================="
