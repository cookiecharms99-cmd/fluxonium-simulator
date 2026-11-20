#!/usr/bin/env bash

# Controlled 150-ns flux-only dimension comparison. The common calibrated
# device config is overridden only by the declared flux dimension.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GENERATIONS="${GENERATIONS:-300}"
POPSIZE="${POPSIZE:-16}"
WORKERS="${WORKERS:-8}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

run_experiment() {
  local label="$1"
  local flux_count=$2

  uv run python main/run_cmaes.py \
    --physics-config config/device_calibrated.json \
    --full-physics \
    --duration 150 \
    --initialization random \
    --seed 1234 \
    --generations "$GENERATIONS" \
    --popsize "$POPSIZE" \
    --workers "$WORKERS" \
    --flux-count "$flux_count" \
    --out-dir "data/output/cmaes/${label}_${RUN_TAG}"
}


run_experiment \
"random_flux_20D" \
20

run_experiment \
"random_flux_40D" \
40

run_experiment \
"random_flux_60D" \
 60


run_experiment \
"random_flux_80D" \
80
