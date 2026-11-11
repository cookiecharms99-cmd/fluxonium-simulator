#!/usr/bin/env bash

# Four controlled 150-ns experiments at X = flux-only mode, 60D vector, random initialization
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GENERATIONS="${GENERATIONS:-600}"
POPSIZE="${POPSIZE:-16}"
WORKERS="${WORKERS:-8}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

run_experiment() {
  local label="$1"
  local seed=$2
  local flux_count=$3

  uv run python main/run_cmaes.py \
    --physics-config config/flux_only_150ns.json \
    --full-physics \
    --duration 150 \
    --initialization random \
    --generations "$GENERATIONS" \
    --popsize "$POPSIZE" \
    --workers "$WORKERS" \
    --out-dir "data/output/cmaes/${label}_${RUN_TAG}"
}

run_experiment \
"random_flux_40D" \
1234 \
40

run_experiment \
"random_flux_60D" \
31415 \
60

run_experiment \
"random_flux_80D" \
894849 \
80
