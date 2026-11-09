#!/usr/bin/env bash

# Four controlled 150-ns experiments. Each one has the same numerical model,
# population, and RNG seed; only the declared active/fixed controls differ.
# Run sequentially so this laptop is not oversubscribed overnight.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GENERATIONS="${GENERATIONS:-200}"
POPSIZE="${POPSIZE:-16}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1234}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

run_experiment() {
  local label="$1"
  local config="$2"
  local initialization="$3"

  uv run python main/run_cmaes.py \
    --physics-config "$config" \
    --initialization "$initialization" \
    --full-physics \
    --duration 150 \
    --generations "$GENERATIONS" \
    --popsize "$POPSIZE" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --out-dir "data/output/cmaes/${label}_${RUN_TAG}"
}

# 1. Fixed random flux waveform; optimize only charge corrections.
run_experiment \
  "charge_on_random_flux_150ns" \
  "config/charge_only_random_flux_150ns.json" \
  "random"

# 2. Idle flux; optimize only charge corrections.
run_experiment \
  "charge_on_idle_flux_150ns" \
  "config/charge_only_idle_150ns.json" \
  "random"

# 3. Joint search starts with random flux coefficients but zero charge correction.
run_experiment \
  "joint_random_flux_zero_charge_150ns" \
  "config/joint_run.json" \
  "random_flux_zero_charge"

# 4. Hold the saved 0.995657 flux waveform fixed; optimize charge corrections.
run_experiment \
  "charge_on_good_flux_150ns" \
  "config/charge_only_good_flux_150ns.json" \
  "random"
