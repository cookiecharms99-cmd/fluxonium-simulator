#!/usr/bin/env bash

# Four controlled 150-ns experiments. Each one has the same calibrated device,
# numerical model, population, and RNG seed; only the declared active/fixed
# controls differ. The runner saves the resulting effective config per run.
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
  local initialization="$2"
  shift 2

  uv run python main/run_cmaes.py \
    --physics-config config/device_calibrated.json \
    --initialization "$initialization" \
    --full-physics \
    --duration 150 \
    --generations "$GENERATIONS" \
    --popsize "$POPSIZE" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    "$@" \
    --out-dir "data/output/cmaes/${label}_${RUN_TAG}"
}

# 1. Fixed random flux waveform; optimize only charge corrections.
run_experiment \
  "charge_on_random_flux_150ns" \
  "random" \
  --flux-count 0 \
  --charge-count 20 \
  --fixed-flux random_sine \
  --fixed-flux-harmonic-count 40 \
  --fixed-flux-seed 20261108 \
  --fixed-flux-scale 0.1

# 2. Idle flux; optimize only charge corrections.
run_experiment \
  "charge_on_idle_flux_150ns" \
  "random" \
  --flux-count 0 \
  --charge-count 20 \
  --fixed-flux idle

# 3. Joint search starts with random flux coefficients but zero charge correction.
run_experiment \
  "joint_random_flux_zero_charge_150ns" \
  "random_flux_zero_charge" \
  --flux-count 40 \
  --charge-count 20

# 4. Hold the saved 0.995657 flux waveform fixed; optimize charge corrections.
run_experiment \
  "charge_on_good_flux_150ns" \
  "random" \
  --flux-count 0 \
  --charge-count 20 \
  --fixed-flux saved_controls \
  --saved-flux-controls data/output/cmaes/reference_6h_legacy_4776599/controls.npz
