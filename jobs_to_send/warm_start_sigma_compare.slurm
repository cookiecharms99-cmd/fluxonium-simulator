#!/usr/bin/env bash
#SBATCH --job-name=warm-sigma
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=cluster_logs/slurm-warm-sigma-%A_%a.out
#SBATCH --error=cluster_logs/slurm-warm-sigma-%A_%a.err

# Fair 150-ns fixed-charge continuation comparison.
#
# Every array task starts from the same saved 40-flux-coefficient pulse and
# uses the same numerical model, seed, 200-generation evaluation budget, and
# worker count. Only sigma0 changes:
#   0: copy the parent run's final sigma (history fallback supports old runs)
#   1: sqrt-loss sigma from the shared initial-center population
#   2: explicit small middle-scale sigma, adjustable through FIXED_SIGMA0
#
# Compare best reward above the shared initial center, sigma evolution, and
# improvement per evaluation. Do not compare raw late-generation rewards only.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/fluxonium-simulator}"
cd "$PROJECT_ROOT"

PARENT_RUN="data/output/cmaes/reference_6h_legacy_4776599"
PARENT_WEIGHTS="${PARENT_RUN}/best_weights.npy"
if [[ ! -f "$PARENT_WEIGHTS" ]]; then
  echo "Missing parent weights: $PARENT_WEIGHTS" >&2
  exit 1
fi

SIGMA_MODES=(parent_final sqrt_loss fixed)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "$TASK_INDEX" =~ ^[0-2]$ ]]; then
  echo "Expected array index 0, 1, or 2; got $TASK_INDEX." >&2
  exit 1
fi
SIGMA_MODE="${SIGMA_MODES[$TASK_INDEX]}"
FIXED_SIGMA0="${FIXED_SIGMA0:-0.005}"

SIGMA_ARGS=(--sigma0-mode "$SIGMA_MODE")
if [[ "$SIGMA_MODE" == "fixed" ]]; then
  SIGMA_ARGS+=(--sigma0 "$FIXED_SIGMA0")
fi

JOB_LABEL="${SLURM_ARRAY_JOB_ID:-local}_$TASK_INDEX"
RUN_LABEL="warm_sigma_${SIGMA_MODE}_${JOB_LABEL}"

uv run python main/run_cmaes.py \
  --physics-config config/device_calibrated.json \
  --initialization warm_start \
  --warm-start "$PARENT_WEIGHTS" \
  --full-physics \
  --duration 150 \
  --generations 200 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  "${SIGMA_ARGS[@]}" \
  --out-dir "data/output/cmaes/${RUN_LABEL}"
