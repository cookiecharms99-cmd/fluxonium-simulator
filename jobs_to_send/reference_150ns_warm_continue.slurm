#!/usr/bin/env bash
#SBATCH --job-name=flux-warm-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=cluster_logs/slurm-flux-warm-40-%j.out
#SBATCH --error=cluster_logs/slurm-flux-warm-40-%j.err

# Continue the best 40-coefficient fixed-charge reference pulse at the same
# duration. This is a local CMA-ES refinement control, not a basis transfer.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/fluxonium-simulator}"
cd "$PROJECT_ROOT"

PARENT_WEIGHTS="data/output/cmaes/reference_6h_legacy_4776599/best_weights.npy"
if [[ ! -f "$PARENT_WEIGHTS" ]]; then
  echo "Missing parent weights: $PARENT_WEIGHTS" >&2
  exit 1
fi

JOB_LABEL="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"

uv run python main/run_cmaes.py \
  --physics-config config/reference_prof.json \
  --initialization warm_start \
  --warm-start "$PARENT_WEIGHTS" \
  --full-physics \
  --duration 150 \
  --generations 10000 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  --out-dir "data/output/cmaes/reference_150ns_warm_continue_${JOB_LABEL}"
