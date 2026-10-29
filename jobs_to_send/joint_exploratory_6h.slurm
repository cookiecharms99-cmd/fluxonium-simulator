#!/usr/bin/env bash
#SBATCH --job-name=flux-joint-explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=cluster_logs/slurm-joint-exploratory-%j.out
#SBATCH --error=cluster_logs/slurm-joint-exploratory-%j.err

# Exploratory joint flux-plus-charge optimization. It uses the calibrated
# cutoff-20 device settings, but 60 free controls rather than the professor's
# fixed-charge reference protocol; do not compare its score directly with the
# flux-only benchmark without declaring the different control budget.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/fluxonium-simulator}"
cd "$PROJECT_ROOT"

JOB_LABEL="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"

exec uv run python main/run_cmaes.py \
  --physics-config config/joint_exploratory.json \
  --control-protocol joint \
  --initialization random \
  --full-physics \
  --duration 150 \
  --generations 10000 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  --out-dir "data/output/cmaes/joint_exploratory_6h_${JOB_LABEL}"
