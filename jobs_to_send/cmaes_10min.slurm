#!/usr/bin/env bash
#SBATCH --job-name=fluxonium-cmaes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --signal=TERM@60
#SBATCH --output=cluster_logs/slurm-%j.out
#SBATCH --error=cluster_logs/slurm-%j.err

# Stop on an error, an unset variable, or a failed pipeline command.
set -euo pipefail

# This works regardless of the directory from which sbatch was called.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# SLURM substitutes its job ID on the cluster. The fallback permits a local
# Bash smoke test without overwriting a previous local run.
JOB_LABEL="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"

uv run python main/run_cmaes.py \
  --generations 20 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  --out-dir "data/output/cmaes/10min_${JOB_LABEL}"
