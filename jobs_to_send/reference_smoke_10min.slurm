#!/usr/bin/env bash
#SBATCH --job-name=flux-ref-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --signal=TERM@60
#SBATCH --output=cluster_logs/slurm-reference-smoke-%j.out
#SBATCH --error=cluster_logs/slurm-reference-smoke-%j.err

# Validate the reference protocol, multiprocessing, artifact creation, and
# cluster environment before submitting the six-hour scientific campaign.
set -euo pipefail



PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/fluxonium-simulator}"
cd "$PROJECT_ROOT"

echo "Current directory on compute node: $(pwd)"
ls -ld main/run_cmaes.py
JOB_LABEL="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"

uv run python main/run_cmaes.py \
  --physics-config config/reference_prof.json \
  --control-protocol reference \
  --initialization legacy_piecewise_lstsq \
  --generations 20 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  --out-dir "data/output/cmaes/reference_smoke_${JOB_LABEL}"
