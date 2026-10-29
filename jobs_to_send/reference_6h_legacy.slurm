#!/usr/bin/env bash
#SBATCH --job-name=flux-ref-legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=cluster_logs/slurm-reference-legacy-%j.out
#SBATCH --error=cluster_logs/slurm-reference-legacy-%j.err

# This is a bounded, full-physics reference campaign. Its large generation
# count is an upper limit; SLURM's six-hour wall time requests a clean
# generation-boundary checkpoint before termination.
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
  --full-physics \
  --duration 150 \
  --generations 10000 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  --out-dir "data/output/cmaes/reference_6h_legacy_${JOB_LABEL}"
