#!/usr/bin/env bash
#SBATCH --job-name=charge-good-flux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=cluster_logs/slurm-charge-good-flux-%j.out
#SBATCH --error=cluster_logs/slurm-charge-good-flux-%j.err

# Hold the saved high-quality 150-ns flux pulse fixed. CMA-ES may adjust only
# 20 charge-envelope corrections, beginning from exactly zero correction.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/fluxonium-simulator}"
cd "$PROJECT_ROOT"

PARENT_CONTROLS="data/output/cmaes/reference_6h_legacy_4776599/controls.npz"
if [[ ! -f "$PARENT_CONTROLS" ]]; then
  echo "Missing saved flux controls: $PARENT_CONTROLS" >&2
  exit 1
fi

JOB_LABEL="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"

uv run python main/run_cmaes.py \
  --physics-config config/charge_only_good_flux_150ns_cutoff20.json \
  --initialization zero \
  --full-physics \
  --duration 150 \
  --generations 10000 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  --out-dir "data/output/cmaes/charge_on_good_flux_150ns_cutoff20_${JOB_LABEL}"
