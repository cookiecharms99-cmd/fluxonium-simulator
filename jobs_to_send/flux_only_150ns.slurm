#!/usr/bin/env bash
#SBATCH --partition=shortq7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=cluster_logs/slurm-flux-dim-%j.out
#SBATCH --error=cluster_logs/slurm-flux-dim-%j.err

# What this does: train flux from random initialization with varying dimensions: 20,40,60,80



set -euo pipefail

# go home :3
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/fluxonium-simulator}"
cd "$PROJECT_ROOT"

# set label
JOB_LABEL="${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)}"

# make sure we can find binaries :D
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"


for dim in 20 40 60 80; do

    echo "launching dimension D = ${dim} okie?"
    RUN_OUT_DIR="data/output/cmaes/flux_150ns_dimtest_${dim}_${JOB_LABEL}"


    uv run python main/run_cmaes.py \
    --physics-config config/flux_only_150ns.json \
    --full-physics \
    --duration 150 \
    --initialization random \
    --generations 250 \
    --popsize 16 \
    --workers 16 \
    --flux-count "$dim"  \
    --seed "$dim" \
    --out-dir "data/output/cmaes/flux_150ns_dimtest_${dim}_${JOB_LABEL}"

    echo "done with the dimension ${dim} :3"
done
echo "finished the runs yippieee :D"
