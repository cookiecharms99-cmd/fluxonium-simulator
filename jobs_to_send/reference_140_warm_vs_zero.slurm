#!/usr/bin/env bash
#SBATCH --job-name=flux-140-init
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@180
#SBATCH --array=0-1
#SBATCH --output=cluster_logs/slurm-140-init-%A_%a.out
#SBATCH --error=cluster_logs/slurm-140-init-%A_%a.err

# Fair duration-continuation comparison:
#   task 0 = neutral zero-coefficient center
#   task 1 = compatible 150-ns reference weights, time-compressed to 140 ns
# Both tasks use the same reference physics, seed, CMA-ES population, worker
# count, and 400-generation (6,400 candidate evaluation) budget.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/fluxonium-simulator}"
cd "$PROJECT_ROOT"

PARENT_WEIGHTS="data/output/cmaes/reference_6h_legacy_4776599/best_weights.npy"
if [[ ! -f "$PARENT_WEIGHTS" ]]; then
  echo "Missing parent warm-start artifact: $PARENT_WEIGHTS" >&2
  echo "Copy the 150-ns reference run into data/output/cmaes/ before submitting." >&2
  exit 1
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    INITIALIZATION="zero"
    EXTRA_ARGS=()
    ;;
  1)
    INITIALIZATION="warm_start"
    EXTRA_ARGS=(--warm-start "$PARENT_WEIGHTS")
    ;;
  *)
    echo "Unexpected array task id: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN_LABEL="140ns_${INITIALIZATION}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Initialization: $INITIALIZATION"
echo "Output: data/output/cmaes/$RUN_LABEL"

exec uv run python main/run_cmaes.py \
  --physics-config config/reference_prof.json \
  --control-protocol reference \
  --initialization "$INITIALIZATION" \
  "${EXTRA_ARGS[@]}" \
  --full-physics \
  --duration 140 \
  --generations 400 \
  --popsize 16 \
  --workers 4 \
  --seed 1234 \
  --out-dir "data/output/cmaes/$RUN_LABEL"
