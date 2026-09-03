#!/bin/bash
#SBATCH --job-name=flux_cmaes
#SBATCH --output=jobs_to_send/logs/cmaes_%j.out
#SBATCH --error=jobs_to_send/logs/cmaes_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=04:00:00


# --- 1. Load Modules ---
module load python/3.12
python3 --version

# --- 2. Activate Environment ---
source .venv/bin/activate

# --- 3. Run Simulation ---
# We use 12 workers to match the cpus-per-task
python funsies_experiments/buildingzone.py \
    --iters 100 \
    --popsize 12 \
    --workers 12 \
    --out_dir "data/cmaes_job_${SLURM_JOB_ID}" \
    --seed ${SLURM_JOB_ID}
echo "pizza is the best"