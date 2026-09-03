#!/bin/bash
#SBATCH --job-name=cmaes_job
#SBATCH --output=fluxonium_cmaes_output.txt
#SBATCH --error=fluxonium_cmaes_error.txt
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mem=32GB

# --- Load modern Python module ---
module load python/3.9
python3 --version   # logs Python version in output file
source ~/quantum_env/bin/activate  # Activate your Python virtual environment!

# --- Go to folder with scripts ---
cd /mnt/beegfs/home/jseepaul2023/research/quantum_project/exploring

# --- Create a timestamped output folder ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="/mnt/beegfs/home/jseepaul2023/research/quantum_project/outputs/run_$TIMESTAMP"
mkdir -p "$OUTDIR"

echo "Output folder: $OUTDIR"

# --- Run your CMA-ES Python script ---
python3 cmaes.py --out_dir "$OUTDIR" --workers 20 --T 150