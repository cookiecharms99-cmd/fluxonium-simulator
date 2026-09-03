# 🚀 Cluster Submission Guide (SLURM)

This guide explains how to run your Fluxonium optimizations on a High-Performance Computing (HPC) cluster using SLURM.

## 1. The SLURM Script (`run_cmaes.slurm`)
The script I created handles the "paperwork" for the cluster. Here is what the headers mean:
- `--job-name`: How the job appears in the queue.
- `--output/error`: Where the "print" statements and errors are saved. `%j` is replaced by the Job ID.
- `--cpus-per-task`: Currently set to `1` because `buildingzone.py` is running serially. 
- `--mem`: Memory limit. Fluxonium sims are usually light, but 8GB is a safe starting point.
- `--time`: Max runtime (HH:MM:SS). The job will be killed if it exceeds this.

## 2. Common Commands

### Submit a job
```bash
sbatch jobs_to_send/run_cmaes.slurm
```

### Check your jobs
```bash
squeue -u your_username
```

### Cancel a job
```bash
scancel <JOB_ID>
```

### View logs (live)
```bash
tail -f jobs_to_send/logs/cmaes_<JOB_ID>.out
```


