# Cluster smoke-job guide

This folder will hold the small SLURM submission scripts used to run the
Fluxonium reset optimizer on the cluster. Start with one tiny smoke job. Its
purpose is to confirm that the cluster can import the project, use the correct
Python environment, run multiprocessing, and save a complete run record.

Do not begin with a long optimization campaign. A successful one-generation
smoke job is the checkpoint before spending significant cluster time.

## Before making a submission script

On the cluster login node, confirm these facts first:

```bash
pwd
git status
uv --version
uv run python --version
```

Write down the absolute path printed by `pwd`. The submission script must
`cd` into that project directory before it invokes Python. If `uv` is not
available on the cluster, stop here and ask the cluster documentation or Dr.
Fu's group how the project environment should be activated; do not silently
switch to a different Python installation.

## The prepared ten-minute jobs

`jobs_to_send/cmaes_10min.slurm` is prepared for the first larger smoke run:
20 generations, population 16, and four worker processes for at most ten
minutes. It finds the repository from its own location, so it does not need a
machine-specific absolute path.

```bash
sbatch jobs_to_send/cmaes_10min.slurm
```

Why these choices:

- `--workers 4` creates four simulator worker processes.
- `--cpus-per-task=4` reserves four CPU cores for them. Keep these numbers
  equal for now.
- Twenty generations with population 16 means at most 320 objective
  evaluations. This is still a bounded smoke campaign, not a final baseline.
- The current runner sets numerical libraries to one thread per worker. That
  prevents two worker processes from each trying to occupy every CPU core.
- `%j` is replaced by the SLURM job ID. It gives the logs and output directory
  a unique name, so concurrent jobs cannot overwrite one another.
- `#SBATCH --signal=TERM@60` asks SLURM to warn the running processes 60
  seconds before its hard time limit. The runner then finishes the current
  generation, saves its checkpoint, and exits cleanly.

The script deliberately does **not** use `--full-physics`; the default smoke
configuration is cheaper and is the appropriate first time-limited run.

`reference_smoke_10min.slurm` performs the equivalent smoke check for the
fixed-charge, 40-flux-coefficient reference protocol. Submit it before the
six-hour campaign:

```bash
sbatch jobs_to_send/reference_smoke_10min.slurm
```

## The six-hour reference campaign

After `reference_smoke_10min.slurm` has produced complete artifacts, submit:

```bash
sbatch jobs_to_send/reference_6h_legacy.slurm
```

This uses cutoff 20, 500 time samples, 40 flux coefficients, the fixed
calibrated charge drive, and the 150-ns professor-inspired pulse projected
into the sine basis as the CMA-ES center. It requests four CPUs and runs up to
six hours. `--generations 10000` is only an upper bound; SLURM supplies the
real wall-clock limit and asks the runner to checkpoint 120 seconds before it.

It is a single-seed reference campaign, not an initialization comparison.
The laptop script `scripts/run_reference_initialization_compare.sh` runs the
equal-budget zero/random/legacy-initialization comparison locally.

## Submit and monitor

From the repository root on the cluster:

```bash
squeue -u "$USER"
```

SLURM will print a job ID after `sbatch`. Watch its standard-output log with:

```bash
tail -f cluster_logs/slurm-<JOB_ID>.out
```

If needed, cancel a queued or running job:

```bash
scancel <JOB_ID>
```

## After the job finishes

Check both the log and the run directory:

```bash
cat cluster_logs/slurm-<JOB_ID>.out
cat data/output/cmaes/10min_<JOB_ID>/summary.json
```

A successful smoke job should report a completed summary and contain:

```text
best_weights.npy
controls.npz
history.csv
initial_weights.npy
metadata.json
summary.json
```

`summary.json` is written at startup and again after every fully completed
generation. During an active run it has `"status": "running"` and reports
the best fully completed generation so far. At a graceful finish it has either
`"status": "completed"`, with a
`fresh_reproduction_population` matching the recorded best final population,
or `"status": "stopped"`. A stopped result still preserves the best fully
completed generation: `best_weights.npy`, `controls.npz`, and `history.csv`.
It cannot preserve an individual candidate evaluation that SLURM kills midway
through its solver call.

The `cluster_logs/` directory must already exist in the repository before
submitting: SLURM opens the requested log file before it runs the script body.

## SLURM versus Bash

Bash is the shell language that runs ordinary command scripts: `cd`, `mkdir`,
and `uv run ...` are all Bash commands here.

SLURM is the cluster scheduler. The `#SBATCH` lines are instructions for
SLURM, not normal Bash commands: they ask it to reserve resources, choose a
time limit, and save logs. When you run `sbatch script.slurm`, SLURM queues the
request. Once the resources are available, it runs the rest of the file as a
Bash script on a compute node.

So a `.slurm` file is usually **a Bash script plus a resource request**. It is
not a different programming language.

## Before scaling up

Only after the smoke job succeeds should we make a larger submission. Record
the job ID, the output-folder path, runtime, objective-evaluation count, and
final `P(g,0)`. Then increase one thing at a time: generations, population
size, requested wall time, or physics fidelity.
