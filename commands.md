# Fluxonium project manual :D

Run `just` for the compact recipe list, or `just manual` to read this guide.
The Justfile is the friendly interface; the Python scripts remain available
when their full `--help` is useful.

## Everyday workflow

```bash
# Run all automated checks.
just test

# Show saved output directories, newest first.
just latest_data 10
just list-runs

# Interactively choose a run with fzf, then inspect it.
just ez_plot --population-cutoff 20
just ez_cutoff_audit
just ez_benchmark_run
```

The `ez_*` recipes need `fzf`. Their direct counterparts work everywhere and
take an exact run path or name. Put `--` between the Just recipe and flags
intended for the underlying Python script:

```bash
just plot reference_6h_legacy_4776599 -- --population-cutoff 20
just audit reference_6h_legacy_4776599 -- --cutoffs 13 20 40
just benchmark reference_6h_legacy_4776599 -- --repetitions 3 --warmups 1
```

## Launching CMA-ES

`just run` forwards normal flags to `main/run_cmaes.py`. It is intentionally
thin: the saved effective configuration and the run directory are the source
of truth, not a second command wrapper.

```bash
# Tiny plumbing check.
just run -- --generations 2 --popsize 4 --workers 2

# Full fixed-charge, flux-only reference run.
just run -- --physics-config config/device_calibrated.json --full-physics \
  --duration 150 --generations 1000 --popsize 16 --workers 4 \
  --label reference-150ns

# Joint exploratory run: the device config stays the same; this run activates
# a 20-coefficient additive charge-correction block.
just run -- --physics-config config/device_calibrated.json --full-physics \
  --charge-count 20 --initialization random_flux_zero_charge \
  --duration 150 --generations 1000 --popsize 16 --workers 4

# Hold a parent flux waveform while optimizing only charge corrections.
just run -- --physics-config config/device_calibrated.json --full-physics \
  --flux-count 0 --charge-count 20 --fixed-flux saved_controls \
  --saved-flux-controls data/output/cmaes/PARENT/controls.npz \
  --initialization zero --duration 150 --generations 1000 --popsize 16 --workers 4

# Continue a compatible flux-only parent with a declared exploration scale.
# `parent_final` reads the parent artifact's final sigma; `sqrt_loss` uses the
# evaluated center; `fixed` requires an explicit number.
just run -- --physics-config config/device_calibrated.json --full-physics \
  --initialization warm_start --warm-start data/output/cmaes/PARENT/best_weights.npy \
  --sigma0-mode parent_final --duration 150 --generations 200 --popsize 16 --workers 4

# Continue a compatible saved 40-flux-weight run.
just run -- --physics-config config/device_calibrated.json --full-physics \
  --initialization warm_start \
  --warm-start data/output/cmaes/reference_6h_legacy_4776599/best_weights.npy \
  --duration 150 --generations 1000 --popsize 16 --workers 4
```

Use `uv run python main/run_cmaes.py --help` to see all runner flags. A warm
start must match the active control-vector shape exactly; cross-duration or
cross-basis transfer needs an explicit projection step.

## Performance checks

The benchmark separates simulator construction, a cached solve, and a fresh
simulator-plus-solve. It can also time a tiny real CMA-ES generation without
writing an experiment artifact:

```bash
just benchmark reference_6h_legacy_4776599 -- \
  --repetitions 1 --warmups 0 \
  --campaign-generations 1 --campaign-workers 1 4 8 \
  --campaign-popsize 16 --campaign-repetitions 1
```

Benchmark JSON files go to `local_scratch/` by default. Do not compare timings
unless cutoff, time grid, worker count, and machine are recorded.

## Cluster helper

```bash
just pull-cluster-data
```

This repository's command contains Jonathan's Athene endpoint. Other users
should make an equivalent local recipe rather than editing shared research
history.
