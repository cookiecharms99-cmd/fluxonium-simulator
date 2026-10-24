# Fluxonium Reset Project Missions

## Where we are

The project has completed a substantial physics audit and now has a clean path from pulse coefficients to parallel CMA-ES evaluation:

```text
CMA-ES coefficients
        ↓
flux and charge controls
        ↓
fluxonium-cavity simulator
        ↓
final P(g,0)
        ↓
CMA-ES distribution update
```

This is still baseline infrastructure rather than a new optimization result. That is expected. A trustworthy baseline is what will make later claims about faster reset, joint controls, or reinforcement learning meaningful.

## Completed foundations

### Physics audit

- Confirmed the intended reset pathway `|e,0> -> |f,0> <-> |g,1> -> |g,0>`.
- Distinguished the fixed LC computational basis from physical fluxonium eigenstates.
- Confirmed half flux as the idle endpoint and quarter flux as the experimental interaction neighborhood.
- Distinguished the experimental `0.25 Phi_0` calibration from the model-derived bare resonance near `0.253980967 Phi_0`.
- Confirmed relevant transition frequencies, charge matrix elements, and the avoided crossing in the audited static model.
- Recorded unresolved frequency-unit and cavity-decay questions for Dr. Fu.

### Checkpoint 1: Serial educational CMA-ES runner

Status: complete ✅

- Loaded simulator and optimizer configuration.
- Implemented the `ask -> evaluate -> tell` loop.
- Converted final `P(g,0)` into the minimization loss `1 - P(g,0)`.
- Added deterministic seeding and generation summaries.

### Checkpoint 2: Local multiprocessing

Status: complete ✅

- Added one simulator per worker process.
- Used `spawn` and `pool.map` safely.
- Limited numerical-library threads to avoid CPU oversubscription.
- Verified identical candidates give matching serial and parallel rewards.
- Added an automated pytest integration check.
- Separated reusable optimization mechanics from the command-line runner.

Current files:

- `source/optimization/cmaes.py`: reusable CMA-ES and multiprocessing machinery.
- `main/run_cmaes.py`: current experiment entry point.
- `tests/test_cmaes_multiprocessing.py`: serial/parallel consistency test.

## Next mission: reproducible small campaign

Status: next

Turn the working smoke campaign into a result that can be stopped, inspected, and reproduced.

### Required outputs

- Create a unique output directory for each run. ✅
- Save the random seed and complete configuration snapshot.
- Save the best coefficient vector.
- Save the reconstructed flux and charge waveforms.
- Save one history row per generation containing:
  - generation number;
  - best reward ever;
  - best reward in the generation;
  - mean and standard deviation of rewards;
  - CMA-ES sigma;
  - population size;
  - cumulative objective-evaluation count;
  - elapsed wall-clock time.
- Save a final summary with the stop reason and final `P(g,0)`.

### Verification

- Reload the saved best coefficients in a fresh simulator.
- Confirm the reproduced final population matches the recorded value within numerical tolerance.
- Run the same seed twice and confirm the optimization history is deterministic.
- Keep this campaign small: approximately 2–5 generations with smoke physics.

## Following mission: cluster smoke job

Status: after reproducible outputs

- Create one new SLURM script rather than reviving the outdated scripts.
- Match `--cpus-per-task` to the CMA-ES worker count.
- Use one numerical thread per worker.
- Confirm the cluster repository path, Python module, and environment setup.
- Submit a one-generation smoke job first.
- Verify logs and output artifacts survive after the job exits.
- Only then increase generations, population, and wall-time request.

## Phase 1 finish line: trusted CMA-ES baseline

Before claiming a new result, reproduce a fair version of the professor's baseline:

- Use the confirmed full-physics cutoff and time grid.
- Optimize 40 flux coefficients.
- Keep the calibrated charge drive fixed, matching the reference protocol.
- Evaluate the intuitive pulse under the same simulator and numerical settings.
- Record seed, runtime, objective-evaluation count, best final population, and pulse duration.
- Save and plot the best pulse.
- Confirm it begins and ends at phase `pi`.
- Compare the optimized result against the intuitive pulse under equal conditions.

This is the point where Phase 1 is genuinely complete.

## Phase 2: new project experiments

After the trusted baseline is frozen, the project can make controlled changes one at a time:

1. Reduce total reset duration while maintaining `P(g,0) >= 0.99`.
2. Compare flux-only optimization with joint flux and charge-drive optimization.
3. Test robustness to calibrated parameter perturbations.
4. Study whether the optimum visits the bare resonance, passes through it, or favors a nearby dressed operating point.
5. Profile the trusted simulator and evaluate reduced models against its trajectories.
6. Compare reinforcement learning only under the same controls, physics, and compute budget as CMA-ES.

## Open physics questions

- Are stored frequencies in cycles/ns or radians/ns at every point in the model?
- What physical measurement or convention gives `kappa = 2*pi/40`?
- Is `0.25 Phi_0` the intended calibrated interaction bias even though the bare static model crosses near `0.253980967 Phi_0`?
- What bottleneck and optimization budget governed the earlier graduate-student CMA-ES result?

These questions do not block the smoke-test infrastructure, but they do block strong quantitative claims about physical timing and fidelity.

## Immediate next coding session

Implement run records and checkpoints in this order:

1. Return generation history from `run_cmaes`.
2. Add an explicit `--out-dir` to `main/run_cmaes.py`.
3. Save metadata, history, best coefficients, and reconstructed controls.
4. Add a reload-and-reproduce test.
5. Run a tiny local campaign and inspect every output manually.

Do not add automatic restarts or the 1,400-line autotuning machinery yet. First make one simple campaign fully reproducible.
