# Fluxonium reset roadmap

This is the practical project roadmap. The detailed audit and historical
checklist remain in `main/mission.md`; this file answers what to do next and
what evidence is needed before changing direction.

## Current position

Completed:

- Physics audit of the intended reset pathway and calibrated operating points.
- Refactored simulator with cached static physics and local multiprocessing.
- Reproducible CMA-ES artifacts: configuration, seed, generation history,
  best coefficients, sampled controls, and fresh-run reproduction.
- A checkpointing CMA-ES runner and a ten-minute, four-worker SLURM script.
- A first equal-budget, single-seed initialization comparison at cutoff 20:
  the projected legacy intuitive center reached about `0.893`, ahead of the
  zero and random centers (about `0.744` and `0.753`). This is useful evidence,
  not yet a multi-seed conclusion.
- Independent `k=1...N` sine bases for flux and charge controls, with
  coefficient-length, endpoint, and duration-cache tests.
- One long local cutoff-20 joint-control run that proved generation-level
  checkpoints survive an interrupted campaign. It is exploratory only, not a
  professor-comparable baseline.
- A fixed-charge, flux-only reference-control path: a dedicated configuration,
  40-dimensional CMA-ES protocol, worker-safe multiprocessing, saved sampled
  controls, and fresh-simulator reproduction. A one-generation cutoff-8 smoke
  run completed successfully.

Not yet established:

- A trusted full-physics CMA-ES baseline comparable with the professor's
  result.
- Confirmed units and provenance for the cavity-decay parameter.
- A cluster result from Athene.
- Validation of a reported approximately `0.995` long-run result; its complete
  artifact has not yet been returned to the local project.

## Phase A — make the execution path trustworthy

Goal: run bounded campaigns locally and on Athene, producing
inspectable artifacts even when a time limit ends the campaign.

1. Maintain two declared execution tiers: a fast cutoff-8 smoke tier for
   plumbing and a cutoff-20 reference tier for scientific comparisons.
2. Resolve Athene access and verify its `uv` environment.
3. Submit one bounded cluster smoke run; compare its artifact layout and result with
   the laptop run.
4. Write a provisional checkpoint summary after every completed generation, so
   a scheduler time limit cannot leave an otherwise useful run without a
   readable best-so-far result. Mark it as provisional; only a final summary
   may claim clean completion or fresh reproduction.
5. Put SLURM stdout/stderr in an ignored `cluster_logs/` directory, with an
   empty tracked placeholder. The directory must exist before `sbatch` because
   SLURM opens log files before executing the script.
6. Provide a run-inspection command that reads each run's exact metadata,
   history, and sampled controls. SHA-256 fields are integrity fingerprints,
   not a human-facing way to identify an experiment.
7. Only after several runs exist, export a concise results table from the
   per-run summaries. Per-run folders remain the source of truth, which avoids
   concurrent cluster jobs corrupting one shared CSV.

Success criterion: a result can be tied to a seed, configuration, code
revision, runtime, evaluation count, saved pulse, and a fresh reproduction.

## Phase B — freeze the reference control model

Goal: make one physically and computationally explicit CMA-ES baseline before
experimenting with better parameterizations.

1. Confirm with Dr. Fu which values are experimental calibration inputs and
   which are model checks, especially the decay rate and frequency units.
2. Keep the experimental interaction neighborhood at `0.25 Phi_0` distinct
   from the model-derived bare resonance near `0.253980967 Phi_0`.
3. Match the professor's comparison protocol: optimize 40 flux coefficients
   while keeping the calibrated charge drive fixed. The execution path is
   implemented; the first cutoff-20 reference scores remain to be recorded.
4. For the exact reference configuration, retain the experimental interaction
   point at `0.25 Phi_0`; reserve `0.253980967 Phi_0` for model-resonance
   diagnostics and a later controlled comparison.
5. Add the remaining focused physics tests: Hamiltonian Hermiticity,
   density-matrix trace, non-negative populations, and cutoff convergence.

Success criterion: the chosen control model, initial state, collapse operator,
cutoffs, time grid, solver settings, and objective are written down and tested.

## Phase C — establish a fair CMA-ES baseline

Goal: answer what the current trusted simulator can achieve under a declared
compute budget.

1. Implement the professor's 150-ns ramp / hold / return flux-phase pulse and
   test its endpoints and `pi/2` hold. A duration-scaled version is a separate
   initialization scheme, not silently the same legacy waveform.
2. Evaluate the direct intuitive pulse under exactly the baseline settings,
   then project its phase deviation from idle into the 40-sine flux basis by
   least squares and evaluate the reconstruction.
3. Compare declared initial-weight schemes one at a time and record their
   provenance and coefficient-vector hash in the run artifacts.
4. Validate the reported `~0.995` reference result: restore its artifact,
   fresh-reproduce the saved best candidate, inspect the control traces, and
   check cutoff convergence. It clears the practical `0.99` reset milestone if
   verified, but remains below the currently configured `0.999` target.
5. Run several seeds at equal evaluation and wall-clock budgets.
6. Compare final `P(g,0)`, duration, runtime, and pulse shapes—not just the
   single best reward.
7. Diagnose whether limited wall time, CMA-ES settings, numerical precision,
   or the control parameterization is the bottleneck.

Initial-weight ideas worth comparing one at a time:

- zero coefficients, as a neutral reference;
- the intuitive pulse projected into the flux sine basis;
- a projected raised-cosine ramp / flat hold / raised-cosine return pulse;
- a saved successful pulse from the same duration.

Success criterion: a reproducible flux-only baseline is fairly comparable to
the intuitive pulse and professor reference.

## Phase D — controlled research experiments

Only change one scientific variable per experiment.

1. **Duration continuation / warm starts.** Start with a solved longer pulse,
   resample it by normalized time `t/T` onto a shorter interval, project it into that shorter interval's
   sine basis, then use those coefficients as the CMA-ES center. Compare this
   directly against zero initialization at the same short duration and budget.
   Record the parent run and both durations. It may help, but it is not
   guaranteed: shortening changes both the pulse bandwidth and the reachable
   dynamics.
2. **CMA-ES restart study.** Once the flux-only baseline and initialization
   comparison exist, compare ordinary CMA-ES against a declared restart rule
   (for example, no improvement for a fixed number of generations, with a
   fresh distribution and larger population). Use the same total evaluation
   budget and retain best-ever results across restarts. Do not use an
   unlogged, ad-hoc sigma increase as a rescue mechanism.
3. **Joint flux and charge optimization.** Introduce it only after the
   flux-only fixed-charge baseline is frozen.
4. **Robustness.** Perturb calibrated parameters and assess how much reset
   quality changes.
5. **Reduced models.** Benchmark any faster approximation against trusted
   `mesolve` trajectories and final populations before using it for RL.
6. **Reinforcement learning.** Treat RL as exploratory. It earns its cost if
   it provides transfer, robustness, or feedback value—not merely one better
   fixed pulse under an unequal compute budget.
7. **Campaign manager / meta-optimization.** Begin with an auditable
   rule-based policy for low-cutoff exploration, high-cutoff validation, and
   declared CMA-ES restart choices. Consider a learned manager only after the
   run database contains enough comparable campaigns to define its state,
   action, computational cost, and reward.

## Questions for Dr. Fu / the literature

- Are frequencies represented consistently in cycles/ns or radians/ns?
- What measurement or convention gives `kappa = 2*pi/40`, and what are its
  units in the collapse operator `sqrt(kappa) (I tensor a)`?
- Is `0.25 Phi_0` the intended calibrated interaction point for the experiment?
- What limited the prior graduate-student CMA-ES run: wall time, precision,
  simulator cost, parameterization, or optimizer convergence?
- What result did the intuitive pulse achieve under the same numerical model
  and compute budget?
