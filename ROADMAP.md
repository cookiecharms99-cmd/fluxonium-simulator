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

Not yet established:

- A trusted full-physics CMA-ES baseline comparable with the professor's
  result.
- Confirmed units and provenance for the cavity-decay parameter.
- A cluster result from Athene.

## Phase A — make the execution path trustworthy

Goal: run the existing smoke campaign locally and on Athene, producing
inspectable artifacts even when a time limit ends the campaign.

1. Run `jobs_to_send/cmaes_10min.slurm` locally with Bash and inspect its
   artifacts.
2. Resolve Athene access and verify its `uv` environment.
3. Submit one bounded cluster run; compare its artifact layout and result with
   the laptop run.
4. Only after several runs exist, add a command that scans their summaries
   into a global `runs.csv`. Per-run folders remain the source of truth, which
   avoids concurrent cluster jobs corrupting one shared CSV.

Success criterion: a result can be tied to a seed, configuration, code
revision, runtime, evaluation count, saved pulse, and a fresh reproduction.

## Phase B — freeze the reference control model

Goal: make one physically and computationally explicit CMA-ES baseline before
experimenting with better parameterizations.

1. Confirm with Dr. Fu which values are experimental calibration inputs and
   which are model checks, especially the decay rate and frequency units.
2. Keep the experimental interaction neighborhood at `0.25 Phi_0` distinct
   from the model-derived bare resonance near `0.253980967 Phi_0`.
3. Implement separate flux and charge-drive sine bases, both indexed from
   `k=1`; validate coefficient lengths and pulse endpoints.
4. Match the professor's comparison protocol: optimize flux coefficients
   while keeping the calibrated charge drive fixed.
5. Add the remaining focused physics tests: Hamiltonian Hermiticity,
   density-matrix trace, non-negative populations, and cutoff convergence.

Success criterion: the chosen control model, initial state, collapse operator,
cutoffs, time grid, solver settings, and objective are written down and tested.

## Phase C — establish a fair CMA-ES baseline

Goal: answer what the current trusted simulator can achieve under a declared
compute budget.

1. Evaluate the intuitive pulse under exactly the baseline settings.
2. Use a declared initial-weight scheme and record it in the run artifacts.
3. Run several seeds at equal evaluation and wall-clock budgets.
4. Compare final `P(g,0)`, duration, runtime, and pulse shapes—not just the
   single best reward.
5. Diagnose whether limited wall time, CMA-ES settings, numerical precision,
   or the control parameterization is the bottleneck.

Initial-weight ideas worth comparing one at a time:

- zero coefficients, as a neutral reference;
- the intuitive pulse projected into the flux sine basis;
- a saved successful pulse from the same duration.

Success criterion: a reproducible flux-only baseline is fairly comparable to
the intuitive pulse and professor reference.

## Phase D — controlled research experiments

Only change one scientific variable per experiment.

1. **Duration continuation / warm starts.** Start with a solved longer pulse,
   resample it onto a shorter interval, project it into that shorter interval's
   sine basis, then use those coefficients as the CMA-ES center. Compare this
   directly against zero initialization at the same short duration and budget.
   It may help, but it is not guaranteed: shortening changes both the pulse
   bandwidth and the reachable dynamics.
2. **Joint flux and charge optimization.** Introduce it only after the
   flux-only fixed-charge baseline is frozen.
3. **Robustness.** Perturb calibrated parameters and assess how much reset
   quality changes.
4. **Reduced models.** Benchmark any faster approximation against trusted
   `mesolve` trajectories and final populations before using it for RL.
5. **Reinforcement learning.** Treat RL as exploratory. It earns its cost if
   it provides transfer, robustness, or feedback value—not merely one better
   fixed pulse under an unequal compute budget.

## Questions for Dr. Fu / the literature

- Are frequencies represented consistently in cycles/ns or radians/ns?
- What measurement or convention gives `kappa = 2*pi/40`, and what are its
  units in the collapse operator `sqrt(kappa) (I tensor a)`?
- Is `0.25 Phi_0` the intended calibrated interaction point for the experiment?
- What limited the prior graduate-student CMA-ES run: wall time, precision,
  simulator cost, parameterization, or optimizer convergence?
- What result did the intuitive pulse achieve under the same numerical model
  and compute budget?
