# Project handoff — 2026-11-20

Read `README.md`, `AGENTS.md`, and this file before changing simulator physics
or the optimization path. `References/fluxonium-project-handoff.docx` remains
the detailed historical physics audit.

## Current checkpoint

`source/simulator.py` is the trusted fluxonium-cavity reference simulator. It
uses a fixed fluxonium LC basis tensor a two-level cavity Fock basis and solves
the open-system dynamics with QuTiP. The objective is final `P(g,0)` from the
50/50 incoherent idle mixture of `|g,0>` and `|e,0>`.

`config/device_calibrated.json` is the one active device/numerics baseline.
CLI flags declare an experiment's active or fixed flux/charge controls, and
the runner saves the resulting effective JSON inside the run directory. This
keeps scenario-specific configs out of version control while preserving exact
reproducibility. Do not reintroduce a duplicate reference simulator.

The repository interface is now the `Justfile`: run `just manual` for the
short manual. `main/flux.py` was intentionally removed. The latest verified
suite result is **37 passed, 1 skipped**.

## Physics conventions that must not drift

- External phase is `2*pi*external_flux_quanta`; half flux is idle phase
  `pi`, and the final measurement is at idle.
- `0.25 Phi_0` is the experimentally calibrated interaction neighborhood.
  The model-derived bare resonance near `0.253980967 Phi_0` is a separate
  diagnostic and must not silently replace it.
- The intended pathway is
  `|e,0> --charge--> |f,0> <--> |g,1> --cavity decay--> |g,0>`.
- The experimental target is `P(g,0) >= 0.999`.
- `kappa = 2*pi/40` is experimentally derived. The cycles/ns versus
  radians/ns convention still needs explicit documentation.
- A cutoff of 13 is acceptable for cheap qualitative screening near 0.99; it
  is not validation for 0.999 claims. Use cutoff 20 for the normal serious
  baseline and audit notable results at higher cutoff.

## Evidence established so far

### Fixed-charge flux baseline

The 150-ns professor-style fixed-charge, flux-only pulse
`reference_6h_legacy_4776599` reproduced around **0.995657** at cutoff 20.
This is the current useful reference point, but it is still below the 0.999
target. Duration-continuation experiments reached roughly 0.99+ at shorter
durations, including an approximately 0.996 result at 110 ns, but need careful
initialization/projection before comparative claims.

### Charge ablations

Charge corrections have genuine but limited demonstrated leverage:

- holding the good saved 150-ns flux waveform fixed and optimizing 20
  charge-correction coefficients raised the saved best from
  **0.9956569513** to **0.9959302193** at cutoff 20;
- the charge-only and random-flux ablations did substantially worse;
- naive joint flux-plus-charge CMA-ES gets lost in the larger search space.

Therefore: the flux waveform is the dominant ingredient. Do not yet claim
that charge control solves the reset problem or that an exploratory amplitude
is hardware-realistic without calibration constraints.

### Dimensions and performance

An exploratory 150-ns flux-only cutoff-15 sweep, 250 generations / 4000
evaluations each, produced 20D **0.9896**, 40D **0.7809**, 60D **0.4307**, and
80D **0.2623**. This suggests higher dimension can make CMA-ES much harder,
but different seeds confound a dimension-only conclusion. A 60D saved result
changed only from 0.4307 at cutoff 15 to about 0.4447 at cutoff 20, so
truncation alone does not explain that failure. Finish cutoff checks and run a
controlled same-seed cutoff-20 comparison.

The Athene jobs were slower per generation than anticipated from laptop
timings. The next comparison must benchmark an allocated compute node and the
laptop under matching numerical settings; do not infer speed from nominal CPU
clock rate alone.

## Important known issues / active development

1. **Warm-start incumbent repair:** the runner now persists and seeds the
   evaluated center as the best-so-far candidate before generation 1. Its
   evaluation is included in the recorded evaluation count, but it is not fed
   into CMA-ES's population update. The next question is scientific: compare
   sensible warm-start sigma choices under equal budgets. The prepared Slurm
   array `jobs_to_send/warm_start_sigma_compare.slurm` holds every other
   setting fixed and compares parent-final sigma, square-root-loss sigma, and
   fixed 0.005.
2. **Cluster finalization:** two six-hour jobs checkpointed cleanly but their
   summaries remained `status: running` and did not write a final fresh
   reproduction after the scheduler deadline. Treat checkpoint artifacts as
   usable but fix/report this termination path before relying on the summary
   status.
3. **New charge modes:** `free_fourier`, additive calibrated correction, and
   `calibrated_times_one_plus_fourier_correction` now have waveform-level
   tests. The last mode is explicitly a *dimensionless modulation*, so zero
   coefficients reproduce the calibrated tone. They still need declared
   experimental configs and a physical amplitude decision before launch.
4. **Cross-duration transfer:** weights are basis/duration-specific. Preserve
   a physical waveform with explicit resampling/projection rather than copying
   coefficient vectors across durations.

## Immediate next task

Make the new charge modes testable and scientifically comparable first:

1. clean the carrier/envelope implementation and add waveform-level tests;
2. define configs and an experimental hypothesis for free versus additive and
   multiplicative charge control;
3. only then submit an equal-budget cutoff-20 charge-mode comparison.

In parallel, review and submit the equal-budget warm-start sigma comparison.
See `TODO_TODAY.md` for the ordered tasks, commands, and ownership.
