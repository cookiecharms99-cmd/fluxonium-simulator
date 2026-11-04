# Duration continuation, control-model check, and performance measurement

Date: 2026-10-29

## Questions

1. Does the saved 150-ns fixed-charge pulse provide a useful 140-ns starting
   point, and can the resulting shorter pulse be trusted numerically?
2. Did a first free joint flux-and-charge CMA-ES run provide evidence for joint
   control?
3. Does removing redundant idle-fluxonium construction improve simulator
   performance, and where is the runtime actually spent?

## Method

- Ran a 140-ns, cutoff-20 fixed-charge continuation from the best 150-ns
  reference coefficients. Because the sine basis is `sin(pi*k*t/T)`, copying
  compatible coefficients preserves the waveform in normalized time `t/T`;
  it is a time-compressed warm start, not an invalid coefficient transfer.
- Reproduced the checkpointed 140-ns best pulse with a fresh simulator and
  audited it across cutoffs 5--40.
- Ran one six-hour, random-start joint-control campaign with 40 flux and 20
  charge coefficients.
- Added an initial-center diagnostic, reusable cutoff-audit command, and a
  benchmark harness that separates construction, solve-only, and fresh
  construction-plus-solve timing.
- Simplified the simulator to construct one persistent idle-fluxonium object
  and one cached idle eigensystem. Benchmarked the current uncommitted version
  against `HEAD:source/simulator.py` in memory, using the same saved 140-ns
  pulse and three measured repetitions after one warmup.

## Results

- The 140-ns continuation checkpoint reached
  `P(g,0) = 0.995994653948` after 1,141 generations and 18,256 candidate
  evaluations. The process later stopped on a filesystem I/O error while
  writing; this was not an optimizer or physics failure.
- A fresh simulator reproduced the saved checkpoint exactly at cutoff 20.
  The cutoff-40 score was `0.995985980652`, differing from cutoff 20 by about
  `8.67e-6`. Cutoff 20 is therefore again trustworthy at roughly `1e-5`
  resolution for this pulse.
- The random-start joint-control run plateaued at `P(g,0) = 0.54785729` after
  493 generations / 7,888 evaluations. It was a 60-parameter exploratory
  search and is not directly comparable with the fixed-charge reference.
- The one-object simulator cache reduced median construction time from
  `0.028 s` (committed old version) to `0.006 s` (current version), about a
  4.5x startup improvement. Shared-simulator solve and end-to-end medians
  differed by only a few percent across three samples, so no solver-speed
  claim is justified. `mesolve` remains the dominant cost.

## Interpretation

The 140-ns saved pulse is a validated numerical checkpoint, but the useful
question is not yet whether it exists: it is whether a warm start reaches good
controls more reliably or cheaply than a neutral zero center under an equal
budget. The submitted 140-ns zero-versus-warm-start Slurm array provides that
comparison.

The joint result does not rule out joint control. The current joint protocol
replaces the calibrated charge tone with a free 20-coefficient envelope, so it
asks CMA-ES to rediscover charge behavior while searching a larger space. Its
physical parameterization must be decided before interpreting another joint
campaign.

The idle-eigensystem cache is worth keeping as a safe startup cleanup, but it
is not a high-leverage optimization target. Future performance work should
profile solver settings or a validated reduced model, while retaining `mesolve`
as the reference.

## Next decision

When the Slurm array returns, compare the zero and warm starts using their
initial-center score, best score, objective-evaluation count, runtime, and
saved pulse shapes. Audit any promising winner at higher cutoff before deciding
whether to attempt 130 ns.
