# Reference benchmark and cutoff audit

Date: 2026-10-28

## Question

Can the saved high-performing fixed-charge reference control be trusted at
fluxonium cutoff 20, and what does it establish for the next phase of work?

## Method

- Recovered the six-hour Athene artifact
  `reference_6h_legacy_4776599`.
- Evaluated its saved best flux coefficients on the exact saved time grid while
  changing only the fluxonium LC-basis cutoff from 5 through 40.
- Kept the reference protocol fixed: 40 flux coefficients, calibrated charge
  drive, 150 ns duration, and 500 samples. The audit configuration was checked
  against the physics configuration saved in the run metadata before running.

## Result

- The interrupted cluster campaign completed 824 generations (13,184 objective
  evaluations) and recorded best `P(g,0) = 0.9956569515` at cutoff 20.
- The same saved control gave `P(g,0) = 0.9956496116` at cutoff 40.
- The cutoff-20 versus cutoff-40 difference is approximately `7.3e-6` for
  this control. Cutoffs 20, 23, 25, 30, 35, and 40 form a tight plateau.
- Cutoff 8 gave approximately `0.799`, so it remains a software-plumbing
  tier, not a scientific scoring tier.
- Solver time rose from about 4.3 s at cutoff 20 to about 81 s at cutoff 40
  for this control. Cutoff 20 is therefore the practical full-physics tier
  for the present campaign.

## Interpretation

This validates cutoff 20 to roughly `1e-5` numerical resolution for this
saved control. That is comfortable for comparisons at the `1e-4` level, but
new candidates near a strict threshold should receive their own higher-cutoff
check. One pulse does not prove convergence for every future pulse.

The professor-style reference protocol has served its purpose as a sanity
check and stable benchmark. It should remain reproducible, but it is not the
only legitimate future control model. The next research direction is duration
continuation: use the successful 150-ns pulse as a documented warm start for
shorter pulses.

## Engineering notes

- The old interrupted run did not leave a useful final summary. The runner now
  writes a provisional `summary.json` at startup and after every fully
  completed generation.
- Scheduler stdout/stderr now belong in `cluster_logs/`.
- `main/plot_run.py` inspects saved histories and sampled controls without
  requiring a fresh optimization run.

## Next decision

Repair the runner's now-stale dependency on the removed legacy
`cmaes_config.json`, then make the duration-continuation initialization and
compare it fairly against zero initialization.
