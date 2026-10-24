# Today / next session

Updated: 2026-10-23

This is the short working list. `ROADMAP.md` explains why the longer-term
items are ordered this way.

## First priority: prove the current campaign works end to end

- [ x] Run the prepared laptop campaign:

  ```bash
  bash jobs_to_send/cmaes_10min.slurm
  ```

- [x ] Inspect its `summary.json`, `history.csv`, and `controls.npz`.
- [ x] Record the wall time, final `P(g,0)`, objective-evaluation count, and
  output-folder path in a short note or commit message.
- [ ] When Athene SSH is reachable, submit the same script with `sbatch` and
      inspect the cluster artifacts and log.

## Communication blockers

- [ x] Send the OIT email about intermittent Athene SSH access, including an
  exact error message and the approximate failed/successful times if possible.
- [ x] Email Dr. Fu to ask whether she is available to meet tomorrow. Bring the
  open questions about the cavity decay rate, units, experimental flux point,
  and the former CMA-ES bottleneck.

## Learning / research, no simulator changes yet

- [ x] Read the RL paper with one question in mind: what state, action,
  reward, simulator cost, and comparison budget would make RL informative for
  _this_ reset problem?
- [ ] Sketch possible initial CMA-ES centers: zero coefficients, a projection
      of the intuitive flux pulse into the sine basis, and a warm start from a
      longer-duration pulse. Do not choose one without recording it in run
      metadata.

## Next code change after the run infrastructure is trusted

- [x ] Give flux and charge drive separate sine bases, both starting at
  harmonic `k=1`; add endpoint and coefficient-length tests.
- [ ] Decide and document the reference baseline: flux-only optimization with
      the calibrated charge drive fixed, matching the professor's protocol.

## Explicitly not today

- Do not build RL training yet.
- Do not combine a new initial-weight scheme, a new charge basis, and a
  performance refactor in one experiment.
- Do not claim a physical improvement until unit conventions, decay rate, and
  a comparable baseline are resolved.
