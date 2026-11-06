# Next session — 2026-11-07

The completed record belongs in `Journal/`; this file is only the next
actionable work.

## First: consolidate control protocols

- [x] Replace duplicated joint/reference population fetchers with one explicit
      protocol-control/evolution pathway. `reference` was independently checked
      against the removed waveform/evolution formulation; its physics is
      unchanged. Joint charge envelopes now use the intended full Fourier basis.

- [x] Make protocol parameter count authoritative rather than relying on a
      basis cache. `FluxoniumResetSimulator.parameter_count()` now derives it
      from the declared protocol and coefficient counts. The eventual protocols
      are fixed-charge flux-only, joint, and charge-only. -> UNDO LATER
- [ ] Do not implement charge-only until we state whether its optimized drive
      replaces the calibrated tone, modulates it, or adds a bounded correction. -> IGNORE

## Then: diagnose the trusted 150-ns reference pulse

- [ ] Compare the pathway timing against the fixed charge-drive, interaction,
      and cavity-decay timescales. Write observations before changing controls.
- [ ] Look through Jonathan's pulse plots together and record only concrete
      hypotheses, not visual vibes masquerading as conclusions. :3

## Then: establish the 0.999 reference problem clearly

- [ ] Reproduce the best 150-ns score at cutoff 20 and audit it at cutoff 40.
- [ ] Decide whether a longer/equal-budget fixed-charge CMA-ES run is still
      improving, or has genuinely plateaued below 0.999.
- [ ] Record the meeting conclusions: target, experimental origin of kappa,
      and any clarification of the 20-minute performance observation.

## CMA-ES engineering, after the diagnosis

- [ ] Seed the evaluated initial center as the CMA-ES incumbent so a good warm
      start is retained even if every first sampled candidate is worse.
- [ ] Add a focused regression test for that incumbent behavior.
- [ ] Compare warm-start `sigma0` values (for example 0.05, 0.1, 0.3) under an
      equal evaluation budget; record seed, center score, and best-so-far score.
- [ ] Do **not** prioritize a log loss: ordinary CMA-ES is rank-based, so a
      monotonic transformation of reward does not change its candidate ranking.
- [ ] Define physically justified flux bounds/clamping before enabling one.
      Compare declared bounded controls against the unclamped reference; do not
      silently add a legacy-style `tanh` limiter.

## Charge-control design (parked until the reference diagnosis is written)

- [ ] Specify whether optimized charge control replaces the calibrated tone or
      acts as a bounded correction/envelope around it.
- [ ] Design a charge basis separately from the flux basis. Unlike flux, the
      charge drive need not return to an idle endpoint, so consider a full
      Fourier representation (constant, sine, and cosine terms) or physically
      meaningful carrier quadratures.
- [ ] Establish flux-only and charge-only ablations before retrying a joint
      optimization.

## Later / hygiene

- [ ] Make a compatibility benchmark for the old code with exactly matched
      cutoff, grid, solver tolerances, controls, and worker count; report
      construction, one evaluation, and one generation separately.
- [ ] Update `PROJECT_HANDOFF.md` and `README.md` after this diagnosis round so
      their target and checkpoint are no longer stale.
- [ ] Keep raw `data/output/` ignored; export a concise validated results index
      later rather than maintaining one concurrently written CSV.
