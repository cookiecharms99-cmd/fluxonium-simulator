# Next session — 2026-11-07

The completed record belongs in `Journal/`; this file is only the next
actionable work.

## First: consolidate control configurations

- [x] Replace duplicated protocol APIs with one configuration-driven
      control/evolution pathway. Runnable configurations declare active flux and
      charge blocks; retired flat configs are rejected by tests.
- [x] Define charge optimization as a full-Fourier correction around the fixed
      calibrated carrier, and record final CMA-ES sigma in run artifacts.

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
