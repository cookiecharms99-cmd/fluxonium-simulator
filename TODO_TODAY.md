# Next session — 2026-11-20

This file is the short, actionable queue. Put completed reasoning, plots, and
numerical results in `Journal/`, then refresh `PROJECT_HANDOFF.md` at a real
milestone.

## Completed in this session

- [x] Consolidated active JSON inputs into `config/device_calibrated.json`.
      Scenario-specific control choices now use runner flags and are frozen as
      the effective `physics_config.json` in each new run directory. Historical
      JSON artifacts remain self-contained inside their existing run data.

## 1. Make the new charge modes trustworthy before launching them

- [x] **AI:** Review and clean the new charge-mode implementation in
      `source/simulator.py`: `free_fourier`,
      `calibrated_plus_fourier_correction`, and
      `calibrated_times_fourier_correction` need clear names and one shared,
      unambiguous definition of carrier versus envelope.
- [x] **Jonathan + AI:** Add focused tests for each charge mode: vector shape,
      zero-weight waveform, and a manually calculated simple waveform. In
      particular, test the zero-coefficient `free_fourier` case before using
      it in a physics run. (Implemented 2026-11-20.)
- [x] **Us:** Write one declared config for each scientifically distinct
      charge experiment. Record whether it means (a) a free physical drive,
      (b) an additive correction to the calibrated drive, or (c) a
      dimensionless multiplicative modulation. Do not compare their raw
      coefficient scales as though they were the same units. -> decided on one consolidated config which gets overwritten with CLI flags :3

## 2. Repair and measure warm starts

- [x] **AI:** Make an explicitly evaluated warm-start center eligible to be
      the saved CMA-ES incumbent. Add a regression test: if every sampled
      first-generation candidate is worse, the saved best remains the center.
      (Implemented and smoke-tested 2026-11-20.)
- [ ] **Us:** Design one equal-budget 150-ns continuation comparison from the
      same saved flux pulse: carry over final sigma, choose a fixed small
      sigma, and use the current heuristic. Record initial-center score,
      best-ever score, final sigma, evaluations, and wall time.
      `jobs_to_send/warm_start_sigma_compare.slurm` is prepared: its array
      compares parent-final sigma, square-root-loss sigma, and fixed 0.005
      over 200 generations at cutoff 20. Review, commit, and submit it.
- [ ] **Later:** For cross-duration or cross-basis continuation, transfer the
      _sampled physical waveform_ through an explicit projection/resampling
      procedure; never paste incompatible coefficient vectors.

## 3. Resolve the dimension / cutoff question with controlled evidence

- [ ] **Us:** Write up the existing cutoff-15 150-ns flux-only sweep as an
      exploratory result: 20D 0.9896, 40D 0.7809, 60D 0.4307, 80D 0.2623.
      Its seeds differed, so it does not isolate dimension alone.
- [ ] **Jonathan + AI:** Run one controlled 20D-versus-40D comparison at
      cutoff 20 with the same seed and equal evaluation budget. Decide before
      launch whether sigma stays 0.3 per coordinate or scales as
      `0.3 * sqrt(20 / dimension)`.
- [ ] **Us:** Finish cutoff-20/40 checks of saved poor 60D/80D candidates
      before interpreting unselected population as physical leakage. -> perhaps corrupted | deleted
- [ ] **AI:** Fix or retire `scripts/run_150ns_random.sh`; its advertised
      `flux_count` currently does not reach the runner as `--flux-count`.

## 4. Make performance comparisons interpretable

- [ ] **Jonathan + AI:** Benchmark a single cutoff-20 evaluation and a small
      one-generation campaign on both laptop and an allocated Athene compute
      node, with matching cutoff, time grid, workers, and environment.
- [ ] **Us:** Add a concise progress metric to run plots/reports: best reward
      versus evaluations and versus elapsed wall time. This is an optimization
      diagnostic, not a neural-network learning-rate claim.

## 5. Experimental discipline and project policy

- [ ] **Us:** Before submitting a charge-mode run, state the hypothesis,
      fixed controls, initialization, cutoff, seed(s), budget, and success
      criterion in the job/script header.
- [ ] **Us:** Decide and document the legacy policy: it is okay to break old
      configs and artifact compatibility when the representation improves, but
      move known obsolete configs under `config/legacy_configs/`, explain the
      break, and add a test for the new declared format.

## Parked discussion prompts — worth thinking about before we overbuild

- [ ] **Jonathan + AI:** Separate what is intrinsically difficult about joint
      flux-plus-charge CMA-ES from what may be repairable engineering: scaling,
      initialization, covariance structure, staged/alternating optimization,
      or restart policy. Do not assume CMA-ES itself is the problem without an
      equal-budget comparison.
- [ ] **Jonathan + AI:** Discuss why low-cutoff optimization may train poorly
      even when a _good saved pulse_ evaluates similarly at low and high
      cutoff. Poor/intermediate candidates can visit higher levels more
      strongly, so objective ranking may be less stable than end-of-run audits
      suggest. Design a small test rather than trusting this story.
- [ ] **Jonathan + AI:** Decide how a warm-start `sigma0` should be chosen:
      carry over the parent's final sigma, use a fixed small exploratory
      sigma, or test the current heuristic / quadratic approximation. Keep it
      an experiment, not a formula we bless from one run.
- [ ] **Jonathan + AI:** Investigate the Athene-versus-laptop slowdown. Check
      numerical settings, environment/library versions, allocated physical
      cores, multiprocessing overhead, and actual compute-node timing before
      attributing it to processor clock speed.
- [ ] **Jonathan + AI:** Consider lightweight optimizer diagnostics: reward
      improvement per evaluation, per generation, and per wall-clock second;
      population reward spread; and sigma evolution. These are descriptive
      signals, not proof that one optimizer “learns better.”
- [ ] **Us:** Revisit whether optimization should eventually use staged modes:
      optimize flux first, then charge on a held physical waveform, then allow
      a deliberately small joint refinement. This is a hypothesis to compare
      against one-shot joint CMA-ES, not a conclusion yet.

## Agentic Cleanup

- [] unstale-ify instruction sets for any agents in AGENTS.md
- [] make sure to emphasize that tests are not rigid | todo: separate pipeline functionality tests from correctness of pipeline tests | also make performance tests to gauge whether changes made hinder performance
- []
