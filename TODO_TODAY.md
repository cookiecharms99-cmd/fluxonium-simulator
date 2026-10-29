# Today / next session

Updated: 2026-10-29

This is the short working list. `ROADMAP.md` holds the longer research order;
`Journal/` records completed milestones and decisions.

## Current position

- [x] Establish a trusted fixed-charge, flux-only reference path as a
      benchmark, rather than treating it as the project's permanent control
      model.
- [x] Recover the six-hour cluster artifact: 824 completed generations,
      best recorded `P(g,0) = 0.99565695` at cutoff 20.
- [x] Audit that saved pulse across cutoffs 5–40. Cutoff 20 agrees with cutoff
      40 to about `7.3e-6` for this pulse, while cutoff 8 is plumbing-only.
- [x] Add per-generation provisional summaries, dedicated cluster logs, and a
      reusable run plotter.
- [x] Clean obvious stale scripts and repair the time-grid/control-basis API.
- [x] Run two exploratory overnight campaigns: - 140-ns fixed-charge continuation from the saved 150-ns reference;
      checkpoint best `P(g,0) = 0.99599465` after 1,141 generations.
      The process later hit a filesystem I/O error while writing, not a
      physics/optimizer failure. - six-hour random-start joint-control exploration; it plateaued at
      `P(g,0) = 0.54785729` after 493 generations. This is one exploratory
      60-parameter search, not evidence that joint control is unphysical.

## First: restore a runnable, explicit CMA-ES configuration

- [x] **Jonathan + AI:** Replace the deleted `cmaes_config.json` dependency in
      `main/run_cmaes.py` with an explicit `--sigma0` argument (default `0.3`)
      or a deliberately minimal active configuration. Do not restore unused
      legacy stopping/bound settings merely to make the file exist.
- [x] **Jonathan + AI:** Run a one-generation `legacy_piecewise_lstsq`
      reference smoke command after that repair. This verifies the recent
      `sine_basis(times, count)` and initializer-signature cleanup end to end.
- [x] **AI:** Add a focused runner-level test for the legacy projected
      initializer, so a future API change cannot silently break that command.

## Next research move: duration continuation

- [ ] **Jonathan + AI:** Promote the cutoff audit into a reusable command:
      accept a run path and cutoffs, load the saved physics configuration from
      `metadata.json`, and write a clearly named local/cluster scratch CSV.
- [x] **Jonathan + AI:** Generalize flux-pulse least-squares projection so it
      can initialize both control protocols: project the flux part, then place
      it in the flux slice of a full joint-control vector when needed.
- [x] **Jonathan + AI:** Implement a first duration warm start by copying a
      compatible 40-coefficient flux vector. For the sine basis
      `sin(pi*k*t/T)`, this exactly preserves the pulse as a function of
      normalized time `t/T`; it is already the desired time-compressed start.
- [x] **AI:** Record `initial_center_population` before the first CMA-ES
      generation. Generation-one history contains sampled candidates, so it
      cannot diagnose the quality of a warm-start center.
- [x] **Jonathan + AI:** Fresh-reproduce the 140-ns checkpointed best pulse
      with a new simulator. It exactly reproduces `P(g,0) = 0.995994653948`
      at cutoff 20 (zero observed delta from the checkpointed reward).
- [x] **Jonathan + AI:** Audit the reproduced 140-ns pulse at higher cutoffs.
      It gives `0.99599465` at cutoff 20 and `0.99598598` at cutoff 40, a
      difference of about `8.67e-6`; cutoff 20 is therefore again trustworthy
      at roughly `1e-5` resolution for this saved pulse.
- [ ] **Jonathan + AI:** Compare the 140-ns warm start against zero
      initialization with the same cutoff, seed policy, population, evaluation
      budget, and wall-clock budget. Only then decide whether continuation is
      helping and whether to attempt 130 ns
- [ ] **Later, if control dimensions or bases differ:** add a separate
      sampled-control resample-and-project warm-start transfer. Do not use it
      for the current same-40-sine-basis duration change, where direct copying
      is mathematically exact.

## Parked: joint-control exploration

- [ ] Do not rerun the same random-start 60-parameter joint search yet. First
      decide whether the optimized charge waveform should _replace_ the
      calibrated charge tone or be a bounded correction/envelope around it.
- [ ] If joint control resumes, define independent flux/charge coefficient
      scales, a physically motivated initial center, and a fair comparison
      against the fixed-charge baseline before interpreting its score.

## Experiment records and project hygiene

- [ ] **Jonathan:** Add brief dated entries to `Journal/` after real research
      milestones or meetings: question, method, artifact/run identifier,
      result, interpretation, and next decision.
- [ ] **Jonathan + AI:** Use searchable commit subjects for substantive work:
      `fix(runner): ...`, `feat(controls): ...`, `test(simulator): ...`,
      `docs(research): ...`, or `refactor(cleanup): ...`.
- [ ] **Jonathan + AI:** Keep raw `data/output/` run folders ignored. Once a
      few more runs are validated, export a concise results index from the
      per-run artifacts rather than letting concurrent jobs append one shared
      CSV.
- [ ] Add `main/plot_run.py` usage to `README.md`.

## Later / deliberately not active

- [ ] Save CMA-ES distribution diagnostics (mean and final generation best),
      in addition to the best-ever candidate.
- [ ] Cache one idle-flux scqubits eigensystem as a measured startup cleanup;
      benchmark it separately from any physics change.
- [ ] Compare a declared restart policy against ordinary CMA-ES under equal
      total evaluation budgets.
- [ ] Consider a rule-based campaign manager only after the artifact index is
      reliable. A learned/NN manager needs many comparable campaigns first.
- [ ] Resolve the decay-rate provenance and cycles/ns versus radians/ns
      convention with Dr. Fu before making experimental-strength claims.
