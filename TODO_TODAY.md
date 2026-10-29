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
- [ ] **Jonathan + AI:** Implement one explicit warm-start scheme: reconstruct
      the successful 150-ns pulse on normalized time `t/T`, resample it at a
      shorter duration, project onto that duration's sine basis, and record
      parent run, source duration, and child duration in metadata.
- [ ] Compare the warm start against zero initialization at the same shorter
      duration, seed policy, cutoff, evaluation budget, and wall-clock budget.

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
