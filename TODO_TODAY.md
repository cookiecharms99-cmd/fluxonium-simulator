# Today / next session

Updated: 2026-10-27

This is the short working list. `ROADMAP.md` explains why the longer-term
items are ordered this way.

## First priority: finish the intuitive initialization path

- [x] Add `config/reference_prof.json` and an explicit `reference` CMA-ES
      protocol: 40 flux coefficients, calibrated charge drive fixed, and the
      same device solver/objective as the joint-control simulator.
- [x] Verify a one-generation cutoff-8 reference smoke run, including saved
      controls and fresh-simulator reproduction.
- [x] Correctly polish and test `source/control_initialization.py`:
  - the 150-ns waveform must reproduce the original ramp / hold / return;
  - use names in phase units (`idle_phase`), a floating output array, and
    clear validation of the timing fractions;
  - retain the duration-scaled form as an explicitly named new variant, not
    an unlabelled claim of exact legacy compatibility at other durations.
- [x] Add a least-squares projection from target external phase to the
      `k=1..40` flux sine basis. Keep this pure pulse-to-coefficients logic in
      `control_initialization.py`; do not store or mutate CMA-ES weights in
      the simulator.
- [x] Add focused tests for the 150-ns pulse endpoints / `pi/2` hold and exact
      recovery of a phase waveform already in the chosen sine basis.
- [x] Make initialization an explicit runner choice: `zero`, seeded `random`,
      or `legacy_piecewise_lstsq`. The legacy scheme is restricted to the
      150-ns fixed-charge reference protocol and is recorded in run metadata.
- [x] Set and document the reference configuration's interaction point as the
      experimental `0.25 Phi_0`; keep the model-derived `0.253980967 Phi_0`
      for separate diagnostic/resonance studies.
- [ ] Evaluate these controls under identical cutoff-20 reference settings:
  1. zero flux coefficients;
  2. direct legacy 150-ns intuitive pulse;
  3. its 40-sine least-squares reconstruction.

## Make experiments legible

- [x] Record the initialization scheme and hash of the initial coefficient
      vector in run metadata. Add a source-run path later for warm starts.
- [ ] Define two named run tiers:
  - smoke: a fast software check (cutoff 8);
  - reference: physically comparable results (cutoff 20).
- [ ] Keep cutoff 15 for occasional profiling only; do not treat it as a
      third scientific baseline.
- [ ] Save enough optimizer state for diagnosis: CMA-ES mean and the final
      generation's best candidate, in addition to the best-ever candidate.

## Small cleanup, after the reference baseline works

- [ ] Cache one idle-flux (`0.5 Phi_0`) scqubits `Fluxonium` instance and its
      eigensystem for static operators, initial-state construction, and the
      final `|g,0>` projector. Keep `new_fluxonium(flux_quanta)` for explicit
      non-idle spectrum/pathway diagnostics; this is a startup cleanup, not a
      change expected to speed up per-candidate `mesolve` appreciably.

## Interpret the overnight exploratory run correctly

- [x] Inspect the interrupted joint-control cutoff-20 run. Its best-ever
      `P(g,0)` was about 0.547857 at generation 13; the later population did
      not improve it.
- [ ] Treat this as an artifact/checkpointing test, not a claim about the
      physics or a professor-comparable result. It optimized both flux and
      charge and produced unconstrained-looking controls.
- [ ] After the flux-only baseline exists, test restart CMA-ES fairly: compare
      an ordinary run against a declared no-improvement restart rule at equal
      total evaluation budget. Preserve best-ever results across restarts.

## Communication / infrastructure

- [ ] Follow up when OIT replies about intermittent Athene SSH access; then
      submit `jobs_to_send/reference_smoke_10min.slurm` and inspect its
      artifacts. Submit the six-hour reference job only after that succeeds.
- [ ] Bring the unit, decay-rate, experimental flux-point, and historical
      CMA-ES-bottleneck questions to Dr. Fu when you meet.
- [ ] Ask Dr. Fu whether the original linear ramp/hold/return pulse encoded
      hardware constraints, measured timing knowledge, or simply a physically
      motivated starting guess.

## Explicitly not today

- Do not build RL training yet.
- Do not mix a new basis, a new initialization scheme, and optimizer changes
  in a single comparison.
- Do not claim a physical improvement until the fixed-charge flux-only
  reference baseline and unit conventions are resolved.
