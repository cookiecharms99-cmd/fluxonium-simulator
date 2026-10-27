# Project handoff — 2026-10-27

This is the compact handoff for the next collaboration session. Read
`README.md`, `AGENTS.md`, and this file before changing the physics or
optimizer path. The detailed historical audit remains in
`References/fluxonium-project-handoff.docx`.

## Where the project is

The trusted simulator is `source/simulator.py`, using a fixed fluxonium LC
basis tensor a two-state cavity Fock basis. The reset objective is final
`P(g,0)` from a 50/50 incoherent idle mixture of `|g,0>` and `|e,0>`.

Two control protocols now exist and deliberately share the same device,
solver, collapse operator, initial state, and measurement:

- `joint`: exploratory optimization of flux and charge sine coefficients;
- `reference`: professor-comparable optimization of 40 flux coefficients
  with the calibrated charge drive fixed to
  `(Omega / n_ef) * sin(omega_ef * t)`.

`main/run_cmaes.py` selects these with `--control-protocol joint|reference`
and accepts `--physics-config`. Its artifacts, checkpoints, saved controls,
and fresh-simulator reproduction honor the selected protocol. Reference-mode
serial and multiprocessing rewards agree in tests.

The runner also has explicit initialization choices: `zero`, seeded `random`,
and `legacy_piecewise_lstsq`. The latter projects the professor-inspired
150-ns flux waveform into the 40-sine basis, is restricted to the fixed-charge
reference protocol at 150 ns, and is recorded in metadata alongside the
initial-weight hash.

The last verified suite result was **9 passed, 1 skipped**. A one-generation,
cutoff-8 reference smoke run completed with a 40-element coefficient vector
and matching fresh reproduction. This establishes execution plumbing only;
it is not a scientific performance result.

## Physics guardrails

- External flux fraction and phase differ: `phase = 2*pi*flux_quanta`.
- Idle is `0.5 Phi_0` / phase `pi`; preparation and final `|g,0>` measurement
  belong there.
- `0.25 Phi_0` is the experimental interaction neighborhood. The model-derived
  bare resonance near `0.253980967 Phi_0` is a separate diagnostic result and
  must not silently replace the experimental value.
- `config/reference_prof.json` stores the experimental interaction value
  `0.25 Phi_0`. The model-derived bare resonance near `0.253980967 Phi_0`
  remains a separate diagnostic value. The legacy intuitive pulse targets
  phase `pi/2`, i.e. `0.25 Phi_0`.
- `kappa = 2*pi/40` and the cycles/ns versus radians/ns convention remain open
  experimental/unit questions. Do not over-interpret numerical results until
  they are clarified with Dr. Fu.

## Current next task: intuitive initialization

`source/control_initialization.py` was started for pulse-construction ideas.
It currently implements the professor-inspired piecewise phase pulse: ramp
from `pi` to `pi/2`, hold, then return to `pi`. At 150 ns its timing fractions
correspond to the legacy 10 ns / 110 ns / 150 ns schedule. Focused tests now
cover that schedule and exact least-squares recovery for a waveform already in
the sine basis. It still needs polish and one project-specific diagnostic:

1. use phase-based names and clean fraction validation;
2. plot and record the approximation error when the 40-sine basis reconstructs
   the sharp-cornered legacy waveform;
3. keep projection in `control_initialization.py`, returning a new weight
   vector. Do not make the simulator own or mutate CMA-ES candidate weights.

The first fair cutoff-20 reference evaluations should be:

1. zero flux coefficients;
2. direct 150-ns intuitive waveform;
3. its 40-sine least-squares reconstruction.

Only then run several equal-budget CMA-ES seeds from declared centers. Record
the initialization scheme and coefficient hash in metadata.

## Architecture decisions worth preserving

- Do not copy a second reference simulator. `reference_controls` and
  `reference_final_ground_population` are the named reference doorway within
  the shared simulator.
- Flux and charge sine bases are independent and start at `k=1`.
- The simulator's temporary scqubits object used for fixed-LC-basis operators
  is not a flux-centered accuracy trick. A later small cleanup can cache one
  idle-flux scqubits object/eigensystem for startup efficiency; cutoff
  convergence, not helper-object flux, controls truncation confidence.
- Treat the earlier 60-parameter joint-control cutoff-20 overnight run as a
  checkpointing/exploration artifact only, not a professor-comparable result.
  Its best `P(g,0)` was about 0.547857 early in the run.
- Do not add CMA-ES restart heuristics until the flux-only reference and
  initial-center comparison exist. Then compare restart rules under equal
  total evaluation budgets.

## Useful commands

```bash
uv run pytest

uv run main/run_cmaes.py \
  --physics-config config/reference_prof.json \
  --control-protocol reference \
  --initialization legacy_piecewise_lstsq \
  --generations 5 --popsize 4 --workers 2
```

Use `--full-physics` for cutoff-20 reference work; without it the runner uses
the cheap cutoff-8 smoke settings. See `TODO_TODAY.md` for the active checklist
and `ROADMAP.md` for the ordered research plan.

Prepared launchers:

- `scripts/run_reference_initialization_compare.sh`: laptop, three equal-budget
  cutoff-20 runs for `zero`, `random`, and `legacy_piecewise_lstsq` centers;
- `jobs_to_send/reference_smoke_10min.slurm`: cluster environment/reference
  smoke check;
- `jobs_to_send/reference_6h_legacy.slurm`: six-hour, full-physics,
  legacy-initialized single-seed reference campaign. Submit this only after the
  smoke job succeeds.
