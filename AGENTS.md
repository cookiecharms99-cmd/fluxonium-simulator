# Collaboration guide

This repository belongs to a physics student working with Dr. Fu on fluxonium active reset. Explain technical and design choices in plain language, including why a change is physically and computationally justified. Do not assume Jonathan already knows software architecture, QuTiP internals, numerical optimization, or reinforcement learning.

## Sources of truth

- Read `README.md` for project scope, current architecture, confirmed findings, and open questions.
- Use `References/fluxonium-project-handoff.docx` for the detailed physics handoff.
- Use `References/Code/` to understand the professor's baseline. Do not modify reference code unless explicitly asked; adapt or compare it elsewhere.
- Treat calibrated experimental values as authoritative when their provenance is known. Use scqubits to construct operators, eigenstates, and model checks, but do not silently replace measured values with simulated values.

## Physics conventions

- Distinguish external flux fraction from external phase: `external_phase = 2 * pi * external_flux_quanta`.
- Half flux (`0.5 Phi_0`, phase `pi`) is the idle endpoint.
- Quarter flux (`0.25 Phi_0`) is the experimental interaction neighborhood. The value `0.253980967 Phi_0` is a model-derived bare resonance; retain both meanings when relevant.
- The state basis is the fixed fluxonium LC basis tensor the cavity Fock basis. Do not describe the fluxonium itself as harmonic merely because its matrices use an LC basis.
- The intended pathway is `|e,0> -> |f,0> <-> |g,1> -> |g,0>`.
- The practical initial state is a 50/50 incoherent mixture of `|g,0>` and `|e,0>`.
- The reset objective is final `P(g,0)`, not a generic unspecified fidelity.
- Keep the cavity collapse convention explicit: `L = sqrt(kappa) * (I tensor a)`. The origin and units of the current `kappa = 2*pi/40` still require confirmation.
- Do not conceal the unresolved cycles/ns versus radians/ns convention. Flag it when it affects interpretation.

## Coding guidance

- Keep device physics, controls, solver execution, and optimizer/file handling separated.
- Prefer names that include physical meaning and units, such as `idle_flux_quanta`, `external_phase`, `duration_ns`, and `cavity_decay_rate`.
- Cache static operators, repeated eigensystems, time grids, and control bases when safe. `mesolve` is expected to dominate runtime; preserve readable abstractions unless profiling identifies them as costly.
- Preserve sparse QuTiP/SciPy operators when they improve the open-system solve. Benchmark before assuming sparse or dense is faster.
- Flux and charge controls should have independent basis definitions. Validate coefficient-vector lengths and pulse endpoints.
- Never change physics and optimize performance in the same unverified step. Compare final populations and representative trajectories before and after performance changes.
- Add focused tests for Hamiltonian Hermiticity, density-matrix trace, non-negative populations, pulse endpoints, deterministic repeated evaluation, and cutoff convergence.
- Use `uv run` for project commands. Do not add dependencies without explaining why they are needed.

## Experimental discipline

- For benchmarks, match cutoff, initial state, collapse operators, controls, time grid, tolerances, solver method, and whether initialization time is included.
- Record random seed, runtime, objective-evaluation count, final population, duration, and output artifact location.
- Keep the professor's CMA-ES result, the intuitive pulse, and new optimizers comparable under equal evaluation and wall-clock budgets.
- Reinforcement learning is exploratory. Keep `mesolve` as the trusted reference until a reduced solver reproduces relevant trajectories and final populations within a declared tolerance.

## Working style

- Preserve unrelated local changes; this repository may have an intentionally dirty worktree.
- Critique assumptions and suggest alternatives when useful, but label facts, inferences, and open questions separately.
- Prefer small, testable changes and explain what was changed, why, and how it was checked.
- Keep the project's friendly tone while using precise physics terminology in code and documentation.
