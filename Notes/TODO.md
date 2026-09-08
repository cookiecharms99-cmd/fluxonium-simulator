# Fluxonium Project — Getting Back In

The goal is not to rebuild everything at once. Each session should finish one small, observable thing.

## First session: make the project runnable again

- [x] Read `README.md`, `plans.md`, and `progress.md` (you already did the hardest part: coming back).
- [x] Set up the Python environment with the dependencies in `pyproject.toml`.v
- [x] Run a tiny simulator smoke test: construct `FluxoniumSimulator` from `params.json` and evaluate one random or zero-weight pulse.

## temporary cleanup mission!

- [] test physics! Analyze one sample as it evolves. Make sure to understand it lol
- [] think about what setup works best for the initial state! 

- [] refactor `simulator.py` -> separation of concerns between setup and simulation
- [] read more carefully :D
- [] 



- [ ] Reread simulator.py. Actually implement gaussian basis. 
- [ ] Ensure everything matches up with our notes. Iterate over suggestions. 
- [ ] Record the result and runtime in `progress.md`.
- [ ] Fix the CMA-ES checkpoint mismatch: `buildingzone.py` calls `build_pulse`, but the simulator now has separate flux and drive builders.
- [ ] Reread everything in ReadFromSmater people and understand the code involved
- [ ] Push to a new github :3
- [ ] Consult over ideal architecture and research practice

## Establish a trustworthy baseline

- [ ] Write one real test for pulse construction: shapes, coefficient lengths, and expected endpoints/bounds.
- [ ] Write one real test that verifies `update_weights` preserves the *normalized-time* pulse shape when changing duration.
- [ ] Check the physics assumptions: initial mixed state, target ground-state projector, collapse operator, units, and the meaning of every value in `params.json`.
- [ ] Run a deliberately tiny fixed-time CMA-ES job (few coefficients, few generations, one worker).
- [ ] Save its best fidelity, pulse coefficients, configuration, random seed, and runtime in a dated run folder.

## Make one clean fixed-time optimizer

- [ ] Promote/clean up `testing/buildingzone.py` into a maintained runner under `main/`.
- [ ] Give the runner a clear command and documented arguments: duration, target fidelity, seed, iterations, workers, and output directory.
- [ ] Save flux pulse and drive envelope separately, plus a simple plot and per-generation history.
- [ ] Add physical control constraints before interpreting results: amplitude, smoothness/bandwidth, and any experimentally relevant limits.
- [ ] Repeat a fixed-duration run with several seeds to learn whether results are reproducible.

## Find the minimum useful duration

- [ ] Choose a concrete target, e.g. final ground-state population >= 0.999.
- [ ] Run a coarse duration scan (for example 150, 125, 100, 75 ns) using the same constraints.
- [ ] Warm-start each shorter run by resampling the nearest successful pulse.
- [ ] Narrow the transition region with smaller duration steps.
- [ ] Report the shortest duration that reliably meets the target across chosen seeds.

## Learn from the solutions

- [ ] Implement the minimal data logger in `source/data_pipe.py` only after the fixed-time runner works.
- [ ] Plot fidelity versus duration, plus representative flux and drive pulses.
- [ ] Track leakage / unwanted state population, not just the final target fidelity.
- [ ] Write a one-page research note: model, constraints, optimization setup, best result, and open questions.

## Rules for keeping momentum

- [ ] Before ending a session, add exactly one next action below. Make it runnable in 30–90 minutes.
- [ ] Prefer a small completed experiment over a broad refactor.
- [ ] Do not add a new optimizer (RL, Bayesian optimization, etc.) until CMA-ES has a specific demonstrated limitation.

## Next action

- [ ] Run a one-candidate simulator smoke test and write down the fidelity and runtime.
