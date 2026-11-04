# Fluxonium Sim Commands and things :D

**HI WELCOME!!**

### Entry and Main Points

- `main/flux.py` &rarr; current entry point beyond the justfile for running things from a central point. Run with args command and then the args for that command.

* `main/run_cmaes.py` &rarr; main runner thingy. Has a bunch of args on defaults to reweak :]

### Analyses :O

- `main/plot_run.py` &rarr; plots the training curve over generations and also the best charge and flux pulses. Needs a run

- `main/audit_controls.py` &rarr; prints and saves a table for cutoff, fidelity, and performance. Checks if the best controls are actually legit. Might take a while on default (fair warning!). Requires a run.

- `main/benchmark.py` &rarr; generates performance table and json file. requires a run (also pretty slow)
