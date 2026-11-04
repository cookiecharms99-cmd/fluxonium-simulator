# Next session

Updated: 2026-10-29

The completed record lives in `Journal/`; this file is deliberately only the
next actionable work.

## Software QOL:

- [x] Make a help (or just) or make command which guides the user on what to run for QOL.
- [x] Simplify directory names. pointers might exist in metadata, or figure out how to pipe to the python script grep results or support pattern matching in the input, since many of our commands run on verbose file names

Notes:
fzf for the win <3

## Result Misison

We noticed that despite the run having a better start, because of sigma, it did not effectively capitalize on it. Decreasing sigma might keep in a plateau though but idk lol. Additionally we do not see the best initial path recorded anywhere so we will need to fix this.

Qualitatively, we are starting to notice some consistent trends with high performers but I don't want to speak too hastily. I noticed something where the initial shape seems roughly the same even on different time frames maybe we can exploit this in the future.

Missions ->

- [] Record cleaning, record best ever with the initial reward as well!
- [] Engineering: decide on how to handle sigma0 when using a warm start. With the research done, make notes for the future on how to avoid plateaus due to sigma as well.

## Then: choose the duration-continuation direction

- [ ] If warm start is clearly more efficient under the equal budget, launch a
      documented 130-ns continuation from the validated 140-ns pulse.
- [ ] If it is not, compare pulse shapes and the recorded center scores before
      changing CMA-ES settings; do not add restart heuristics yet.

## Parked until a physics decision

- [ ] Decide whether joint charge control should replace the calibrated charge
      tone or act as a bounded correction/envelope around it. Only then design
      its parameter scales, initial center, and fair baseline comparison.
- [ ] Confirm decay-rate provenance and the cycles/ns versus radians/ns
      convention with Dr. Fu before making experimental-strength claims.

## Small project hygiene

- [ ] Add `main/plot_run.py`, `source/audit_controls.py`, and
      `main/benchmark_simulator.py` usage to `README.md` when the immediate
      comparison is complete.
- [ ] Keep raw `data/output/` ignored; later export a concise results index
      from validated per-run artifacts rather than using one concurrent CSV.
