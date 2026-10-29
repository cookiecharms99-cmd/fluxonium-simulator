"""
Mission! Verify that the reported fidelity is mostly good and isn't a cutoff error
"""

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
from simulator import FluxoniumResetSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIRECTORY = PROJECT_ROOT / "data" / "output" / "cmaes"


def load_json(path: Path) -> dict:
    """Load one run-record JSON file with a helpful missing-file error."""
    if not path.is_file():
        raise FileNotFoundError(f"Expected run artifact does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def resolve_run_directory(value: str) -> Path:
    """Accept a full path, project-relative path, or a run-directory name."""
    requested = Path(value)
    candidates = (
        requested,
        PROJECT_ROOT / requested,
        DEFAULT_RUNS_DIRECTORY / requested,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find run directory {value!r}. Tried a direct path, a "
        f"project-relative path, and {DEFAULT_RUNS_DIRECTORY}."
    )


RUN_NAME = "reference_6h_legacy_4776599"

RUN_DIRECTORY = resolve_run_directory(RUN_NAME)

weights = np.load(RUN_DIRECTORY / "best_weights.npy")
with np.load(RUN_DIRECTORY / "controls.npz") as data:
    times = data["times"]

physics_config = load_json(PROJECT_ROOT / "config" / "reference_prof.json")

cutoffs = [
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    23,
    25,
    30,
    35,
    40,
]

output_data = [["Cutoff", "Final g0 population", "Time to compute"]]
for cutoff in cutoffs:
    test_config = copy.deepcopy(physics_config)
    test_config["model_settings"]["cutoff"] = cutoff
    sim = FluxoniumResetSimulator(test_config)
    start = time.time()
    final_population = sim.reference_final_ground_population(
        flux_weights=weights, times=times
    )
    end = time.time()
    elapsed = end - start
    output_data.append([cutoff, final_population, elapsed])
    print(
        f"Cutoff {cutoff} | Final pop {final_population:.8f} | Time to compute {elapsed:.3f}s!"
    )
save_destination = PROJECT_ROOT / "local_scratch" / f"cutoff_audit_{RUN_NAME}.csv"
print(f"Saving to... {save_destination} ")
with open(save_destination, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(output_data)
