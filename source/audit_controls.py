"""Re-evaluate a saved CMA-ES pulse at several fluxonium LC-basis cutoffs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from simulator import FluxoniumResetSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIRECTORY = PROJECT_ROOT / "data" / "output" / "cmaes"
DEFAULT_CUTOFFS = (
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
)


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


def resolve_output_directory(value: str) -> Path:
    """Resolve an audit-output directory relative to the project when needed."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def final_population_for_protocol(
    simulator: FluxoniumResetSimulator,
    control_protocol: str,
    weights: np.ndarray,
    times: np.ndarray,
) -> float:
    """Evaluate saved weights through the same control doorway as their run."""
    if control_protocol == "reference":
        return simulator.reference_final_ground_population(weights, times)
    if control_protocol == "joint":
        return simulator.final_ground_population(weights, times)
    raise ValueError(
        f"Unsupported control protocol in run metadata: {control_protocol!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="Run directory, project-relative path, or name under data/output/cmaes/.",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=DEFAULT_CUTOFFS,
        help="Fluxonium LC-basis cutoffs to evaluate (default: standard convergence ladder).",
    )
    parser.add_argument(
        "--output-dir",
        default="local_scratch",
        help="Directory for the CSV, relative to the project unless absolute.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(cutoff < 2 for cutoff in args.cutoffs):
        raise ValueError("Every cutoff must be at least 2.")

    run_directory = resolve_run_directory(args.run)
    metadata = load_json(run_directory / "metadata.json")
    try:
        physics_config = metadata["physics_config"]
        control_protocol = metadata["run_parameters"]["control_protocol"]
    except KeyError as error:
        raise KeyError(
            "Run metadata is missing the saved physics configuration or control protocol."
        ) from error

    weights = np.load(run_directory / "best_weights.npy", allow_pickle=False)
    with np.load(run_directory / "controls.npz", allow_pickle=False) as data:
        times = data["times"]

    output_rows = [("cutoff", "final_ground_population", "elapsed_seconds")]
    for cutoff in args.cutoffs:
        test_config = copy.deepcopy(physics_config)
        test_config["model_settings"]["cutoff"] = cutoff
        simulator = FluxoniumResetSimulator(test_config)

        start = perf_counter()
        final_population = final_population_for_protocol(
            simulator, control_protocol, weights, times
        )
        elapsed = perf_counter() - start
        output_rows.append((cutoff, final_population, elapsed))
        print(
            f"Cutoff {cutoff:>2} | final P(g, 0) = {final_population:.10f} "
            f"| {elapsed:.3f} s"
        )

    output_directory = resolve_output_directory(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"cutoff_audit_{run_directory.name}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(output_rows)
    print(f"Saved cutoff audit to: {output_path}")


if __name__ == "__main__":
    main()
