"""Benchmark simulator construction and saved-control evaluation separately."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from source.run_artifacts import environment_snapshot, utc_timestamp, write_json  # noqa: E402
from source.simulator import FluxoniumResetSimulator  # noqa: E402

DEFAULT_RUNS_DIRECTORY = PROJECT_ROOT / "data" / "output" / "cmaes"


def resolve_run_directory(value: str) -> Path:
    """Accept a full path, project-relative path, or CMA-ES run name."""
    requested = Path(value)
    for candidate in (
        requested,
        PROJECT_ROOT / requested,
        DEFAULT_RUNS_DIRECTORY / requested,
    ):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find a run directory for {value!r}.")


def load_simulator_class(revision: str | None) -> tuple[type, str]:
    """Load the working simulator or one historical Git version in memory."""
    if revision is None:
        return FluxoniumResetSimulator, "working_tree"

    completed = subprocess.run(
        ["git", "show", f"{revision}:source/simulator.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Could not load source/simulator.py from Git revision {revision!r}: "
            f"{completed.stderr.strip()}"
        )

    historical_module = types.ModuleType("historical_simulator")
    historical_module.__file__ = f"{revision}:source/simulator.py"
    exec(
        compile(completed.stdout, historical_module.__file__, "exec"),
        historical_module.__dict__,
    )
    return historical_module.FluxoniumResetSimulator, revision


def measure(
    operation: Callable[[], object], repetitions: int, warmups: int
) -> list[float]:
    """Return elapsed seconds after unrecorded warmup executions."""
    for _ in range(warmups):
        operation()

    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        operation()
        samples.append(perf_counter() - start)
    return samples


def timing_summary(samples: list[float]) -> dict[str, float | list[float]]:
    """Keep raw samples and simple robust summary statistics together."""
    values = np.asarray(samples, dtype=float)
    return {
        "samples_seconds": values.tolist(),
        "median_seconds": float(np.median(values)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }


def evaluate_saved_weights(
    simulator: Any,
    control_protocol: str,
    weights: np.ndarray,
    times: np.ndarray,
) -> float:
    """Evaluate through the same control doorway used by the saved campaign."""
    if control_protocol == "reference":
        return simulator.reference_final_ground_population(weights, times)
    if control_protocol == "joint":
        return simulator.final_ground_population(weights, times)
    raise ValueError(f"Unsupported control protocol: {control_protocol!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run directory or name under data/output/cmaes/.")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-dir", default="local_scratch")
    parser.add_argument(
        "--simulator-revision",
        default=None,
        help=(
            "Git revision whose source/simulator.py should be benchmarked in memory. "
            "Omit for the current working tree."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("--repetitions must be positive and --warmups non-negative.")

    run_directory = resolve_run_directory(args.run)
    with (run_directory / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    physics_config = metadata["physics_config"]
    control_protocol = metadata["run_parameters"]["control_protocol"]
    weights = np.load(run_directory / "best_weights.npy", allow_pickle=False)
    with np.load(run_directory / "controls.npz", allow_pickle=False) as data:
        times = data["times"]
    simulator_class, simulator_source = load_simulator_class(args.simulator_revision)

    construction = measure(
        lambda: simulator_class(physics_config), args.repetitions, args.warmups
    )

    shared_simulator = simulator_class(physics_config)
    solve_only = measure(
        lambda: evaluate_saved_weights(
            shared_simulator, control_protocol, weights, times
        ),
        args.repetitions,
        args.warmups,
    )

    def construct_and_solve() -> float:
        simulator = simulator_class(physics_config)
        return evaluate_saved_weights(simulator, control_protocol, weights, times)

    end_to_end = measure(construct_and_solve, args.repetitions, args.warmups)
    benchmark = {
        "run_directory": str(run_directory),
        "control_protocol": control_protocol,
        "cutoff": int(physics_config["model_settings"]["cutoff"]),
        "duration_ns": float(metadata["run_parameters"]["duration_ns"]),
        "simulator_source": simulator_source,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "construction": timing_summary(construction),
        "solve_on_shared_simulator": timing_summary(solve_only),
        "fresh_simulator_plus_solve": timing_summary(end_to_end),
        "environment": environment_snapshot(PROJECT_ROOT),
    }

    output_directory = Path(args.output_dir)
    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"simulator_benchmark_{run_directory.name}_{utc_timestamp()}.json"
    )
    write_json(output_path, benchmark)

    print(f"Construction median: {benchmark['construction']['median_seconds']:.3f} s")
    print(
        "Shared-simulator solve median: "
        f"{benchmark['solve_on_shared_simulator']['median_seconds']:.3f} s"
    )
    print(
        "Fresh simulator + solve median: "
        f"{benchmark['fresh_simulator_plus_solve']['median_seconds']:.3f} s"
    )
    print(f"Saved benchmark to: {output_path}")


if __name__ == "__main__":
    main()
