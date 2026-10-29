"""Inspect one saved CMA-ES run without requiring the original simulator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def load_history(path: Path) -> dict[str, np.ndarray]:
    """Return numeric columns from an optional generation-history CSV."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def print_run_summary(run_directory: Path, metadata: dict, summary: dict) -> None:
    """Print the few fields useful for deciding what to inspect next."""
    parameters = metadata.get("run_parameters", {})
    print(f"Run: {run_directory}")
    print(f"Status: {summary.get('status', metadata.get('status', 'unknown'))}")
    print(f"Protocol: {parameters.get('control_protocol', 'unknown')}")
    print(f"Initialization: {parameters.get('initialization_scheme', 'unknown')}")
    print(f"Seed: {parameters.get('seed', 'unknown')}")
    print(
        f"Duration: {summary.get('duration_ns', parameters.get('duration_ns', 'unknown'))} ns"
    )
    print(f"Best P(g,0): {summary.get('best_final_ground_population')}")
    print(f"Completed generations: {summary.get('completed_generations', 'unknown')}")
    print(f"Objective evaluations: {summary.get('objective_evaluations', 'unknown')}")
    print(f"Fresh reproduction: {summary.get('fresh_reproduction_population')}")


def plot_run(run_directory: Path, save_path: Path | None, show: bool) -> None:
    """Plot reward history and sampled controls from one artifact directory."""
    metadata = load_json(run_directory / "metadata.json")
    summary = load_json(run_directory / "summary.json")
    # returns an error on current
    print_run_summary(run_directory, metadata, summary)

    controls_path = run_directory / "controls.npz"
    if not controls_path.is_file():
        raise FileNotFoundError(
            "No controls.npz exists yet; this run has not completed a generation."
        )
    with np.load(controls_path) as controls:
        times = controls["times"]
        external_phase = controls["external_phase"]
        charge_drive = controls["charge_drive"]
    history = load_history(run_directory / "history.csv")

    figure, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    if history:
        axes[0].plot(
            history["generation"], history["best_reward_ever"], label="best ever"
        )
        axes[0].plot(
            history["generation"],
            history["best_reward_generation"],
            alpha=0.6,
            label="generation best",
        )
        axes[0].set_ylabel("P(g, 0)")
        axes[0].set_xlabel("generation")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No completed generation history yet", ha="center")
        axes[0].set_axis_off()

    axes[1].plot(times, external_phase)
    axes[1].set_ylabel("external phase")
    axes[1].set_xlabel("time (ns)")

    axes[2].plot(times, charge_drive)
    axes[2].set_ylabel("charge drive")
    axes[2].set_xlabel("time (ns)")
    figure.suptitle(run_directory.name)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160)
        print(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one saved CMA-ES run.")
    parser.add_argument(
        "run", help="Run path or directory name under data/output/cmaes/."
    )
    parser.add_argument(
        "--save", type=Path, help="Optional PNG path; directories are created."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open a plot window."
    )
    args = parser.parse_args()
    plot_run(resolve_run_directory(args.run), args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
