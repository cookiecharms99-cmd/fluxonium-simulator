"""Inspect one saved CMA-ES run and optionally reproduce pathway populations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from source.simulator import FluxoniumResetSimulator, PathwayDiagnostics  # noqa: E402

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
    for candidate in (
        requested,
        PROJECT_ROOT / requested,
        DEFAULT_RUNS_DIRECTORY / requested,
    ):
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


def print_run_summary(
    run_directory: Path,
    metadata: dict,
    summary: dict,
    population_cutoff: int | None,
) -> None:
    """Print the few fields useful for deciding what to inspect next."""
    parameters = metadata.get("run_parameters", {})
    print(f"Run: {run_directory}")
    print(f"Status: {summary.get('status', metadata.get('status', 'unknown'))}")
    controls = (
        metadata.get("physics_config", {})
        .get("model_settings", {})
        .get("controls", "legacy flat control counts")
    )
    print(f"Controls: {controls}")
    print(f"Initialization: {parameters.get('initialization_scheme', 'unknown')}")
    print(f"Seed: {parameters.get('seed', 'unknown')}")
    print(
        f"Duration: {summary.get('duration_ns', parameters.get('duration_ns', 'unknown'))} ns"
    )
    print(f"Best P(g,0): {summary.get('best_final_ground_population')}")
    print(f"Completed generations: {summary.get('completed_generations', 'unknown')}")
    print(f"Objective evaluations: {summary.get('objective_evaluations', 'unknown')}")
    print(f"Fresh reproduction: {summary.get('fresh_reproduction_population')}")
    if population_cutoff is not None:
        print(f"Reproducing pathway diagnostics at cutoff {population_cutoff}...")


def reproduce_pathway_diagnostics(
    metadata: dict,
    best_weights: np.ndarray,
    times: np.ndarray,
    population_cutoff: int,
) -> PathwayDiagnostics:
    """Re-evaluate one saved candidate without mutating its stored metadata."""
    if population_cutoff < 2:
        raise ValueError("--population-cutoff must be at least 2.")

    physics_config = copy.deepcopy(metadata["physics_config"])
    physics_config["model_settings"]["cutoff"] = population_cutoff
    simulator = FluxoniumResetSimulator(physics_config)
    return simulator.pathway_diagnostics(best_weights, times)


def plot_run(
    run_directory: Path,
    save_path: Path | None,
    show: bool,
    population_cutoff: int | None,
) -> None:
    """Plot reward history, controls, and optional pathway diagnostics."""
    metadata = load_json(run_directory / "metadata.json")
    summary = load_json(run_directory / "summary.json")
    print_run_summary(run_directory, metadata, summary, population_cutoff)

    controls_path = run_directory / "controls.npz"
    if not controls_path.is_file():
        raise FileNotFoundError(
            "No controls.npz exists yet; this run has not completed a generation."
        )
    with np.load(controls_path) as controls:
        times = controls["times"]
        external_phase = controls["external_phase"]
        charge_drive = controls["charge_drive"]

    weights_path = run_directory / "best_weights.npy"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Expected run artifact does not exist: {weights_path}")
    best_weights = np.load(weights_path, allow_pickle=False)

    history = load_history(run_directory / "history.csv")
    diagnostics = (
        None
        if population_cutoff is None
        else reproduce_pathway_diagnostics(
            metadata, best_weights, times, population_cutoff
        )
    )
    figure, axes = plt.subplots(
        3 if diagnostics is None else 5,
        1,
        figsize=(9, 9 if diagnostics is None else 13),
        constrained_layout=True,
    )

    if history:
        axes[0].plot(
            history["generation"],
            np.log(1 - history["best_reward_ever"] + 1e-6),
            label="best ever",
        )
        axes[0].plot(
            history["generation"],
            np.log(1 - history["best_reward_generation"] + 1e-6),
            alpha=0.6,
            label="generation best",
        )
        axes[0].set_ylabel("log(1-P(g, 0))")
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

    if diagnostics is not None:
        colors = {
            "g0": "tab:green",
            "e0": "tab:orange",
            "f0": "tab:red",
            "g1": "tab:purple",
        }
        for name, population in diagnostics.instantaneous_populations.items():
            axes[3].plot(
                times,
                population,
                label=rf"$P[{name[0]}(\varphi(t)),{name[1]}]$",
                color=colors[name],
            )
        selected_sum = np.sum(
            np.stack(tuple(diagnostics.instantaneous_populations.values())), axis=0
        )
        axes[3].plot(times, selected_sum, "k--", alpha=0.55, label="selected sum")
        axes[3].set_ylabel("population")
        axes[3].set_xlabel("time (ns)")
        axes[3].set_ylim(-0.02, 1.02)
        axes[3].set_title(
            f"instantaneous bare-fluxonium populations; final P(g, 0) = "
            f"{diagnostics.instantaneous_populations['g0'][-1]:.6f} "
            f"| cutoff {population_cutoff}"
        )
        axes[3].legend(loc="upper left", fontsize=5, framealpha=0.01)

        axes[4].plot(
            times, diagnostics.cavity_photon_number, label=r"$\langle n_c\rangle$"
        )
        axes[4].plot(times, diagnostics.trace, "k--", alpha=0.7, label=r"Tr($\rho$)")
        axes[4].set_ylabel("diagnostic")
        axes[4].set_xlabel("time (ns)")
        axes[4].set_ylim(-0.02, 1.05)
        axes[4].set_title("basis-independent cavity occupation and trace")
        axes[4].legend(loc="upper left", fontsize=7)

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
    parser.add_argument(
        "--population-cutoff",
        "--plot-pop",
        dest="population_cutoff",
        type=int,
        help="Reproduce and plot instantaneous pathway diagnostics at this cutoff.",
    )
    args = parser.parse_args()
    plot_run(
        resolve_run_directory(args.run),
        args.save,
        show=not args.no_show,
        population_cutoff=args.population_cutoff,
    )


if __name__ == "__main__":
    main()
