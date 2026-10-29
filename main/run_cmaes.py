"""Run a small, reproducible CMA-ES campaign with the project simulator."""

from __future__ import annotations

import os

# One numerical thread per process avoids oversubscribing the allocated CPUs.
# These settings must be applied before importing NumPy/SciPy/QuTiP.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Import stuff! -------------
import argparse
import copy
import json
import signal
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# ----------------------------------------

# Import project stuff! -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import source.control_initialization as control_init
from source.optimization.cmaes import (  # noqa: E402
    CMAESProgress,
    parameter_count,
    run_cmaes,
)
from source.run_artifacts import (  # noqa: E402
    create_run_directory,
    environment_snapshot,
    save_controls,
    save_history,
    sha256_file,
    write_json,
)
from source.simulator import (
    FluxoniumResetSimulator,  # noqa: E402
    sine_basis,
)

# -----------------------------------------------------

# Define uh.... macros? :3 -----------------------------
DEFAULT_DURATION = 150.0
# -----------------------------------


# ---------- Auxillary Funcs to load and setup ----------------
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_smoke_config(physics_config: dict) -> dict:
    """Return a cheaper in-memory configuration without editing the source JSON."""
    smoke_config = copy.deepcopy(physics_config)
    smoke_config["model_settings"]["cutoff"] = 8
    smoke_config["model_settings"]["num_t_steps"] = 250
    return smoke_config


def evaluate_protocol_center(
    simulator: FluxoniumResetSimulator,
    control_protocol: str,
    weights: np.ndarray,
    times: np.ndarray,
) -> float:
    """Evaluate the declared CMA-ES center before population sampling begins."""
    if control_protocol == "joint":
        return simulator.final_ground_population(weights, times)
    if control_protocol == "reference":
        return simulator.reference_final_ground_population(weights, times)
    raise ValueError(f"Unknown control protocol: {control_protocol!r}.")


def fill_initial_weights(
    simulator: FluxoniumResetSimulator,
    initialization: str,
    control_protocol: str,
    seed: int,
    times: np.ndarray,
    initial_weights: np.ndarray,
    warm_start_path: str | Path | None = None,
) -> None:
    """Fill a protocol-sized CMA-ES center using one declared scheme.

    Reference mode always has a flux-only candidate vector. It may still use
    a physics config that declares joint-control charge harmonics: those are
    intentionally ignored because reference mode supplies its fixed charge
    drive instead.
    """
    expected_count = parameter_count(simulator, control_protocol)
    if initial_weights.shape != (expected_count,):
        raise ValueError(
            f"Expected {expected_count} initial weights for {control_protocol!r} "
            f"protocol, got {initial_weights.shape}."
        )
    if initialization == "zero":
        initial_weights.fill(0.0)
    elif initialization == "random":
        initial_weights[:] = np.random.default_rng(seed).normal(
            0.0, 0.1, size=expected_count
        )
    elif initialization == "piecewise_lstsq":
        initial_flux_pulse = control_init.piecewise_pulse_init(times=times)
        flux_basis = sine_basis(times=times, count=simulator.n_flux_controls)
        initial_weights[: simulator.n_flux_controls] = (
            control_init.lst_sq_projection_flux(
                pulse=initial_flux_pulse,
                basis_arr=flux_basis,
                idle_phase=simulator.idle_phase,
            )
        )
    elif initialization == "warm_start":
        if warm_start_path is None:
            raise ValueError(
                "--warm-start is required when --initialization warm_start is selected."
            )
        source_path = Path(warm_start_path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Warm-start weights file does not exist: {source_path}"
            )
        source_weights = np.asarray(np.load(source_path, allow_pickle=False), dtype=float)
        if source_weights.shape != initial_weights.shape:
            raise ValueError(
                "Warm-start weight shape does not match the selected control "
                f"protocol: source {source_weights.shape}, expected "
                f"{initial_weights.shape}. Projection between bases is not yet "
                "implemented."
            )
        initial_weights[:] = source_weights
    else:
        raise ValueError(f"Unknown initialization scheme: {initialization!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the current educational CMA-ES campaign."
    )
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument(
        "--physics-config",
        default="config/mini_audit.json",
        help="Physics/model JSON path, relative to the project root by default.",
    )
    parser.add_argument(
        "--control-protocol",
        choices=("joint", "reference"),
        default="joint",
        help="Joint flux/charge optimization or the fixed-charge flux-only reference.",
    )

    parser.add_argument(
        "--initialization",
        choices=("zero", "random", "piecewise_lstsq", "warm_start"),
        default="random",
        help=(
            "CMA-ES center: zero coefficients, seeded random coefficients, or "
            "the 150-ns professor-inspired piecewise flux pulse projected into "
            "the sine basis."
        ),
    )
    parser.add_argument(
        "--full-physics",
        action="store_true",
        help="Use the configured cutoff and time grid instead of the cheap smoke settings.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help=(
            "Empty directory for this run, relative to the project root unless "
            "absolute. Defaults to a timestamped directory under "
            "data/output/cmaes/."
        ),
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        default=0.3,
        help=("Provide the initial std dev of the dist for cmaes :3"),
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help=(
            "Path to compatible .npy warm-start weights. Required only when "
            "--initialization warm_start is selected; basis projection is not "
            "yet supported."
        ),
    )
    return parser


# signal exit handling
#


# --------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    physics_path = Path(args.physics_config)
    if not physics_path.is_absolute():
        physics_path = PROJECT_ROOT / physics_path
    # optimizer_path = PROJECT_ROOT / "config" / "model_config" / "cmaes_config.json"

    physics_config = load_json(physics_path)
    # optimizer_config = load_json(optimizer_path)
    if not args.full_physics:
        physics_config = make_smoke_config(physics_config)

    simulator = FluxoniumResetSimulator(physics_config)
    num_steps = int(physics_config["model_settings"]["num_t_steps"])
    times = np.linspace(0.0, args.duration, num_steps)
    coefficient_count = parameter_count(simulator, args.control_protocol)
    initial_weights = np.zeros(coefficient_count)
    fill_initial_weights(
        initialization=args.initialization,
        control_protocol=args.control_protocol,
        seed=args.seed,
        simulator=simulator,
        times=times,
        initial_weights=initial_weights,
        warm_start_path=args.warm_start,
    )

    output_path = create_run_directory(PROJECT_ROOT, args.out_dir)
    mode = "full configured physics" if args.full_physics else "smoke physics"

    print(
        f"CMA-ES {args.control_protocol} run ({mode}): {args.generations} generations x "
        f"{args.popsize} candidates on {args.workers} worker(s)"
    )
    print(f"Initialization: {args.initialization}")

    # ------ Print output path--------
    try:
        display_output_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_output_path = output_path
    print(f"Saving run record to: {display_output_path}")

    # -------------------------------------
    #
    #
    # -----------------Save meta data -------
    metadata = {
        "status": "running",
        "command": sys.argv,
        "mode": "full" if args.full_physics else "smoke",
        "physics_config": physics_config,
        "optimizer_config": "Currently deprecated",
        "run_parameters": {
            "generations": args.generations,
            "population_size": args.popsize,
            "workers": args.workers,
            "seed": args.seed,
            "sigma0": float(args.sigma0),
            "duration_ns": args.duration,
            "time_steps": num_steps,
            "control_protocol": args.control_protocol,
            "initialization_scheme": args.initialization,
            "warm_start_weights_path": (
                args.warm_start if args.initialization == "warm_start" else None
            ),
            "warm_start_transfer": (
                "direct_compatible_weight_copy"
                if args.initialization == "warm_start"
                else None
            ),
            "physics_config_path": str(physics_path.relative_to(PROJECT_ROOT)),
            "initial_weight_count": coefficient_count,
        },
        "environment": environment_snapshot(PROJECT_ROOT),
    }

    np.save(output_path / "initial_weights.npy", initial_weights)
    metadata["initial_weights_sha256"] = sha256_file(
        output_path / "initial_weights.npy"
    )

    termination_requested = False
    start = perf_counter()
    latest_progress: CMAESProgress | None = None
    initial_center_population = evaluate_protocol_center(
        simulator, args.control_protocol, initial_weights, times
    )
    metadata["initial_center_population"] = initial_center_population
    write_json(output_path / "metadata.json", metadata)
    print(f"Initial center P(g, 0): {initial_center_population:.6f}")

    def checkpoint_artifact_hashes() -> dict[str, str]:
        """Fingerprint artifacts that exist at a completed-generation boundary."""
        names = [
            "initial_weights.npy",
            "best_weights.npy",
            "history.csv",
            "controls.npz",
        ]
        return {
            name: sha256_file(output_path / name)
            for name in names
            if (output_path / name).is_file()
        }

    def write_provisional_summary(progress: CMAESProgress | None) -> None:
        """Record inspectable best-so-far state without claiming a final result."""
        completed_generations = 0 if progress is None else len(progress.history)
        summary = {
            "status": "running",
            "stop_reason": None,
            "best_final_ground_population": (
                None if progress is None else progress.best_reward
            ),
            "initial_center_population": initial_center_population,
            "elapsed_seconds": perf_counter() - start,
            "completed_generations": completed_generations,
            "requested_generations": args.generations,
            "objective_evaluations": (
                0 if progress is None else progress.evaluation_count
            ),
            "duration_ns": args.duration,
            "fresh_reproduction_population": None,
            "reproduction_delta": None,
            "artifacts_sha256": checkpoint_artifact_hashes(),
        }
        write_json(output_path / "summary.json", summary)

    # If a job ends before its first complete generation, this still explains
    # the run directory rather than leaving it without a summary at all.
    write_provisional_summary(None)

    def request_termination(_signal_number, _frame) -> None:
        nonlocal termination_requested
        if not termination_requested:
            print(
                "Termination signal received. The current generation will finish, "
                "then the best completed result will be saved."
            )
        termination_requested = True

    previous_sigterm_handler = signal.signal(signal.SIGTERM, request_termination)

    def save_checkpoint(progress: CMAESProgress) -> None:
        """Persist the newest fully completed generation before starting another."""
        nonlocal latest_progress
        latest_progress = progress
        np.save(output_path / "best_weights.npy", progress.best_weights)
        save_history(output_path / "history.csv", progress.history)
        save_controls(
            output_path / "controls.npz",
            times,
            (
                simulator.controls(progress.best_weights, times)
                if args.control_protocol == "joint"
                else simulator.reference_controls(progress.best_weights, times)
            ),
        )
        metadata["completed_generations"] = len(progress.history)
        metadata["completed_objective_evaluations"] = progress.evaluation_count
        write_json(output_path / "metadata.json", metadata)
        write_provisional_summary(progress)

    try:
        result = run_cmaes(
            simulator,
            physics_config,
            times,
            generations=args.generations,
            population_size=args.popsize,
            sigma0=args.sigma0,
            seed=args.seed,
            workers=args.workers,
            control_protocol=args.control_protocol,
            initial_weights=initial_weights,
            on_generation=save_checkpoint,
            should_stop=lambda: termination_requested,
        )
        elapsed = perf_counter() - start

        summary = {
            "status": "completed" if result.completed else "stopped",
            "stop_reason": result.stop_reason,
            "best_final_ground_population": result.best_reward,
            "initial_center_population": initial_center_population,
            "elapsed_seconds": elapsed,
            "completed_generations": len(result.history),
            "requested_generations": args.generations,
            "objective_evaluations": result.evaluation_count,
            "duration_ns": args.duration,
            "artifacts_sha256": {
                name: sha256_file(output_path / name)
                for name in [
                    "initial_weights.npy",
                    "best_weights.npy",
                    "history.csv",
                    "controls.npz",
                ]
            },
        }

        if result.completed:
            # A fresh simulator confirms that the saved pulse, not worker-local
            # state, produces the result written to disk.
            reproduction_simulator = FluxoniumResetSimulator(physics_config)
            if args.control_protocol == "joint":
                reproduced_reward = reproduction_simulator.final_ground_population(
                    result.best_weights, times
                )
            else:
                reproduced_reward = (
                    reproduction_simulator.reference_final_ground_population(
                        result.best_weights, times
                    )
                )
            reproduction_delta = reproduced_reward - result.best_reward
            if not np.isclose(
                reproduced_reward, result.best_reward, rtol=1e-10, atol=1e-12
            ):
                raise RuntimeError(
                    "Fresh-simulator reproduction did not match the recorded best reward: "
                    f"{reproduced_reward:.12f} versus {result.best_reward:.12f}."
                )
            summary["fresh_reproduction_population"] = reproduced_reward
            summary["reproduction_delta"] = reproduction_delta
        else:
            summary["fresh_reproduction_population"] = None
            summary["reproduction_delta"] = None
        metadata["status"] = summary["status"]
        write_json(output_path / "metadata.json", metadata)
        write_json(output_path / "summary.json", summary)

        print(f"Best final P(g, 0): {result.best_reward:.6f}")
        print(f"Best coefficient vector shape: {result.best_weights.shape}")
        if result.completed:
            print(f"Fresh reproduction: {reproduced_reward:.6f}")
        else:
            print("Stopped cleanly after a completed generation.")
        print(f"Elapsed time: {elapsed:.2f} s")

    except Exception as error:
        metadata["status"] = "failed"
        metadata["failure"] = f"{type(error).__name__}: {error}"
        write_json(output_path / "metadata.json", metadata)
        write_json(
            output_path / "summary.json",
            {
                "status": "failed",
                "stop_reason": f"{type(error).__name__}: {error}",
                "best_final_ground_population": (
                    None if latest_progress is None else latest_progress.best_reward
                ),
                "initial_center_population": initial_center_population,
                "elapsed_seconds": perf_counter() - start,
                "completed_generations": (
                    0 if latest_progress is None else len(latest_progress.history)
                ),
                "requested_generations": args.generations,
                "objective_evaluations": (
                    0 if latest_progress is None else latest_progress.evaluation_count
                ),
                "duration_ns": args.duration,
                "fresh_reproduction_population": None,
                "reproduction_delta": None,
                "artifacts_sha256": checkpoint_artifact_hashes(),
            },
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    main()
