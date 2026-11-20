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
import csv
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
    weights: np.ndarray,
    times: np.ndarray,
) -> float:
    """Evaluate the declared CMA-ES center before population sampling begins."""
    return simulator.final_ground_population(weights, times)


def sqrt_loss_sigma0(
    initial_fidelity: float | None = None, default_sigma0: float = 0.3, min_sigma0: float = 0.015
) -> float:
    """Scale sigma0 as sqrt(initial infidelity), with conservative limits."""
    if initial_fidelity is not None and initial_fidelity > 0.50:
        loss = max(1 - initial_fidelity, 1e-6)
        scaled_sigma0 = default_sigma0 * np.sqrt(loss / 0.50)
        return float(np.clip(scaled_sigma0, a_min=min_sigma0, a_max=default_sigma0))
    return default_sigma0


def load_parent_final_sigma(warm_start_path: str | Path) -> tuple[float, Path]:
    """Load a parent run's final sigma, including pre-final-sigma artifacts.

    New runs record ``final_sigma`` directly. Older completed/checkpointed
    artifacts may only have the last ``sigma`` in ``history.csv``; that is the
    closest faithful record of the parent distribution's final spread.
    """
    parent_directory = Path(warm_start_path).resolve().parent
    for artifact_name in ("metadata.json", "summary.json"):
        artifact_path = parent_directory / artifact_name
        if not artifact_path.is_file():
            continue
        with artifact_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        for field in ("final_sigma", "latest_sigma"):
            value = payload.get(field)
            if value is not None and np.isfinite(value) and value > 0.0:
                return float(value), artifact_path

    history_path = parent_directory / "history.csv"
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            value = float(rows[-1]["sigma"])
            if np.isfinite(value) and value > 0.0:
                return value, history_path

    raise ValueError(
        "Could not find a positive final sigma in the warm-start parent run. "
        "Expected final_sigma/latest_sigma in metadata.json or summary.json, "
        "or a final sigma column in history.csv."
    )


def select_sigma0(
    mode: str,
    *,
    initial_fidelity: float,
    fixed_sigma0: float | None,
    warm_start_path: str | Path | None,
) -> tuple[float, str]:
    """Choose a declared initial CMA-ES spread and describe its source."""
    if mode == "sqrt_loss":
        if fixed_sigma0 is not None:
            raise ValueError("--sigma0 is only valid with --sigma0-mode fixed.")
        return sqrt_loss_sigma0(initial_fidelity), "sqrt_loss(initial_center_population)"

    if mode == "fixed":
        if fixed_sigma0 is None:
            raise ValueError("--sigma0-mode fixed requires a positive --sigma0.")
        if not np.isfinite(fixed_sigma0) or fixed_sigma0 <= 0.0:
            raise ValueError("--sigma0 must be a finite positive number.")
        return float(fixed_sigma0), "explicit --sigma0"

    if mode == "parent_final":
        if fixed_sigma0 is not None:
            raise ValueError("--sigma0 is only valid with --sigma0-mode fixed.")
        if warm_start_path is None:
            raise ValueError("--sigma0-mode parent_final requires --warm-start.")
        sigma0, source_path = load_parent_final_sigma(warm_start_path)
        return sigma0, f"parent final sigma from {source_path}"

    raise ValueError(f"Unknown sigma0 mode: {mode!r}.")


def apply_control_overrides(physics_config: dict, args: argparse.Namespace) -> None:
    """Apply declared experiment choices to the copied device configuration.

    The JSON file identifies the calibrated device and numerical defaults. CLI
    flags identify one experiment's active/fixed controls; the resulting full
    configuration is written into that run's artifact directory.
    """
    controls = physics_config["model_settings"]["controls"]
    flux = controls["flux"]
    charge = controls["charge"]

    if args.flux_count is not None:
        flux["coefficient_count"] = args.flux_count
    if args.charge_count is not None:
        charge["coefficient_count"] = args.charge_count
    if args.charge_mode is not None:
        charge["mode"] = args.charge_mode
    if args.fixed_flux is not None:
        flux["fixed_default"] = args.fixed_flux
    if args.saved_flux_controls is not None:
        if args.fixed_flux not in (None, "saved_controls"):
            raise ValueError("--saved-flux-controls requires --fixed-flux saved_controls.")
        flux["fixed_default"] = "saved_controls"
        flux["saved_controls_path"] = args.saved_flux_controls
    if args.fixed_flux_harmonic_count is not None:
        flux["fixed_harmonic_count"] = args.fixed_flux_harmonic_count
    if args.fixed_flux_seed is not None:
        flux["random_seed"] = args.fixed_flux_seed
    if args.fixed_flux_scale is not None:
        flux["random_scale"] = args.fixed_flux_scale

    random_flux_arguments = (
        args.fixed_flux_harmonic_count,
        args.fixed_flux_seed,
        args.fixed_flux_scale,
    )
    if any(value is not None for value in random_flux_arguments):
        if flux["fixed_default"] != "random_sine":
            raise ValueError(
                "Fixed random-flux parameters require --fixed-flux random_sine."
            )
        if any(value is None for value in random_flux_arguments):
            raise ValueError(
                "--fixed-flux random_sine needs --fixed-flux-harmonic-count, "
                "--fixed-flux-seed, and --fixed-flux-scale."
            )


def fill_initial_weights(
    simulator: FluxoniumResetSimulator,
    initialization: str,
    seed: int,
    times: np.ndarray,
    initial_weights: np.ndarray,
    warm_start_path: str | Path | None = None,
) -> None:
    """Fill this configuration's CMA-ES center using one declared scheme."""
    if initial_weights.shape != (simulator.n_controls,):
        raise ValueError(
            f"Expected {simulator.n_controls} initial weights, got "
            f"{initial_weights.shape}."
        )
    if initialization == "zero":
        initial_weights.fill(0.0)
    elif initialization == "random":
        initial_weights[:] = np.random.default_rng(seed).normal(
            0.0, 0.1, size=simulator.n_controls
        )
    elif initialization == "random_flux_zero_charge":
        if simulator.n_flux_controls == 0:
            raise ValueError("random_flux_zero_charge needs active flux coefficients.")
        initial_weights[: simulator.n_flux_controls] = np.random.default_rng(
            seed
        ).normal(0.0, 0.1, size=simulator.n_flux_controls)
        initial_weights[simulator.n_flux_controls :] = 0.0
    elif initialization == "piecewise_lstsq":
        if simulator.n_flux_controls == 0:
            raise ValueError(
                "piecewise_lstsq needs active flux coefficients; choose a flux "
                "configuration or use zero/random/warm_start."
            )
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
        source_weights = np.asarray(
            np.load(source_path, allow_pickle=False), dtype=float
        )
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
        default="config/device_calibrated.json",
        help="Physics/model JSON path, relative to the project root by default.",
    )
    parser.add_argument(
        "--initialization",
        choices=(
            "zero",
            "random",
            "random_flux_zero_charge",
            "piecewise_lstsq",
            "warm_start",
        ),
        default="random",
        help=(
            "CMA-ES center: zero, seeded random, random-flux/zero-charge, or "
            "the professor-inspired piecewise flux pulse projected into the sine basis."
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
        "--label",
        help=(
            "Optional filesystem-safe prefix for an automatically created run "
            "directory, for example reference-140ns-warm-full. Ignored when "
            "--out-dir is supplied."
        ),
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        help="Explicit initial CMA-ES standard deviation; requires --sigma0-mode fixed.",
    )
    parser.add_argument(
        "--sigma0-mode",
        choices=("sqrt_loss", "parent_final", "fixed"),
        default="sqrt_loss",
        help=(
            "Choose sigma0 from initial infidelity, copy the warm-start parent's "
            "final sigma, or use explicit --sigma0."
        ),
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
    parser.add_argument(
        "--flux-count",
        type=int,
        help=(
            "Override the configuration's active flux-coefficient count for this run. "
            "Use 0 to hold the configured fixed flux waveform."
        ),
    )
    parser.add_argument(
        "--charge-count",
        type=int,
        help="Override the configuration's active charge-coefficient count for this run.",
    )
    parser.add_argument(
        "--charge-mode",
        choices=(
            "free_fourier",
            "calibrated_plus_fourier_correction",
            "calibrated_times_one_plus_fourier_correction",
        ),
        help="Charge-control meaning for an active charge block.",
    )
    parser.add_argument(
        "--fixed-flux",
        choices=("idle", "piecewise", "random_sine", "saved_controls"),
        help="Waveform used when --flux-count 0 holds flux fixed.",
    )
    parser.add_argument(
        "--saved-flux-controls",
        help="controls.npz to use with --fixed-flux saved_controls.",
    )
    parser.add_argument(
        "--fixed-flux-harmonic-count",
        type=int,
        help="Harmonic count for --fixed-flux random_sine.",
    )
    parser.add_argument(
        "--fixed-flux-seed",
        type=int,
        help="Random seed for --fixed-flux random_sine.",
    )
    parser.add_argument(
        "--fixed-flux-scale",
        type=float,
        help="Coefficient standard deviation for --fixed-flux random_sine.",
    )
    return parser


# --------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    physics_path = Path(args.physics_config)
    if not physics_path.is_absolute():
        physics_path = PROJECT_ROOT / physics_path
    # optimizer_path = PROJECT_ROOT / "config" / "model_config" / "cmaes_config.json"

    physics_config = load_json(physics_path)

    if args.flux_count is not None and args.flux_count < 0:
        raise ValueError("--flux-count must be non-negative.")
    if args.charge_count is not None and args.charge_count < 0:
        raise ValueError("--charge-count must be non-negative.")
    apply_control_overrides(physics_config, args)
    # optimizer_config = load_json(optimizer_path)
    if not args.full_physics:
        physics_config = make_smoke_config(physics_config)

    simulator = FluxoniumResetSimulator(physics_config)
    num_steps = int(physics_config["model_settings"]["num_t_steps"])
    times = np.linspace(0.0, args.duration, num_steps)
    coefficient_count = simulator.n_controls

    initial_weights = np.zeros(coefficient_count)
    fill_initial_weights(
        initialization=args.initialization,
        seed=args.seed,
        simulator=simulator,
        times=times,
        initial_weights=initial_weights,
        warm_start_path=args.warm_start,
    )

    output_path = create_run_directory(PROJECT_ROOT, args.out_dir, label=args.label)
    write_json(output_path / "physics_config.json", physics_config)
    mode = "full configured physics" if args.full_physics else "smoke physics"

    print(
        f"CMA-ES run ({mode}; {simulator.n_flux_controls} flux + "
        f"{simulator.n_charge_controls} charge coefficients): {args.generations} generations x "
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
            "duration_ns": args.duration,
            "time_steps": num_steps,
            "flux_count_override": args.flux_count,
            "charge_count_override": args.charge_count,
            "charge_mode_override": args.charge_mode,
            "fixed_flux_override": args.fixed_flux,
            "sigma0_mode": args.sigma0_mode,
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
            "effective_physics_config_artifact": "physics_config.json",
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
        simulator, initial_weights, times
    )
    metadata["initial_center_population"] = initial_center_population
    sigma0, sigma0_source = select_sigma0(
        args.sigma0_mode,
        initial_fidelity=initial_center_population,
        fixed_sigma0=args.sigma0,
        warm_start_path=args.warm_start,
    )
    metadata["run_parameters"]["sigma0"] = sigma0
    metadata["run_parameters"]["sigma0_source"] = sigma0_source
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
                initial_center_population if progress is None else progress.best_reward
            ),
            "final_sigma": sigma0 if progress is None else progress.sigma,
            "initial_center_population": initial_center_population,
            "elapsed_seconds": perf_counter() - start,
            "completed_generations": completed_generations,
            "requested_generations": args.generations,
            "objective_evaluations": (
                1 if progress is None else progress.evaluation_count
            ),
            "duration_ns": args.duration,
            "fresh_reproduction_population": None,
            "reproduction_delta": None,
            "artifacts_sha256": checkpoint_artifact_hashes(),
        }
        write_json(output_path / "summary.json", summary)

    # The evaluated center is a real incumbent before generation 1. Persist it
    # now, so a timeout during the first population cannot erase it.
    np.save(output_path / "best_weights.npy", initial_weights)
    save_controls(
        output_path / "controls.npz",
        times,
        simulator.controls(initial_weights, times),
    )

    # If a job ends before its first complete generation, this records the
    # evaluated center rather than leaving the run without a useful summary.
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
            simulator.controls(progress.best_weights, times),
        )
        metadata["completed_generations"] = len(progress.history)
        metadata["completed_objective_evaluations"] = progress.evaluation_count
        metadata["latest_sigma"] = progress.sigma
        write_json(output_path / "metadata.json", metadata)
        write_provisional_summary(progress)

    try:
        result = run_cmaes(
            simulator,
            physics_config,
            times,
            generations=args.generations,
            population_size=args.popsize,
            sigma0=sigma0,
            seed=args.seed,
            workers=args.workers,
            initial_weights=initial_weights,
            initial_reward=initial_center_population,
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
            "final_sigma": result.final_sigma,
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
            reproduced_reward = reproduction_simulator.final_ground_population(
                result.best_weights, times
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
        metadata["final_sigma"] = result.final_sigma
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
                    initial_center_population
                    if latest_progress is None
                    else latest_progress.best_reward
                ),
                "initial_center_population": initial_center_population,
                "elapsed_seconds": perf_counter() - start,
                "completed_generations": (
                    0 if latest_progress is None else len(latest_progress.history)
                ),
                "requested_generations": args.generations,
                "objective_evaluations": (
                    1 if latest_progress is None else latest_progress.evaluation_count
                ),
                "final_sigma": (
                    sigma0 if latest_progress is None else latest_progress.sigma
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
