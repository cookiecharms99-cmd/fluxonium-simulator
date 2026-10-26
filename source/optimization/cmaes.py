"""Reusable CMA-ES and multiprocessing mechanics.

This module knows how to evaluate pulse coefficients and update CMA-ES. It
does not load configuration files, choose experiment paths, or save outputs;
those responsibilities belong to the experiment runner.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import cma
import numpy as np

from source.simulator import FluxoniumResetSimulator

ControlProtocol = Literal["joint", "reference"]

WORKER_SIMULATOR: FluxoniumResetSimulator | None = None
WORKER_TIMES: np.ndarray | None = None
WORKER_CONTROL_PROTOCOL: ControlProtocol | None = None


@dataclass(frozen=True)
class CMAESResult:
    """The information needed to inspect and reproduce one CMA-ES campaign."""

    best_reward: float
    best_weights: np.ndarray
    initial_weights: np.ndarray
    history: list[dict[str, float | int]]
    evaluation_count: int
    completed: bool
    stop_reason: str


@dataclass(frozen=True)
class CMAESProgress:
    """Best-so-far state after one fully completed generation."""

    best_reward: float
    best_weights: np.ndarray
    history: list[dict[str, float | int]]
    evaluation_count: int


GenerationCallback = Callable[[CMAESProgress], None]
StopRequested = Callable[[], bool]


def parameter_count(
    simulator: FluxoniumResetSimulator, control_protocol: ControlProtocol
) -> int:
    """Return the coefficient-vector length required by one control protocol."""
    if control_protocol == "joint":
        return simulator.n_controls
    if control_protocol == "reference":
        return simulator.n_flux_controls
    raise ValueError(f"Unknown control protocol: {control_protocol!r}.")


def evaluate_candidate(
    simulator: FluxoniumResetSimulator,
    candidate: np.ndarray,
    times: np.ndarray,
    control_protocol: ControlProtocol,
) -> float:
    """Evaluate one candidate under the selected physical control protocol."""
    if control_protocol == "joint":
        return simulator.final_ground_population(weights=candidate, times=times)
    if control_protocol == "reference":
        return simulator.reference_final_ground_population(
            flux_weights=candidate, times=times
        )
    raise ValueError(f"Unknown control protocol: {control_protocol!r}.")


def initialize_worker(
    physics_config: dict, times: np.ndarray, control_protocol: ControlProtocol
) -> None:
    """Construct one private simulator inside each worker process."""
    global WORKER_SIMULATOR, WORKER_TIMES, WORKER_CONTROL_PROTOCOL
    WORKER_SIMULATOR = FluxoniumResetSimulator(physics_config)
    WORKER_TIMES = times
    WORKER_CONTROL_PROTOCOL = control_protocol


def evaluate_worker(candidate: np.ndarray) -> float:
    """Evaluate one candidate using this worker's cached simulator."""
    if (
        WORKER_SIMULATOR is None
        or WORKER_TIMES is None
        or WORKER_CONTROL_PROTOCOL is None
    ):
        raise RuntimeError("Worker physics was not initialized.")
    return evaluate_candidate(
        WORKER_SIMULATOR,
        candidate,
        WORKER_TIMES,
        WORKER_CONTROL_PROTOCOL,
    )


def evaluate_serial(
    simulator: FluxoniumResetSimulator,
    candidates: Sequence[np.ndarray],
    times: np.ndarray,
    *,
    control_protocol: ControlProtocol = "joint",
) -> np.ndarray:
    """Evaluate one candidate population in the current process."""
    return np.asarray(
        [
            evaluate_candidate(simulator, candidate, times, control_protocol)
            for candidate in candidates
        ],
        dtype=float,
    )


def evaluate_parallel(
    physics_config: dict,
    candidates: Sequence[np.ndarray],
    times: np.ndarray,
    *,
    workers: int,
    control_protocol: ControlProtocol = "joint",
) -> np.ndarray:
    """Evaluate one population using worker-local simulator instances."""
    if workers <= 0:
        raise ValueError("workers must be positive.")

    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=initialize_worker,
        initargs=(physics_config, times, control_protocol),
    ) as pool:
        return np.asarray(pool.map(evaluate_worker, candidates), dtype=float)


def run_generations(
    strategy,
    evaluate,
    *,
    generations: int,
    on_generation: GenerationCallback | None = None,
    should_stop: StopRequested | None = None,
) -> tuple[float, np.ndarray, list[dict[str, float | int]], bool]:
    """Run the shared ask/evaluate/tell loop with a supplied evaluator."""
    best_reward = -np.inf
    best_weights: np.ndarray | None = None
    history: list[dict[str, float | int]] = []
    start = perf_counter()

    for generation in range(generations):
        candidates = strategy.ask()
        rewards = np.asarray(evaluate(candidates), dtype=float)
        if rewards.shape != (len(candidates),):
            raise ValueError("Evaluator returned one reward per candidate incorrectly.")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("A candidate evaluation returned a non-finite reward.")

        # pycma minimizes, while the reset objective maximizes P(g, 0).
        strategy.tell(candidates, (1.0 - rewards).tolist())

        generation_best_index = int(np.argmax(rewards))
        generation_best = float(rewards[generation_best_index])
        if generation_best > best_reward:
            best_reward = generation_best
            best_weights = np.asarray(
                candidates[generation_best_index], dtype=float
            ).copy()

        history.append(
            {
                "generation": generation + 1,
                "best_reward_ever": best_reward,
                "best_reward_generation": generation_best,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "sigma": float(strategy.sigma),
                "population_size": len(candidates),
                "evaluation_count": (generation + 1) * len(candidates),
                "elapsed_seconds": perf_counter() - start,
            }
        )

        if on_generation is not None:
            on_generation(
                CMAESProgress(
                    best_reward=best_reward,
                    best_weights=best_weights,
                    history=history.copy(),
                    evaluation_count=(generation + 1) * len(candidates),
                )
            )

        print(
            f"Generation {generation + 1:02d} | "
            f"best ever={best_reward:.6f} | "
            f"best generation={generation_best:.6f} | "
            f"mean={np.mean(rewards):.6f} | "
            f"sigma={strategy.sigma:.4f}"
        )

        if should_stop is not None and should_stop():
            print("Stopping after the completed generation; checkpoint is saved.")
            break

    if best_weights is None:
        raise RuntimeError("CMA-ES completed without evaluating a candidate.")
    return best_reward, best_weights, history, len(history) == generations


def run_cmaes(
    simulator: FluxoniumResetSimulator,
    physics_config: dict,
    times: np.ndarray,
    *,
    generations: int,
    population_size: int,
    sigma0: float,
    seed: int,
    workers: int = 1,
    control_protocol: ControlProtocol = "joint",
    initial_weights: np.ndarray | None = None,
    on_generation: GenerationCallback | None = None,
    should_stop: StopRequested | None = None,
) -> CMAESResult:
    """Run a minimal serial or multiprocessing CMA-ES campaign.

    Supplying ``initial_weights`` permits fair serial/parallel comparisons.
    Otherwise the initial center is generated reproducibly from ``seed``.
    """
    if generations <= 0:
        raise ValueError("generations must be positive.")
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")

    coefficient_count = parameter_count(simulator, control_protocol)
    if initial_weights is None:
        rng = np.random.default_rng(seed)
        initial_weights = rng.normal(0.0, 0.1, size=coefficient_count)
    else:
        initial_weights = np.asarray(initial_weights, dtype=float).copy()

    if initial_weights.shape != (coefficient_count,):
        raise ValueError(
            f"Expected {coefficient_count} initial weights for "
            f"{control_protocol!r} protocol, "
            f"got {initial_weights.shape}."
        )

    strategy = cma.CMAEvolutionStrategy(
        initial_weights,
        sigma0,
        {
            "popsize": population_size,
            "seed": seed,
            "verbose": -9,
        },
    )

    if workers == 1:
        best_reward, best_weights, history, completed = run_generations(
            strategy,
            lambda candidates: evaluate_serial(
                simulator, candidates, times, control_protocol=control_protocol
            ),
            generations=generations,
            on_generation=on_generation,
            should_stop=should_stop,
        )
    else:
        # Keep one pool alive for the whole campaign. Recreating it each generation
        # would repeatedly rebuild every worker's simulator.
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=initialize_worker,
            initargs=(physics_config, times, control_protocol),
        ) as pool:
            best_reward, best_weights, history, completed = run_generations(
                strategy,
                lambda candidates: pool.map(evaluate_worker, candidates),
                generations=generations,
                on_generation=on_generation,
                should_stop=should_stop,
            )

    return CMAESResult(
        best_reward=best_reward,
        best_weights=best_weights,
        initial_weights=initial_weights,
        history=history,
        evaluation_count=len(history) * population_size,
        completed=completed,
        stop_reason=(
            "requested_generation_count_reached"
            if completed
            else "termination_requested_after_completed_generation"
        ),
    )
