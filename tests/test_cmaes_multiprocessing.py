"""Integration checks for deterministic multiprocessing evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from source.optimization.cmaes import evaluate_parallel, evaluate_serial
from source.simulator import FluxoniumResetSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def small_physics_config() -> dict:
    """Use a cheap model because this test checks plumbing, not convergence."""
    with (PROJECT_ROOT / "config" / "mini_audit.json").open(
        "r", encoding="utf-8"
    ) as file:
        config = copy.deepcopy(json.load(file))

    settings = config["model_settings"]
    settings["cutoff"] = 8
    settings["n_flux"] = 4
    settings["n_charge"] = 2
    settings["num_t_steps"] = 60
    return config


def test_parallel_rewards_match_serial_rewards() -> None:
    physics_config = small_physics_config()
    simulator = FluxoniumResetSimulator(physics_config)
    times = np.linspace(0.0, 5.0, physics_config["model_settings"]["num_t_steps"])

    rng = np.random.default_rng(1234)
    candidates = [
        rng.normal(0.0, 0.1, size=simulator.n_controls) for _ in range(2)
    ]

    serial_rewards = evaluate_serial(simulator, candidates, times)
    parallel_rewards = evaluate_parallel(
        physics_config,
        candidates,
        times,
        workers=2,
    )

    np.testing.assert_allclose(
        parallel_rewards,
        serial_rewards,
        rtol=1e-10,
        atol=1e-12,
    )


def test_reference_parallel_rewards_match_serial_rewards() -> None:
    physics_config = small_physics_config()
    simulator = FluxoniumResetSimulator(physics_config)
    times = np.linspace(0.0, 5.0, physics_config["model_settings"]["num_t_steps"])

    rng = np.random.default_rng(5678)
    candidates = [
        rng.normal(0.0, 0.1, size=simulator.n_flux_controls) for _ in range(2)
    ]

    serial_rewards = evaluate_serial(
        simulator, candidates, times, control_protocol="reference"
    )
    parallel_rewards = evaluate_parallel(
        physics_config,
        candidates,
        times,
        workers=2,
        control_protocol="reference",
    )

    np.testing.assert_allclose(
        parallel_rewards,
        serial_rewards,
        rtol=1e-10,
        atol=1e-12,
    )
