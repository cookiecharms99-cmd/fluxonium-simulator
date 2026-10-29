"""Regression checks for runner-selected CMA-ES initialization schemes."""

from __future__ import annotations

import numpy as np
import pytest

from main.run_cmaes import (
    DEFAULT_DURATION,
    evaluate_protocol_center,
    fill_initial_weights,
)
from source.simulator import sine_basis


class TinyReferenceSimulator:
    """Only the runner-facing simulator attributes needed for this fast test."""

    n_controls = 6
    n_flux_controls = 4
    idle_phase = np.pi

    def final_ground_population(self, weights, times) -> float:
        return 0.11

    def reference_final_ground_population(self, weights, times) -> float:
        return 0.22


def test_runner_builds_piecewise_projected_reference_center() -> None:
    """Catch API drift between the runner and pulse/basis helper functions."""
    simulator = TinyReferenceSimulator()
    times = np.linspace(0.0, DEFAULT_DURATION, 501)
    weights = np.zeros(simulator.n_flux_controls)

    fill_initial_weights(
        initialization="piecewise_lstsq",
        control_protocol="reference",
        seed=1234,
        simulator=simulator,
        times=times,
        initial_weights=weights,
    )

    assert weights.shape == (simulator.n_flux_controls,)
    reconstructed_phase = simulator.idle_phase + weights @ sine_basis(
        times, count=simulator.n_flux_controls
    )
    np.testing.assert_allclose(
        reconstructed_phase[[0, -1]], simulator.idle_phase, atol=1e-12
    )


def test_runner_piecewise_center_keeps_joint_charge_weights_zero() -> None:
    """The same flux guess may seed joint control without enabling charge tuning."""
    simulator = TinyReferenceSimulator()
    weights = np.zeros(simulator.n_controls)

    fill_initial_weights(
        initialization="piecewise_lstsq",
        control_protocol="joint",
        seed=1234,
        simulator=simulator,
        times=np.linspace(0.0, DEFAULT_DURATION, 11),
        initial_weights=weights,
    )

    assert np.any(weights[: simulator.n_flux_controls] != 0.0)
    np.testing.assert_array_equal(weights[simulator.n_flux_controls :], 0.0)


def test_runner_rejects_wrong_protocol_vector_length() -> None:
    with pytest.raises(ValueError, match="Expected 4 initial weights"):
        fill_initial_weights(
            initialization="zero",
            control_protocol="reference",
            seed=1234,
            simulator=TinyReferenceSimulator(),
            times=np.linspace(0.0, DEFAULT_DURATION, 11),
            initial_weights=np.zeros(TinyReferenceSimulator.n_controls),
        )


def test_runner_evaluates_the_center_through_the_selected_protocol() -> None:
    simulator = TinyReferenceSimulator()
    weights = np.zeros(simulator.n_flux_controls)
    times = np.linspace(0.0, DEFAULT_DURATION, 11)

    assert evaluate_protocol_center(simulator, "joint", weights, times) == 0.11
    assert evaluate_protocol_center(simulator, "reference", weights, times) == 0.22


def test_runner_warm_start_copies_compatible_weights(tmp_path) -> None:
    source_weights = np.array([0.2, -0.1, 0.05, 0.03])
    source_path = tmp_path / "parent_best_weights.npy"
    np.save(source_path, source_weights)
    child_weights = np.zeros_like(source_weights)

    fill_initial_weights(
        initialization="warm_start",
        control_protocol="reference",
        seed=1234,
        simulator=TinyReferenceSimulator(),
        times=np.linspace(0.0, DEFAULT_DURATION, 11),
        initial_weights=child_weights,
        warm_start_path=source_path,
    )

    np.testing.assert_array_equal(child_weights, source_weights)


def test_runner_warm_start_rejects_missing_or_incompatible_source(tmp_path) -> None:
    simulator = TinyReferenceSimulator()
    common_arguments = {
        "initialization": "warm_start",
        "control_protocol": "reference",
        "seed": 1234,
        "simulator": simulator,
        "times": np.linspace(0.0, DEFAULT_DURATION, 11),
        "initial_weights": np.zeros(simulator.n_flux_controls),
    }

    with pytest.raises(ValueError, match="--warm-start is required"):
        fill_initial_weights(**common_arguments)

    mismatched_path = tmp_path / "wrong_size.npy"
    np.save(mismatched_path, np.zeros(simulator.n_controls))
    with pytest.raises(ValueError, match="shape does not match"):
        fill_initial_weights(**common_arguments, warm_start_path=mismatched_path)
