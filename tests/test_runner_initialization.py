"""Regression checks for runner-selected CMA-ES initialization schemes."""

from __future__ import annotations

import numpy as np
import pytest

from main.run_cmaes import (
    DEFAULT_DURATION,
    apply_control_overrides,
    build_parser,
    evaluate_protocol_center,
    fill_initial_weights,
    load_parent_final_sigma,
    select_sigma0,
)
from source.simulator import sine_basis


class TinySimulator:
    """Only the runner-facing simulator attributes needed for this fast test."""

    n_controls = 6
    n_flux_controls = 4
    idle_phase = np.pi

    def final_ground_population(self, weights, times) -> float:
        return 0.11


def test_runner_builds_piecewise_projected_center() -> None:
    """Catch API drift between the runner and pulse/basis helper functions."""
    simulator = TinySimulator()
    times = np.linspace(0.0, DEFAULT_DURATION, 501)
    weights = np.zeros(simulator.n_controls)

    fill_initial_weights(
        initialization="piecewise_lstsq",
        seed=1234,
        simulator=simulator,
        times=times,
        initial_weights=weights,
    )

    reconstructed_phase = simulator.idle_phase + weights[: simulator.n_flux_controls] @ sine_basis(
        times, count=simulator.n_flux_controls
    )
    np.testing.assert_allclose(
        reconstructed_phase[[0, -1]], simulator.idle_phase, atol=1e-12
    )
    np.testing.assert_array_equal(weights[simulator.n_flux_controls :], 0.0)


def test_runner_piecewise_center_keeps_joint_charge_weights_zero() -> None:
    """The same flux guess may seed joint control without enabling charge tuning."""
    simulator = TinySimulator()
    weights = np.zeros(simulator.n_controls)

    fill_initial_weights(
        initialization="piecewise_lstsq",
        seed=1234,
        simulator=simulator,
        times=np.linspace(0.0, DEFAULT_DURATION, 11),
        initial_weights=weights,
    )

    assert np.any(weights[: simulator.n_flux_controls] != 0.0)
    np.testing.assert_array_equal(weights[simulator.n_flux_controls :], 0.0)


def test_runner_can_seed_random_flux_without_a_charge_correction() -> None:
    simulator = TinySimulator()
    weights = np.zeros(simulator.n_controls)

    fill_initial_weights(
        initialization="random_flux_zero_charge",
        seed=1234,
        simulator=simulator,
        times=np.linspace(0.0, DEFAULT_DURATION, 11),
        initial_weights=weights,
    )

    assert np.any(weights[: simulator.n_flux_controls] != 0.0)
    np.testing.assert_array_equal(weights[simulator.n_flux_controls :], 0.0)


def test_runner_rejects_wrong_vector_length() -> None:
    with pytest.raises(ValueError, match="Expected 6 initial weights"):
        fill_initial_weights(
            initialization="zero",
            seed=1234,
            simulator=TinySimulator(),
            times=np.linspace(0.0, DEFAULT_DURATION, 11),
            initial_weights=np.zeros(4),
        )


def test_runner_evaluates_the_configured_center() -> None:
    simulator = TinySimulator()
    weights = np.zeros(simulator.n_controls)
    times = np.linspace(0.0, DEFAULT_DURATION, 11)

    assert evaluate_protocol_center(simulator, weights, times) == 0.11


def test_runner_control_flags_create_a_saved_flux_charge_experiment() -> None:
    """Scenario choices belong to the recorded effective config, not new JSON files."""
    physics_config = {
        "model_settings": {
            "controls": {
                "flux": {"coefficient_count": 40, "fixed_default": "idle"},
                "charge": {
                    "coefficient_count": 0,
                    "mode": "calibrated_plus_fourier_correction",
                },
            }
        }
    }
    args = build_parser().parse_args(
        [
            "--flux-count",
            "0",
            "--charge-count",
            "20",
            "--fixed-flux",
            "saved_controls",
            "--saved-flux-controls",
            "parent/controls.npz",
        ]
    )

    apply_control_overrides(physics_config, args)

    assert physics_config["model_settings"]["controls"] == {
        "flux": {
            "coefficient_count": 0,
            "fixed_default": "saved_controls",
            "saved_controls_path": "parent/controls.npz",
        },
        "charge": {
            "coefficient_count": 20,
            "mode": "calibrated_plus_fourier_correction",
        },
    }


def test_runner_rejects_incomplete_fixed_random_flux_flags() -> None:
    physics_config = {
        "model_settings": {
            "controls": {
                "flux": {"coefficient_count": 0, "fixed_default": "idle"},
                "charge": {
                    "coefficient_count": 0,
                    "mode": "calibrated_plus_fourier_correction",
                },
            }
        }
    }
    args = build_parser().parse_args(
        ["--fixed-flux", "random_sine", "--fixed-flux-seed", "7"]
    )

    with pytest.raises(ValueError, match="needs --fixed-flux-harmonic-count"):
        apply_control_overrides(physics_config, args)


def test_parent_final_sigma_prefers_recorded_metadata(tmp_path) -> None:
    warm_start_path = tmp_path / "best_weights.npy"
    np.save(warm_start_path, np.zeros(4))
    (tmp_path / "metadata.json").write_text('{"final_sigma": 0.0125}')

    sigma0, source = load_parent_final_sigma(warm_start_path)

    assert sigma0 == 0.0125
    assert source.name == "metadata.json"


def test_parent_final_sigma_falls_back_to_old_history(tmp_path) -> None:
    warm_start_path = tmp_path / "best_weights.npy"
    np.save(warm_start_path, np.zeros(4))
    (tmp_path / "history.csv").write_text(
        "generation,sigma\n1,0.1\n2,0.003\n", encoding="utf-8"
    )

    sigma0, source = load_parent_final_sigma(warm_start_path)

    assert sigma0 == 0.003
    assert source.name == "history.csv"


def test_sigma0_modes_are_explicit_and_recordable(tmp_path) -> None:
    parent_weights = tmp_path / "best_weights.npy"
    np.save(parent_weights, np.zeros(4))
    (tmp_path / "summary.json").write_text('{"final_sigma": 0.004}')

    sqrt_sigma, sqrt_source = select_sigma0(
        "sqrt_loss",
        initial_fidelity=0.995,
        fixed_sigma0=None,
        warm_start_path=None,
    )
    fixed_sigma, fixed_source = select_sigma0(
        "fixed",
        initial_fidelity=0.995,
        fixed_sigma0=0.005,
        warm_start_path=None,
    )
    parent_sigma, parent_source = select_sigma0(
        "parent_final",
        initial_fidelity=0.995,
        fixed_sigma0=None,
        warm_start_path=parent_weights,
    )

    assert sqrt_sigma == pytest.approx(0.03)
    assert sqrt_source == "sqrt_loss(initial_center_population)"
    assert fixed_sigma == 0.005
    assert fixed_source == "explicit --sigma0"
    assert parent_sigma == 0.004
    assert "summary.json" in parent_source


def test_runner_warm_start_copies_compatible_weights(tmp_path) -> None:
    source_weights = np.linspace(-0.1, 0.2, TinySimulator.n_controls)
    source_path = tmp_path / "parent_best_weights.npy"
    np.save(source_path, source_weights)
    child_weights = np.zeros_like(source_weights)

    fill_initial_weights(
        initialization="warm_start",
        seed=1234,
        simulator=TinySimulator(),
        times=np.linspace(0.0, DEFAULT_DURATION, 11),
        initial_weights=child_weights,
        warm_start_path=source_path,
    )

    np.testing.assert_array_equal(child_weights, source_weights)


def test_runner_warm_start_rejects_missing_or_incompatible_source(tmp_path) -> None:
    simulator = TinySimulator()
    common_arguments = {
        "initialization": "warm_start",
        "seed": 1234,
        "simulator": simulator,
        "times": np.linspace(0.0, DEFAULT_DURATION, 11),
        "initial_weights": np.zeros(simulator.n_controls),
    }

    with pytest.raises(ValueError, match="--warm-start is required"):
        fill_initial_weights(**common_arguments)

    mismatched_path = tmp_path / "wrong_size.npy"
    np.save(mismatched_path, np.zeros(simulator.n_controls - 1))
    with pytest.raises(ValueError, match="shape does not match"):
        fill_initial_weights(**common_arguments, warm_start_path=mismatched_path)
