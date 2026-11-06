"""Checks for independent flux and full-Fourier charge-control bases."""

import json
from pathlib import Path

import numpy as np
import pytest

from source.simulator import FluxoniumResetSimulator, charge_fourier_basis

duration = 20.0

project_root = Path(__file__).resolve().parents[1]
params_path = project_root / "config" / "mini_audit.json"

with open(params_path, "r") as file:
    params = json.load(file)
simulator = FluxoniumResetSimulator(params)


def test_first_charge_coefficient_is_the_unconstrained_constant_envelope() -> None:
    time_arr = np.linspace(0.0, duration, params["model_settings"]["num_t_steps"])
    test_weights = np.zeros(simulator.n_controls)
    test_weights[simulator.n_flux_controls] = 1
    # x =  [00000...100000...]
    controls = simulator.controls(test_weights, time_arr)

    flux_pulse = controls.external_phase
    expected_pulse = simulator.idle_phase * np.ones_like(time_arr)
    np.testing.assert_allclose(flux_pulse, expected_pulse)
    drive = controls.charge_drive
    expected_drive = np.sin(float(params["physics"]["omega_ef"]) * time_arr)
    np.testing.assert_allclose(drive, expected_drive)

    # Flux returns to its idle phase, while charge has no matching constraint.
    np.testing.assert_allclose(
        controls.external_phase[[0, -1]],
        simulator.idle_phase,
        rtol=0.0,
        atol=1e-12,
    )
    assert controls.charge_drive[-1] != pytest.approx(0.0)


def test_charge_fourier_basis_alternates_sine_and_cosine() -> None:
    time_arr = np.linspace(0.0, duration, 101)
    basis = charge_fourier_basis(time_arr, count=5)

    np.testing.assert_allclose(basis[0], 1.0)
    np.testing.assert_allclose(basis[1], np.sin(2.0 * np.pi * time_arr / duration))
    np.testing.assert_allclose(basis[2], np.cos(2.0 * np.pi * time_arr / duration))
    np.testing.assert_allclose(basis[3], np.sin(4.0 * np.pi * time_arr / duration))
    np.testing.assert_allclose(basis[4], np.cos(4.0 * np.pi * time_arr / duration))


def test_basis_cache_refreshes_for_a_new_duration() -> None:
    """A single simulator may safely be reused during duration continuation."""
    weights = np.zeros(simulator.n_controls)
    weights[simulator.n_flux_controls] = 1.0

    simulator.controls(
        weights,
        np.linspace(0.0, 20.0, params["model_settings"]["num_t_steps"]),
    )
    short_times = np.linspace(0.0, 10.0, params["model_settings"]["num_t_steps"])
    short_controls = simulator.controls(weights, short_times)

    expected_drive = np.sin(float(params["physics"]["omega_ef"]) * short_times)
    np.testing.assert_allclose(short_controls.charge_drive, expected_drive)


@pytest.mark.parametrize(
    "times",
    [
        np.array([1.0, 2.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, np.nan]),
    ],
)
def test_controls_reject_an_ambiguous_time_grid(times: np.ndarray) -> None:
    """A pulse is defined in its own protocol clock, from 0 ns onward."""
    with pytest.raises(ValueError):
        simulator.controls(np.zeros(simulator.n_controls), times)
