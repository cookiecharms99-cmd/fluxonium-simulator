"""Checks for independent flux and charge-control sine bases."""

import json
from pathlib import Path

import numpy as np
from source.simulator import FluxoniumResetSimulator

duration = 20.0

project_root = Path(__file__).resolve().parents[1]
params_path = project_root / "config" / "mini_audit.json"

with open(params_path, "r") as file:
    params = json.load(file)
simulator = FluxoniumResetSimulator(params)


def test_first_charge_coefficient_uses_the_first_charge_harmonic() -> None:
    time_arr = np.linspace(0.0, duration, params["model_settings"]["num_t_steps"])
    test_weights = np.zeros(simulator.n_controls)
    test_weights[simulator.n_flux_controls] = 1
    # x =  [00000...100000...]
    controls = simulator.controls(test_weights, time_arr)

    flux_pulse = controls.external_phase
    expected_pulse = simulator.idle_phase * np.ones_like(time_arr)
    np.testing.assert_allclose(flux_pulse, expected_pulse)
    drive = controls.charge_drive
    expected_drive = np.sin(np.pi * time_arr / duration) * np.sin(
        float(params["physics"]["omega_ef"]) * time_arr
    )
    np.testing.assert_allclose(drive, expected_drive)
    # They should clamp at endpoints as we desire
    np.testing.assert_allclose(
        controls.external_phase[[0, -1]],
        simulator.idle_phase,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        controls.charge_drive[[0, -1]],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_basis_cache_refreshes_for_a_new_duration() -> None:
    """A single simulator may safely be reused during duration continuation."""
    weights = np.zeros(simulator.n_controls)
    weights[simulator.n_flux_controls] = 1.0

    simulator.controls(
        weights,
        np.linspace(0.0, 20.0, params["model_settings"]["num_t_steps"]),
    )
    short_times = np.linspace(
        0.0, 10.0, params["model_settings"]["num_t_steps"]
    )
    short_controls = simulator.controls(weights, short_times)

    expected_drive = np.sin(np.pi * short_times / 10.0) * np.sin(
        float(params["physics"]["omega_ef"]) * short_times
    )
    np.testing.assert_allclose(short_controls.charge_drive, expected_drive)
