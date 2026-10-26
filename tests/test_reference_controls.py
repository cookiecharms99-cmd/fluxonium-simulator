"""Checks if the reference controls are made properly."""

import json
from pathlib import Path

import numpy as np

from source.simulator import FluxoniumResetSimulator

duration = 20.0

project_root = Path(__file__).resolve().parents[1]
params_path = project_root / "config" / "reference_prof.json"

with open(params_path, "r") as file:
    params = json.load(file)
simulator = FluxoniumResetSimulator(params)


def test_reference_controls_use_fixed_charge_drive() -> None:
    times = np.linspace(0.0, duration, 101)
    flux_weights = np.zeros(simulator.n_flux_controls)

    controls = simulator.reference_controls(flux_weights, times)

    expected_charge = (params["physics"]["Omega"] / params["physics"]["n_ef"]) * np.sin(
        params["physics"]["omega_ef"] * times
    )

    np.testing.assert_allclose(controls.charge_drive, expected_charge)
    np.testing.assert_allclose(
        controls.external_phase,
        simulator.idle_phase,
        atol=1e-12,
    )
