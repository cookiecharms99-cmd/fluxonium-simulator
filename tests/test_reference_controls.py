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

    controls = simulator.controls(
        flux_weights, times, control_protocol="reference"
    )

    expected_charge = (params["physics"]["Omega"] / params["physics"]["n_ef"]) * np.sin(
        params["physics"]["omega_ef"] * times
    )

    np.testing.assert_allclose(controls.charge_drive, expected_charge)
    np.testing.assert_allclose(
        controls.external_phase,
        simulator.idle_phase,
        atol=1e-12,
    )


def test_reference_pathway_populations_are_labelled_and_start_in_idle_mixture() -> None:
    times = np.linspace(0.0, 1.0, 11)
    flux_weights = np.zeros(simulator.n_flux_controls)

    populations = simulator.pathway_populations(
        flux_weights, times, control_protocol="reference"
    )

    assert tuple(populations) == ("g0", "e0", "f0", "g1")
    assert all(curve.shape == times.shape for curve in populations.values())
    assert all(np.all(np.isfinite(curve)) for curve in populations.values())
    np.testing.assert_allclose(populations["g0"][0], 0.5, atol=1e-12)
    np.testing.assert_allclose(populations["e0"][0], 0.5, atol=1e-12)
    np.testing.assert_allclose(populations["f0"][0], 0.0, atol=1e-12)
    np.testing.assert_allclose(populations["g1"][0], 0.0, atol=1e-12)
