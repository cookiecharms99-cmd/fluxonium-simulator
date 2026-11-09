"""Checks for independent flux and full-Fourier charge-control bases."""

import json
import copy
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


def test_runnable_configs_use_explicit_control_blocks() -> None:
    """A new input may not silently fall back to the retired flat schema."""
    for name in (
        "mini_audit.json",
        "reference_prof.json",
        "joint_exploratory.json",
        "joint_run.json",
        "charge_only_test.json",
        "charge_only_idle_150ns.json",
        "charge_only_random_flux_150ns.json",
        "charge_only_good_flux_150ns.json",
    ):
        with (project_root / "config" / name).open(encoding="utf-8") as file:
            settings = json.load(file)["model_settings"]
        assert set(settings["controls"]) == {"flux", "charge"}
        assert "n_flux" not in settings
        assert "n_charge" not in settings


def test_simulator_rejects_the_retired_flat_control_schema() -> None:
    legacy_params = copy.deepcopy(params)
    legacy_params["model_settings"].pop("controls")
    legacy_params["model_settings"]["n_flux"] = 40
    legacy_params["model_settings"]["n_charge"] = 0

    with pytest.raises(ValueError, match="model_settings.controls"):
        FluxoniumResetSimulator(legacy_params)


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
    expected_drive = (
        params["physics"]["Omega"] / params["physics"]["n_ef"] + 1.0
    ) * np.sin(float(params["physics"]["omega_ef"]) * time_arr)
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

    expected_drive = (
        params["physics"]["Omega"] / params["physics"]["n_ef"] + 1.0
    ) * np.sin(float(params["physics"]["omega_ef"]) * short_times)
    np.testing.assert_allclose(short_controls.charge_drive, expected_drive)


def test_config_can_select_charge_only_controls_with_idle_flux() -> None:
    """A zero-size flux block means a declared fixed flux waveform, not an error."""
    charge_only_params = copy.deepcopy(params)
    charge_only_params["model_settings"]["controls"]["flux"]["coefficient_count"] = 0
    charge_only_params["model_settings"]["controls"]["charge"]["coefficient_count"] = 2
    charge_only_simulator = FluxoniumResetSimulator(charge_only_params)
    times = np.linspace(0.0, duration, 101)

    controls = charge_only_simulator.controls(np.array([0.3, -0.1]), times)
    expected_envelope = (
        params["physics"]["Omega"] / params["physics"]["n_ef"]
        + 0.3
        - 0.1 * np.sin(2.0 * np.pi * times / duration)
    )
    np.testing.assert_allclose(controls.external_phase, charge_only_simulator.idle_phase)
    np.testing.assert_allclose(
        controls.charge_drive,
        expected_envelope * np.sin(params["physics"]["omega_ef"] * times),
    )


def test_random_sine_fixed_flux_is_seeded_and_deterministic() -> None:
    random_flux_params = copy.deepcopy(params)
    random_flux_params["model_settings"]["controls"]["flux"] = {
        "coefficient_count": 0,
        "fixed_default": "random_sine",
        "fixed_harmonic_count": 4,
        "random_seed": 1234,
        "random_scale": 0.1,
    }
    random_flux_params["model_settings"]["controls"]["charge"]["coefficient_count"] = 0
    random_flux_simulator = FluxoniumResetSimulator(random_flux_params)
    times = np.linspace(0.0, duration, 101)

    first = random_flux_simulator.controls(np.empty(0), times).external_phase
    second = random_flux_simulator.controls(np.empty(0), times).external_phase

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[[0, -1]], random_flux_simulator.idle_phase)
    assert not np.allclose(first, random_flux_simulator.idle_phase)


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
