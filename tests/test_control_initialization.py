"""Checks for physically meaningful and numerically sound pulse initialization."""

import numpy as np

from source.control_initialization import (
    lst_sq_projection_flux,
    piecewise_pulse_init,
)
from source.simulator import sine_basis


def test_legacy_piecewise_pulse_has_expected_150_ns_schedule() -> None:
    """The professor-inspired pulse ramps, holds at quarter flux, then returns."""
    times = np.array([0.0, 5.0, 10.0, 50.0, 110.0, 130.0, 150.0])

    pulse = piecewise_pulse_init(times)

    np.testing.assert_allclose(
        pulse,
        [np.pi, 3 * np.pi / 4, np.pi / 2, np.pi / 2, np.pi / 2, 3 * np.pi / 4, np.pi],
        atol=1e-12,
    )


def test_least_squares_projection_recovers_known_sine_weights() -> None:
    """Projection exactly recovers a target phase already in the chosen basis."""
    times = np.linspace(0.0, 150.0, 501)
    basis = sine_basis(times, count=4)
    known_weights = np.array([0.2, -0.1, 0.05, 0.03])
    target_external_phase = np.pi + known_weights @ basis

    recovered_weights = lst_sq_projection_flux(target_external_phase, basis)

    np.testing.assert_allclose(recovered_weights, known_weights, atol=1e-12)
