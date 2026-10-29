"""
Here we manage the initialization of control vectors for starting.
For now, only a fresh start. Perhaps a more artifact-y util handles 'warm starts'.
"""

import numpy as np


def piecewise_pulse_init(
    times: np.ndarray,
    stop1_pct: float = 10 / 150.0,
    stop2_pct: float = 110 / 150.0,
    idle_flux=np.pi,
):
    """
    Builds a starting point for initial flux pulse!
    Piecewise curve. Starts at idle flux, then heads down near resonance (sorta), then hovers for a while, then heads back up to idle
    """
    if (
        (stop1_pct > stop2_pct)
        or (stop1_pct >= 1)
        or (stop1_pct <= 0)
        or (stop2_pct >= 1)
        or (stop2_pct <= 0)
    ):
        raise ValueError("Stops must be in order and must be in (0,1) :3")

    duration = times[-1]
    stop1 = stop1_pct * duration
    stop2 = stop2_pct * duration
    pulse = np.empty_like(times, dtype=float)
    mask1 = times < stop1
    mask2 = (times >= stop1) & (times < stop2)
    mask3 = times >= stop2

    # go down to resonance, then hover, then head back to idle
    pulse[mask1] = idle_flux * (1 - 0.5 * times[mask1] / stop1)
    pulse[mask2] = idle_flux * 0.5
    pulse[mask3] = idle_flux / 2 * (1 + (times[mask3] - stop2) / (duration - stop2))
    return pulse


def lst_sq_projection_flux(pulse: np.ndarray, basis_arr: np.ndarray, idle_phase=np.pi):
    """
    Compute the best fit weight vector for a given flux pulse using least squares.
    Outputs the flux component vector, not the entire weights vector!
    Ex usage:
    ```

    flux_weights = lst_sq_projection(flux_pulse, sim._flux_basis)
    weights[:sim.n_flux_controls] = flux_weights
    ```

    """

    # x * basis + idle = pulse
    # basis = n,t
    # x = 1,n
    # pulse = 1,t?
    pulse_shifted = pulse - idle_phase
    # it says np lstsq solves Ax = b, but i have xB = p, solving for x.
    # I can maybe do umm... solving B.T x.T = p.T -> x,_,_,_ = solve(B.T, p.T), ret x.T? Sure!
    weights_guess, _, _, _ = np.linalg.lstsq(basis_arr.T, pulse_shifted, rcond=None)
    return weights_guess


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simulator import sine_basis

    times = np.linspace(0, 150, 500)
    duration = times[-1]
    pulse = piecewise_pulse_init(times)
    idle_phase = np.pi

    # Now we will see how well least squares reconstructs pulse, using a blunt verison to make things visible!
    flux_basis_blunt = sine_basis(times, count=7)
    flux_basis_fine = sine_basis(times, count=40)
    weights_blunt = lst_sq_projection_flux(pulse, flux_basis_blunt)
    weights_fine = lst_sq_projection_flux(pulse, flux_basis_fine)
    approx_pulse_blunt = idle_phase + weights_blunt @ flux_basis_blunt
    approx_pulse_fine = idle_phase + weights_fine @ flux_basis_fine
    plt.title("Least Squares Projection onto initial pulse array (n = 7,40)")
    plt.xlabel("time")
    plt.ylabel("phase")
    plt.plot(times, pulse)
    plt.plot(times, approx_pulse_blunt)
    plt.plot(times, approx_pulse_fine)
    plt.show()
