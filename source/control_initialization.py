import numpy as np


def piecewise_pulse_init(
    times: np.ndarray,
    duration: float,
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
    stop1 = stop1_pct * duration
    stop2 = stop2_pct * duration
    pulse = np.empty_like(times)
    mask1 = times < stop1
    mask2 = (times >= stop1) & (times < stop2)
    mask3 = times >= stop2

    # go down to resonance, then hover, then head back to idle
    pulse[mask1] = idle_flux * (1 - 0.5 * times[mask1] / stop1)
    pulse[mask2] = idle_flux * 0.5
    pulse[mask3] = idle_flux / 2 * (1 + (times[mask3] - stop2) / (duration - stop2))
    return pulse


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    times = np.linspace(0, 150, 500)
    pulse = piecewise_pulse_init(times, duration=150)
    plt.plot(times, pulse)
    plt.show()
