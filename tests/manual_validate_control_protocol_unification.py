"""One-time equivalence check for removal of the former reference methods.

Run directly with ``uv run python tests/manual_validate_control_protocol_unification.py``.
It is deliberately not named ``test_*.py``: this records the migration check,
not a second long-lived production test path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.simulator import ControlWaveforms, FluxoniumResetSimulator, sine_basis


def main() -> None:
    with (ROOT / "config" / "reference_prof.json").open(encoding="utf-8") as file:
        params = json.load(file)

    simulator = FluxoniumResetSimulator(params)
    times = np.linspace(0.0, 1.0, 11)
    flux_weights = np.linspace(-0.08, 0.08, simulator.n_flux_controls)

    # Independent transcription of the deleted reference_controls formula.
    legacy_waveform = ControlWaveforms(
        external_phase=simulator.idle_phase
        + flux_weights @ sine_basis(times, simulator.n_flux_controls),
        charge_drive=(params["physics"]["Omega"] / params["physics"]["n_ef"])
        * np.sin(params["physics"]["omega_ef"] * times),
    )
    unified_waveform = simulator.controls(flux_weights, times)
    np.testing.assert_allclose(
        unified_waveform.external_phase, legacy_waveform.external_phase
    )
    np.testing.assert_allclose(
        unified_waveform.charge_drive, legacy_waveform.charge_drive
    )

    legacy_result = simulator._evolve_waveform(
        legacy_waveform, times, e_ops=[simulator.ground_projector]
    )
    unified_population = simulator.final_ground_population(
        flux_weights,
        times,
    )
    np.testing.assert_allclose(
        unified_population,
        np.real(legacy_result.expect[0][-1]),
        rtol=1e-10,
        atol=1e-12,
    )
    print("Unified reference pathway exactly matches the deleted formulation.")


if __name__ == "__main__":
    main()
