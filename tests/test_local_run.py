"""Reproduce the saved result from ``cmaes_10min_local_20261023T213448Z``.

``controls.npz`` contains the sampled external-phase and charge-drive
waveforms, rather than the CMA-ES coefficient vector accepted by the
simulator.  This test therefore evaluates ``best_weights.npy`` and also
checks that those weights regenerate the waveforms in ``controls.npz``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from source.simulator import FluxoniumResetSimulator

pytestmark = pytest.mark.skip(
    reason=(
        "This archived run used the old charge basis k=41..60; it cannot be "
        "reproduced by the corrected independent charge basis."
    )
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = PROJECT_ROOT / "data/output/cmaes/10min_local_20261023T213448Z"


def test_local_cmaes_result_reproduces_recorded_best_reward() -> None:
    """Saved coefficients must reproduce the best P(g, 0) in the run history."""
    with (RUN_DIRECTORY / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    with (RUN_DIRECTORY / "history.csv").open(newline="", encoding="utf-8") as file:
        history = list(csv.DictReader(file))

    assert history, "The run record must include at least one completed generation."
    recorded_best_reward = float(history[-1]["best_reward_ever"])

    with np.load(RUN_DIRECTORY / "controls.npz") as saved_controls:
        times = saved_controls["times"]
        saved_external_phase = saved_controls["external_phase"]
        saved_charge_drive = saved_controls["charge_drive"]
    best_weights = np.load(RUN_DIRECTORY / "best_weights.npy")

    simulator = FluxoniumResetSimulator(metadata["physics_config"])
    regenerated_controls = simulator.controls(best_weights, times)
    np.testing.assert_allclose(
        regenerated_controls.external_phase, saved_external_phase
    )
    np.testing.assert_allclose(regenerated_controls.charge_drive, saved_charge_drive)

    reproduced_reward = simulator.final_ground_population(best_weights, times)
    assert reproduced_reward == pytest.approx(
        recorded_best_reward, rel=1e-10, abs=1e-12
    )
