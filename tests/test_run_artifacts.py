"""Fast checks for the on-disk CMA-ES run record helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from source.run_artifacts import (
    create_run_directory,
    save_controls,
    save_history,
    sha256_file,
)
from source.simulator import ControlWaveforms


def test_run_artifacts_are_saved_without_overwriting(tmp_path) -> None:
    run_directory = create_run_directory(tmp_path, "one_run")
    history = [
        {
            "generation": 1,
            "best_reward_ever": 0.8,
            "best_reward_generation": 0.8,
            "mean_reward": 0.7,
            "std_reward": 0.1,
            "sigma": 0.2,
            "population_size": 4,
            "evaluation_count": 4,
            "elapsed_seconds": 0.5,
        }
    ]
    save_history(run_directory / "history.csv", history)
    save_controls(
        run_directory / "controls.npz",
        np.array([0.0, 1.0]),
        ControlWaveforms(
            external_phase=np.array([np.pi, np.pi]),
            charge_drive=np.array([0.0, 0.0]),
        ),
    )

    assert "best_reward_ever" in (run_directory / "history.csv").read_text()
    with np.load(run_directory / "controls.npz") as controls:
        np.testing.assert_allclose(controls["external_phase"], [np.pi, np.pi])
    assert len(sha256_file(run_directory / "controls.npz")) == 64

    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "one_run")
