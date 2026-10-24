"""Small, filesystem-only helpers for reproducible optimization run records."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from source.simulator import ControlWaveforms


def utc_timestamp() -> str:
    """Return a sortable UTC timestamp suitable for a directory name."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def create_run_directory(
    project_root: Path, requested_path: str | None, model="cmaes"
) -> Path:
    """Create one empty run directory, refusing to overwrite an existing run."""
    if requested_path is None:
        run_directory = project_root / "data" / "output" / model / f"{utc_timestamp()}"
        suffix = 1
        while run_directory.exists():
            run_directory = (
                project_root
                / "data"
                / "output"
                / model
                / (f"{utc_timestamp()}_{suffix}")
            )
            suffix += 1
    else:
        run_directory = Path(requested_path)
        if not run_directory.is_absolute():
            run_directory = project_root / run_directory

    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def sha256_file(path: Path) -> str:
    """Return a content fingerprint without loading the full artifact at once."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write human-readable JSON with stable key order."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def git_revision(project_root: Path) -> str | None:
    """Return the checked-out revision when Git metadata is available."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def environment_snapshot(project_root: Path) -> dict[str, str | None]:
    """Capture the small amount of execution context needed for a rerun."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": git_revision(project_root),
    }


def save_history(path: Path, history: list[dict[str, float | int]]) -> None:
    """Write one transparent CSV row per completed CMA-ES generation."""
    if not history:
        raise ValueError("Cannot save an empty optimization history.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_controls(path: Path, times: np.ndarray, controls: ControlWaveforms) -> None:
    """Store the sampled waveforms alongside their time coordinate."""
    np.savez(
        path,
        times=np.asarray(times, dtype=float),
        external_phase=np.asarray(controls.external_phase, dtype=float),
        charge_drive=np.asarray(controls.charge_drive, dtype=float),
    )
