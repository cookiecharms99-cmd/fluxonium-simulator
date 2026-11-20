"""Analyze convergence dynamics and milestones for one or more CMA-ES runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main.plot_run import load_history, resolve_run_directory  # noqa: E402
from source.analysis.run_analysis import (  # noqa: E402
    analyze_history,
    format_convergence_summary,
)


def analyze_run(run_path: Path) -> None:
    """Analyze and print convergence summary for a given run folder."""
    history_path = run_path / "history.csv"
    if not history_path.is_file():
        print(f"No history.csv found in {run_path}")
        return

    history = load_history(history_path)
    summary = analyze_history(history)
    if summary is None:
        print(f"Could not parse valid history from {history_path}")
        return

    print(f"Run: {run_path.name}")
    print(format_convergence_summary(summary))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze convergence dynamics and infidelity decay for CMA-ES runs."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="One or more run paths or directory names under data/output/cmaes/.",
    )
    args = parser.parse_args()

    for idx, run_name in enumerate(args.runs):
        if idx > 0:
            print("\n" + "=" * 50 + "\n")
        run_dir = resolve_run_directory(run_name)
        analyze_run(run_dir)


if __name__ == "__main__":
    main()
