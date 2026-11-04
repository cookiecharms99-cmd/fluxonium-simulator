"""Friendly front door for the project's common research workflows.

The original scripts remain available for advanced use. This command provides
short names, safe run lookup, readable automatic output labels, and a compact
map of the supported workflows.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = PROJECT_ROOT / "data" / "output" / "cmaes"


def resolve_run_selector(value: str) -> Path:
    """Resolve an exact path/name or one unique case-insensitive name fragment."""
    requested = Path(value)
    for candidate in (requested, PROJECT_ROOT / requested, RUNS_DIRECTORY / requested):
        if candidate.is_dir():
            return candidate.resolve()

    matches = sorted(
        path for path in RUNS_DIRECTORY.iterdir() if path.is_dir() and value.lower() in path.name.lower()
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"No run matches {value!r}. Use `uv run python main/flux.py list-runs` "
            "to see available artifact names."
        )
    options = "\n".join(f"  - {path.name}" for path in matches)
    raise ValueError(f"Run selector {value!r} is ambiguous. Choose one of:\n{options}")


def execute(command: list[str], *, dry_run: bool) -> None:
    """Show the underlying trusted command, then run it unless asked not to."""
    print("Underlying command:")
    print("  " + shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def command_overview() -> str:
    return """Common workflows

  commands                 Show this guide.
  list-runs [--match TEXT] List saved CMA-ES artifacts.
  run                      Launch a named CMA-ES run with friendly defaults.
  audit --run TEXT         Check one saved pulse across numerical cutoffs.
  plot --run TEXT          Inspect reward history and saved controls.
  benchmark --run TEXT     Separate simulator startup time from solve time.

Every --run value accepts an exact path, an exact output-directory name, or a
unique fragment such as `warm_start_4779342`. The command refuses ambiguous
fragments instead of guessing.

For the full advanced flags of any underlying tool, use its normal --help.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("commands", help="Print a compact workflow guide.")

    list_runs = subparsers.add_parser("list-runs", help="List saved CMA-ES artifacts.")
    list_runs.add_argument("--match", help="Only show names containing this text.")

    run = subparsers.add_parser("run", help="Launch CMA-ES with friendly defaults.")
    run.add_argument("--protocol", choices=("reference", "joint"), default="reference")
    run.add_argument(
        "--init",
        choices=("zero", "random", "piecewise_lstsq", "warm_start"),
        default="zero",
        help="CMA-ES center; warm_start additionally requires --warm-start.",
    )
    run.add_argument("--warm-start", help="Compatible saved .npy coefficient vector.")
    run.add_argument("--duration", type=float, default=150.0, help="Protocol duration in ns.")
    run.add_argument("--generations", type=int, default=5)
    run.add_argument("--popsize", type=int, default=4)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--sigma0", type=float, default=0.3)
    run.add_argument("--full-physics", action="store_true")
    run.add_argument("--label", help="Optional readable output-label prefix.")
    run.add_argument("--out-dir", help="Explicit output directory; overrides automatic naming.")
    run.add_argument("--dry-run", action="store_true", help="Show, but do not run, the command.")

    audit = subparsers.add_parser("audit", help="Audit one saved pulse across cutoffs.")
    audit.add_argument("--run", required=True)
    audit.add_argument("--cutoffs", type=int, nargs="+")
    audit.add_argument("--output-dir", default="local_scratch")
    audit.add_argument("--dry-run", action="store_true")

    plot = subparsers.add_parser("plot", help="Plot one saved run.")
    plot.add_argument("--run", required=True)
    plot.add_argument("--save", type=Path)
    plot.add_argument("--no-show", action="store_true")
    plot.add_argument("--dry-run", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark one saved run.")
    benchmark.add_argument("--run", required=True)
    benchmark.add_argument("--repetitions", type=int, default=3)
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--simulator-revision")
    benchmark.add_argument("--output-dir", default="local_scratch")
    benchmark.add_argument("--dry-run", action="store_true")
    return parser


def list_runs(match: str | None) -> None:
    if not RUNS_DIRECTORY.is_dir():
        print("No data/output/cmaes directory exists yet.")
        return
    names = sorted(path.name for path in RUNS_DIRECTORY.iterdir() if path.is_dir())
    if match is not None:
        names = [name for name in names if match.lower() in name.lower()]
    if not names:
        print("No matching run artifacts.")
        return
    print("Saved runs:")
    for name in names:
        print(f"  {name}")


def run_campaign(args: argparse.Namespace) -> None:
    if args.init == "warm_start" and args.warm_start is None:
        raise ValueError("--warm-start is required when --init warm_start is selected.")
    if args.init != "warm_start" and args.warm_start is not None:
        raise ValueError("--warm-start is only valid with --init warm_start.")

    physics_config = (
        "config/reference_prof.json"
        if args.protocol == "reference"
        else "config/joint_exploratory.json"
    )
    tier = "full" if args.full_physics else "smoke"
    automatic_label = f"{args.protocol}-{args.duration:g}ns-{args.init}-{tier}"
    command = [
        sys.executable,
        "main/run_cmaes.py",
        "--physics-config",
        physics_config,
        "--control-protocol",
        args.protocol,
        "--initialization",
        args.init,
        "--duration",
        str(args.duration),
        "--generations",
        str(args.generations),
        "--popsize",
        str(args.popsize),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--sigma0",
        str(args.sigma0),
    ]
    if args.warm_start is not None:
        command.extend(("--warm-start", args.warm_start))
    if args.full_physics:
        command.append("--full-physics")
    if args.out_dir is not None:
        command.extend(("--out-dir", args.out_dir))
    else:
        command.extend(("--label", args.label or automatic_label))
    execute(command, dry_run=args.dry_run)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "commands":
        print(command_overview())
    elif args.command == "list-runs":
        list_runs(args.match)
    elif args.command == "run":
        run_campaign(args)
    elif args.command == "audit":
        command = [
            sys.executable,
            "source/audit_controls.py",
            "--run",
            str(resolve_run_selector(args.run)),
            "--output-dir",
            args.output_dir,
        ]
        if args.cutoffs is not None:
            command.extend(("--cutoffs", *(str(cutoff) for cutoff in args.cutoffs)))
        execute(command, dry_run=args.dry_run)
    elif args.command == "plot":
        command = [sys.executable, "main/plot_run.py", str(resolve_run_selector(args.run))]
        if args.save is not None:
            command.extend(("--save", str(args.save)))
        if args.no_show:
            command.append("--no-show")
        execute(command, dry_run=args.dry_run)
    elif args.command == "benchmark":
        command = [
            sys.executable,
            "main/benchmark_simulator.py",
            str(resolve_run_selector(args.run)),
            "--repetitions",
            str(args.repetitions),
            "--warmups",
            str(args.warmups),
            "--output-dir",
            args.output_dir,
        ]
        if args.simulator_revision is not None:
            command.extend(("--simulator-revision", args.simulator_revision))
        execute(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
