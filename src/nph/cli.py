from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import simulate_subgroup
from .paths import repo_root, reports_dir, results_dir


def _cmd_simulate(args: argparse.Namespace) -> int:
    sim_argv = [
        "--outdir",
        str(args.outdir),
        "--max-minutes",
        str(args.max_minutes),
        "--seed",
        str(args.seed),
    ]
    if args.n_sims is not None:
        sim_argv.extend(["--n-sims", str(args.n_sims)])
    simulate_subgroup.main(sim_argv)
    return 0


def _cmd_make_report(args: argparse.Namespace) -> int:
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = repo_root() / script_path

    cmd = [sys.executable, str(script_path), *(args.script_args or [])]
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nph", description="CLI for NPH simulations and reports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("simulate", help="Run subgroup simulation scenarios")
    sim.add_argument("--outdir", type=Path, default=results_dir(), help="Output directory (default: repo results/)")
    sim.add_argument("--max-minutes", type=float, default=25.0, help="Runtime budget in minutes")
    sim.add_argument("--n-sims", type=int, default=None, help="Override automatic simulation count")
    sim.add_argument("--seed", type=int, default=20260227, help="Base random seed")
    sim.set_defaults(func=_cmd_simulate)

    report = subparsers.add_parser("make-report", help="Run report generation script")
    report.add_argument(
        "--script",
        default="scripts/make_report.py",
        help="Report script path (relative to repo root unless absolute)",
    )
    report.add_argument("script_args", nargs="*", help="Additional arguments passed to report script")
    report.set_defaults(func=_cmd_make_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def simulate_entrypoint() -> int:
    return main(["simulate", *sys.argv[1:]])


def make_report_entrypoint() -> int:
    return main(["make-report", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
