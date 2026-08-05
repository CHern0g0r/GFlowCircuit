"""Experiment 4: factorial logZ learning-rate study for the TB GFlowNet."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.tb_logz_calibration import EXPERIMENT_4_RATES, run_experiment as _run_shared


RUN_SCHEMA_VERSION = 3
EXPECTED_INITIALIZATIONS = ("z0", "zcal")
EXPECTED_RATES = EXPERIMENT_4_RATES


def run_experiment(args: argparse.Namespace) -> int:
    """Run one initialization/rate/circuit/seed cell through the shared engine."""
    args.variant = args.initialization
    args.experiment_name = "log_z_learning_rate"
    args.run_schema_version = RUN_SCHEMA_VERSION
    return _run_shared(args)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="run one Experiment 4 cell")
    parser.add_argument("--initialization", choices=EXPECTED_INITIALIZATIONS, required=True)
    parser.add_argument("--log-z-learning-rate", type=float, choices=EXPECTED_RATES, required=True)
    parser.add_argument("--config-name", default="tb_zhuDOP")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=800)
    parser.add_argument("--schedule-trajectories", type=int, default=800)
    parser.add_argument("--milestones", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.set_defaults(handler=run_experiment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    from src.experiments.tb_logz_learning_rate_report import add_report_parsers
    add_report_parsers(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_INITIALIZATIONS",
    "EXPECTED_RATES",
    "RUN_SCHEMA_VERSION",
    "build_parser",
    "main",
    "run_experiment",
]
