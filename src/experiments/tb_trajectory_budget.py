"""Experiment 5: select a common TB trajectory budget on bc0 and dalu."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.experiments.tb_logz_calibration import run_experiment as _run_shared


RUN_SCHEMA_VERSION = 4
EXPERIMENT_NAME = "trajectory_budget_selection"
INITIALIZATION = "zcal"
LOG_Z_LEARNING_RATE = 0.01
MAX_TRAJECTORIES = 6_400
SCHEDULE_TRAJECTORIES = 800
MILESTONES = (200, 400, 800, 1_600, 3_200, 6_400)
EXPECTED_CIRCUITS = ("bc0", "dalu")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)


def _validate_experiment5_args(args: argparse.Namespace) -> None:
    if getattr(args, "_allow_test_budget", False):
        return
    expected = {
        "max_trajectories": MAX_TRAJECTORIES,
        "schedule_trajectories": SCHEDULE_TRAJECTORIES,
        "milestones": list(MILESTONES),
    }
    actual = {
        "max_trajectories": int(args.max_trajectories),
        "schedule_trajectories": int(args.schedule_trajectories),
        "milestones": [int(value) for value in args.milestones],
    }
    if actual != expected:
        raise ValueError(f"Experiment 5 scientific budget is fixed: expected {expected}, got {actual}")


def run_experiment(args: argparse.Namespace) -> int:
    """Run one circuit/seed continuously through all trajectory milestones."""
    _validate_experiment5_args(args)
    args.variant = INITIALIZATION
    args.log_z_learning_rate = LOG_Z_LEARNING_RATE
    args.experiment_name = EXPERIMENT_NAME
    args.run_schema_version = RUN_SCHEMA_VERSION
    return _run_shared(args)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run", help="run one Experiment 5 circuit/seed")
    parser.add_argument("--config-name", default="tb_zhuDOP")
    parser.add_argument("--circuit", choices=EXPECTED_CIRCUITS, required=True)
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=MAX_TRAJECTORIES)
    parser.add_argument("--schedule-trajectories", type=int, default=SCHEDULE_TRAJECTORIES)
    parser.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.set_defaults(handler=run_experiment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    from src.experiments.tb_trajectory_budget_report import add_report_parser
    add_report_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
