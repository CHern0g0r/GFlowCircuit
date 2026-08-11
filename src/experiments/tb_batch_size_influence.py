"""Experiment 5.5: measure TB batch-size influence at fixed trajectory budget."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from src.experiments.tb_logz_calibration import run_experiment as _run_shared


RUN_SCHEMA_VERSION = 5
EXPERIMENT_NAME = "batch_size_influence"
INITIALIZATION = "zcal"
LOG_Z_LEARNING_RATE = 0.01
BATCH_SIZES = (1, 4, 8, 16, 32)
EXPECTED_CIRCUITS = ("bc0", "dalu")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
MAX_TRAJECTORIES = 1_600
SCHEDULE_TRAJECTORIES = 800
MILESTONES = (256, 512, 1_024, 1_600)
PREFLIGHT_TRAJECTORIES = 128
PREFLIGHT_MILESTONES = (128,)


def _configure(args: argparse.Namespace, *, preflight: bool) -> None:
    if int(args.batch_size) not in BATCH_SIZES:
        raise ValueError(f"batch size must be one of {BATCH_SIZES}")
    if args.circuit not in EXPECTED_CIRCUITS:
        raise ValueError(f"circuit must be one of {EXPECTED_CIRCUITS}")
    allow_test_budget = bool(getattr(args, "_allow_test_budget", False))
    if allow_test_budget:
        if int(args.max_trajectories) < 64 or int(args.max_trajectories) % int(args.batch_size):
            raise ValueError("test budget must be at least 64 and divisible by batch size")
        args.milestones = [int(value) for value in args.milestones]
    elif preflight:
        args.seed = 0
        args.max_trajectories = PREFLIGHT_TRAJECTORIES
        args.milestones = list(PREFLIGHT_MILESTONES)
    else:
        if int(args.seed) not in EXPECTED_SEEDS:
            raise ValueError(f"seed must be one of {EXPECTED_SEEDS}")
        args.max_trajectories = MAX_TRAJECTORIES
        args.milestones = list(MILESTONES)
    args.schedule_trajectories = SCHEDULE_TRAJECTORIES
    args.variant = INITIALIZATION
    args.log_z_learning_rate = LOG_Z_LEARNING_RATE
    args.experiment_name = EXPERIMENT_NAME
    args.run_schema_version = RUN_SCHEMA_VERSION


def run_experiment(args: argparse.Namespace) -> int:
    """Run one fixed scientific batch/circuit/seed cell."""
    _configure(args, preflight=False)
    return _run_shared(args)


def run_preflight(args: argparse.Namespace) -> int:
    """Run one 128-trajectory seed-zero timing and infrastructure cell."""
    _configure(args, preflight=True)
    return _run_shared(args)


def _add_shared_arguments(parser: argparse.ArgumentParser, *, include_seed: bool) -> None:
    parser.add_argument("--batch-size", type=int, choices=BATCH_SIZES, required=True)
    parser.add_argument("--circuit", choices=EXPECTED_CIRCUITS, required=True)
    if include_seed:
        parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument("--config-name", default="tb_zhuDOP")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one fixed Experiment 5.5 cell")
    _add_shared_arguments(run, include_seed=True)
    run.set_defaults(handler=run_experiment)
    preflight = subparsers.add_parser("preflight", help="run one seed-zero 128-trajectory preflight")
    _add_shared_arguments(preflight, include_seed=False)
    preflight.set_defaults(handler=run_preflight)
    from src.experiments.tb_batch_size_influence_report import add_report_parser
    add_report_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"batch-size experiment {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_SIZES", "EXPECTED_CIRCUITS", "EXPECTED_SEEDS", "MAX_TRAJECTORIES",
    "MILESTONES", "RUN_SCHEMA_VERSION", "build_parser", "main", "run_experiment",
    "run_preflight",
]
