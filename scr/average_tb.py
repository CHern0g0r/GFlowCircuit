#!/usr/bin/env python3
"""Aggregate TensorBoard scalars across runs and write summary dashboards.

This script reads one or more experiment TensorBoard directories, aggregates
scalar tags across repeated runs, and writes a new TensorBoard log per
experiment with:

1) median curve + interquartile range (q25..q75) band
2) mean curve + standard error band

The output event files include a Custom Scalars layout so both bands are easy
to open in TensorBoard.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires TensorBoard writer from PyTorch. "
        "Install with: pip install torch tensorboard"
    ) from exc

SCALARS = event_accumulator.SCALARS


@dataclass(frozen=True)
class Experiment:
    name: str
    logdir: Path


def _has_event_files(path: Path) -> bool:
    return any(p.is_file() and p.name.startswith("events.out.tfevents") for p in path.iterdir())


def _collect_run_dirs(logdir: Path) -> list[Path]:
    """Collect run directories from a TensorBoard logdir."""
    logdir = logdir.resolve()
    if not logdir.is_dir():
        raise SystemExit(f"Logdir is not a directory: {logdir}")

    if _has_event_files(logdir):
        return [logdir]

    run_dirs: list[Path] = []
    for child in sorted(logdir.iterdir()):
        if child.is_dir() and child.name.startswith("run_") and _has_event_files(child):
            run_dirs.append(child)

    if run_dirs:
        return run_dirs

    # Be permissive for non-run_* layouts if directory nesting differs.
    fallback = [child for child in sorted(logdir.iterdir()) if child.is_dir() and _has_event_files(child)]
    if fallback:
        return fallback

    raise SystemExit(f"No TensorBoard event files found under: {logdir}")


def _load_run_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars") or []
    series: dict[str, dict[int, float]] = {}
    for tag in tags:
        by_step: dict[int, float] = {}
        # If several points share a step, keep the latest point.
        for event in sorted(ea.Scalars(tag), key=lambda e: e.step):
            by_step[int(event.step)] = float(event.value)
        if by_step:
            series[tag] = by_step
    return series


def _infer_name(logdir: Path) -> str:
    p = logdir.resolve()
    name = p.parent.name if p.name == "tensorboard" else p.name
    return name.replace(":", "-")


def _build_experiments(args: argparse.Namespace) -> list[Experiment]:
    names = args.name or []
    logdirs = [Path(p).resolve() for p in args.logdir]
    if names and len(names) != len(logdirs):
        raise SystemExit("If --name is provided, it must be repeated exactly once per --logdir.")

    experiments: list[Experiment] = []
    seen_names: set[str] = set()
    for idx, logdir in enumerate(logdirs):
        name = names[idx] if names else _infer_name(logdir)
        safe_name = name.replace("/", "_").replace(" ", "_").replace(":", "-")
        if safe_name in seen_names:
            raise SystemExit(f"Duplicate experiment name after normalization: {safe_name}")
        seen_names.add(safe_name)
        experiments.append(Experiment(name=safe_name, logdir=logdir))
    return experiments


def _aggregate_tag(per_run: list[dict[str, dict[int, float]]], tag: str) -> dict[int, dict[str, float]]:
    steps = sorted({step for run in per_run if tag in run for step in run[tag]})
    by_step: dict[int, dict[str, float]] = {}
    for step in steps:
        vals = [run[tag][step] for run in per_run if tag in run and step in run[tag]]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        n = int(arr.size)
        mean = float(fmean(vals))
        std = float(arr.std(ddof=0)) if n > 1 else 0.0
        se = float(std / math.sqrt(n)) if n > 0 else 0.0
        q25, median, q75 = np.quantile(arr, [0.25, 0.5, 0.75]).tolist()
        by_step[step] = {
            "n_runs": float(n),
            "mean": mean,
            "std": std,
            "se": se,
            "mean_minus_se": mean - se,
            "mean_plus_se": mean + se,
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return by_step


def _write_aggregated_experiment(exp: Experiment, output_dir: Path) -> tuple[int, int]:
    run_dirs = _collect_run_dirs(exp.logdir)
    per_run = [_load_run_scalars(rd) for rd in run_dirs]
    all_tags = sorted({tag for run in per_run for tag in run})
    if not all_tags:
        raise SystemExit(f"No scalar tags found for experiment: {exp.name}")

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    written_points = 0

    layout: dict[str, dict[str, list[object]]] = {}
    for tag in all_tags:
        agg = _aggregate_tag(per_run, tag)
        if not agg:
            continue
        prefix = f"agg/{tag}"
        for step, stats in sorted(agg.items()):
            writer.add_scalar(f"{prefix}/n_runs", stats["n_runs"], step)
            writer.add_scalar(f"{prefix}/mean", stats["mean"], step)
            writer.add_scalar(f"{prefix}/std", stats["std"], step)
            writer.add_scalar(f"{prefix}/se", stats["se"], step)
            writer.add_scalar(f"{prefix}/mean_minus_se", stats["mean_minus_se"], step)
            writer.add_scalar(f"{prefix}/mean_plus_se", stats["mean_plus_se"], step)
            writer.add_scalar(f"{prefix}/q25", stats["q25"], step)
            writer.add_scalar(f"{prefix}/median", stats["median"], step)
            writer.add_scalar(f"{prefix}/q75", stats["q75"], step)
            writer.add_scalar(f"{prefix}/min", stats["min"], step)
            writer.add_scalar(f"{prefix}/max", stats["max"], step)
            written_points += 1

        group_name = tag.replace("/", "_")
        layout[group_name] = {
            "median_iqr": ["Margin", [f"{prefix}/median", f"{prefix}/q25", f"{prefix}/q75"]],
            "mean_se": [
                "Margin",
                [f"{prefix}/mean", f"{prefix}/mean_minus_se", f"{prefix}/mean_plus_se"],
            ],
        }

    writer.add_custom_scalars(layout)
    writer.flush()
    writer.close()
    return len(run_dirs), written_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate TensorBoard runs and emit per-tag median/IQR + mean/SE "
            "summary event files."
        )
    )
    parser.add_argument(
        "--logdir",
        action="append",
        required=True,
        help=(
            "TensorBoard directory for one experiment. "
            "Repeat this flag to process several algorithms."
        ),
    )
    parser.add_argument(
        "--name",
        action="append",
        help=(
            "Optional experiment name aligned with --logdir order. "
            "If omitted, the name is inferred from the folder."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Output root for aggregated TensorBoard runs. "
            "Default: <first-logdir-parent>/tb_aggregated"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = _build_experiments(args)

    # Single experiment: write directly to "<experiment>/tb_aggregated".
    # Multiple experiments: keep one output root and create one subdir per name.
    if args.output_root is not None:
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"Writing aggregated TensorBoard runs to: {output_root}")
        for exp in experiments:
            out_dir = output_root if len(experiments) == 1 else output_root / exp.name
            out_dir.mkdir(parents=True, exist_ok=True)
            n_runs, n_points = _write_aggregated_experiment(exp, output_dir=out_dir)
            print(
                f"[{exp.name}] runs={n_runs} "
                f"aggregated_steps={n_points} "
                f"output={out_dir}"
            )
        tb_logdir = output_root
    else:
        print("Writing aggregated TensorBoard runs:")
        if len(experiments) == 1:
            exp = experiments[0]
            out_dir = exp.logdir.resolve().parent / "tb_aggregated"
            out_dir.mkdir(parents=True, exist_ok=True)
            n_runs, n_points = _write_aggregated_experiment(exp, output_dir=out_dir)
            print(
                f"[{exp.name}] runs={n_runs} "
                f"aggregated_steps={n_points} "
                f"output={out_dir}"
            )
            tb_logdir = out_dir
        else:
            output_root = experiments[0].logdir.resolve().parent / "tb_aggregated"
            output_root.mkdir(parents=True, exist_ok=True)
            for exp in experiments:
                out_dir = output_root / exp.name
                out_dir.mkdir(parents=True, exist_ok=True)
                n_runs, n_points = _write_aggregated_experiment(exp, output_dir=out_dir)
                print(
                    f"[{exp.name}] runs={n_runs} "
                    f"aggregated_steps={n_points} "
                    f"output={out_dir}"
                )
            tb_logdir = output_root

    print("\nOpen TensorBoard with:")
    print(f"  tensorboard --logdir {tb_logdir}")
    print("Then go to Custom Scalars to view 'median_iqr' and 'mean_se' charts per tag.")


if __name__ == "__main__":
    main()
