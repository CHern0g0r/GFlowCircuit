#!/home/fedor.chernogorskii/envs/ospiel/bin/python3
"""Export mean/std TensorBoard scalars across runs into CSV.

Typical usage:
  python export_tb_mean_std_csv.py --logdir /path/to/experiment/tensorboard

The script scans run directories (run_*) with event files, collects scalar events,
and writes a long-form CSV with one row per (tag, step):
  tag,step,n_runs,mean,std,min,max
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean, pstdev

from tensorboard.backend.event_processing import event_accumulator

SCALARS = event_accumulator.SCALARS


def _has_event_files(path: Path) -> bool:
    return any(p.is_file() and p.name.startswith("events.out.tfevents") for p in path.iterdir())


def _collect_run_dirs(logdir: Path) -> tuple[list[Path], bool]:
    logdir = logdir.resolve()
    if not logdir.is_dir():
        raise SystemExit(f"Logdir is not a directory: {logdir}")

    # Case 1: direct run dir
    if _has_event_files(logdir):
        return [logdir], False

    # Case 2: experiment tensorboard dir with run_* children
    run_dirs: list[Path] = []
    for child in sorted(logdir.iterdir()):
        if child.is_dir() and child.name.startswith("run_") and _has_event_files(child):
            run_dirs.append(child)

    if run_dirs:
        return run_dirs, True

    # Case 3: no run_* subdirs and no direct event files in logdir.
    # Keep behavior permissive for callers that pass an experiment root:
    # treat the provided directory as a single source.
    return [logdir], False


def _load_run_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars") or []
    series: dict[str, dict[int, float]] = {}
    for tag in tags:
        # If multiple events share a step, keep the last one.
        by_step: dict[int, float] = {}
        for event in sorted(ea.Scalars(tag), key=lambda e: e.step):
            by_step[int(event.step)] = float(event.value)
        if by_step:
            series[tag] = by_step
    return series


def _aggregate_runs(
    run_dirs: list[Path],
    has_run_folders: bool,
) -> list[dict[str, str | int | float | None]]:
    per_run = [_load_run_scalars(rd) for rd in run_dirs]
    all_tags = sorted({tag for run in per_run for tag in run.keys()})
    rows: list[dict[str, str | int | float | None]] = []

    for tag in all_tags:
        all_steps = sorted({step for run in per_run if tag in run for step in run[tag].keys()})
        for step in all_steps:
            vals = [run[tag][step] for run in per_run if tag in run and step in run[tag]]
            if not vals:
                continue
            mean_v = float(fmean(vals))
            if has_run_folders:
                std_v: float | None = float(pstdev(vals)) if len(vals) > 1 else 0.0
            else:
                std_v = None
            rows.append(
                {
                    "tag": tag,
                    "step": int(step),
                    "n_runs": int(len(vals)),
                    "mean": mean_v,
                    "std": std_v,
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                }
            )
    return rows


def _default_output(logdir: Path) -> Path:
    root = logdir.resolve()
    stem = root.parent.name if root.name == "tensorboard" else root.name
    safe_stem = stem.replace(":", "-")
    return root.parent / f"{safe_stem}_mean_std.csv"


def _fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.10g}"


def _fmt_optional(x: float | None) -> str:
    if x is None:
        return ""
    return _fmt(x)


def main(logdir: Path, output: Path | None = None) -> None:
    run_dirs, has_run_folders = _collect_run_dirs(logdir)
    rows = _aggregate_runs(run_dirs, has_run_folders=has_run_folders)
    if not rows:
        raise SystemExit("No scalar data found in TensorBoard runs.")

    out = output.resolve() if output else _default_output(logdir)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "step", "n_runs", "mean", "std", "min", "max"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tag": row["tag"],
                    "step": row["step"],
                    "n_runs": row["n_runs"],
                    "mean": _fmt(float(row["mean"])),
                    "std": _fmt_optional(row["std"] if isinstance(row["std"], float) or row["std"] is None else None),
                    "min": _fmt(float(row["min"])),
                    "max": _fmt(float(row["max"])),
                }
            )

    print(f"Wrote {out}")
    print(f"Aggregated {len(run_dirs)} run(s).")


if __name__ == "__main__":
    logdir = Path("/home/fedor.chernogorskii/workspace/circ/rl_project/GFlowCircuit/outputs/2026-04-22/12:40_gflownet_tb_2670/tensorboard")
    output = Path('/home/fedor.chernogorskii/workspace/circ/rl_project/results/csv/gflownet.csv')
    main(logdir, output)
