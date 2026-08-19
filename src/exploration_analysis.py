"""Metrics, selection, and reports for exploration-tuning stages."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

import numpy as np

from src.exploration_protocol import ExplorationProtocol, Setting


@dataclass(frozen=True)
class Point:
    size: int
    depth: int
    normalized_size: float
    normalized_depth: float


def pareto_front_min(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set((float(x), float(y)) for x, y in points))
    return [
        point
        for point in unique
        if not any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and other != point
            for other in unique
        )
    ]


def hypervolume_min_2d(
    points: Iterable[tuple[float, float]],
    reference: tuple[float, float] = (1.0, 1.0),
) -> float:
    front = pareto_front_min(points)
    clipped = sorted(
        {
            (max(0.0, min(reference[0], x)), max(0.0, min(reference[1], y)))
            for x, y in front
            if x < reference[0] and y < reference[1]
        }
    )
    volume = 0.0
    for index, (x, y) in enumerate(clipped):
        next_x = clipped[index + 1][0] if index + 1 < len(clipped) else reference[0]
        volume += max(0.0, next_x - x) * max(0.0, reference[1] - y)
    return float(volume)


def bootstrap_mean_ci(
    differences: Iterable[float],
    *,
    repetitions: int = 10000,
    seed: int = 20260818,
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap requires at least one difference")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(repetitions), values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _read_points(path: Path) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    references = [row for row in rows if not str(row.get("run_id", "")).strip()]
    samples = [row for row in rows if str(row.get("run_id", "")).strip()]
    if len(references) != 1:
        raise ValueError(f"{path} must contain exactly one original-circuit row")
    initial = (int(float(references[0]["size"])), int(float(references[0]["depth"])))
    endpoints = [(int(float(row["size"])), int(float(row["depth"]))) for row in samples]
    return initial, endpoints


def _task_status(task_root: Path) -> Mapping[str, Any]:
    status_path = task_root / "task_status.json"
    if not status_path.is_file():
        raise ValueError(f"task status is missing: {status_path}")
    return json.loads(status_path.read_text(encoding="utf-8"))


def _resolved_attempt(status: Mapping[str, Any]) -> Path:
    if status.get("state") == "reused":
        return Path(str(status["source_attempt"]))
    if status.get("state") != "complete":
        raise ValueError(f"task is not complete: {status.get('task_id', 'unknown')}")
    return Path(str(status["attempt_dir"]))


def collect_task_records(artifact_root: Path) -> list[dict[str, Any]]:
    manifest_path = artifact_root / "stage_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"stage manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        task_root = artifact_root / "tasks" / str(task["task_id"])
        status = _task_status(task_root)
        attempt = _resolved_attempt(status)
        execution_status = dict(status)
        attempt_metadata = attempt / "attempt.json"
        if status.get("state") == "reused" and attempt_metadata.is_file():
            execution_status = json.loads(attempt_metadata.read_text(encoding="utf-8"))
        records.append(
            {
                **task,
                "status": dict(status),
                "execution_status": execution_status,
                "attempt_dir": str(attempt),
            }
        )
    return records


def _discovery_summary(attempt_dir: Path) -> tuple[float | None, int | None]:
    path = attempt_dir / "train" / "discovery_metrics.csv"
    if not path.is_file():
        return None, None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("row_type") == "circuit"]
    if not rows:
        return None, None
    final = next((row for row in rows if str(row.get("is_final", "")).lower() == "true"), rows[-1])
    return float(final["hypervolume"]), int(float(final["local_trajectory"]))


def _optimizer_updates(task: Mapping[str, Any], attempt_dir: Path) -> int:
    config_path = attempt_dir / "train" / ".hydra" / "config.yaml"
    try:
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return int(task["episodes"])
    episodes = int(task["episodes"])
    algorithm = str(task["algorithm"])
    if algorithm == "reinforce":
        return episodes * int(cfg.get("num_steps", 20))
    if algorithm == "ppo":
        ppo = cfg.get("algorithm", {}).get("ppo", cfg.get("ppo", {}))
        batches = math.ceil(int(ppo.get("rollout_steps", 80)) / int(ppo.get("minibatch_size", 64)))
        return episodes * int(ppo.get("ppo_epochs", 20)) * batches
    if algorithm == "pcn":
        pcn = cfg.get("pcn", cfg.get("algorithm", {}).get("pcn", {}))
        seeded = min(episodes, int(pcn.get("random_seed_episodes", 32)))
        collections = math.ceil(max(0, episodes - seeded) / max(1, int(pcn.get("collect_episodes_per_iter", 8))))
        return collections * int(pcn.get("train_updates_per_iter", 64))
    return episodes


def per_seed_metrics(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], list[Point]]]:
    rows: list[dict[str, Any]] = []
    grouped_points: dict[tuple[str, str, str], list[Point]] = defaultdict(list)
    for task in records:
        attempt = Path(str(task["attempt_dir"]))
        initial, endpoints = _read_points(attempt / "train" / "points.csv")
        initial_size, initial_depth = initial
        points = [
            Point(
                size=size,
                depth=depth,
                normalized_size=size / initial_size,
                normalized_depth=depth / initial_depth,
            )
            for size, depth in endpoints
        ]
        coordinates = [(point.normalized_size, point.normalized_depth) for point in points]
        front = pareto_front_min(coordinates)
        discovery_hv, discovered_trajectories = _discovery_summary(attempt)
        key = (str(task["algorithm"]), str(task["setting"]["setting_id"]), str(task["circuit"]))
        grouped_points[key].extend(points)
        unique_endpoints = len(set(endpoints))
        rows.append(
            {
                "algorithm": task["algorithm"],
                "setting_id": task["setting"]["setting_id"],
                "circuit": task["circuit"],
                "seed": int(task["seed"]),
                "sample_count": len(points),
                "hypervolume": hypervolume_min_2d(coordinates),
                "distinct_endpoints": unique_endpoints,
                "nondominated_endpoints": len(front),
                "best_size_reduction": max((1.0 - point.normalized_size for point in points), default=0.0),
                "best_depth_reduction": max((1.0 - point.normalized_depth for point in points), default=0.0),
                "mean_product_improvement": fmean(
                    [1.0 - point.normalized_size * point.normalized_depth for point in points]
                ) if points else 0.0,
                "training_discovery_hypervolume": discovery_hv,
                "discovered_trajectories": discovered_trajectories,
                "optimizer_updates": _optimizer_updates(task, attempt),
                "wall_time_seconds": float(task["execution_status"].get("wall_time_seconds", 0.0)),
            }
        )
    return rows, grouped_points


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(fmean(data)) if data else 0.0


def summarize_settings(
    seed_rows: list[dict[str, Any]],
    grouped_points: Mapping[tuple[str, str, str], list[Point]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(str(row["algorithm"]), str(row["setting_id"]), str(row["circuit"]))].append(row)
    summary: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        algorithm, setting_id, circuit = key
        points = list(grouped_points[key])
        coordinates = [(point.normalized_size, point.normalized_depth) for point in points]
        pooled_front = pareto_front_min(coordinates)
        front_set = set(pooled_front)
        front_hits = sum(
            (point.normalized_size, point.normalized_depth) in front_set for point in points
        )
        hvs = [float(row["hypervolume"]) for row in rows]
        summary.append(
            {
                "algorithm": algorithm,
                "setting_id": setting_id,
                "circuit": circuit,
                "n_seeds": len(rows),
                "mean_hypervolume": _mean(hvs),
                "std_hypervolume": float(pstdev(hvs)) if len(hvs) > 1 else 0.0,
                "pooled_hypervolume": hypervolume_min_2d(coordinates),
                "distinct_endpoints": len(set((point.size, point.depth) for point in points)),
                "nondominated_endpoints": len(pooled_front),
                "best_size_reduction": max((1.0 - point.normalized_size for point in points), default=0.0),
                "best_depth_reduction": max((1.0 - point.normalized_depth for point in points), default=0.0),
                "mean_product_improvement": _mean(float(row["mean_product_improvement"]) for row in rows),
                "pooled_front_hit_rate": front_hits / len(points) if points else 0.0,
                "mean_wall_time_seconds": _mean(float(row["wall_time_seconds"]) for row in rows),
                "mean_optimizer_updates": _mean(float(row["optimizer_updates"]) for row in rows),
            }
        )
        lookup = {(point.normalized_size, point.normalized_depth): point for point in points}
        for normalized_size, normalized_depth in pooled_front:
            point = lookup[(normalized_size, normalized_depth)]
            pooled_rows.append(
                {
                    "algorithm": algorithm,
                    "setting_id": setting_id,
                    "circuit": circuit,
                    "pooled_hypervolume": hypervolume_min_2d(coordinates),
                    "size": point.size,
                    "depth": point.depth,
                    "normalized_size": normalized_size,
                    "normalized_depth": normalized_depth,
                }
            )
    return summary, pooled_rows


def select_settings(
    summary_rows: list[dict[str, Any]],
    settings: Iterable[Setting],
    *,
    circuits: Iterable[str],
    tie_threshold: float,
) -> dict[str, Any]:
    by_setting = {(setting.algorithm, setting.setting_id): setting for setting in settings}
    circuits = list(circuits)
    algorithms = sorted({algorithm for algorithm, _ in by_setting})
    output: dict[str, Any] = {}
    for algorithm in algorithms:
        candidates = [setting for (name, _), setting in by_setting.items() if name == algorithm]
        circuit_rows = {
            (str(row["setting_id"]), str(row["circuit"])): row
            for row in summary_rows
            if row["algorithm"] == algorithm
        }
        maxima: dict[str, float] = {}
        for circuit in circuits:
            maxima[circuit] = max(
                (float(circuit_rows[(setting.setting_id, circuit)]["mean_hypervolume"]) for setting in candidates),
                default=0.0,
            )
        ranking: list[dict[str, Any]] = []
        for setting in candidates:
            relative = []
            products = []
            for circuit in circuits:
                row = circuit_rows.get((setting.setting_id, circuit))
                if row is None:
                    raise ValueError(f"missing summary for {algorithm}/{setting.setting_id}/{circuit}")
                maximum = maxima[circuit]
                relative.append(1.0 if maximum == 0.0 else float(row["mean_hypervolume"]) / maximum)
                products.append(float(row["mean_product_improvement"]))
            ranking.append(
                {
                    "setting": setting.to_dict(),
                    "selection_score": _mean(relative),
                    "minimum_relative_score": min(relative),
                    "mean_product_improvement": _mean(products),
                }
            )
        pending = list(ranking)
        ranked: list[dict[str, Any]] = []
        while pending:
            best_score = max(float(row["selection_score"]) for row in pending)
            tied = [
                row for row in pending
                if best_score - float(row["selection_score"]) <= tie_threshold + 1e-12
            ]
            tied.sort(
                key=lambda row: (
                    -float(row["minimum_relative_score"]),
                    -float(row["mean_product_improvement"]),
                    float(row["setting"]["exploration_rank"]),
                    str(row["setting"]["setting_id"]),
                )
            )
            winner = tied[0]
            ranked.append(winner)
            pending.remove(winner)
        winner = ranked[0]
        noncontrol = next((row for row in ranked if not row["setting"]["control"]), None)
        if noncontrol is None:
            raise ValueError(f"{algorithm} has no non-control candidate")
        output[algorithm] = {
            "selected": winner["setting"],
            "best_noncontrol": noncontrol["setting"],
            "ranked": [row["setting"] for row in ranked],
            "ranking_metrics": ranked,
        }
    return output


def confirmation_differences(
    seed_rows: list[dict[str, Any]],
    settings: Iterable[Setting],
    *,
    repetitions: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    roles: dict[tuple[str, str], str] = {}
    for setting in settings:
        role = "selected" if setting.setting_id.startswith("selected_") else "comparator"
        roles[(setting.algorithm, setting.setting_id)] = role
    grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in seed_rows:
        key = (str(row["algorithm"]), str(row["circuit"]), int(row["seed"]))
        grouped[key][roles[(str(row["algorithm"]), str(row["setting_id"]))]] = float(row["hypervolume"])
    by_pair: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (algorithm, circuit, seed), values in grouped.items():
        if set(values) != {"selected", "comparator"}:
            raise ValueError(f"unpaired confirmation seed: {algorithm}/{circuit}/{seed}")
        by_pair[(algorithm, circuit)].append((seed, values["selected"] - values["comparator"]))
    output: list[dict[str, Any]] = []
    for (algorithm, circuit), values in sorted(by_pair.items()):
        ordered = sorted(values)
        differences = [difference for _, difference in ordered]
        low, high = bootstrap_mean_ci(
            differences,
            repetitions=repetitions,
            seed=bootstrap_seed,
        )
        output.append(
            {
                "algorithm": algorithm,
                "circuit": circuit,
                "n_pairs": len(differences),
                "mean_difference": _mean(differences),
                "ci95_low": low,
                "ci95_high": high,
                "seed_differences": json.dumps(dict(ordered), sort_keys=True),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discovery_curve_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in records:
        path = Path(str(task["attempt_dir"])) / "train" / "discovery_metrics.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("row_type") != "circuit":
                    continue
                output.append(
                    {
                        "algorithm": task["algorithm"],
                        "setting_id": task["setting"]["setting_id"],
                        "circuit": task["circuit"],
                        "seed": int(task["seed"]),
                        "local_trajectory": int(float(row["local_trajectory"])),
                        "hypervolume": float(row["hypervolume"]),
                        "nondominated_count": float(row["nondominated_count"]),
                        "failed_trajectories": float(row["failed_trajectories"]),
                        "infeasible_trajectories": float(row["infeasible_trajectories"]),
                        "is_final": str(row.get("is_final", "")).lower() == "true",
                    }
                )
    return output


def analyze_stage(
    *,
    protocol: ExplorationProtocol,
    stage: str,
    artifact_root: Path,
    settings: list[Setting],
) -> dict[str, Any]:
    records = collect_task_records(artifact_root)
    seed_rows, grouped_points = per_seed_metrics(records)
    summary_rows, pooled_rows = summarize_settings(seed_rows, grouped_points)
    selection = (
        {}
        if stage == "smoke"
        else select_settings(
            summary_rows,
            settings,
            circuits=protocol.stages[stage]["circuits"],
            tie_threshold=float(protocol.data["common"]["practical_tie_threshold"]),
        )
    )
    selection_payload: dict[str, Any] = {
        "stage": stage,
        "protocol_hash": protocol.protocol_hash,
        "algorithms": selection,
    }
    _write_csv(artifact_root / "per_seed_metrics.csv", seed_rows)
    _write_csv(artifact_root / "setting_summary.csv", summary_rows)
    _write_csv(artifact_root / "pooled_front.csv", pooled_rows)
    _write_csv(artifact_root / "discovery_curves.csv", discovery_curve_rows(records))
    if stage == "confirmation":
        paired = confirmation_differences(
            seed_rows,
            settings,
            repetitions=int(protocol.data["common"]["bootstrap_repetitions"]),
            bootstrap_seed=int(protocol.data["common"]["bootstrap_seed"]),
        )
        _write_csv(artifact_root / "paired_differences.csv", paired)
        selection_payload["paired_differences"] = paired
    (artifact_root / "selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        f"# Exploration tuning: {stage}",
        "",
        f"Protocol hash: `{protocol.protocol_hash}`",
        "",
        "| Algorithm | Selected setting | Best non-control |",
        "| --- | --- | --- |",
    ]
    for algorithm, result in sorted(selection.items()):
        report_lines.append(
            f"| {algorithm} | `{result['selected']['setting_id']}` | "
            f"`{result['best_noncontrol']['setting_id']}` |"
        )
    if stage == "confirmation":
        report_lines.extend(
            [
                "",
                "## Paired confirmation differences",
                "",
                "| Algorithm | Circuit | Mean selected - comparator HV | 95% bootstrap CI |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for row in selection_payload["paired_differences"]:
            report_lines.append(
                f"| {row['algorithm']} | {row['circuit']} | {row['mean_difference']:.6g} | "
                f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] |"
            )
    report_lines.extend(
        [
            "",
            "Review `setting_summary.csv`, `per_seed_metrics.csv`, and `pooled_front.csv` before submitting the next stage.",
            "Pooled hypervolume is reported but is not used for selection.",
        ]
    )
    (artifact_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return selection_payload


__all__ = [
    "analyze_stage",
    "bootstrap_mean_ci",
    "collect_task_records",
    "confirmation_differences",
    "discovery_curve_rows",
    "hypervolume_min_2d",
    "pareto_front_min",
    "select_settings",
    "summarize_settings",
]
