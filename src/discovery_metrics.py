"""Online, training-only circuit discovery metrics.

The archive in this module is deliberately independent of policy evaluation and
of algorithm-specific replay/archive implementations. It retains unique,
feasible terminal size/depth pairs and measures their strict 2-D hypervolume
relative to the original circuit.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


@dataclass(frozen=True)
class DiscoveryPoint:
    size: int
    depth: int
    first_local_trajectory: int


def _weakly_dominates(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    """Return whether lhs is no worse in both costs and better in at least one."""
    return lhs[0] <= rhs[0] and lhs[1] <= rhs[1] and lhs != rhs


class CircuitDiscoveryArchive:
    """Constrained Pareto archive for one circuit's training trajectories."""

    def __init__(self, *, circuit: str, initial_size: int, initial_depth: int) -> None:
        if int(initial_size) <= 0 or int(initial_depth) <= 0:
            raise ValueError("initial circuit size and depth must be positive")
        self.circuit = str(circuit)
        self.initial_size = int(initial_size)
        self.initial_depth = int(initial_depth)
        self.attempted_trajectories = 0
        self.failed_trajectories = 0
        self.infeasible_trajectories = 0
        origin = DiscoveryPoint(self.initial_size, self.initial_depth, 0)
        self._points: dict[tuple[int, int], DiscoveryPoint] = {
            (origin.size, origin.depth): origin
        }

    @property
    def points(self) -> tuple[DiscoveryPoint, ...]:
        return tuple(sorted(self._points.values(), key=lambda p: (p.size, p.depth)))

    @property
    def nondominated_count(self) -> int:
        return len(self._points)

    def record_failure(self) -> int:
        self.attempted_trajectories += 1
        self.failed_trajectories += 1
        return self.attempted_trajectories

    def record_terminal(
        self,
        *,
        initial_size: int,
        initial_depth: int,
        final_size: int,
        final_depth: int,
    ) -> bool:
        """Count a trajectory and insert its endpoint if it extends the archive."""
        if (int(initial_size), int(initial_depth)) != (self.initial_size, self.initial_depth):
            raise ValueError(
                f"initial metrics changed for {self.circuit}: "
                f"expected {(self.initial_size, self.initial_depth)}, "
                f"got {(int(initial_size), int(initial_depth))}"
            )

        self.attempted_trajectories += 1
        size = int(final_size)
        depth = int(final_depth)
        if size < 0 or depth < 0:
            self.failed_trajectories += 1
            return False
        if size > self.initial_size or depth > self.initial_depth:
            self.infeasible_trajectories += 1
            return False

        key = (size, depth)
        if key in self._points:
            return False
        if any(_weakly_dominates(existing, key) for existing in self._points):
            return False

        dominated = [existing for existing in self._points if _weakly_dominates(key, existing)]
        for existing in dominated:
            del self._points[existing]
        self._points[key] = DiscoveryPoint(size, depth, self.attempted_trajectories)
        return True

    def hypervolume(self) -> float:
        """Strict 2-D minimization hypervolume with normalized reference (1, 1)."""
        current_y = 1.0
        volume = 0.0
        for point in self.points:
            x = float(point.size) / float(self.initial_size)
            y = float(point.depth) / float(self.initial_depth)
            if y < current_y:
                volume += max(0.0, 1.0 - x) * (current_y - y)
                current_y = y
        return float(volume)

    def snapshot(self) -> dict[str, int | float | str]:
        return {
            "circuit": self.circuit,
            "local_trajectory": self.attempted_trajectories,
            "hypervolume": self.hypervolume(),
            "nondominated_count": self.nondominated_count,
            "failed_trajectories": self.failed_trajectories,
            "infeasible_trajectories": self.infeasible_trajectories,
        }


class TrainingDiscoveryTracker:
    """Maintain per-circuit archives and budget-aligned metric snapshots."""

    def __init__(
        self,
        *,
        initial_metrics: Mapping[str, tuple[int, int]],
        emit_every_trajectories: int = 50,
        tensorboard_logger: Any | None = None,
    ) -> None:
        interval = int(emit_every_trajectories)
        if interval <= 0:
            raise ValueError("emit_every_trajectories must be positive")
        if not initial_metrics:
            raise ValueError("at least one training circuit is required")
        self.emit_every_trajectories = interval
        self._tb = tensorboard_logger
        self.archives = {
            str(circuit): CircuitDiscoveryArchive(
                circuit=str(circuit),
                initial_size=int(metrics[0]),
                initial_depth=int(metrics[1]),
            )
            for circuit, metrics in initial_metrics.items()
        }
        self._labels = {
            circuit: f"{idx}_{Path(circuit).stem}"
            for idx, circuit in enumerate(self.archives)
        }
        self._circuit_rows: dict[tuple[str, int], dict[str, Any]] = {}
        self._aggregate_rows: dict[int, dict[str, Any]] = {}
        self._metric_rows: list[dict[str, Any]] = []
        self._finalized = False

    def record_terminal(
        self,
        *,
        circuit: str,
        initial_size: int,
        initial_depth: int,
        final_size: int,
        final_depth: int,
    ) -> bool:
        self._ensure_active()
        archive = self._archive(circuit)
        inserted = archive.record_terminal(
            initial_size=initial_size,
            initial_depth=initial_depth,
            final_size=final_size,
            final_depth=final_depth,
        )
        self._capture_milestone(archive)
        return inserted

    def record_failure(self, *, circuit: str) -> None:
        self._ensure_active()
        archive = self._archive(circuit)
        archive.record_failure()
        self._capture_milestone(archive)

    def finalize(self) -> dict[str, list[dict[str, Any]]]:
        if not self._finalized:
            for archive in self.archives.values():
                self._capture(archive, is_final=True)
            counts = {archive.attempted_trajectories for archive in self.archives.values()}
            if len(counts) == 1:
                self._capture_aggregate(next(iter(counts)))
            self._finalized = True
        return {
            "discovery_front": self.front_rows(),
            "discovery_metrics": list(self._metric_rows),
        }

    def front_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for circuit, archive in self.archives.items():
            for point in archive.points:
                rows.append(
                    {
                        "circuit": circuit,
                        "size": point.size,
                        "depth": point.depth,
                        "normalized_size": float(point.size) / float(archive.initial_size),
                        "normalized_depth": float(point.depth) / float(archive.initial_depth),
                        "first_local_trajectory": point.first_local_trajectory,
                    }
                )
        return rows

    def _ensure_active(self) -> None:
        if self._finalized:
            raise RuntimeError("cannot record trajectories after discovery tracker finalization")

    def _archive(self, circuit: str) -> CircuitDiscoveryArchive:
        try:
            return self.archives[str(circuit)]
        except KeyError as exc:
            raise KeyError(f"unknown training circuit: {circuit}") from exc

    def _capture_milestone(self, archive: CircuitDiscoveryArchive) -> None:
        count = archive.attempted_trajectories
        if count % self.emit_every_trajectories == 0:
            self._capture(archive, is_final=False)

    def _capture(self, archive: CircuitDiscoveryArchive, *, is_final: bool) -> None:
        count = archive.attempted_trajectories
        key = (archive.circuit, count)
        existing = self._circuit_rows.get(key)
        if existing is not None:
            existing["is_final"] = bool(existing["is_final"] or is_final)
        else:
            row: dict[str, Any] = {
                "row_type": "circuit",
                **archive.snapshot(),
                "is_final": bool(is_final),
            }
            self._circuit_rows[key] = row
            self._metric_rows.append(row)
            if self._tb is not None:
                label = self._labels[archive.circuit]
                self._tb.add_scalars(
                    count,
                    {
                        f"discovery/{label}/hypervolume": float(row["hypervolume"]),
                        f"discovery/{label}/nondominated_count": float(row["nondominated_count"]),
                    },
                )
        self._capture_aggregate(count)

    def _capture_aggregate(self, count: int) -> None:
        source_rows = [
            self._circuit_rows.get((circuit, int(count))) for circuit in self.archives
        ]
        if any(row is None for row in source_rows):
            return
        rows = [row for row in source_rows if row is not None]
        is_final = all(bool(row["is_final"]) for row in rows)
        existing = self._aggregate_rows.get(int(count))
        if existing is not None:
            existing["is_final"] = bool(existing["is_final"] or is_final)
            return
        row = {
            "row_type": "mean",
            "circuit": "__mean__",
            "local_trajectory": int(count),
            "hypervolume": float(fmean(float(item["hypervolume"]) for item in rows)),
            "nondominated_count": float(
                fmean(float(item["nondominated_count"]) for item in rows)
            ),
            "failed_trajectories": float(
                fmean(float(item["failed_trajectories"]) for item in rows)
            ),
            "infeasible_trajectories": float(
                fmean(float(item["infeasible_trajectories"]) for item in rows)
            ),
            "is_final": is_final,
        }
        self._aggregate_rows[int(count)] = row
        self._metric_rows.append(row)
        if self._tb is not None:
            self._tb.add_scalars(
                int(count),
                {
                    "discovery/mean_hypervolume": float(row["hypervolume"]),
                    "discovery/mean_nondominated_count": float(row["nondominated_count"]),
                },
            )


def initial_metrics_from_resyn2_cache(
    *,
    circuits: list[str],
    resyn2_baselines: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, int]]:
    """Extract original circuit metrics from the cache already built by src.run."""
    out: dict[str, tuple[int, int]] = {}
    for circuit in circuits:
        variants = resyn2_baselines[circuit]["resyn2_variants"]
        if not isinstance(variants, Mapping):
            raise TypeError("resyn2_variants must be a mapping")
        reference = variants["resyn2_1"]
        if not isinstance(reference, Mapping):
            raise TypeError("resyn2_1 metrics must be a mapping")
        out[circuit] = (int(reference["initial_size"]), int(reference["initial_depth"]))
    return out


def build_training_discovery_tracker(
    *,
    enabled: bool,
    circuits: list[str],
    resyn2_baselines: Mapping[str, Mapping[str, Any]],
    emit_every_trajectories: int,
    tensorboard_logger: Any | None,
) -> TrainingDiscoveryTracker | None:
    if not enabled:
        return None
    return TrainingDiscoveryTracker(
        initial_metrics=initial_metrics_from_resyn2_cache(
            circuits=circuits,
            resyn2_baselines=resyn2_baselines,
        ),
        emit_every_trajectories=emit_every_trajectories,
        tensorboard_logger=tensorboard_logger,
    )


def finalize_training_discovery(
    tracker: TrainingDiscoveryTracker | None,
) -> dict[str, list[dict[str, Any]]]:
    if tracker is None:
        return {"discovery_front": [], "discovery_metrics": []}
    return tracker.finalize()


def record_training_trajectory(
    tracker: TrainingDiscoveryTracker | None,
    trajectory: Any,
    *,
    circuit: str | None = None,
) -> None:
    """Record a sampler result represented as either an object or a mapping."""
    if tracker is None:
        return

    def field(name: str) -> Any:
        if isinstance(trajectory, Mapping):
            return trajectory[name]
        return getattr(trajectory, name)

    resolved_circuit: str | None = None
    try:
        resolved_circuit = str(circuit if circuit is not None else field("file_path"))
        initial_size = int(field("initial_size"))
        initial_depth = int(field("initial_depth"))
        final_size = int(field("final_size"))
        final_depth = int(field("final_depth"))
    except (AttributeError, KeyError, TypeError, ValueError):
        if resolved_circuit is None:
            raise
        tracker.record_failure(circuit=resolved_circuit)
        return

    tracker.record_terminal(
        circuit=resolved_circuit,
        initial_size=initial_size,
        initial_depth=initial_depth,
        final_size=final_size,
        final_depth=final_depth,
    )


def write_discovery_artifacts(
    *,
    output_dir: Path,
    algorithm: str,
    runs: list[dict[str, Any]],
) -> dict[str, str]:
    """Persist final fronts and milestone metrics for all training seeds."""
    front_fields = [
        "algorithm",
        "run_idx",
        "seed",
        "circuit",
        "size",
        "depth",
        "normalized_size",
        "normalized_depth",
        "first_local_trajectory",
    ]
    metric_fields = [
        "algorithm",
        "run_idx",
        "seed",
        "row_type",
        "circuit",
        "local_trajectory",
        "hypervolume",
        "nondominated_count",
        "failed_trajectories",
        "infeasible_trajectories",
        "is_final",
    ]
    front_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for run in runs:
        metadata = {
            "algorithm": str(algorithm),
            "run_idx": int(run["run_idx"]),
            "seed": int(run["seed"]),
        }
        front_rows.extend({**metadata, **row} for row in run.get("discovery_front", []))
        metric_rows.extend({**metadata, **row} for row in run.get("discovery_metrics", []))

    if not front_rows and not metric_rows:
        return {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, fields, rows in (
        ("discovery_front.csv", front_fields, front_rows),
        ("discovery_metrics.csv", metric_fields, metric_rows),
    ):
        path = output_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = str(path)
    return paths


__all__ = [
    "CircuitDiscoveryArchive",
    "DiscoveryPoint",
    "TrainingDiscoveryTracker",
    "build_training_discovery_tracker",
    "finalize_training_discovery",
    "initial_metrics_from_resyn2_cache",
    "record_training_trajectory",
    "write_discovery_artifacts",
]
