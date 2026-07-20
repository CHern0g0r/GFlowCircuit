from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.discovery_metrics import (
    CircuitDiscoveryArchive,
    TrainingDiscoveryTracker,
    record_training_trajectory,
    write_discovery_artifacts,
)


class _FakeTensorBoard:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, float]]] = []

    def add_scalars(self, step: int, scalars: dict[str, float]) -> None:
        self.calls.append((int(step), dict(scalars)))


class CircuitDiscoveryArchiveTest(unittest.TestCase):
    def test_constrained_dominance_duplicates_and_failure_counts(self) -> None:
        archive = CircuitDiscoveryArchive(circuit="c.blif", initial_size=100, initial_depth=100)
        self.assertEqual([(p.size, p.depth) for p in archive.points], [(100, 100)])
        self.assertEqual(archive.hypervolume(), 0.0)

        self.assertTrue(
            archive.record_terminal(
                initial_size=100,
                initial_depth=100,
                final_size=90,
                final_depth=100,
            )
        )
        self.assertFalse(
            archive.record_terminal(
                initial_size=100,
                initial_depth=100,
                final_size=90,
                final_depth=100,
            )
        )
        self.assertFalse(
            archive.record_terminal(
                initial_size=100,
                initial_depth=100,
                final_size=80,
                final_depth=110,
            )
        )
        self.assertTrue(
            archive.record_terminal(
                initial_size=100,
                initial_depth=100,
                final_size=95,
                final_depth=90,
            )
        )
        archive.record_failure()

        self.assertEqual(archive.attempted_trajectories, 5)
        self.assertEqual(archive.failed_trajectories, 1)
        self.assertEqual(archive.infeasible_trajectories, 1)
        self.assertEqual([(p.size, p.depth) for p in archive.points], [(90, 100), (95, 90)])
        self.assertTrue(math.isclose(archive.hypervolume(), 0.005, abs_tol=1e-12))

        self.assertTrue(
            archive.record_terminal(
                initial_size=100,
                initial_depth=100,
                final_size=85,
                final_depth=90,
            )
        )
        self.assertEqual([(p.size, p.depth) for p in archive.points], [(85, 90)])
        self.assertEqual(archive.points[0].first_local_trajectory, 6)
        self.assertTrue(math.isclose(archive.hypervolume(), 0.015, abs_tol=1e-12))

    def test_two_point_hypervolume(self) -> None:
        archive = CircuitDiscoveryArchive(circuit="c.blif", initial_size=100, initial_depth=100)
        archive.record_terminal(
            initial_size=100,
            initial_depth=100,
            final_size=80,
            final_depth=95,
        )
        archive.record_terminal(
            initial_size=100,
            initial_depth=100,
            final_size=90,
            final_depth=90,
        )
        self.assertTrue(math.isclose(archive.hypervolume(), 0.015, abs_tol=1e-12))

    def test_initial_metrics_must_remain_stable(self) -> None:
        archive = CircuitDiscoveryArchive(circuit="c.blif", initial_size=100, initial_depth=20)
        with self.assertRaisesRegex(ValueError, "initial metrics changed"):
            archive.record_terminal(
                initial_size=101,
                initial_depth=20,
                final_size=90,
                final_depth=19,
            )


class TrainingDiscoveryTrackerTest(unittest.TestCase):
    def test_local_milestones_and_synchronized_mean(self) -> None:
        logger = _FakeTensorBoard()
        tracker = TrainingDiscoveryTracker(
            initial_metrics={"a.blif": (100, 100), "b.blif": (200, 100)},
            emit_every_trajectories=2,
            tensorboard_logger=logger,
        )
        tracker.record_terminal(
            circuit="a.blif",
            initial_size=100,
            initial_depth=100,
            final_size=90,
            final_depth=90,
        )
        tracker.record_terminal(
            circuit="a.blif",
            initial_size=100,
            initial_depth=100,
            final_size=80,
            final_depth=95,
        )
        self.assertFalse(any(row["row_type"] == "mean" for row in tracker._metric_rows))

        tracker.record_failure(circuit="b.blif")
        tracker.record_terminal(
            circuit="b.blif",
            initial_size=200,
            initial_depth=100,
            final_size=180,
            final_depth=90,
        )
        result = tracker.finalize()

        rows_at_two = [row for row in result["discovery_metrics"] if row["local_trajectory"] == 2]
        self.assertEqual(len(rows_at_two), 3)
        mean = next(row for row in rows_at_two if row["row_type"] == "mean")
        self.assertTrue(mean["is_final"])
        self.assertTrue(math.isclose(float(mean["hypervolume"]), 0.0125, abs_tol=1e-12))
        self.assertEqual(float(mean["nondominated_count"]), 1.5)
        self.assertTrue(any("discovery/mean_hypervolume" in scalars for _, scalars in logger.calls))

        a_points = [row for row in result["discovery_front"] if row["circuit"] == "a.blif"]
        self.assertEqual(
            {(row["size"], row["depth"], row["first_local_trajectory"]) for row in a_points},
            {(80, 95, 2), (90, 90, 1)},
        )

    def test_unequal_final_budgets_do_not_create_final_mean(self) -> None:
        tracker = TrainingDiscoveryTracker(
            initial_metrics={"a": (10, 10), "b": (10, 10)},
            emit_every_trajectories=2,
        )
        for _ in range(3):
            tracker.record_terminal(
                circuit="a",
                initial_size=10,
                initial_depth=10,
                final_size=9,
                final_depth=9,
            )
        for _ in range(2):
            tracker.record_terminal(
                circuit="b",
                initial_size=10,
                initial_depth=10,
                final_size=9,
                final_depth=9,
            )
        result = tracker.finalize()
        final_means = [
            row
            for row in result["discovery_metrics"]
            if row["row_type"] == "mean" and row["is_final"]
        ]
        self.assertEqual(final_means, [])
        final_circuit_counts = {
            int(row["local_trajectory"])
            for row in result["discovery_metrics"]
            if row["row_type"] == "circuit" and row["is_final"]
        }
        self.assertEqual(final_circuit_counts, {2, 3})

    def test_csv_artifacts_include_run_metadata(self) -> None:
        tracker = TrainingDiscoveryTracker(
            initial_metrics={"a": (10, 10)},
            emit_every_trajectories=1,
        )
        tracker.record_terminal(
            circuit="a",
            initial_size=10,
            initial_depth=10,
            final_size=9,
            final_depth=9,
        )
        result = tracker.finalize()
        with TemporaryDirectory() as directory:
            paths = write_discovery_artifacts(
                output_dir=Path(directory),
                algorithm="test_algorithm",
                runs=[{"run_idx": 2, "seed": 7, **result}],
            )
            front = Path(paths["discovery_front.csv"]).read_text(encoding="utf-8")
            metrics = Path(paths["discovery_metrics.csv"]).read_text(encoding="utf-8")
        self.assertIn("test_algorithm,2,7,a,9,9", front)
        self.assertIn("test_algorithm,2,7,circuit,a,1", metrics)

    def test_invalid_mapping_counts_as_failure_when_circuit_is_known(self) -> None:
        tracker = TrainingDiscoveryTracker(
            initial_metrics={"a": (10, 10)},
            emit_every_trajectories=50,
        )
        record_training_trajectory(
            tracker,
            {"initial_size": 10, "initial_depth": 10},
            circuit="a",
        )
        result = tracker.finalize()
        row = next(
            row
            for row in result["discovery_metrics"]
            if row["row_type"] == "circuit"
        )
        self.assertEqual(row["local_trajectory"], 1)
        self.assertEqual(row["failed_trajectories"], 1)


if __name__ == "__main__":
    unittest.main()
