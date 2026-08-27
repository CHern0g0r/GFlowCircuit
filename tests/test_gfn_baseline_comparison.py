from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.gfn_baseline_comparison import (
    ComparisonProtocol,
    _next_attempt_dir,
    _resolved_health_config,
    _training_command,
    _validate_health_run,
    compare_artifacts,
    hierarchical_bootstrap_ci,
    strict_hypervolume,
    strict_pareto_front,
    validate_protocol,
)

PROTOCOL = Path("cfg/exp/gfn_baseline_comparison/protocol.yaml")


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ComparisonProtocol.load(PROTOCOL)

    def test_exact_matrix_and_budget(self) -> None:
        result = validate_protocol(self.protocol, compose=False)
        self.assertEqual(
            self.protocol.circuits,
            ("C1355", "C5315", "adder", "apex1", "bc0", "dalu", "k2", "max"),
        )
        self.assertEqual(result["tasks"], 8)
        self.assertEqual(result["trained_models"], 80)
        self.assertEqual(result["training_trajectories_per_model"], 800)
        self.assertEqual(result["unique_evaluation_rollouts"], 160_000)

    def test_training_command_uses_calibrated_health_runner(self) -> None:
        task = self.protocol.task(2)
        command = _training_command(
            self.protocol,
            task,
            seed=7,
            output_dir=Path("/tmp/out"),
            python_executable="python",
            device_name="cuda",
        )
        self.assertIn("src.experiments.tb_logz_calibration", command)
        self.assertEqual(command[command.index("--variant") + 1], "zcal")
        self.assertEqual(command[command.index("--log-z-learning-rate") + 1], "0.01")
        self.assertEqual(command[command.index("--max-trajectories") + 1], "800")
        self.assertEqual(command[command.index("--schedule-trajectories") + 1], "800")
        self.assertTrue(command[command.index("--circuit") + 1].endswith("adder.aig"))

    def test_attempt_directories_are_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "attempt_001").mkdir()
            (root / "attempt_004").mkdir()
            self.assertEqual(_next_attempt_dir(root).name, "attempt_005")


class MetricTests(unittest.TestCase):
    def test_strict_front_and_hypervolume(self) -> None:
        points = [(0.8, 0.9), (0.9, 0.8), (0.95, 0.95), (1.2, 0.5)]
        self.assertEqual(strict_pareto_front(points), [(0.8, 0.9), (0.9, 0.8)])
        self.assertAlmostEqual(strict_hypervolume(points), 0.03)
        self.assertEqual(strict_hypervolume([(0.8, 1.0), (1.0, 0.7)]), 0.0)

    def test_hierarchical_bootstrap_is_deterministic(self) -> None:
        values = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float)
        first = hierarchical_bootstrap_ci(values, repetitions=500, seed=17)
        second = hierarchical_bootstrap_ci(values, repetitions=500, seed=17)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], values.mean())
        self.assertGreaterEqual(first[1], values.mean())


class HealthArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ComparisonProtocol.load(PROTOCOL)
        self.task = self.protocol.task(0)

    def _write_run(self, run_dir: Path, *, training_trajectories: int = 800) -> None:
        checkpoint_path = run_dir / "checkpoints" / "trajectory_800.pt"
        checkpoint_path.parent.mkdir(parents=True)
        config = OmegaConf.to_container(_resolved_health_config(self.protocol), resolve=True)
        counters = {
            "training_trajectories": training_trajectories,
            "training_transitions": 16000,
            "training_presentations": 800,
            "optimizer_updates": 200,
            "calibration_trajectories": 64,
            "calibration_target_presentations": 64,
            "calibration_training_presentations": 64,
            "new_training_trajectories": 736,
        }
        resolved = {
            "trajectories_per_update": 4,
            "policy_learning_rate": 0.001,
            "log_z_learning_rate_resolved": 0.01,
            "reward_alpha": 4.0,
            "exploration_epsilon_start": 0.5,
            "exploration_epsilon_end": 0.01,
            "exploration_warmup_updates": 20,
            "schedule_trajectories": 800,
            "calibration_trajectories": 64,
            "project_config": config,
        }
        torch.save(
            {
                "variant": "zcal",
                "numerical_failure": None,
                "counters": counters,
                "calibration_phase_complete": True,
                "archive": {"records": [{} for _ in range(800)]},
                "resolved_config": resolved,
                "policy": {"log_z": torch.tensor(1.0)},
            },
            checkpoint_path,
        )
        (run_dir / "run_summary.json").write_text(
            json.dumps({
                "complete": True,
                "numerical_failure": None,
                "variant": "zcal",
                "circuit": self.task.circuit,
                "seed": 0,
                "counters": counters,
                "final_checkpoint": str(checkpoint_path),
            }),
            encoding="utf-8",
        )

    def test_accepts_complete_800_trajectory_health_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_run(run_dir)
            checkpoint = _validate_health_run(self.protocol, self.task, seed=0, run_dir=run_dir)
            self.assertEqual(checkpoint.name, "trajectory_800.pt")

    def test_rejects_non_800_trajectory_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_run(run_dir, training_trajectories=796)
            with self.assertRaisesRegex(ValueError, "trajectory accounting mismatch"):
                _validate_health_run(self.protocol, self.task, seed=0, run_dir=run_dir)


def _write_attempt(
    root: Path,
    *,
    method: str,
    circuit: str,
    training_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    offset: int,
) -> None:
    attempt = root / "tasks" / f"{method}__{circuit}" / "attempt_001"
    attempt.mkdir(parents=True)
    (attempt / "attempt.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    budget_dir = attempt / "evaluation" / "budgets"
    budget_dir.mkdir(parents=True)
    reference = {
        "method": method,
        "circuit_name": circuit,
        "run_id": None,
        "training_seed": None,
        "evaluation_seed": None,
        "sample_id": None,
        "size": 100,
        "depth": 100,
    }
    for budget in budgets:
        rows = [{**reference, "sample_budget": budget}]
        for training_seed in training_seeds:
            for evaluation_seed in evaluation_seeds:
                for sample_id in range(budget):
                    rows.append({
                        "method": method,
                        "circuit_name": circuit,
                        "run_id": training_seed,
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "sample_id": sample_id,
                        "size": 95 - offset - sample_id,
                        "depth": 95 - offset,
                        "sample_budget": budget,
                    })
        pd.DataFrame(rows).to_csv(budget_dir / f"points_n{budget:03d}.csv", index=False)


class ComparisonReportTests(unittest.TestCase):
    def test_synthetic_artifacts_generate_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            methods = ("gflownet", "reinforce", "drills", "ppo")
            roots = {method: root / method for method in methods}
            for method in methods:
                _write_attempt(
                    roots[method],
                    method=method,
                    circuit="toy",
                    training_seeds=(0, 1),
                    evaluation_seeds=(0, 1),
                    budgets=(1, 2),
                    offset=2 if method == "gflownet" else 0,
                )
            protocol = SimpleNamespace(
                circuits=("toy",),
                training_seeds=(0, 1),
                evaluation_seeds=(0, 1),
                sample_budgets=(1, 2),
                max_samples=2,
                protocol_hash="test",
                data={
                    "campaign": "test",
                    "caveat": "Synthetic test.",
                    "common": {"bootstrap_repetitions": 200, "bootstrap_seed": 1},
                },
            )
            output = root / "report"
            result = compare_artifacts(
                protocol,
                gfn_root=roots["gflownet"],
                baseline_roots={
                    "reinforce": roots["reinforce"],
                    "drills": roots["drills"],
                    "ppo": roots["ppo"],
                },
                output_dir=output,
            )
            self.assertEqual(result["methods"], list(methods))
            for name in (
                "seed_metrics.csv",
                "summary.csv",
                "pairwise.csv",
                "pooled_fronts.csv",
                "artifact_manifest.json",
                "comparison_report.md",
                "comparison_summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            pairwise = pd.read_csv(output / "pairwise.csv")
            final = pairwise.loc[
                (pairwise["baseline"] == "reinforce")
                & (pairwise["circuit"] == "toy")
                & (pairwise["sample_budget"] == 2)
            ].iloc[0]
            self.assertGreater(float(final["mean_difference_hypervolume"]), 0.0)


if __name__ == "__main__":
    unittest.main()
