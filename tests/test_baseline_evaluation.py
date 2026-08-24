from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml

from src.baseline_evaluation import (
    BaselineEvaluationProtocol,
    _next_attempt_dir,
    _train_command,
    run_task,
    validate_attempt,
    validate_protocol,
)


PROTOCOL = Path("cfg/exp/baseline_evaluation/protocol.yaml")


def _set_path(mapping: dict, dotted_path: str, value: object) -> None:
    current = mapping
    components = dotted_path.split(".")
    for component in components[:-1]:
        current = current.setdefault(component, {})
    current[components[-1]] = value


class BaselineEvaluationProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = BaselineEvaluationProtocol.load(PROTOCOL)

    def test_matrix_and_rollout_counts(self) -> None:
        result = validate_protocol(self.protocol, python_executable="python", compose=False)
        self.assertEqual(self.protocol.methods, ("reinforce", "drills", "ppo"))
        self.assertEqual(len(self.protocol.tasks()), 27)
        self.assertEqual(len({task.task_id for task in self.protocol.tasks()}), 27)
        self.assertEqual(result["trained_models"], 270)
        self.assertEqual(result["unique_evaluation_rollouts"], 540_000)
        self.assertEqual(result["sample_budgets"], [10, 50, 100, 200])

    def test_stable_circuit_index_and_minimal_train_overrides(self) -> None:
        task = self.protocol.task(method="ppo", circuit_index=2)
        self.assertEqual(task.circuit, "adder")
        command = _train_command(task, train_dir=Path("/tmp/train"), python_executable="python")
        overrides = [value for value in command if "=" in value]
        self.assertEqual(
            overrides,
            [
                "run_name=baseline_ppo_adder",
                "dataset_cfg=cfg/data/epfl_arithmetic/adder.yaml",
                "hydra.run.dir=/tmp/train",
            ],
        )

    def test_next_attempt_is_always_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory)
            (task_root / "attempt_001").mkdir()
            self.assertEqual(_next_attempt_dir(task_root).name, "attempt_002")

    def test_run_task_creates_a_new_attempt_on_every_invocation(self) -> None:
        counts = {"checkpoint_count": 10, "evaluation_seed_count": 10, "unique_sample_count": 20_000}
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.baseline_evaluation._run_command", return_value=0
        ), patch("src.baseline_evaluation._validate_training_artifacts"), patch(
            "src.baseline_evaluation._sample_task"
        ), patch("src.baseline_evaluation.validate_attempt", return_value=counts):
            first = run_task(
                self.protocol,
                method="reinforce",
                circuit_index=0,
                artifact_root=Path(directory),
                project_commit="a" * 40,
                python_executable="python",
                device_name="cpu",
            )
            second = run_task(
                self.protocol,
                method="reinforce",
                circuit_index=0,
                artifact_root=Path(directory),
                project_commit="a" * 40,
                python_executable="python",
                device_name="cpu",
            )
        self.assertEqual(Path(first["attempt_dir"]).name, "attempt_001")
        self.assertEqual(Path(second["attempt_dir"]).name, "attempt_002")


class BaselineEvaluationArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = BaselineEvaluationProtocol.load(PROTOCOL)
        self.task = self.protocol.task(method="reinforce", circuit_index=0)

    def _write_complete_attempt(self, attempt: Path) -> None:
        train = attempt / "train"
        config: dict = {
            "dataset_cfg": self.task.dataset_cfg,
            "run_name": f"baseline_{self.task.method}_{self.task.circuit}",
        }
        for path, value in self.task.expected.items():
            _set_path(config, path, value)
        (train / ".hydra").mkdir(parents=True)
        (train / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

        report_runs = []
        for run_idx, seed in enumerate(self.protocol.training_seeds):
            checkpoint = train / "saved_models" / f"run_{run_idx}" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"run_idx": run_idx, "seed": seed, "policy_state_dict": {}}, checkpoint)
            report_runs.append({"run_idx": run_idx, "seed": seed})
        (train / self.task.report_file).write_text(
            json.dumps({"algorithm": self.task.report_algorithm, "runs": report_runs}),
            encoding="utf-8",
        )

        raw_frames = []
        for evaluation_seed in self.protocol.evaluation_seeds:
            rows = []
            for run_id, training_seed in enumerate(self.protocol.training_seeds):
                checkpoint = train / "saved_models" / f"run_{run_id}" / "last.pt"
                for sample_id in range(self.protocol.max_samples):
                    rows.append(
                        {
                            "method": self.task.method,
                            "circuit_name": self.task.circuit,
                            "circuit": self.task.circuit_path,
                            "run_id": run_id,
                            "training_seed": training_seed,
                            "evaluation_seed": evaluation_seed,
                            "sample_id": sample_id,
                            "size": 1000 - sample_id,
                            "depth": 100 - (sample_id % 20),
                            "source_checkpoint": str(checkpoint),
                        }
                    )
            frame = pd.DataFrame(rows)
            raw_path = attempt / "evaluation" / "raw" / f"seed_{evaluation_seed:02d}.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(raw_path, index=False)
            raw_frames.append(frame)

        raw = pd.concat(raw_frames, ignore_index=True)
        reference = {
            "method": self.task.method,
            "circuit_name": self.task.circuit,
            "circuit": self.task.circuit_path,
            "run_id": None,
            "training_seed": None,
            "evaluation_seed": None,
            "sample_id": None,
            "size": 1000,
            "depth": 100,
            "source_checkpoint": None,
        }
        pd.concat([pd.DataFrame([reference]), raw], ignore_index=True).to_csv(
            attempt / "points.csv", index=False
        )
        budget_dir = attempt / "evaluation" / "budgets"
        budget_dir.mkdir(parents=True)
        for budget in self.protocol.sample_budgets:
            view = raw.loc[raw["sample_id"] < budget].copy()
            view["sample_budget"] = budget
            budget_reference = {**reference, "sample_budget": budget}
            pd.concat([pd.DataFrame([budget_reference]), view], ignore_index=True).to_csv(
                budget_dir / f"points_n{budget:03d}.csv", index=False
            )

    def test_complete_attempt_and_nested_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_001"
            self._write_complete_attempt(attempt)
            result = validate_attempt(self.task, attempt_dir=attempt, protocol=self.protocol)
        self.assertEqual(result["checkpoint_count"], 10)
        self.assertEqual(result["evaluation_seed_count"], 10)
        self.assertEqual(result["unique_sample_count"], 20_000)

    def test_rejects_budget_view_that_is_not_exact_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_001"
            self._write_complete_attempt(attempt)
            path = attempt / "evaluation" / "budgets" / "points_n010.csv"
            frame = pd.read_csv(path)
            sample_index = frame.index[frame["run_id"].notna()][0]
            frame.loc[sample_index, "size"] = -1
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "not the exact nested prefix"):
                validate_attempt(self.task, attempt_dir=attempt, protocol=self.protocol)


class PairedSamplingTests(unittest.TestCase):
    def test_explicit_evaluation_seed_is_reused_across_training_runs(self) -> None:
        try:
            from src.sample_exp import sample_paired_evaluation_seed
        except ModuleNotFoundError as exc:
            self.skipTest(f"sampling runtime dependency unavailable: {exc}")

        calls: list[int] = []

        def fake_sample(**kwargs):
            calls.append(int(kwargs["seed"]))
            return [{"size": 10 + index, "depth": 5} for index in range(kwargs["num_samples"])]

        fake_cfg = {"num_steps": 20, "seed": 0}
        checkpoints = [(0, Path("/tmp/run_0.pt")), (1, Path("/tmp/run_1.pt"))]
        with patch("src.sample_exp._load_cfg", return_value=fake_cfg), patch(
            "src.sample_exp._discover_run_checkpoints", return_value=checkpoints
        ), patch("src.sample_exp._sample_trajectories", side_effect=fake_sample):
            frame = sample_paired_evaluation_seed(
                experiment_dir=Path("/tmp/experiment"),
                circuit_path=Path("/tmp/C1355.blif"),
                method="reinforce",
                circuit_name="C1355",
                num_samples=200,
                evaluation_seed=7,
                device=torch.device("cpu"),
            )
        self.assertEqual(calls, [7, 7])
        self.assertEqual(len(frame), 400)
        self.assertEqual(set(frame["evaluation_seed"]), {7})
        self.assertEqual(set(frame["training_seed"]), {0, 1})
        self.assertEqual(list(frame.loc[frame["training_seed"] == 0, "sample_id"]), list(range(200)))


if __name__ == "__main__":
    unittest.main()
