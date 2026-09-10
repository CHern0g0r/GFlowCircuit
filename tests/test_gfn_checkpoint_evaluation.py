from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from src.gfn_checkpoint_evaluation import (
    GFNCheckpointEvaluationProtocol,
    _sample_task,
    run_task,
    validate_attempt,
    validate_protocol,
    validate_training_artifacts,
)


PROTOCOL = Path("cfg/exp/gfn_checkpoint_evaluation/protocol.yaml")


def _set_path(mapping: dict, dotted_path: str, value: object) -> None:
    current = mapping
    components = dotted_path.split(".")
    for component in components[:-1]:
        current = current.setdefault(component, {})
    current[components[-1]] = value


def _write_training_fixture(
    protocol: GFNCheckpointEvaluationProtocol,
    *,
    root: Path,
    circuit_index: int = 0,
) -> None:
    task = protocol.task(circuit_index)
    training = protocol.data["training"]
    provenance = {
        "project_commit": training["project_commit"],
        "config": training["config"],
        "config_sha256": training["config_sha256"],
        "slurm_array_job_id": training["slurm_array_job_id"],
        "slurm_array_task_id": str(task.provenance_task_index),
    }
    (root / f"provenance-{task.provenance_task_index}.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / f"provenance-{task.provenance_task_index}.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in provenance.items()),
        encoding="utf-8",
    )

    circuit_root = root / "circuits" / task.circuit
    config: dict = {"dataset_cfg": task.dataset_cfg}
    for dotted_path, value in training["expected"].items():
        _set_path(config, dotted_path, value)
    config_path = circuit_root / "hydra" / ".hydra" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    runs = []
    for run_idx, seed in enumerate(protocol.training_seeds):
        checkpoint = circuit_root / "saved_models" / f"run_{run_idx}" / "last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "run_idx": run_idx,
                "seed": seed,
                "policy_state_dict": {"weight": torch.tensor([float(run_idx)])},
                "tb_training": {
                    "log_z_initialization": "calibrated",
                    "calibration_target": 1.5,
                    "calibration_trajectories": 64,
                    "new_on_policy_trajectories": 736,
                    "training_trajectories": 800,
                    "training_presentations": 800,
                    "optimizer_updates": 200,
                },
            },
            checkpoint,
        )
        runs.append({"run_idx": run_idx, "seed": seed})
    (circuit_root / training["report_file"]).write_text(
        json.dumps({"algorithm": "gflownet_tb", "runs": runs}),
        encoding="utf-8",
    )


def _write_attempt(
    protocol: GFNCheckpointEvaluationProtocol,
    *,
    training_root: Path,
    attempt: Path,
    circuit_index: int = 0,
) -> None:
    task = protocol.task(circuit_index)
    raw_frames: list[pd.DataFrame] = []
    for evaluation_seed in protocol.evaluation_seeds:
        rows = []
        for run_id, training_seed in enumerate(protocol.training_seeds):
            checkpoint = (
                training_root
                / "circuits"
                / task.circuit
                / "saved_models"
                / f"run_{run_id}"
                / "last.pt"
            ).resolve()
            for sample_id in range(protocol.max_samples):
                rows.append(
                    {
                        "method": "gflownet_tb",
                        "circuit_name": task.circuit,
                        "circuit": str((protocol.repo_root / task.circuit_path).resolve()),
                        "run_id": run_id,
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "sample_id": sample_id,
                        "size": 1000 - sample_id,
                        "depth": 100 - sample_id % 20,
                        "source_checkpoint": str(checkpoint),
                    }
                )
        frame = pd.DataFrame(rows)
        path = attempt / "raw" / f"seed_{evaluation_seed:02d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        raw_frames.append(frame)

    raw = pd.concat(raw_frames, ignore_index=True)
    reference = {
        "method": "gflownet_tb",
        "circuit_name": task.circuit,
        "circuit": str((protocol.repo_root / task.circuit_path).resolve()),
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
    budget_dir = attempt / "budgets"
    budget_dir.mkdir(parents=True, exist_ok=True)
    for budget in protocol.sample_budgets:
        view = raw.loc[raw["sample_id"] < budget].copy()
        view["sample_budget"] = budget
        budget_reference = {**reference, "sample_budget": budget}
        pd.concat([pd.DataFrame([budget_reference]), view], ignore_index=True).to_csv(
            budget_dir / f"points_n{budget:03d}.csv", index=False
        )


class GFNCheckpointProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = GFNCheckpointEvaluationProtocol.load(PROTOCOL)

    def test_exact_matrix_and_rollout_count(self) -> None:
        result = validate_protocol(self.protocol)
        self.assertEqual(self.protocol.circuits, ("C1355", "C5315", "adder", "apex1", "bc0", "dalu", "k2", "max"))
        self.assertEqual(self.protocol.training_seeds, tuple(range(10)))
        self.assertEqual(self.protocol.evaluation_seeds, tuple(range(10)))
        self.assertEqual(self.protocol.sample_budgets, (10, 50, 100, 200))
        self.assertEqual(result["checkpoints"], 80)
        self.assertEqual(result["unique_evaluation_rollouts"], 160_000)

    def test_split_training_layout_and_nonfinite_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_training_fixture(self.protocol, root=root)
            result = validate_training_artifacts(
                self.protocol,
                self.protocol.task(0),
                training_artifact_root=root,
            )
            self.assertEqual(result["checkpoint_count"], 10)
            self.assertEqual(
                Path(result["config_path"]),
                (root / "circuits/C1355/hydra/.hydra/config.yaml").resolve(),
            )

            checkpoint = root / "circuits/C1355/saved_models/run_0/last.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["policy_state_dict"]["weight"] = torch.tensor([float("nan")])
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_training_artifacts(
                    self.protocol,
                    self.protocol.task(0),
                    training_artifact_root=root,
                )


class GFNCheckpointSamplingTests(unittest.TestCase):
    def test_split_sampler_reuses_evaluation_seed_and_forwards_batch_size(self) -> None:
        try:
            from src.sample_exp import sample_paired_evaluation_seed_from_paths
        except ModuleNotFoundError as exc:
            self.skipTest(f"sampling runtime dependency unavailable: {exc}")

        calls: list[tuple[int, int | None]] = []

        def fake_sample(**kwargs):
            calls.append((int(kwargs["seed"]), kwargs["gflownet_batch_size"]))
            return [{"size": 10 + index, "depth": 5} for index in range(kwargs["num_samples"])]

        with patch("src.sample_exp._load_cfg", return_value={"num_steps": 20, "seed": 0}), patch(
            "src.sample_exp._discover_run_checkpoints_from_root",
            return_value=[(0, Path("/tmp/run_0.pt")), (1, Path("/tmp/run_1.pt"))],
        ), patch("src.sample_exp._sample_trajectories", side_effect=fake_sample):
            frame = sample_paired_evaluation_seed_from_paths(
                config_path=Path("/tmp/config.yaml"),
                saved_models_dir=Path("/tmp/saved_models"),
                circuit_path=Path("/tmp/C1355.blif"),
                method="gflownet_tb",
                circuit_name="C1355",
                num_samples=200,
                evaluation_seed=7,
                device=torch.device("cpu"),
                gflownet_batch_size=20,
            )
        self.assertEqual(calls, [(7, 20), (7, 20)])
        self.assertEqual(len(frame), 400)
        self.assertEqual(list(frame.loc[frame["training_seed"] == 0, "sample_id"]), list(range(200)))

    def test_gflownet_sampling_is_chunked_in_order(self) -> None:
        try:
            from src.sample_exp import _sample_trajectories
        except ModuleNotFoundError as exc:
            self.skipTest(f"sampling runtime dependency unavailable: {exc}")

        batches: list[int] = []

        class Trajectory:
            def __init__(self, value: int) -> None:
                self.final_size = value
                self.final_depth = value + 1

        def fake_batch(**kwargs):
            size = len(kwargs["file_paths"])
            start = sum(batches)
            batches.append(size)
            return [Trajectory(start + index) for index in range(size)]

        loaded = {
            "algorithm": "gflownet_tb",
            "policy": object(),
            "reward_class": object(),
            "mo_reward_class": None,
            "pcn": {},
            "available_actions": list(range(7)),
        }
        with patch("src.sample_exp._load_policy", return_value=loaded), patch(
            "src.algorithms.gflownet_tb.sampler.sample_tb_trajectories", side_effect=fake_batch
        ):
            rows = _sample_trajectories(
                checkpoint_path=Path("/tmp/checkpoint.pt"),
                cfg=OmegaConf.create(
                    {"tb": {"reward_alpha": 4.0, "reward_eps": 1e-8, "reward_improvement_clip": 2.0}}
                ),
                circuit_path=Path("/tmp/C1355.blif"),
                num_steps=20,
                num_samples=45,
                device=torch.device("cpu"),
                seed=3,
                pcn_sampling_mode="target",
                pcn_zero_variance_jitter=0.05,
                gflownet_batch_size=20,
            )
        self.assertEqual(batches, [20, 20, 5])
        self.assertEqual([row["size"] for row in rows], list(range(45)))


class GFNCheckpointArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = GFNCheckpointEvaluationProtocol.load(PROTOCOL)
        self.task = self.protocol.task(0)

    def test_counts_prefixes_publication_and_idempotence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_root = root / "training"
            job_root = root / "job"
            _write_training_fixture(self.protocol, root=training_root)

            def fake_sample(protocol, task, *, attempt_dir, training_paths, device):
                _write_attempt(protocol, training_root=training_root, attempt=attempt_dir)

            with patch("src.gfn_checkpoint_evaluation._sample_task", side_effect=fake_sample):
                first = run_task(
                    self.protocol,
                    circuit_index=0,
                    training_artifact_root=training_root,
                    job_artifact_root=job_root,
                    evaluator_commit="a" * 40,
                    device_name="cpu",
                )
            self.assertEqual(first["state"], "complete")
            self.assertEqual(first["unique_sample_count"], 20_000)
            canonical = training_root / "circuits/C1355/points.csv"
            self.assertTrue(canonical.is_file())
            self.assertEqual(len(pd.read_csv(canonical)), 20_001)

            second = run_task(
                self.protocol,
                circuit_index=0,
                training_artifact_root=training_root,
                job_artifact_root=job_root,
                evaluator_commit="a" * 40,
                device_name="cpu",
            )
            self.assertEqual(second["state"], "already_complete")
            self.assertEqual(len(list((training_root / "circuits/C1355/evaluation").glob("attempt_*"))), 1)

    def test_corrupt_budget_prefix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            training_root = Path(directory) / "training"
            attempt = training_root / "circuits/C1355/evaluation/attempt_001"
            _write_attempt(self.protocol, training_root=training_root, attempt=attempt)
            path = attempt / "budgets/points_n010.csv"
            frame = pd.read_csv(path)
            sample_index = frame.index[frame["run_id"].notna()][0]
            frame.loc[sample_index, "size"] = -1
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "not the exact nested prefix"):
                validate_attempt(
                    self.protocol,
                    self.task,
                    attempt_dir=attempt,
                    training_artifact_root=training_root,
                )


if __name__ == "__main__":
    unittest.main()
