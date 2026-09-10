from __future__ import annotations

import sys
import types
import unittest
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

pyspiel_stub = types.ModuleType("pyspiel")
pyspiel_stub.State = object
sys.modules.setdefault("pyspiel", pyspiel_stub)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.sampler import _epsilon_mixed_probs, sample_tb_trajectories
from src.algorithms.gflownet_tb.trainer import (
    TBGFlowNetTrainer,
    _build_tb_optimizer,
    _tb_exploration_epsilon,
    calibrated_log_z_target,
)
from src.algorithms.gflownet_tb.eval import evaluate_tb
from src.metrics import TensorBoardLogger


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)


class TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


def _policy() -> TBGFlowNetPolicy:
    return TBGFlowNetPolicy(encoder=TinyEncoder(), head=TinyHead(), num_actions=2)


class CalibrationPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def forward(self, observations):
        return self.logits.expand(len(observations), -1)

    def log_prob_legal_batch(self, logits, legal_rows, actions):
        indices = torch.as_tensor(actions, dtype=torch.long, device=logits.device)[:, None]
        return torch.log_softmax(logits, dim=-1).gather(1, indices).squeeze(1)


class FakeTensorBoard:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, float]]] = []
        self.closed = False

    def add_scalars(self, step: int, scalars: dict[str, float]) -> None:
        self.calls.append((int(step), dict(scalars)))

    def close(self) -> None:
        self.closed = True


def _fake_trajectory(policy, file_path: str, action: int):
    observation = object()
    logits = policy([observation])
    log_pf = policy.log_prob_legal_batch(logits, [[0, 1]], [action]).sum()
    return SimpleNamespace(
        file_path=file_path,
        steps=[SimpleNamespace(observation=observation, legal_actions=[0, 1], action=action)],
        initial_size=100,
        initial_depth=100,
        final_size=90 - action,
        final_depth=90,
        final_return=0.1,
        comparable_return=0.1,
        terminal_reward=1.0,
        log_reward=1.0 + action,
        log_pf_sum=log_pf,
        log_pb_sum=torch.tensor(0.0),
    )


class GFlowNetTBTest(unittest.TestCase):
    def test_optimizer_uses_separate_log_z_learning_rate(self) -> None:
        policy = _policy()

        optimizer = _build_tb_optimizer(
            policy,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
        )

        self.assertEqual(len(optimizer.param_groups), 2)
        policy_group, log_z_group = optimizer.param_groups
        self.assertEqual(policy_group["lr"], 1e-3)
        self.assertEqual(log_z_group["lr"], 1e-2)

        policy_param_ids = {id(param) for param in policy_group["params"]}
        log_z_param_ids = {id(param) for param in log_z_group["params"]}
        encoder_param_ids = {id(param) for param in policy.encoder.parameters()}
        head_param_ids = {id(param) for param in policy.head.parameters()}

        self.assertEqual(log_z_param_ids, {id(policy.log_z)})
        self.assertNotIn(id(policy.log_z), policy_param_ids)
        self.assertTrue(encoder_param_ids.issubset(policy_param_ids))
        self.assertTrue(head_param_ids.issubset(policy_param_ids))

    def test_optimizer_rejects_nonpositive_learning_rates(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _build_tb_optimizer(
                _policy(),
                learning_rate=0.0,
                log_z_learning_rate=1e-2,
            )

        with self.assertRaisesRegex(ValueError, "log_z_learning_rate"):
            _build_tb_optimizer(
                _policy(),
                learning_rate=1e-3,
                log_z_learning_rate=-1e-2,
            )

    def test_exploration_schedule_warmup_then_linear_decay(self) -> None:
        kwargs = {
            "episodes": 5,
            "enabled": True,
            "epsilon_start": 0.5,
            "epsilon_end": 0.05,
            "warmup_episodes": 2,
            "decay_episodes": 3,
        }

        self.assertAlmostEqual(_tb_exploration_epsilon(episode=1, **kwargs), 0.5)
        self.assertAlmostEqual(_tb_exploration_epsilon(episode=2, **kwargs), 0.5)
        self.assertAlmostEqual(_tb_exploration_epsilon(episode=3, **kwargs), 0.35)
        self.assertAlmostEqual(_tb_exploration_epsilon(episode=5, **kwargs), 0.05)
        self.assertAlmostEqual(_tb_exploration_epsilon(episode=6, **kwargs), 0.05)

    def test_exploration_schedule_defaults_decay_to_end_of_training(self) -> None:
        self.assertAlmostEqual(
            _tb_exploration_epsilon(
                episode=200,
                episodes=200,
                enabled=True,
                epsilon_start=0.5,
                epsilon_end=0.01,
                warmup_episodes=20,
                decay_episodes=None,
            ),
            0.01,
        )

    def test_exploration_schedule_disabled_and_zero_are_valid(self) -> None:
        self.assertEqual(
            _tb_exploration_epsilon(
                episode=1,
                episodes=10,
                enabled=False,
                epsilon_start=0.5,
                epsilon_end=0.01,
                warmup_episodes=2,
                decay_episodes=None,
            ),
            0.0,
        )
        self.assertEqual(
            _tb_exploration_epsilon(
                episode=1,
                episodes=10,
                enabled=True,
                epsilon_start=0.0,
                epsilon_end=0.0,
                warmup_episodes=0,
                decay_episodes=1,
            ),
            0.0,
        )

    def test_exploration_schedule_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "exploration_epsilon_start"):
            _tb_exploration_epsilon(
                episode=1,
                episodes=10,
                enabled=True,
                epsilon_start=1.1,
                epsilon_end=0.0,
                warmup_episodes=0,
                decay_episodes=1,
            )
        with self.assertRaisesRegex(ValueError, "exploration_epsilon_end"):
            _tb_exploration_epsilon(
                episode=1,
                episodes=10,
                enabled=True,
                epsilon_start=0.5,
                epsilon_end=-0.1,
                warmup_episodes=0,
                decay_episodes=1,
            )
        with self.assertRaisesRegex(ValueError, "exploration_decay_episodes"):
            _tb_exploration_epsilon(
                episode=1,
                episodes=10,
                enabled=True,
                epsilon_start=0.5,
                epsilon_end=0.0,
                warmup_episodes=0,
                decay_episodes=0,
            )

    def test_epsilon_mixed_probs_policy_and_uniform_endpoints(self) -> None:
        policy_probs = torch.tensor(
            [
                [0.7, 0.0, 0.3],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        legal_actions = [[0, 2], [1]]

        self.assertTrue(torch.allclose(_epsilon_mixed_probs(policy_probs, legal_actions, 0.0), policy_probs))
        expected_uniform = torch.tensor(
            [
                [0.5, 0.0, 0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(_epsilon_mixed_probs(policy_probs, legal_actions, 1.0), expected_uniform))

        mixed = _epsilon_mixed_probs(policy_probs, legal_actions, 0.25)
        self.assertTrue(torch.allclose(mixed[:, 1], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.allclose(mixed.sum(dim=1), torch.ones(2)))

    def test_sampler_and_eval_keep_epsilon_training_only_by_default(self) -> None:
        self.assertEqual(sample_tb_trajectories.__kwdefaults__["epsilon_uniform"], 0.0)
        self.assertNotIn("epsilon_uniform", signature(evaluate_tb).parameters)

    def test_calibrated_log_z_target(self) -> None:
        policy = CalibrationPolicy()
        trajectories = [
            _fake_trajectory(policy, "c.blif", 0),
            _fake_trajectory(policy, "c.blif", 1),
        ]
        target = calibrated_log_z_target(policy, trajectories)
        self.assertAlmostEqual(target, 1.5 + float(torch.log(torch.tensor(2.0))), places=6)

    def test_src_run_trainer_logs_calibration_eval_and_discovery_metrics(self) -> None:
        policy = CalibrationPolicy()
        logger = FakeTensorBoard()
        trainer = TBGFlowNetTrainer(
            policy=policy,
            reward_class=object,
            train_circuits=["c.blif"],
            test_circuits=["c.blif"],
            resyn2_baselines={
                "c.blif": {
                    "resyn2_variants": {
                        "resyn2_1": {"initial_size": 100, "initial_depth": 100}
                    }
                }
            },
            device=torch.device("cpu"),
            seed=0,
        )
        trainer._tb = logger
        sample_index = 0

        def fake_sample(**kwargs):
            nonlocal sample_index
            rows = []
            for file_path in kwargs["file_paths"]:
                rows.append(_fake_trajectory(policy, file_path, sample_index % 2))
                sample_index += 1
            return rows

        evaluation = {
            "mean_final_return": 0.1,
            "mean_comparable_return": 0.1,
            "mean_size_reduction": 10.0,
            "mean_depth_reduction": 10.0,
            "mean_size_reduction_pct": 10.0,
            "mean_terminal_reward": 1.0,
            "mean_final_size": 90.0,
            "mean_final_depth": 90.0,
            "mean_final_qor": 8100.0,
            "best_final_return": 0.1,
            "best_comparable_return": 0.1,
            "win_rate_vs_resyn2_1": 1.0,
            "win_rate_vs_resyn2_2": 1.0,
            "mean_normalized_improvement_vs_resyn2_2": 0.1,
        }
        with patch("src.algorithms.gflownet_tb.trainer.sample_tb_trajectories", side_effect=fake_sample):
            with patch.object(trainer, "evaluate", return_value=evaluation):
                result = trainer.train(
                    episodes=2,
                    num_steps=1,
                    eval_every=1,
                    learning_rate=0.001,
                    log_z_learning_rate=0.01,
                    trajectories_per_episode=2,
                    reward_alpha=4.0,
                    reward_eps=1e-8,
                    reward_improvement_clip=2.0,
                    exploration_epsilon_enabled=True,
                    exploration_epsilon_start=0.5,
                    exploration_epsilon_end=0.01,
                    exploration_warmup_episodes=1,
                    exploration_decay_episodes=None,
                    best_of_eval_rollouts=1,
                    log_z_initialization="calibrated",
                    calibration_trajectories=2,
                    calibration_epsilon=0.5,
                    discovery_metrics_enabled=True,
                    discovery_emit_every_trajectories=1,
                )

        self.assertEqual(result["training_summary"]["calibration_trajectories"], 2)
        self.assertEqual(result["training_summary"]["new_on_policy_trajectories"], 2)
        self.assertEqual(result["training_summary"]["training_trajectories"], 4)
        tags = {key for _, scalars in logger.calls for key in scalars}
        self.assertIn("train/log_z_calibration_target", tags)
        self.assertIn("train/training_trajectories", tags)
        self.assertIn("eval/mean_final_return", tags)
        self.assertIn("discovery/mean_hypervolume", tags)
        self.assertTrue(logger.closed)

        with TemporaryDirectory() as directory:
            persisted = TensorBoardLogger(Path(directory))
            for step, scalars in logger.calls:
                persisted.add_scalars(step, scalars)
            persisted.close()
            events = EventAccumulator(directory)
            events.Reload()
            persisted_tags = set(events.Tags()["scalars"])
        self.assertTrue(
            {
                "train/log_z_calibration_target",
                "train/policy_loss",
                "eval/mean_final_return",
                "discovery/mean_hypervolume",
                "discovery/mean_nondominated_count",
            }.issubset(persisted_tags)
        )


if __name__ == "__main__":
    unittest.main()
