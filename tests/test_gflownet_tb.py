from __future__ import annotations

import sys
import types
import unittest
from inspect import signature
from pathlib import Path

import torch
from torch import nn

pyspiel_stub = types.ModuleType("pyspiel")
pyspiel_stub.State = object
sys.modules.setdefault("pyspiel", pyspiel_stub)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.sampler import _epsilon_mixed_probs, sample_tb_trajectories
from src.algorithms.gflownet_tb.trainer import _build_tb_optimizer, _tb_exploration_epsilon
from src.algorithms.gflownet_tb.eval import evaluate_tb


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


if __name__ == "__main__":
    unittest.main()
