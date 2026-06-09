from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

pyspiel_stub = types.ModuleType("pyspiel")
pyspiel_stub.State = object
sys.modules.setdefault("pyspiel", pyspiel_stub)

import torch
from torch import nn
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.ppo.loss import compute_gae, ppo_minibatch_loss
from src.algorithms.ppo.types import PPOTransition
from src.algorithms.reinforce.policy import Policy
from src.utils import Observation


class FixedPolicy(Policy):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__(num_actions=2)
        self.logits = nn.Parameter(logits.clone().detach().float())

    def forward(self, obs):
        if isinstance(obs, list):
            return self.logits.unsqueeze(0).repeat(len(obs), 1)
        return self.logits.unsqueeze(0)


class FixedValue(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, obs):
        if isinstance(obs, list):
            return self.value.repeat(len(obs))
        return self.value.unsqueeze(0)


def _obs() -> Observation:
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
    )
    return Observation(
        obs_tensor=torch.zeros(1, dtype=torch.float32),
        graph=graph,
        legal_actions=[0, 1],
    )


class PPOTest(unittest.TestCase):
    def test_gae_lambda_zero_is_one_step_td(self) -> None:
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([False, True])
        values = torch.tensor([0.5, 1.0])
        next_values = torch.tensor([1.0, 0.0])

        advantages, returns = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=0.9,
            gae_lambda=0.0,
        )

        expected_advantages = torch.tensor([1.4, 1.0])
        self.assertTrue(torch.allclose(advantages, expected_advantages))
        self.assertTrue(torch.allclose(returns, expected_advantages + values))

    def test_gae_lambda_one_is_monte_carlo_style(self) -> None:
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([False, True])
        values = torch.tensor([0.5, 1.0])
        next_values = torch.tensor([1.0, 0.0])

        advantages, returns = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=0.9,
            gae_lambda=1.0,
        )

        expected_returns = torch.tensor([2.8, 2.0])
        self.assertTrue(torch.allclose(returns, expected_returns))
        self.assertTrue(torch.allclose(advantages, expected_returns - values))

    def test_gae_mid_lambda_matches_hand_computation(self) -> None:
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([False, True])
        values = torch.tensor([0.5, 1.0])
        next_values = torch.tensor([1.0, 0.0])

        advantages, _ = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=0.9,
            gae_lambda=0.95,
        )

        expected = torch.tensor([1.4 + 0.9 * 0.95 * 1.0, 1.0])
        self.assertTrue(torch.allclose(advantages, expected))

    def test_ppo_clip_fraction_detects_large_ratio(self) -> None:
        obs = _obs()
        transition = PPOTransition(
            observation=obs,
            action=0,
            reward=1.0,
            next_observation=obs,
            done=True,
            old_log_prob=torch.tensor(math.log(0.5)),
            old_value=torch.tensor(0.0),
            next_value=torch.tensor(0.0),
        )
        policy = FixedPolicy(torch.tensor([2.0, 0.0]))
        value = FixedValue(0.0)

        stats = ppo_minibatch_loss(
            policy=policy,  # type: ignore[arg-type]
            value_network=value,
            transitions=[transition],
            advantages=torch.tensor([1.0]),
            returns=torch.tensor([1.0]),
            indices=torch.tensor([0]),
            clip_eps=0.2,
            value_loss_coef=0.5,
            entropy_beta=0.0,
        )

        self.assertTrue(torch.isclose(stats.clip_fraction, torch.tensor(1.0)))
        self.assertLess(float(stats.actor_loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
