from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.drills_a2c.loss import drills_a2c_loss
from src.algorithms.drills_a2c.types import DrillsA2CStep
from src.algorithms.reinforce.policy import Policy
from src.models.rewards import DrillsSizeDepthReward
from src.utils import Observation


class FixedPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(num_actions=2)
        self.logits = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self, obs):
        if isinstance(obs, list):
            return self.logits.unsqueeze(0).repeat(len(obs), 1)
        return self.logits.unsqueeze(0)


class ObsValue(nn.Module):
    def forward(self, obs):
        if isinstance(obs, list):
            return torch.stack([item.obs_tensor[0] for item in obs])
        return obs.obs_tensor[0].unsqueeze(0)


def _obs(value: float) -> Observation:
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
    )
    return Observation(
        obs_tensor=torch.tensor([value], dtype=torch.float32),
        graph=graph,
        legal_actions=[0, 1],
    )


class DrillsA2CTest(unittest.TestCase):
    def test_drills_size_depth_reward_table(self) -> None:
        cases = [
            (90, 10, 100, 10, 3.0),
            (100, 10, 100, 10, 0.0),
            (110, 10, 100, 10, -1.0),
            (90, 11, 100, 12, 3.0),
            (100, 11, 100, 12, 2.0),
            (110, 11, 100, 12, 1.0),
            (90, 12, 100, 12, 2.0),
            (100, 12, 100, 12, 0.0),
            (110, 12, 100, 12, -2.0),
            (90, 13, 100, 12, -1.0),
            (100, 13, 100, 12, -2.0),
            (110, 13, 100, 12, -3.0),
        ]
        reward = DrillsSizeDepthReward(initial_size=100, initial_depth=10, depth_constraint_ratio=1.0)
        for size, depth, prev_size, prev_depth, expected in cases:
            with self.subTest(size=size, depth=depth, prev_size=prev_size, prev_depth=prev_depth):
                self.assertEqual(reward(size, depth, prev_size, prev_depth), expected)

    def test_drills_a2c_loss_uses_one_step_td_targets(self) -> None:
        policy = FixedPolicy()
        value = ObsValue()
        steps = [
            DrillsA2CStep(observation=_obs(1.0), action=0, reward=1.0, next_observation=_obs(2.0), done=False),
            DrillsA2CStep(observation=_obs(2.0), action=1, reward=-1.0, next_observation=_obs(3.0), done=True),
        ]

        loss = drills_a2c_loss(
            policy=policy,
            value_network=value,
            steps=steps,
            gamma=0.5,
            value_loss_coef=0.5,
            entropy_beta=0.0,
            normalize_advantages=False,
        )

        self.assertTrue(torch.isclose(loss.mean_target, torch.tensor(0.5)))
        self.assertTrue(torch.isclose(loss.mean_advantage, torch.tensor(-1.0)))
        self.assertTrue(torch.isclose(loss.critic_loss, torch.tensor(2.5)))
        self.assertTrue(torch.isclose(loss.actor_loss, torch.tensor(-math.log(2.0))))
        self.assertTrue(torch.isclose(loss.entropy, torch.tensor(math.log(2.0))))


if __name__ == "__main__":
    unittest.main()
