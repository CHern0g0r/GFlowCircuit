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

from src.algorithms.pcn.archive import PCNArchive
from src.algorithms.pcn.loss import pcn_cross_entropy_loss
from src.algorithms.pcn.policy import PCNPolicy
from src.algorithms.pcn.types import PCNDatapoint, PCNStep, PCNTrajectory
from src.models.Linear import LinearHead
from src.models.mo_rewards import (
    SizeDepthImprovementReward,
    crowding_distance,
    discounted_vector_returns,
    hypervolume_2d_max,
    non_dominated_mask_max,
)
from src.utils import Observation


def _obs(value: float = 0.0, legal_actions: list[int] | None = None) -> Observation:
    graph = Data(
        x=torch.zeros((1, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
    )
    return Observation(
        obs_tensor=torch.tensor([value], dtype=torch.float32),
        graph=graph,
        legal_actions=legal_actions or [0, 1],
    )


class ObsEncoder(nn.Module):
    def forward(self, obs):
        if isinstance(obs, list):
            return torch.stack([item.obs_tensor for item in obs], dim=0)
        return obs.obs_tensor.unsqueeze(0)


def _trajectory(return_vec: torch.Tensor, action: int = 0) -> PCNTrajectory:
    step = PCNStep(
        observation=_obs(0.0),
        action=action,
        legal_actions=[0, 1],
        reward_vec=return_vec.clone(),
    )
    return PCNTrajectory(
        file_path="dummy",
        steps=[step],
        initial_size=10,
        initial_depth=5,
        final_size=9,
        final_depth=4,
        return_vec=return_vec.clone(),
        horizon=1,
    )


class PCNTest(unittest.TestCase):
    def test_size_depth_improvement_reward(self) -> None:
        reward = SizeDepthImprovementReward(initial_size=100, initial_depth=10)
        got = reward(size=90, depth=8, prev_size=95, prev_depth=9)
        self.assertTrue(torch.allclose(got, torch.tensor([0.05, 0.1])))

    def test_discounted_vector_returns(self) -> None:
        rewards = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        got = discounted_vector_returns(rewards, gamma=0.5)
        expected = torch.tensor([[1.25, 1.25], [0.5, 2.5], [1.0, 1.0]])
        self.assertTrue(torch.allclose(got, expected))

    def test_non_dominated_mask_max(self) -> None:
        points = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.5, 2.0], [0.5, 0.5]])
        got = non_dominated_mask_max(points)
        self.assertEqual(got.tolist(), [False, True, True, False])

    def test_crowding_and_hypervolume(self) -> None:
        points = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        crowding = crowding_distance(points)
        self.assertEqual(tuple(crowding.shape), (3,))
        self.assertGreater(float(crowding[0]), 0.0)
        self.assertTrue(math.isclose(hypervolume_2d_max(points), 0.25, rel_tol=1e-6))

    def test_pcn_loss_uses_legal_actions(self) -> None:
        policy = PCNPolicy(
            encoder=ObsEncoder(),
            head=LinearHead(obs_dim=4, num_actions=3),
            num_actions=3,
            encoder_out_dim=1,
            objective_dim=2,
            num_steps=2,
            embedding_dim=4,
        )
        batch = [
            PCNDatapoint(
                observation=_obs(1.0, legal_actions=[1, 2]),
                action=1,
                legal_actions=[1, 2],
                desired_return=torch.tensor([0.1, 0.2]),
                desired_horizon=1.0,
            )
        ]
        loss = pcn_cross_entropy_loss(policy, batch)
        self.assertTrue(torch.isfinite(loss))
        batch[0].action = 0
        with self.assertRaises(ValueError):
            pcn_cross_entropy_loss(policy, batch)

    def test_pcn_policy_batches_and_condition_changes_logits(self) -> None:
        policy = PCNPolicy(
            encoder=ObsEncoder(),
            head=LinearHead(obs_dim=4, num_actions=2),
            num_actions=2,
            encoder_out_dim=1,
            objective_dim=2,
            num_steps=2,
            embedding_dim=4,
        )
        obs = [_obs(1.0), _obs(1.0)]
        logits = policy(
            obs,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            torch.tensor([1.0, 1.0]),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertFalse(torch.allclose(logits[0], logits[1]))

    def test_archive_samples_datapoint(self) -> None:
        archive = PCNArchive(capacity=2)
        archive.add(_trajectory(torch.tensor([1.0, 0.0])))
        archive.add(_trajectory(torch.tensor([0.0, 1.0]), action=1))
        archive.add(_trajectory(torch.tensor([0.2, 0.2])))
        self.assertLessEqual(len(archive), 2)
        rng = __import__("numpy").random.default_rng(0)
        batch = archive.sample_datapoints(batch_size=2, rng=rng)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].desired_return.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
