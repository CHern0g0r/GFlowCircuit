from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _FakeState:
    def __init__(self, metrics: list[tuple[int, int]]) -> None:
        self.metrics = metrics
        self.idx = 0

    def observation_tensor(self, player: int):
        del player
        obs = np.zeros(16, dtype=np.float32)
        obs[2] = float(self.metrics[self.idx][0])
        obs[3] = float(self.metrics[self.idx][1])
        return obs

    def legal_actions(self):
        return [] if self.is_terminal() else [0, 1]

    def is_terminal(self) -> bool:
        return self.idx >= len(self.metrics) - 1

    def apply_action(self, action: int) -> None:
        del action
        self.idx += 1


class _FakeGame:
    def __init__(self, metrics: list[tuple[int, int]]) -> None:
        self.metrics = metrics

    def new_initial_state(self):
        return _FakeState(self.metrics)


pyspiel_stub = types.ModuleType("pyspiel")
pyspiel_stub.State = _FakeState
pyspiel_stub.load_game = lambda name, params: _FakeGame([(100, 10), (90, 9), (85, 8)])
pyspiel_stub.circuit_graph = lambda state: (
    np.zeros((1, 1), dtype=np.float32),
    np.empty((0, 2), dtype=np.int64),
    np.empty((0, 1), dtype=np.float32),
)
sys.modules["pyspiel"] = pyspiel_stub

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.modrl_ppo.eval import pareto_front_min
from src.algorithms.modrl_ppo.sampler import sample_modrl_ppo_trajectory
from src.algorithms.modrl_ppo.scalarization import scalar_reward_from_step, scalarize_linear, tlo_utility
from src.algorithms.modrl_ppo.types import PreferenceSpec
from src.algorithms.reinforce.policy import Policy
from src.models.mo_rewards import SizeDepthImprovementReward


class FixedPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(num_actions=2)
        self.logits = nn.Parameter(torch.tensor([2.0, 0.0], dtype=torch.float32))

    def forward(self, obs):
        if isinstance(obs, list):
            return self.logits.unsqueeze(0).repeat(len(obs), 1)
        return self.logits.unsqueeze(0)


class FixedValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs):
        if isinstance(obs, list):
            return self.value.repeat(len(obs))
        return self.value.unsqueeze(0)


class MODRLPPORest(unittest.TestCase):
    def test_linear_scalar_reward_equals_weighted_vector_reward(self) -> None:
        reward_vec = torch.tensor([0.2, 0.4])
        got = scalarize_linear(reward_vec, (0.25, 0.75))
        self.assertTrue(torch.isclose(got, torch.tensor(0.35)))

    def test_tlo_scalar_rewards_telescope_to_final_utility(self) -> None:
        preference = PreferenceSpec(
            id="tlo_size_depth",
            mode="tlo",
            constraint_objective="size",
            optimize_objective="depth",
            threshold=0.05,
            lex_scale=10.0,
        )
        cumulative = torch.zeros(2)
        rewards = [torch.tensor([0.03, 0.01]), torch.tensor([0.03, 0.5])]
        scalars = []
        for reward in rewards:
            scalars.append(scalar_reward_from_step(preference=preference, reward_vec=reward, cumulative_before=cumulative))
            cumulative = cumulative + reward

        self.assertTrue(torch.isclose(sum(scalars), tlo_utility(cumulative, preference)))
        self.assertTrue(torch.isclose(sum(scalars), torch.tensor(1.01)))

    def test_tlo_constraint_gain_below_threshold_dominates_secondary_gain(self) -> None:
        preference = PreferenceSpec(
            id="tlo_size_depth",
            mode="tlo",
            constraint_objective="size",
            optimize_objective="depth",
            threshold=0.05,
            lex_scale=10.0,
        )
        cumulative = torch.zeros(2)
        constraint_gain = scalar_reward_from_step(
            preference=preference,
            reward_vec=torch.tensor([0.01, 0.0]),
            cumulative_before=cumulative,
        )
        secondary_gain = scalar_reward_from_step(
            preference=preference,
            reward_vec=torch.tensor([0.0, 0.09]),
            cumulative_before=cumulative,
        )
        self.assertGreater(float(constraint_gain), float(secondary_gain))

    def test_pareto_front_filters_dominated_points_and_dedupes(self) -> None:
        candidates = [
            {"final_size": 10, "final_depth": 10, "scalar_return": 0.1},
            {"final_size": 9, "final_depth": 10, "scalar_return": 0.2},
            {"final_size": 10, "final_depth": 9, "scalar_return": 0.3},
            {"final_size": 9, "final_depth": 10, "scalar_return": 0.4},
            {"final_size": 11, "final_depth": 11, "scalar_return": 0.5},
        ]

        front = pareto_front_min(candidates)

        self.assertEqual([(item["final_size"], item["final_depth"]) for item in front], [(9, 10), (10, 9)])
        self.assertEqual(front[0]["scalar_return"], 0.4)

    def test_sampler_stores_vector_scalar_rewards_and_actions(self) -> None:
        preference = PreferenceSpec(id="linear_balanced", mode="linear", weights=(0.5, 0.5))
        trajectory = sample_modrl_ppo_trajectory(
            file_path="fake.aig",
            num_steps=2,
            policy=FixedPolicy(),  # type: ignore[arg-type]
            value_network=FixedValue(),
            mo_reward_class=SizeDepthImprovementReward,
            preference=preference,
            sample_actions=False,
        )

        self.assertEqual(trajectory.actions_applied, [0, 0])
        self.assertTrue(torch.allclose(trajectory.return_vec, torch.tensor([0.15, 0.2])))
        self.assertTrue(math.isclose(trajectory.scalar_return, 0.175, rel_tol=1e-6))
        self.assertEqual([transition.action for transition in trajectory.transitions], [0, 0])
        self.assertTrue(all(math.isclose(transition.reward, value, rel_tol=1e-6) for transition, value in zip(
            trajectory.transitions,
            [0.1, 0.075],
        )))


if __name__ == "__main__":
    unittest.main()
