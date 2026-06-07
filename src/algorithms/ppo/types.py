from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils import Observation


@dataclass
class PPOTransition:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    done: bool
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    next_value: torch.Tensor


@dataclass
class PPOTrajectory:
    file_path: str
    transitions: list[PPOTransition]
    initial_size: int
    initial_depth: int
    final_size: int
    final_depth: int
    final_return: float
    comparable_return: float


@dataclass
class PPORollout:
    trajectories: list[PPOTrajectory]
    transitions: list[PPOTransition]

