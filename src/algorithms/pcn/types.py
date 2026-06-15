from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils import Observation


@dataclass
class PCNStep:
    observation: Observation
    action: int
    legal_actions: list[int]
    reward_vec: torch.Tensor


@dataclass
class PCNTrajectory:
    file_path: str
    steps: list[PCNStep]
    initial_size: int
    initial_depth: int
    final_size: int
    final_depth: int
    return_vec: torch.Tensor
    horizon: int


@dataclass
class PCNDatapoint:
    observation: Observation
    action: int
    legal_actions: list[int]
    desired_return: torch.Tensor
    desired_horizon: float


@dataclass
class PCNTarget:
    desired_return: torch.Tensor
    desired_horizon: float

