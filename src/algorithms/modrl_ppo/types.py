from __future__ import annotations

from dataclasses import dataclass

import torch

from src.algorithms.ppo.types import PPOTransition


@dataclass(frozen=True)
class PreferenceSpec:
    id: str
    mode: str
    weights: tuple[float, ...] | None = None
    constraint_objective: str | None = None
    optimize_objective: str | None = None
    threshold: float | None = None
    lex_scale: float = 10.0


@dataclass
class MODRLPPOTrajectory:
    file_path: str
    preference: PreferenceSpec
    transitions: list[PPOTransition]
    initial_size: int
    initial_depth: int
    final_size: int
    final_depth: int
    return_vec: torch.Tensor
    scalar_return: float
    actions_applied: list[int]


@dataclass
class MODRLPPORollout:
    trajectories: list[MODRLPPOTrajectory]
    transitions: list[PPOTransition]

