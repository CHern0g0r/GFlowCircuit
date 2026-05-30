from __future__ import annotations

from dataclasses import dataclass

from src.utils import Observation


@dataclass
class DrillsA2CStep:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    done: bool


@dataclass
class DrillsA2CTrajectory:
    file_path: str
    steps: list[DrillsA2CStep]
    initial_size: int
    initial_depth: int
    final_size: int
    final_depth: int
    final_return: float
    comparable_return: float
    feasible_depth: bool

