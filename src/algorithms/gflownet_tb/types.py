from __future__ import annotations

from dataclasses import dataclass

from src.utils import Observation


@dataclass
class TBStep:
    observation: Observation
    action: int
    legal_actions: list[int]
    log_pf: float


@dataclass
class TBTrajectory:
    file_path: str
    steps: list[TBStep]
    initial_size: int
    initial_depth: int
    final_size: int
    final_depth: int
    final_return: float
    td_final_return: float
    log_pf_sum: float
    log_pb_sum: float
    log_reward: float
    terminal_reward: float
