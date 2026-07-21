from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TBTerminalReward:
    raw_improvement: float
    clipped_improvement: float
    reward: float
    log_reward: float


def transform_terminal_reward(
    *,
    improvement: float,
    reward_alpha: float,
    reward_eps: float,
    reward_improvement_clip: float,
) -> TBTerminalReward:
    """Apply the production TB terminal reward transformation."""
    raw_improvement = float(improvement)
    clipped_improvement = max(
        -float(reward_improvement_clip),
        min(float(reward_improvement_clip), raw_improvement),
    )
    reward = max(float(reward_eps), math.exp(float(reward_alpha) * clipped_improvement))
    return TBTerminalReward(
        raw_improvement=raw_improvement,
        clipped_improvement=clipped_improvement,
        reward=reward,
        log_reward=math.log(reward),
    )


__all__ = ["TBTerminalReward", "transform_terminal_reward"]
