from __future__ import annotations

from src.algorithms.drills_a2c.policy import DrillsA2CPolicy


class PPOPolicy(DrillsA2CPolicy):
    """Actor network for PPO.

    PPO uses the same encoder/head policy surface as the existing actor-critic
    baseline; the algorithm-specific behavior lives in the loss/trainer.
    """

