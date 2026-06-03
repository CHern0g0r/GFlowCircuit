from __future__ import annotations

import torch
import torch.nn as nn

from src.algorithms.reinforce.policy import Policy
from src.utils import Observation


class DrillsA2CPolicy(Policy):
    """Actor network for the DRiLLS-style A2C baseline."""

    def __init__(self, encoder: nn.Module, head: nn.Module, num_actions: int, **kwargs) -> None:
        super().__init__(num_actions, **kwargs)
        self.encoder = encoder
        self.head = head

    def forward(self, obs: Observation | list[Observation]) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(obs, list):
            obs = [item.observation_to_device(device) for item in obs]
        else:
            obs = obs.observation_to_device(device)
        emb = self.encoder(obs)
        logits = self.head(emb)
        return logits

