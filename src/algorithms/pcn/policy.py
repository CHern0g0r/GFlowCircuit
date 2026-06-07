from __future__ import annotations

import torch
import torch.nn as nn

from src.algorithms.reinforce.policy import Policy
from src.utils import Observation


class PCNPolicy(Policy):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        head: nn.Module,
        num_actions: int,
        encoder_out_dim: int,
        objective_dim: int,
        num_steps: int,
        embedding_dim: int = 64,
        condition_scale: float = 1.0,
    ) -> None:
        super().__init__(num_actions=num_actions)
        self.encoder = encoder
        self.head = head
        self.objective_dim = int(objective_dim)
        self.num_steps = int(num_steps)
        self.condition_scale = float(condition_scale)
        self.state_proj = nn.Sequential(
            nn.Linear(int(encoder_out_dim), int(embedding_dim)),
            nn.Sigmoid(),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(int(objective_dim) + 1, int(embedding_dim)),
            nn.Sigmoid(),
        )

    def _normalize_condition(
        self,
        desired_return: torch.Tensor,
        desired_horizon: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        desired_return = desired_return.to(device=device, dtype=torch.float32)
        desired_horizon = desired_horizon.to(device=device, dtype=torch.float32)
        if desired_return.dim() == 1:
            desired_return = desired_return.unsqueeze(0)
        if desired_horizon.dim() == 0:
            desired_horizon = desired_horizon.reshape(1, 1)
        elif desired_horizon.dim() == 1:
            desired_horizon = desired_horizon.unsqueeze(-1)
        if desired_return.shape[0] == 1 and batch_size > 1:
            desired_return = desired_return.expand(batch_size, -1)
        if desired_horizon.shape[0] == 1 and batch_size > 1:
            desired_horizon = desired_horizon.expand(batch_size, -1)
        if desired_return.shape != (batch_size, self.objective_dim):
            raise ValueError(
                f"desired_return must have shape {(batch_size, self.objective_dim)}, "
                f"got {tuple(desired_return.shape)}"
            )
        if desired_horizon.shape != (batch_size, 1):
            raise ValueError(f"desired_horizon must have shape {(batch_size, 1)}, got {tuple(desired_horizon.shape)}")
        return desired_return, desired_horizon / float(max(1, self.num_steps))

    def forward(
        self,
        obs: Observation | list[Observation],
        desired_return: torch.Tensor,
        desired_horizon: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(obs, list):
            obs_dev = [item.observation_to_device(device) for item in obs]
            batch_size = len(obs_dev)
        else:
            obs_dev = obs.observation_to_device(device)
            batch_size = 1

        state_emb = self.encoder(obs_dev)
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
        desired_return, desired_horizon = self._normalize_condition(
            desired_return,
            desired_horizon,
            batch_size=batch_size,
            device=device,
        )
        cond = torch.cat([desired_return, desired_horizon], dim=-1) * self.condition_scale
        fused = self.state_proj(state_emb) * self.condition_proj(cond)
        return self.head(fused)

