import torch
import torch.nn as nn
from src.utils import Observation


class IdEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs: Observation) -> torch.Tensor:
        return obs.obs_tensor


class LinearHead(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.fc = nn.Linear(obs_dim, num_actions)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)
