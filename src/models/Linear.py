import torch
import torch.nn as nn
from src.utils import Observation


class IdEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()

    def forward(self, obs_batch: list[Observation] | Observation) -> torch.Tensor:
        if isinstance(obs_batch, Observation):
            obs_batch = [obs_batch]
        return torch.stack([obs.obs_tensor for obs in obs_batch], dim=0)


class LinearHead(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(obs_dim, num_actions)

    def forward(self, emb_batch: torch.Tensor) -> torch.Tensor:
        if emb_batch.dim() == 1:
            emb_batch = emb_batch.unsqueeze(0)
        return self.fc(emb_batch)


class MLPHead(nn.Module):
    """Simple configurable MLP head for policy logits."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        **kwargs,
    ):
        super().__init__()
        dims = [obs_dim, *list(hidden_dims), num_actions]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, emb_batch: torch.Tensor) -> torch.Tensor:
        if emb_batch.dim() == 1:
            emb_batch = emb_batch.unsqueeze(0)
        return self.net(emb_batch)


class ValueMLP(nn.Module):
    """Vector-only value baseline network."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (32, 32),
        **kwargs,
    ):
        super().__init__()
        dims = [in_dim, *list(hidden_dims), 1]
        layers: list[nn.Module] = []
        for in_dim_i, out_dim_i in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim_i, out_dim_i))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_batch: list[Observation] | Observation) -> torch.Tensor:
        if isinstance(obs_batch, Observation):
            obs_batch = [obs_batch]
        x = torch.stack([obs.obs_tensor for obs in obs_batch], dim=0)
        return self.net(x).squeeze(-1)
