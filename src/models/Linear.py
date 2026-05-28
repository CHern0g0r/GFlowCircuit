import torch
import torch.nn as nn
from src.utils import Observation


def _activation_layer(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("identity", "none", "linear"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp_layers(
    dims: list[int],
    *,
    activation: str = "relu",
    dropout: float = 0.0,
    layer_norm: bool = False,
) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(in_dim, out_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(_activation_layer(activation))
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return layers


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
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        dims = [obs_dim, *list(hidden_dims), num_actions]
        self.net = nn.Sequential(
            *_mlp_layers(
                dims,
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
            )
        )

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
        input: str = "obs_tensor",
        **kwargs,
    ):
        super().__init__()
        self.input = str(input)
        dims = [in_dim, *list(hidden_dims), 1]
        self.net = nn.Sequential(*_mlp_layers(dims))

    def forward(self, obs_batch: list[Observation] | Observation) -> torch.Tensor:
        if isinstance(obs_batch, Observation):
            obs_batch = [obs_batch]
        if self.input in ("zhu10", "vector", "vector_tensor"):
            vectors = []
            for obs in obs_batch:
                if obs.vector_tensor is None:
                    raise ValueError("ValueMLP input='zhu10' requires Observation.vector_tensor")
                vectors.append(obs.vector_tensor)
            x = torch.stack(vectors, dim=0)
        elif self.input == "obs_tensor":
            x = torch.stack([obs.obs_tensor for obs in obs_batch], dim=0)
        else:
            raise ValueError(f"Unsupported ValueMLP input: {self.input}")
        return self.net(x).squeeze(-1)


class VectorMLPEncoder(nn.Module):
    """MLP encoder for handcrafted vector features such as the Zhu 10D state."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (),
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        dims = [in_dim, *list(hidden_dims), out_dim]
        self.net = nn.Sequential(
            *_mlp_layers(
                dims,
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
            )
        )

    def forward(self, obs_batch: list[Observation] | Observation) -> torch.Tensor:
        if isinstance(obs_batch, Observation):
            obs_batch = [obs_batch]
        vectors = []
        for obs in obs_batch:
            if obs.vector_tensor is None:
                raise ValueError("VectorMLPEncoder requires Observation.vector_tensor")
            vectors.append(obs.vector_tensor)
        return self.net(torch.stack(vectors, dim=0))


class HybridEncoder(nn.Module):
    """Concatenate graph and vector branch embeddings."""

    def __init__(
        self,
        graph_encoder: nn.Module,
        vector_encoder: nn.Module,
        merge: str = "concat",
        out_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if merge != "concat":
            raise ValueError(f"Unsupported hybrid merge mode: {merge}")
        self.graph_encoder = graph_encoder
        self.vector_encoder = vector_encoder
        self.merge = merge
        self.out_dim = out_dim

    def forward(self, obs_batch: list[Observation] | Observation) -> torch.Tensor:
        graph_emb = self.graph_encoder(obs_batch)
        vector_emb = self.vector_encoder(obs_batch)
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if vector_emb.dim() == 1:
            vector_emb = vector_emb.unsqueeze(0)
        if graph_emb.shape[0] != vector_emb.shape[0]:
            raise ValueError(
                f"Hybrid branch batch sizes differ: graph={graph_emb.shape[0]}, vector={vector_emb.shape[0]}"
            )
        return torch.cat([graph_emb, vector_emb], dim=-1)
