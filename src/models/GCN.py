"""Graph Convolutional Network encoder for circuit states.

This module provides a small reusable GCN encoder that accepts either a
torch_geometric `Data` object or explicit `(x, edge_index, batch)` tensors and
returns node embeddings and a pooled graph embedding.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Batch
from src.utils import Observation
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool

PoolingType = Literal["mean", "sum", "max"]


class GCNEncoder(nn.Module):
    """Basic multi-layer GCN encoder for graph state representations.

    Args:
        in_dim: Input node feature dimension.
        hidden_dim: Hidden feature dimension for intermediate GCN layers.
        out_dim: Output node embedding dimension.
        num_layers: Number of GCN layers (>= 1).
        dropout: Dropout probability between hidden layers.
        pooling: Graph pooling strategy for graph-level embedding.
        use_layer_norm: If True, adds LayerNorm after each hidden layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        pooling: PoolingType = "mean",
        use_layer_norm: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.pooling = pooling

        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(out_dim)

        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.norms = nn.ModuleList()
        for i in range(len(self.convs) - 1):
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(dims[i + 1]))
            else:
                self.norms.append(nn.Identity())

    def forward(
        self,
        obs_batch: list[Observation] | Observation,
    ) -> torch.Tensor:
        if isinstance(obs_batch, Observation):
            obs_batch = [obs_batch]
        pyg_batch = Batch.from_data_list([obs.graph for obs in obs_batch])

        # GCNConv normalization requires floating-point node features.
        h = pyg_batch.x.float()
        edge_index = pyg_batch.edge_index.long()
        batch_index = pyg_batch.batch
        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if layer_idx < len(self.convs) - 1:
                h = self.norms[layer_idx](h)
                h = torch.relu(h)
                if self.dropout > 0.0:
                    h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        if self.pooling == "mean":
            g = global_mean_pool(h, batch_index)
        elif self.pooling == "sum":
            g = global_add_pool(h, batch_index)
        elif self.pooling == "max":
            g = global_max_pool(h, batch_index)
        else:
            raise ValueError(f"Unsupported pooling '{self.pooling}'. Use one of: mean, sum, max.")

        return g


__all__ = ["GCNEncoder", "GCNEncoderOutput"]
