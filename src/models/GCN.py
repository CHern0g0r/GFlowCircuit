"""Graph Convolutional Network encoder for circuit states.

This module provides a small reusable GCN encoder that accepts either a
torch_geometric `Data` object or explicit `(x, edge_index, batch)` tensors and
returns node embeddings and a pooled graph embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool

PoolingType = Literal["mean", "sum", "max"]


@dataclass
class GCNEncoderOutput:
    """Container for encoder forward outputs."""

    node_embeddings: Tensor
    graph_embedding: Tensor


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
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim, hidden_dim and out_dim must be > 0")

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
        data: Data | None = None,
        *,
        x: Tensor | None = None,
        edge_index: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> GCNEncoderOutput:
        """Encode graph(s) into node and graph embeddings.

        You may either pass:
        - `data=<torch_geometric.data.Data>` (or Batch), or
        - explicit tensors via `x=...`, `edge_index=...`, optional `batch=...`.
        """

        if data is not None:
            x_in = data.x
            edge_index_in = data.edge_index
            batch_in = getattr(data, "batch", None)
        else:
            x_in = x
            edge_index_in = edge_index
            batch_in = batch

        if x_in is None or edge_index_in is None:
            raise ValueError("Provide either `data` or both `x` and `edge_index`.")

        if x_in.dim() != 2:
            raise ValueError(f"`x` must be 2D [num_nodes, feat_dim], got shape {tuple(x_in.shape)}.")
        if edge_index_in.dim() != 2 or edge_index_in.size(0) != 2:
            raise ValueError(
                f"`edge_index` must be 2D [2, num_edges], got shape {tuple(edge_index_in.shape)}."
            )

        # GCNConv normalization requires floating-point node features.
        h = x_in.float()
        edge_index_in = edge_index_in.long()
        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index_in)
            if layer_idx < len(self.convs) - 1:
                h = self.norms[layer_idx](h)
                h = torch.relu(h)
                if self.dropout > 0.0:
                    h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        if batch_in is None:
            batch_in = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        if self.pooling == "mean":
            g = global_mean_pool(h, batch_in)
        elif self.pooling == "sum":
            g = global_add_pool(h, batch_in)
        elif self.pooling == "max":
            g = global_max_pool(h, batch_in)
        else:
            raise ValueError(f"Unsupported pooling '{self.pooling}'. Use one of: mean, sum, max.")

        return GCNEncoderOutput(node_embeddings=h, graph_embedding=g)


__all__ = ["GCNEncoder", "GCNEncoderOutput"]
