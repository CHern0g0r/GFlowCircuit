"""Zhu 2020 graph encoder.

The Zhu/abcRL graph branch uses six one-hot vertex classes, a small GCN stack
``6 -> 12 -> 12 -> 12 -> 4``, and mean pooling to produce a graph embedding.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool

from src.utils import Observation

PoolingType = Literal["mean", "sum", "max"]


class ZhuGCNEncoder(nn.Module):
    """Paper-style graph branch for Zhu 2020.

    Vertex classes are represented as a 6D one-hot vector in this order:
    constant-1, primary output, primary input, zero inverters, one inverter,
    two inverters. If ``pyspiel.circuit_graph`` already returns six node
    features, those are used directly. Otherwise the encoder can accept integer
    class ids or derive a best-effort structural encoding from graph topology
    and incoming edge attributes.
    """

    num_vertex_classes = 6

    def __init__(
        self,
        hidden_dim: int = 12,
        out_dim: int = 4,
        num_layers: int = 4,
        pooling: PoolingType = "mean",
        node_feature_mode: str = "auto",
        inverter_edge_attr_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("ZhuGCNEncoder num_layers must be >= 1")
        self.pooling = pooling
        self.node_feature_mode = str(node_feature_mode)
        self.inverter_edge_attr_index = int(inverter_edge_attr_index)

        dims = [self.num_vertex_classes]
        if int(num_layers) == 1:
            dims.append(int(out_dim))
        else:
            dims.extend([int(hidden_dim)] * (int(num_layers) - 1))
            dims.append(int(out_dim))

        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, obs_batch: list[Observation] | Observation) -> torch.Tensor:
        if isinstance(obs_batch, Observation):
            obs_batch = [obs_batch]
        pyg_batch = Batch.from_data_list([obs.graph for obs in obs_batch])

        h = self._encode_vertex_classes(
            x=pyg_batch.x,
            edge_index=pyg_batch.edge_index,
            edge_attr=pyg_batch.edge_attr,
        )
        edge_index = pyg_batch.edge_index.long()
        batch_index = pyg_batch.batch

        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if layer_idx < len(self.convs) - 1:
                h = torch.relu(h)

        if self.pooling == "mean":
            return global_mean_pool(h, batch_index)
        if self.pooling == "sum":
            return global_add_pool(h, batch_index)
        if self.pooling == "max":
            return global_max_pool(h, batch_index)
        raise ValueError(f"Unsupported pooling '{self.pooling}'. Use one of: mean, sum, max.")

    def _encode_vertex_classes(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        mode = self.node_feature_mode
        if mode == "auto":
            if (
                x.dim() == 2
                and x.shape[1] == self.num_vertex_classes
                and self._looks_like_one_hot_prefix(x)
            ):
                return x.float()
            if (
                x.dim() == 2
                and x.shape[1] > self.num_vertex_classes
                and self._looks_like_one_hot_prefix(x[:, : self.num_vertex_classes])
            ):
                return x[:, : self.num_vertex_classes].float()
            if x.dim() == 1 or (x.dim() == 2 and x.shape[1] == 1):
                class_ids = x.reshape(-1).long()
                return self._one_hot(class_ids, x.device)
            return self._structure_features(
                num_nodes=int(x.shape[0]),
                edge_index=edge_index,
                edge_attr=edge_attr,
                device=x.device,
            )

        if mode == "one_hot":
            if x.dim() != 2 or x.shape[1] < self.num_vertex_classes:
                raise ValueError("node_feature_mode='one_hot' requires graph.x with at least 6 columns")
            return x[:, : self.num_vertex_classes].float()

        if mode == "class_id":
            class_ids = x.reshape(-1).long()
            return self._one_hot(class_ids, x.device)

        if mode == "structure":
            return self._structure_features(
                num_nodes=int(x.shape[0]),
                edge_index=edge_index,
                edge_attr=edge_attr,
                device=x.device,
            )

        raise ValueError(f"Unsupported ZhuGCNEncoder node_feature_mode: {mode}")

    def _looks_like_one_hot_prefix(self, x: torch.Tensor) -> bool:
        if x.numel() == 0:
            return True
        x = x.float()
        binary = torch.all((x == 0.0) | (x == 1.0))
        row_has_class = torch.all(x.sum(dim=1) >= 1.0)
        return bool(binary and row_has_class)

    def _one_hot(self, class_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        if class_ids.numel() == 0:
            return torch.empty((0, self.num_vertex_classes), dtype=torch.float32, device=device)
        if torch.any(class_ids < 0) or torch.any(class_ids >= self.num_vertex_classes):
            raise ValueError(
                "ZhuGCNEncoder class ids must be in [0, 5], got "
                f"min={int(class_ids.min())}, max={int(class_ids.max())}"
            )
        return torch.nn.functional.one_hot(class_ids, num_classes=self.num_vertex_classes).float()

    def _structure_features(
        self,
        *,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        features = torch.zeros((num_nodes, self.num_vertex_classes), dtype=torch.float32, device=device)
        if num_nodes == 0:
            return features

        edge_index = edge_index.long()
        src = edge_index[0]
        dst = edge_index[1]
        indegree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        outdegree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        indegree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.long))
        outdegree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.long))

        # Structural fallback cannot reliably distinguish constant-1 from PI.
        features[indegree == 0, 2] = 1.0
        features[outdegree == 0, 1] = 1.0

        logic_mask = (indegree > 0) & (outdegree > 0)
        inverter_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = edge_attr.float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            attr_idx = min(self.inverter_edge_attr_index, int(edge_attr.shape[1]) - 1)
            inverted = edge_attr[:, attr_idx] > 0.5
            inverter_counts.scatter_add_(0, dst, inverted.long())
        inverter_counts = torch.clamp(inverter_counts, min=0, max=2)
        for count, feature_idx in ((0, 3), (1, 4), (2, 5)):
            mask = logic_mask & (inverter_counts == count)
            features[mask, feature_idx] = 1.0

        missing = features.sum(dim=1) == 0
        features[missing, 3] = 1.0
        multi = features.sum(dim=1) > 1
        if torch.any(multi):
            # Prefer explicit PO over PI/logic labels for output wrapper nodes.
            features[multi] = 0.0
            features[multi, 1] = 1.0
        return features


__all__ = ["ZhuGCNEncoder"]
