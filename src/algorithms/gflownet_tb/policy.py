from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import Observation


class TBGFlowNetPolicy(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, num_actions: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.num_actions = int(num_actions)
        self.log_z = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, obs: Observation) -> torch.Tensor:
        device = self.log_z.device
        if obs.obs_tensor.device != device or obs.graph.x.device != device:
            obs = Observation(
                obs_tensor=obs.obs_tensor.to(device),
                graph=obs.graph.to(device),
                legal_actions=obs.legal_actions,
            )
        emb = self.encoder(obs)
        logits = self.head(emb)
        if logits.dim() == 2:
            if logits.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 logits, got {tuple(logits.shape)}")
            logits = logits.squeeze(0)
        if logits.dim() != 1:
            raise ValueError(f"Expected 1D logits, got {tuple(logits.shape)}")
        return logits

    def masked_probs(self, logits: torch.Tensor, legal_actions: list[int]) -> torch.Tensor:
        probs = torch.zeros(self.num_actions, dtype=logits.dtype, device=logits.device)
        if not legal_actions:
            return probs
        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=logits.device)
        legal_logits = logits.index_select(0, legal_idx)
        legal_probs = torch.softmax(legal_logits, dim=0)
        probs.scatter_(0, legal_idx, legal_probs)
        return probs

    def log_prob_legal(self, logits: torch.Tensor, legal_actions: list[int], action: int) -> torch.Tensor:
        if not legal_actions:
            raise ValueError("legal_actions must be non-empty")
        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=logits.device)
        legal_logits = logits.index_select(0, legal_idx)
        log_p = F.log_softmax(legal_logits, dim=0)
        try:
            position = legal_actions.index(action)
        except ValueError as exc:
            raise ValueError(f"Action {action} is not legal: {legal_actions}") from exc
        return log_p[position]
