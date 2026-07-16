from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import Observation
from src.algorithms.reinforce.policy import Policy

LegalSpec = list[int] | list[list[int]]


class TBGFlowNetPolicy(Policy):
    def __init__(self, encoder: nn.Module, head: nn.Module, num_actions: int) -> None:
        super().__init__(num_actions=num_actions)
        self.encoder = encoder
        self.head = head
        self.log_z = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, obs: Observation | list[Observation]) -> torch.Tensor:
        device = self.log_z.device

        is_single = isinstance(obs, Observation)
        if is_single:
            obs = obs.observation_to_device(device)
            batch_size = 1
        else:
            if not obs:
                raise ValueError("obs batch must be non-empty")
            obs = [o.observation_to_device(device) for o in obs]
            batch_size = len(obs)

        emb = self.encoder(obs)
        logits = self.head(emb)

        if is_single:
            if logits.dim() == 2:
                logits = logits.squeeze(0)
            if logits.dim() != 1:
                raise ValueError(f"Expected 1D logits, got {tuple(logits.shape)}")
        else:
            if logits.dim() != 2:
                raise ValueError(f"Expected 2D logits, got {tuple(logits.shape)}")
            if logits.shape[0] != batch_size:
                raise ValueError(f"Expected batch size {batch_size}, got {logits.shape[0]}")
        return logits

    def masked_probs(self, logits: torch.Tensor, legal_actions: LegalSpec) -> torch.Tensor:
        if logits.dim() == 1:
            if len(logits) != self.num_actions:
                raise ValueError(f"logits length {len(logits)} != num_actions {self.num_actions}")
            if legal_actions and isinstance(legal_actions[0], list):
                raise ValueError("1D logits require legal_actions: list[int]")
            return self._masked_probs_one_row(logits, legal_actions)  # type: ignore[arg-type]

        if logits.dim() != 2:
            raise ValueError(f"logits must be 1D or 2D, got shape {tuple(logits.shape)}")

        batch_size, num_actions = logits.shape
        if num_actions != self.num_actions:
            raise ValueError(f"logits last dim {num_actions} != num_actions {self.num_actions}")

        legal_rows = self._normalize_legal_batch(legal_actions, batch_size)
        probs = torch.zeros(batch_size, self.num_actions, dtype=logits.dtype, device=logits.device)
        for row_idx, row_legal_actions in enumerate(legal_rows):
            probs[row_idx] = self._masked_probs_one_row(logits[row_idx], row_legal_actions)
        return probs

    def _masked_probs_one_row(self, logits: torch.Tensor, legal_actions: list[int]) -> torch.Tensor:
        probs = torch.zeros(self.num_actions, dtype=logits.dtype, device=logits.device)
        if not legal_actions:
            return probs
        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=logits.device)
        legal_logits = logits.index_select(0, legal_idx)
        legal_probs = torch.softmax(legal_logits, dim=0)
        probs.scatter_(0, legal_idx, legal_probs)
        return probs

    def log_prob_legal(self, logits: torch.Tensor, legal_actions: list[int], action: int) -> torch.Tensor:
        if logits.dim() == 2:
            if logits.shape[0] != 1:
                raise ValueError(
                    "log_prob_legal: batched logits only supported for batch size 1; "
                    f"got shape {tuple(logits.shape)}"
                )
            logits = logits.squeeze(0)
        if logits.dim() != 1:
            raise ValueError(f"log_prob_legal expects 1D logits after squeeze, got {tuple(logits.shape)}")
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

    def log_prob_legal_batch(
        self,
        logits: torch.Tensor,
        legal_actions: list[list[int]],
        actions: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"log_prob_legal_batch expects 2D logits, got {tuple(logits.shape)}")
        batch_size, num_actions = logits.shape
        if num_actions != self.num_actions:
            raise ValueError(f"logits last dim {num_actions} != num_actions {self.num_actions}")
        if len(legal_actions) != batch_size:
            raise ValueError(f"legal_actions has {len(legal_actions)} rows but logits batch size is {batch_size}")
        if isinstance(actions, torch.Tensor):
            action_rows = [int(a) for a in actions.detach().cpu().tolist()]
        else:
            action_rows = [int(a) for a in actions]
        if len(action_rows) != batch_size:
            raise ValueError(f"actions has {len(action_rows)} rows but logits batch size is {batch_size}")
        return torch.stack(
            [
                self.log_prob_legal(logits[row_idx], legal_actions[row_idx], action_rows[row_idx])
                for row_idx in range(batch_size)
            ]
        )
