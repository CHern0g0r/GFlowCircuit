from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import Observation, StepSample

LegalSpec = list[int] | list[list[int]]


class Policy(nn.Module):
    """Base policy helper with legal-action masking utilities.

    This is used by REINFORCE. GFlowNet-TB keeps its own policy implementation.
    """

    def __init__(self, num_actions: int, **kwargs) -> None:
        super().__init__()
        self.num_actions = int(num_actions)

    def _normalize_legal_batch(self, legal_actions: LegalSpec, batch_size: int) -> list[list[int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not legal_actions and batch_size == 1:
            raise ValueError("legal_actions must be non-empty")

        if batch_size == 1:
            if isinstance(legal_actions[0], int):
                return [legal_actions]  # type: ignore[list-item]
            if isinstance(legal_actions[0], list):
                if len(legal_actions) != 1:
                    raise ValueError("Expected one legal set for batch size 1")
                return [legal_actions[0]]  # type: ignore[index]
            raise TypeError("legal_actions must be list[int] or list[list[int]]")

        if not legal_actions or isinstance(legal_actions[0], int):
            raise ValueError("Batched logits (B>1) require legal_actions as list[list[int]] with length B")

        rows = legal_actions  # type: ignore[assignment]
        if len(rows) != batch_size:
            raise ValueError(f"legal_actions has {len(rows)} rows but logits batch size is {batch_size}")
        return rows

    def masked_action_distribution(self, logits: torch.Tensor, legal_actions: LegalSpec) -> torch.Tensor:
        """Softmax over legal logits only; zeros on illegal actions.

        - logits [num_actions] -> probs [num_actions]
        - logits [B, num_actions] -> probs [B, num_actions]
        """
        if logits.dim() == 1:
            if len(logits) != self.num_actions:
                raise ValueError(f"logits length {len(logits)} != num_actions {self.num_actions}")
            if len(legal_actions) > 0 and isinstance(legal_actions[0], list):
                raise ValueError("1D logits require legal_actions: list[int]")
            la = legal_actions  # type: ignore[assignment]
            return self._masked_probs_one_row(logits, la)

        if logits.dim() != 2:
            raise ValueError(f"logits must be 1D or 2D, got shape {tuple(logits.shape)}")

        b, a = logits.shape
        if a != self.num_actions:
            raise ValueError(f"logits last dim {a} != num_actions {self.num_actions}")

        device = logits.device
        dtype = logits.dtype
        legal_rows = self._normalize_legal_batch(legal_actions, b)
        probs_out = torch.zeros(b, self.num_actions, dtype=dtype, device=device)
        for i in range(b):
            probs_out[i] = self._masked_probs_one_row(logits[i], legal_rows[i])
        return probs_out

    def _masked_probs_one_row(self, logits_row: torch.Tensor, legal_actions: list[int]) -> torch.Tensor:
        probs = torch.zeros(self.num_actions, dtype=logits_row.dtype, device=logits_row.device)
        if not legal_actions:
            return probs
        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=logits_row.device)
        legal_logits = logits_row.index_select(0, legal_idx)
        legal_probs = torch.softmax(legal_logits, dim=0)
        probs.scatter_(0, legal_idx, legal_probs)
        return probs

    def log_prob_legal(self, logits: torch.Tensor, legal_actions: list[int], action: int) -> torch.Tensor:
        """log π(a|s) where π is softmax over logits restricted to legal_actions."""
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
            raise ValueError("log_prob_legal requires non-empty legal_actions")

        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=logits.device)
        legal_logits = logits.index_select(0, legal_idx)
        log_p = F.log_softmax(legal_logits, dim=0)
        try:
            pos = legal_actions.index(action)
        except ValueError as exc:
            raise ValueError(f"action {action} not in legal_actions {legal_actions}") from exc
        return log_p[pos]


class ReinforcePolicy(Policy):
    """Encoder + head; REINFORCE update with masked softmax over legal actions."""

    def __init__(self, encoder: nn.Module, head: nn.Module, num_actions: int, **kwargs) -> None:
        super().__init__(num_actions, **kwargs)
        self.encoder = encoder
        self.head = head

    def forward(self, obs: Observation) -> torch.Tensor:
        device = next(self.parameters()).device
        obs = obs.observation_to_device(device)
        emb = self.encoder(obs)
        logits = self.head(emb)
        return logits

    def reinforce_loss(self, step: StepSample, advantage: torch.Tensor) -> torch.Tensor:
        logits = self.forward(step.observation)
        log_pi = self.log_prob_legal(logits, step.observation.legal_actions, step.action)
        return -(advantage * log_pi)

