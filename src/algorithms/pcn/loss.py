from __future__ import annotations

import torch
import torch.nn.functional as F

from src.algorithms.pcn.policy import PCNPolicy
from src.algorithms.pcn.types import PCNDatapoint


def pcn_cross_entropy_loss(policy: PCNPolicy, batch: list[PCNDatapoint]) -> torch.Tensor:
    if not batch:
        raise ValueError("PCN batch must be non-empty")
    device = next(policy.parameters()).device
    observations = [item.observation for item in batch]
    desired_returns = torch.stack([item.desired_return.to(dtype=torch.float32) for item in batch], dim=0).to(device)
    desired_horizons = torch.tensor([float(item.desired_horizon) for item in batch], dtype=torch.float32, device=device)
    logits = policy(observations, desired_returns, desired_horizons)
    if logits.dim() != 2:
        raise ValueError(f"PCN policy must return batched logits, got {tuple(logits.shape)}")

    losses: list[torch.Tensor] = []
    for row_idx, item in enumerate(batch):
        legal_actions = list(item.legal_actions)
        if not legal_actions:
            continue
        if int(item.action) not in legal_actions:
            raise ValueError(f"PCN target action {item.action} is not legal: {legal_actions}")
        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=device)
        legal_logits = logits[row_idx].index_select(0, legal_idx)
        log_probs = F.log_softmax(legal_logits, dim=0)
        target_pos = legal_actions.index(int(item.action))
        losses.append(-log_probs[target_pos])

    if not losses:
        raise ValueError("PCN batch has no datapoints with legal actions")
    return torch.stack(losses).mean()

