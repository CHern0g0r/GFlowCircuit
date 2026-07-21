from __future__ import annotations

import torch


def epsilon_mixed_probs(
    policy_probs: torch.Tensor,
    legal_actions: list[list[int]],
    epsilon_uniform: float,
) -> torch.Tensor:
    """Mix learned probabilities with a uniform distribution over legal actions."""
    epsilon_uniform = float(epsilon_uniform)
    if epsilon_uniform < 0.0 or epsilon_uniform > 1.0:
        raise ValueError(f"epsilon_uniform must be in [0, 1], got {epsilon_uniform}")
    if policy_probs.dim() != 2:
        raise ValueError(f"policy_probs must be 2D, got shape {tuple(policy_probs.shape)}")
    if len(legal_actions) != int(policy_probs.shape[0]):
        raise ValueError(
            f"legal_actions has {len(legal_actions)} rows but policy_probs has "
            f"batch size {int(policy_probs.shape[0])}"
        )
    if epsilon_uniform == 0.0:
        return policy_probs

    uniform_probs = torch.zeros_like(policy_probs)
    for row_idx, row_legal_actions in enumerate(legal_actions):
        if not row_legal_actions:
            continue
        legal_idx = torch.tensor(row_legal_actions, dtype=torch.long, device=policy_probs.device)
        uniform_probs[row_idx, legal_idx] = 1.0 / float(len(row_legal_actions))
    return (1.0 - epsilon_uniform) * policy_probs + epsilon_uniform * uniform_probs


# Backwards-compatible alias for code that imported the former private helper.
_epsilon_mixed_probs = epsilon_mixed_probs


__all__ = ["epsilon_mixed_probs"]
