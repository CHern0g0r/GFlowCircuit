from __future__ import annotations

from collections.abc import Sequence

import torch


EpsilonInput = float | Sequence[float] | torch.Tensor


def _epsilon_tensor(
    epsilon_uniform: EpsilonInput,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize scalar or per-row epsilon values to a validated column."""
    values = torch.as_tensor(epsilon_uniform, dtype=dtype, device=device)
    if values.ndim == 0:
        values = values.expand(batch_size)
    elif values.ndim != 1 or int(values.numel()) != int(batch_size):
        raise ValueError(
            "epsilon_uniform must be scalar or contain one value per policy row; "
            f"got shape {tuple(values.shape)} for batch size {batch_size}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("epsilon_uniform values must be finite")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError(f"epsilon_uniform must be in [0, 1], got {values.tolist()}")
    return values.reshape(batch_size, 1)


def epsilon_mixed_probs(
    policy_probs: torch.Tensor,
    legal_actions: list[list[int]],
    epsilon_uniform: EpsilonInput,
) -> torch.Tensor:
    """Mix learned probabilities with a uniform distribution over legal actions."""
    if policy_probs.dim() != 2:
        raise ValueError(f"policy_probs must be 2D, got shape {tuple(policy_probs.shape)}")
    if len(legal_actions) != int(policy_probs.shape[0]):
        raise ValueError(
            f"legal_actions has {len(legal_actions)} rows but policy_probs has "
            f"batch size {int(policy_probs.shape[0])}"
        )
    epsilon = _epsilon_tensor(
        epsilon_uniform,
        batch_size=int(policy_probs.shape[0]),
        device=policy_probs.device,
        dtype=policy_probs.dtype,
    )
    if bool((epsilon == 0.0).all()):
        return policy_probs

    uniform_probs = torch.zeros_like(policy_probs)
    for row_idx, row_legal_actions in enumerate(legal_actions):
        if not row_legal_actions:
            continue
        legal_idx = torch.tensor(row_legal_actions, dtype=torch.long, device=policy_probs.device)
        uniform_probs[row_idx, legal_idx] = 1.0 / float(len(row_legal_actions))
    return (1.0 - epsilon) * policy_probs + epsilon * uniform_probs


# Backwards-compatible alias for code that imported the former private helper.
_epsilon_mixed_probs = epsilon_mixed_probs


__all__ = ["EpsilonInput", "epsilon_mixed_probs"]
