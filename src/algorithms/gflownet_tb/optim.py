from __future__ import annotations

import torch
from torch import nn


def build_tb_optimizer(
    policy: nn.Module,
    *,
    learning_rate: float,
    log_z_learning_rate: float,
) -> torch.optim.Adam:
    """Build Adam with independent policy and ``logZ`` parameter groups."""
    learning_rate = float(learning_rate)
    log_z_learning_rate = float(log_z_learning_rate)
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if log_z_learning_rate <= 0.0:
        raise ValueError(f"log_z_learning_rate must be positive, got {log_z_learning_rate}")

    log_z = getattr(policy, "log_z", None)
    if not isinstance(log_z, nn.Parameter):
        raise TypeError("policy.log_z must be a torch.nn.Parameter")

    log_z_id = id(log_z)
    policy_params = [
        param
        for _, param in policy.named_parameters()
        if param.requires_grad and id(param) != log_z_id
    ]
    if not policy_params:
        raise ValueError("policy must have at least one trainable parameter other than log_z")
    if log_z_id in {id(param) for param in policy_params}:
        raise RuntimeError("policy.log_z must not be included in the policy optimizer group")

    return torch.optim.Adam(
        [
            {"params": policy_params, "lr": learning_rate},
            {"params": [log_z], "lr": log_z_learning_rate},
        ]
    )


# Backwards-compatible alias for code that imported the former private helper.
_build_tb_optimizer = build_tb_optimizer


__all__ = ["build_tb_optimizer"]
