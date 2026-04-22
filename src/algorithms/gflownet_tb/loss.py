from __future__ import annotations

import torch


def trajectory_balance_residual(
    log_z: torch.Tensor,
    log_pf_sum: torch.Tensor,
    log_reward: torch.Tensor,
    log_pb_sum: torch.Tensor,
) -> torch.Tensor:
    return log_z + log_pf_sum - log_reward - log_pb_sum


def trajectory_balance_loss(
    log_z: torch.Tensor,
    log_pf_sums: torch.Tensor,
    log_rewards: torch.Tensor,
    log_pb_sums: torch.Tensor,
) -> torch.Tensor:
    delta = trajectory_balance_residual(
        log_z=log_z,
        log_pf_sum=log_pf_sums,
        log_reward=log_rewards,
        log_pb_sum=log_pb_sums,
    )
    return torch.mean(delta.square())
