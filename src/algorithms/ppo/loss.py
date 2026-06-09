from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.algorithms.ppo.policy import PPOPolicy
from src.algorithms.ppo.types import PPOTransition


@dataclass
class PPOBatchStats:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    mean_advantage: torch.Tensor
    mean_return: torch.Tensor


def compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.numel() == 0:
        empty = rewards.detach().clone()
        return empty, empty
    dones_f = dones.to(dtype=rewards.dtype)
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for idx in range(rewards.numel() - 1, -1, -1):
        not_done = 1.0 - dones_f[idx]
        delta = rewards[idx] + float(gamma) * not_done * next_values[idx] - values[idx]
        next_advantage = delta + float(gamma) * float(gae_lambda) * not_done * next_advantage
        advantages[idx] = next_advantage
    returns = advantages + values
    return advantages, returns


def ppo_minibatch_loss(
    *,
    policy: PPOPolicy,
    value_network: torch.nn.Module,
    transitions: list[PPOTransition],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    indices: torch.Tensor,
    clip_eps: float,
    value_loss_coef: float,
    entropy_beta: float,
) -> PPOBatchStats:
    device = next(policy.parameters()).device
    batch_indices = [int(i) for i in indices.tolist()]
    observations = [transitions[i].observation.observation_to_device(device) for i in batch_indices]
    actions = [int(transitions[i].action) for i in batch_indices]
    old_log_probs = torch.stack([transitions[i].old_log_prob.to(device).reshape(()) for i in batch_indices])
    mb_advantages = advantages.index_select(0, indices.to(advantages.device)).to(device)
    mb_returns = returns.index_select(0, indices.to(returns.device)).to(device)

    logits = policy(observations)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    values = value_network(observations)
    if values.dim() == 0:
        values = values.unsqueeze(0)

    log_probs = []
    entropies = []
    for row_idx, transition_idx in enumerate(batch_indices):
        legal_actions = list(transitions[transition_idx].observation.legal_actions)
        log_probs.append(policy.log_prob_legal(logits[row_idx], legal_actions, actions[row_idx]))
        probs = policy.masked_action_distribution(logits[row_idx], legal_actions)
        entropies.append(-torch.sum(probs * torch.log(probs + 1e-12)))
    new_log_probs = torch.stack(log_probs)
    entropy = torch.stack(entropies).mean()

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    actor_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
    value_loss = F.mse_loss(values.reshape(-1), mb_returns)
    total_loss = actor_loss + float(value_loss_coef) * value_loss - float(entropy_beta) * entropy

    approx_kl = (old_log_probs - new_log_probs).mean().detach()
    clip_fraction = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean().detach()
    return PPOBatchStats(
        total_loss=total_loss,
        actor_loss=actor_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        mean_advantage=mb_advantages.detach().mean(),
        mean_return=mb_returns.detach().mean(),
    )

