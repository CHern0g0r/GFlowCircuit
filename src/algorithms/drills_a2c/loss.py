from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.algorithms.drills_a2c.policy import DrillsA2CPolicy
from src.algorithms.drills_a2c.types import DrillsA2CStep


@dataclass
class DrillsA2CLoss:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    entropy: torch.Tensor
    mean_advantage: torch.Tensor
    mean_target: torch.Tensor


def drills_a2c_loss(
    *,
    policy: DrillsA2CPolicy,
    value_network: torch.nn.Module,
    steps: list[DrillsA2CStep],
    gamma: float,
    value_loss_coef: float,
    entropy_beta: float,
    normalize_advantages: bool = False,
) -> DrillsA2CLoss:
    if not steps:
        zero = torch.zeros((), dtype=torch.float32, device=next(policy.parameters()).device, requires_grad=True)
        return DrillsA2CLoss(
            total_loss=zero,
            actor_loss=zero,
            critic_loss=zero,
            entropy=zero,
            mean_advantage=zero,
            mean_target=zero,
        )

    device = next(policy.parameters()).device
    observations = [step.observation.observation_to_device(device) for step in steps]
    next_observations = [step.next_observation.observation_to_device(device) for step in steps]
    rewards = torch.tensor([step.reward for step in steps], dtype=torch.float32, device=device)
    dones = torch.tensor([step.done for step in steps], dtype=torch.bool, device=device)

    values = value_network(observations)
    if values.dim() == 0:
        values = values.unsqueeze(0)
    next_values = value_network(next_observations)
    if next_values.dim() == 0:
        next_values = next_values.unsqueeze(0)
    next_values = next_values.detach()
    next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
    targets = rewards + float(gamma) * next_values
    advantages = targets - values
    actor_advantages = advantages.detach()
    if normalize_advantages and len(steps) > 1:
        actor_advantages = (actor_advantages - actor_advantages.mean()) / (actor_advantages.std(unbiased=False) + 1e-8)

    logits = policy(observations)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    log_probs = []
    entropies = []
    for idx, step in enumerate(steps):
        legal_actions = list(step.observation.legal_actions)
        log_probs.append(policy.log_prob_legal(logits[idx], legal_actions, int(step.action)))
        probs = policy.masked_action_distribution(logits[idx], legal_actions)
        entropies.append(-torch.sum(probs * torch.log(probs + 1e-12)))

    log_prob_t = torch.stack(log_probs)
    entropy_t = torch.stack(entropies)
    actor_loss = -(log_prob_t * actor_advantages).mean()
    critic_loss = F.mse_loss(values, targets.detach())
    entropy = entropy_t.mean()
    total_loss = actor_loss + float(value_loss_coef) * critic_loss - float(entropy_beta) * entropy

    return DrillsA2CLoss(
        total_loss=total_loss,
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy=entropy,
        mean_advantage=advantages.detach().mean(),
        mean_target=targets.detach().mean(),
    )
