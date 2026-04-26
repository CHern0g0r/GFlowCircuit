from __future__ import annotations

import math

import pyspiel
import torch
from torch.distributions import Categorical

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.types import TBStep, TBTrajectory
from src.train.utils import OBS_DEPTH_IDX, OBS_SIZE_IDX
from src.utils import Observation


def sample_tb_trajectory(
    *,
    file_path: str,
    num_steps: int,
    policy: TBGFlowNetPolicy,
    reward_class: type,
    reward_alpha: float,
    reward_eps: float,
    reward_improvement_clip: float,
    sample_actions: bool,
) -> TBTrajectory:
    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    state = game.new_initial_state()

    obs0 = Observation.from_state(state)
    initial_size = int(obs0.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(obs0.obs_tensor[OBS_DEPTH_IDX])
    reward_func = reward_class(initial_size, initial_depth)

    steps: list[TBStep] = []
    log_pf_terms: list[torch.Tensor] = []
    total_reward = 0.0

    while not state.is_terminal():
        obs = Observation.from_state(state)
        logits = policy(obs)
        legal_actions = list(obs.legal_actions)
        probs = policy.masked_probs(logits, legal_actions)
        if not legal_actions:
            break
        if sample_actions:
            action = int(Categorical(probs=probs).sample().item())
        else:
            action = int(probs.argmax().item())
        log_pf = policy.log_prob_legal(logits, legal_actions, action)
        log_pf_terms.append(log_pf)
        prev_size = int(obs.obs_tensor[OBS_SIZE_IDX])
        prev_depth = int(obs.obs_tensor[OBS_DEPTH_IDX])
        state.apply_action(action)
        next_obs = Observation.from_state(state)
        step_reward = float(
            reward_func(
                int(next_obs.obs_tensor[OBS_SIZE_IDX]),
                int(next_obs.obs_tensor[OBS_DEPTH_IDX]),
                prev_size,
                prev_depth,
            )
        )
        total_reward += step_reward
        steps.append(
            TBStep(
                observation=obs,
                action=action,
                legal_actions=legal_actions,
                log_pf=log_pf,
            )
        )

    final_obs = Observation.from_state(state)
    final_size = int(final_obs.obs_tensor[OBS_SIZE_IDX])
    final_depth = int(final_obs.obs_tensor[OBS_DEPTH_IDX])

    td_final_return = float(
        reward_func(
            final_size,
            final_depth,
            initial_size,
            initial_depth,
        )
    )
    improvement = max(-reward_improvement_clip, min(reward_improvement_clip, td_final_return))
    terminal_reward = max(reward_eps, math.exp(float(reward_alpha) * improvement))
    log_reward = float(math.log(terminal_reward))

    if log_pf_terms:
        log_pf_sum = torch.stack(log_pf_terms).sum()
    else:
        log_pf_sum = torch.zeros((), dtype=torch.float32, device=policy.log_z.device)

    return TBTrajectory(
        file_path=file_path,
        steps=steps,
        initial_size=initial_size,
        initial_depth=initial_depth,
        final_size=final_size,
        final_depth=final_depth,
        final_return=float(total_reward),
        td_final_return=td_final_return,
        log_pf_sum=log_pf_sum,
        log_pb_sum=torch.zeros_like(log_pf_sum),
        log_reward=log_reward,
        terminal_reward=terminal_reward,
    )
