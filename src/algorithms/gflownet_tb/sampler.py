from __future__ import annotations

import math

import pyspiel
import torch
from torch.distributions import Categorical

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.types import TBStep, TBTrajectory
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX
from src.eval_metrics import comparable_return
from src.utils import Observation, ZhuVectorState, resolve_vector_action_ids


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
    available_actions: list[int] | None = None,
) -> TBTrajectory:
    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    state = game.new_initial_state()

    obs0 = Observation.from_state(state, available_actions=available_actions)
    initial_size = int(obs0.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(obs0.obs_tensor[OBS_DEPTH_IDX])
    reward_func = reward_class(initial_size, initial_depth)
    vector_state = ZhuVectorState(
        initial_size=initial_size,
        initial_depth=initial_depth,
        num_steps=int(num_steps),
        action_ids=resolve_vector_action_ids(policy.num_actions, available_actions),
    )

    steps: list[TBStep] = []
    log_pf_terms: list[torch.Tensor] = []
    total_reward = 0.0

    while not state.is_terminal():
        obs = Observation.from_state(state, available_actions=available_actions)
        obs = obs.with_vector(
            vector_state.vector(
                current_size=int(obs.obs_tensor[OBS_SIZE_IDX]),
                current_depth=int(obs.obs_tensor[OBS_DEPTH_IDX]),
                step=len(steps),
            )
        )
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
        vector_state.record_action(action=action, previous_size=prev_size, previous_depth=prev_depth)
        next_obs = Observation.from_state(state, available_actions=available_actions)
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

    final_obs = Observation.from_state(state, available_actions=available_actions)
    final_size = int(final_obs.obs_tensor[OBS_SIZE_IDX])
    final_depth = int(final_obs.obs_tensor[OBS_DEPTH_IDX])

    comp_return = comparable_return(
        reward_class=reward_class,
        initial_size=initial_size,
        initial_depth=initial_depth,
        final_size=final_size,
        final_depth=final_depth,
    )
    improvement = max(-reward_improvement_clip, min(reward_improvement_clip, comp_return))
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
        comparable_return=comp_return,
        log_pf_sum=log_pf_sum,
        log_pb_sum=torch.zeros_like(log_pf_sum),
        log_reward=log_reward,
        terminal_reward=terminal_reward,
    )
