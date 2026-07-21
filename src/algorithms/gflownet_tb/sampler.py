from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pyspiel
import torch
from torch.distributions import Categorical

from src.algorithms.gflownet_tb.behavior import epsilon_mixed_probs
from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.reward import transform_terminal_reward
from src.algorithms.gflownet_tb.types import TBStep, TBTrajectory
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX
from src.eval_metrics import comparable_return
from src.utils import Observation, ZhuVectorState, resolve_vector_action_ids


@dataclass
class _TBRollout:
    file_path: str
    state: pyspiel.State
    initial_size: int
    initial_depth: int
    reward_func: Callable[..., float]
    vector_state: ZhuVectorState
    steps: list[TBStep] = field(default_factory=list)
    log_pf_terms: list[torch.Tensor] = field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False


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
    epsilon_uniform: float = 0.0,
    action_generator: torch.Generator | None = None,
) -> TBTrajectory:
    return sample_tb_trajectories(
        file_paths=[file_path],
        num_steps=num_steps,
        policy=policy,
        reward_class=reward_class,
        reward_alpha=reward_alpha,
        reward_eps=reward_eps,
        reward_improvement_clip=reward_improvement_clip,
        sample_actions=sample_actions,
        available_actions=available_actions,
        epsilon_uniform=epsilon_uniform,
        action_generator=action_generator,
    )[0]


def _new_rollout(
    *,
    file_path: str,
    num_steps: int,
    policy: TBGFlowNetPolicy,
    reward_class: type,
    available_actions: list[int] | None,
) -> _TBRollout:
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
    return _TBRollout(
        file_path=file_path,
        state=state,
        initial_size=initial_size,
        initial_depth=initial_depth,
        reward_func=reward_func,
        vector_state=vector_state,
    )


def _finish_rollout(
    *,
    rollout: _TBRollout,
    policy: TBGFlowNetPolicy,
    reward_class: type,
    reward_alpha: float,
    reward_eps: float,
    reward_improvement_clip: float,
    available_actions: list[int] | None,
) -> TBTrajectory:
    final_obs = Observation.from_state(rollout.state, available_actions=available_actions)
    final_size = int(final_obs.obs_tensor[OBS_SIZE_IDX])
    final_depth = int(final_obs.obs_tensor[OBS_DEPTH_IDX])

    comp_return = comparable_return(
        reward_class=reward_class,
        initial_size=rollout.initial_size,
        initial_depth=rollout.initial_depth,
        final_size=final_size,
        final_depth=final_depth,
    )
    transformed_reward = transform_terminal_reward(
        improvement=comp_return,
        reward_alpha=reward_alpha,
        reward_eps=reward_eps,
        reward_improvement_clip=reward_improvement_clip,
    )

    if rollout.log_pf_terms:
        log_pf_sum = torch.stack(rollout.log_pf_terms).sum()
    else:
        log_pf_sum = torch.zeros((), dtype=torch.float32, device=policy.log_z.device)

    return TBTrajectory(
        file_path=rollout.file_path,
        steps=rollout.steps,
        initial_size=rollout.initial_size,
        initial_depth=rollout.initial_depth,
        final_size=final_size,
        final_depth=final_depth,
        final_return=float(rollout.total_reward),
        comparable_return=comp_return,
        log_pf_sum=log_pf_sum,
        log_pb_sum=torch.zeros_like(log_pf_sum),
        log_reward=transformed_reward.log_reward,
        terminal_reward=transformed_reward.reward,
    )


# Backwards compatibility for callers that imported the former private helper.
_epsilon_mixed_probs = epsilon_mixed_probs


def _sample_behavior_actions(
    behavior_probs: torch.Tensor,
    action_generator: torch.Generator | None,
) -> torch.Tensor:
    """Draw behavior actions while preserving the legacy default RNG path."""
    if action_generator is None:
        return Categorical(probs=behavior_probs).sample()
    return torch.multinomial(
        behavior_probs,
        num_samples=1,
        replacement=True,
        generator=action_generator,
    ).squeeze(-1)


def sample_tb_trajectories(
    *,
    file_paths: list[str],
    num_steps: int,
    policy: TBGFlowNetPolicy,
    reward_class: type,
    reward_alpha: float,
    reward_eps: float,
    reward_improvement_clip: float,
    sample_actions: bool,
    available_actions: list[int] | None = None,
    epsilon_uniform: float = 0.0,
    action_generator: torch.Generator | None = None,
) -> list[TBTrajectory]:
    rollouts = [
        _new_rollout(
            file_path=file_path,
            num_steps=num_steps,
            policy=policy,
            reward_class=reward_class,
            available_actions=available_actions,
        )
        for file_path in file_paths
    ]

    while True:
        active_indices: list[int] = []
        obs_batch: list[Observation] = []
        legal_rows: list[list[int]] = []

        for idx, rollout in enumerate(rollouts):
            if rollout.done or rollout.state.is_terminal():
                rollout.done = True
                continue

            obs = Observation.from_state(rollout.state, available_actions=available_actions)
            legal_actions = list(obs.legal_actions)
            if not legal_actions:
                rollout.done = True
                continue

            obs = obs.with_vector(
                rollout.vector_state.vector(
                    current_size=int(obs.obs_tensor[OBS_SIZE_IDX]),
                    current_depth=int(obs.obs_tensor[OBS_DEPTH_IDX]),
                    step=len(rollout.steps),
                )
            )
            active_indices.append(idx)
            obs_batch.append(obs)
            legal_rows.append(legal_actions)

        if not active_indices:
            break

        logits = policy(obs_batch)
        probs = policy.masked_probs(logits, legal_rows)
        if sample_actions:
            behavior_probs = epsilon_mixed_probs(probs, legal_rows, epsilon_uniform)
            actions = _sample_behavior_actions(behavior_probs, action_generator)
        else:
            actions = probs.argmax(dim=-1)
        log_pf_batch = policy.log_prob_legal_batch(logits, legal_rows, actions)

        for row_idx, rollout_idx in enumerate(active_indices):
            rollout = rollouts[rollout_idx]
            obs = obs_batch[row_idx]
            legal_actions = legal_rows[row_idx]
            action = int(actions[row_idx].item())
            log_pf = log_pf_batch[row_idx]
            rollout.log_pf_terms.append(log_pf)

            prev_size = int(obs.obs_tensor[OBS_SIZE_IDX])
            prev_depth = int(obs.obs_tensor[OBS_DEPTH_IDX])
            rollout.state.apply_action(action)
            rollout.vector_state.record_action(action=action, previous_size=prev_size, previous_depth=prev_depth)
            next_obs = Observation.from_state(rollout.state, available_actions=available_actions)
            step_reward = float(
                rollout.reward_func(
                    int(next_obs.obs_tensor[OBS_SIZE_IDX]),
                    int(next_obs.obs_tensor[OBS_DEPTH_IDX]),
                    prev_size,
                    prev_depth,
                )
            )
            rollout.total_reward += step_reward
            rollout.steps.append(
                TBStep(
                    observation=obs,
                    action=action,
                    legal_actions=legal_actions,
                    log_pf=log_pf,
                )
            )

    return [
        _finish_rollout(
            rollout=rollout,
            policy=policy,
            reward_class=reward_class,
            reward_alpha=reward_alpha,
            reward_eps=reward_eps,
            reward_improvement_clip=reward_improvement_clip,
            available_actions=available_actions,
        )
        for rollout in rollouts
    ]
