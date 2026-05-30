from __future__ import annotations

import pyspiel
import torch
from torch.distributions import Categorical

from src.algorithms.drills_a2c.policy import DrillsA2CPolicy
from src.algorithms.drills_a2c.types import DrillsA2CStep, DrillsA2CTrajectory
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX
from src.eval_metrics import comparable_return
from src.utils import Observation, ZhuVectorState, resolve_vector_action_ids


def _with_zhu_vector(
    *,
    obs: Observation,
    vector_state: ZhuVectorState,
    step: int,
) -> Observation:
    return obs.with_vector(
        vector_state.vector(
            current_size=int(obs.obs_tensor[OBS_SIZE_IDX]),
            current_depth=int(obs.obs_tensor[OBS_DEPTH_IDX]),
            step=int(step),
        )
    )


def _feasible_depth(*, reward_func: object, initial_depth: int, final_depth: int) -> bool:
    threshold = getattr(reward_func, "depth_constraint", float(initial_depth))
    return float(final_depth) <= float(threshold)


def sample_drills_a2c_trajectory(
    *,
    file_path: str,
    num_steps: int,
    policy: DrillsA2CPolicy,
    reward_class: type,
    sample_actions: bool,
    available_actions: list[int] | None = None,
) -> DrillsA2CTrajectory:
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

    steps: list[DrillsA2CStep] = []
    total_reward = 0.0

    while not state.is_terminal():
        raw_obs = Observation.from_state(state, available_actions=available_actions)
        obs = _with_zhu_vector(obs=raw_obs, vector_state=vector_state, step=len(steps))
        logits = policy(obs)
        legal_actions = list(obs.legal_actions)
        if not legal_actions:
            break
        probs = policy.masked_action_distribution(logits, legal_actions)
        if sample_actions:
            action = int(Categorical(probs=probs).sample().item())
        else:
            action = int(probs.argmax().item())

        prev_size = int(obs.obs_tensor[OBS_SIZE_IDX])
        prev_depth = int(obs.obs_tensor[OBS_DEPTH_IDX])
        state.apply_action(action)
        vector_state.record_action(action=action, previous_size=prev_size, previous_depth=prev_depth)

        next_raw_obs = Observation.from_state(state, available_actions=available_actions)
        next_obs = _with_zhu_vector(
            obs=next_raw_obs,
            vector_state=vector_state,
            step=len(steps) + 1,
        )
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
            DrillsA2CStep(
                observation=obs,
                action=action,
                reward=step_reward,
                next_observation=next_obs,
                done=bool(state.is_terminal()),
            )
        )

    final_obs = Observation.from_state(state, available_actions=available_actions)
    final_size = int(final_obs.obs_tensor[OBS_SIZE_IDX])
    final_depth = int(final_obs.obs_tensor[OBS_DEPTH_IDX])

    return DrillsA2CTrajectory(
        file_path=file_path,
        steps=steps,
        initial_size=initial_size,
        initial_depth=initial_depth,
        final_size=final_size,
        final_depth=final_depth,
        final_return=float(total_reward),
        comparable_return=comparable_return(
            reward_class=reward_class,
            initial_size=initial_size,
            initial_depth=initial_depth,
            final_size=final_size,
            final_depth=final_depth,
        ),
        feasible_depth=_feasible_depth(
            reward_func=reward_func,
            initial_depth=initial_depth,
            final_depth=final_depth,
        ),
    )

