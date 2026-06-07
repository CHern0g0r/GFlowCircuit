from __future__ import annotations

import numpy as np
import pyspiel
import torch
from torch.distributions import Categorical

from src.algorithms.pcn.policy import PCNPolicy
from src.algorithms.pcn.types import PCNStep, PCNTrajectory
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX
from src.models.mo_rewards import MultiObjectiveReward, discounted_vector_returns
from src.utils import Observation, ZhuVectorState, resolve_vector_action_ids


def _with_zhu_vector(*, obs: Observation, vector_state: ZhuVectorState, step: int) -> Observation:
    return obs.with_vector(
        vector_state.vector(
            current_size=int(obs.obs_tensor[OBS_SIZE_IDX]),
            current_depth=int(obs.obs_tensor[OBS_DEPTH_IDX]),
            step=int(step),
        )
    )


def sample_pcn_trajectory(
    *,
    file_path: str,
    num_steps: int,
    policy: PCNPolicy,
    mo_reward_class: type[MultiObjectiveReward],
    sample_actions: bool,
    rng: np.random.Generator,
    available_actions: list[int] | None = None,
    desired_return: torch.Tensor | None = None,
    desired_horizon: float | None = None,
    desired_return_clip: bool = False,
    gamma: float = 1.0,
) -> PCNTrajectory:
    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    state = game.new_initial_state()

    obs0 = Observation.from_state(state, available_actions=available_actions)
    initial_size = int(obs0.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(obs0.obs_tensor[OBS_DEPTH_IDX])
    reward_func = mo_reward_class(initial_size, initial_depth)
    vector_state = ZhuVectorState(
        initial_size=initial_size,
        initial_depth=initial_depth,
        num_steps=int(num_steps),
        action_ids=resolve_vector_action_ids(policy.num_actions, available_actions),
    )

    command_return = None if desired_return is None else desired_return.detach().clone().to(dtype=torch.float32)
    command_horizon = float(desired_horizon if desired_horizon is not None else num_steps)
    max_return = reward_func.max_return_vector()
    steps: list[PCNStep] = []
    reward_vecs: list[torch.Tensor] = []

    while not state.is_terminal():
        raw_obs = Observation.from_state(state, available_actions=available_actions)
        obs = _with_zhu_vector(obs=raw_obs, vector_state=vector_state, step=len(steps))
        legal_actions = list(obs.legal_actions)
        if not legal_actions:
            break

        if command_return is None:
            action = int(legal_actions[int(rng.integers(0, len(legal_actions)))])
        else:
            with torch.no_grad():
                logits = policy(
                    obs,
                    command_return,
                    torch.tensor(float(command_horizon), dtype=torch.float32),
                )
                probs = policy.masked_action_distribution(logits.squeeze(0), legal_actions)
            if sample_actions:
                action = int(Categorical(probs=probs).sample().item())
            else:
                action = int(probs.argmax().item())

        prev_size = int(obs.obs_tensor[OBS_SIZE_IDX])
        prev_depth = int(obs.obs_tensor[OBS_DEPTH_IDX])
        state.apply_action(action)
        vector_state.record_action(action=action, previous_size=prev_size, previous_depth=prev_depth)

        next_obs = Observation.from_state(state, available_actions=available_actions)
        reward_vec = reward_func(
            int(next_obs.obs_tensor[OBS_SIZE_IDX]),
            int(next_obs.obs_tensor[OBS_DEPTH_IDX]),
            prev_size,
            prev_depth,
        )
        reward_vecs.append(reward_vec)
        if command_return is not None:
            command_return = command_return - reward_vec
            if desired_return_clip:
                command_return = torch.minimum(command_return, max_return)
            command_horizon = float(max(command_horizon - 1.0, 1.0))

        steps.append(
            PCNStep(
                observation=obs,
                action=action,
                legal_actions=legal_actions,
                reward_vec=reward_vec.detach().cpu(),
            )
        )

    final_obs = Observation.from_state(state, available_actions=available_actions)
    final_size = int(final_obs.obs_tensor[OBS_SIZE_IDX])
    final_depth = int(final_obs.obs_tensor[OBS_DEPTH_IDX])
    if reward_vecs:
        return_vec = discounted_vector_returns(reward_vecs, gamma=float(gamma))[0].detach().cpu()
    else:
        return_vec = torch.zeros(reward_func.objective_dim, dtype=torch.float32)

    return PCNTrajectory(
        file_path=file_path,
        steps=steps,
        initial_size=initial_size,
        initial_depth=initial_depth,
        final_size=final_size,
        final_depth=final_depth,
        return_vec=return_vec,
        horizon=len(steps),
    )
