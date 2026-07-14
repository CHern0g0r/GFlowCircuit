from __future__ import annotations

import pyspiel
import torch
from torch.distributions import Categorical

from src.algorithms.modrl_ppo.scalarization import scalar_reward_from_step
from src.algorithms.modrl_ppo.types import MODRLPPOTrajectory, PreferenceSpec
from src.algorithms.ppo.policy import PPOPolicy
from src.algorithms.ppo.types import PPOTransition
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX
from src.models.mo_rewards import MultiObjectiveReward
from src.utils import Observation, ZhuVectorState, resolve_vector_action_ids


def _with_zhu_vector(*, obs: Observation, vector_state: ZhuVectorState, step: int) -> Observation:
    return obs.with_vector(
        vector_state.vector(
            current_size=int(obs.obs_tensor[OBS_SIZE_IDX]),
            current_depth=int(obs.obs_tensor[OBS_DEPTH_IDX]),
            step=int(step),
        )
    )


def sample_modrl_ppo_trajectory(
    *,
    file_path: str,
    num_steps: int,
    policy: PPOPolicy,
    value_network: torch.nn.Module,
    mo_reward_class: type[MultiObjectiveReward],
    preference: PreferenceSpec,
    sample_actions: bool,
    available_actions: list[int] | None = None,
) -> MODRLPPOTrajectory:
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

    transitions: list[PPOTransition] = []
    reward_vecs: list[torch.Tensor] = []
    scalar_rewards: list[float] = []
    actions_applied: list[int] = []
    cumulative_return = torch.zeros(int(reward_func.objective_dim), dtype=torch.float32)
    device = next(policy.parameters()).device

    while not state.is_terminal():
        raw_obs = Observation.from_state(state, available_actions=available_actions)
        obs = _with_zhu_vector(obs=raw_obs, vector_state=vector_state, step=len(transitions))
        obs_dev = obs.observation_to_device(device)
        legal_actions = list(obs.legal_actions)
        if not legal_actions:
            break

        with torch.no_grad():
            logits = policy(obs_dev)
            probs = policy.masked_action_distribution(logits, legal_actions)
            value = value_network(obs_dev)
            if value.dim() == 0:
                value = value.unsqueeze(0)

        if sample_actions:
            action = int(Categorical(probs=probs).sample().item())
        else:
            action = int(probs.argmax().item())

        with torch.no_grad():
            old_log_prob = policy.log_prob_legal(logits, legal_actions, action)

        prev_size = int(obs.obs_tensor[OBS_SIZE_IDX])
        prev_depth = int(obs.obs_tensor[OBS_DEPTH_IDX])
        state.apply_action(action)
        vector_state.record_action(action=action, previous_size=prev_size, previous_depth=prev_depth)

        next_raw_obs = Observation.from_state(state, available_actions=available_actions)
        next_obs = _with_zhu_vector(obs=next_raw_obs, vector_state=vector_state, step=len(transitions) + 1)
        next_obs_dev = next_obs.observation_to_device(device)
        with torch.no_grad():
            next_value = value_network(next_obs_dev)
            if next_value.dim() == 0:
                next_value = next_value.unsqueeze(0)

        reward_vec = reward_func(
            int(next_obs.obs_tensor[OBS_SIZE_IDX]),
            int(next_obs.obs_tensor[OBS_DEPTH_IDX]),
            prev_size,
            prev_depth,
        ).detach().to(dtype=torch.float32)
        scalar_reward = scalar_reward_from_step(
            preference=preference,
            reward_vec=reward_vec,
            cumulative_before=cumulative_return,
        )
        cumulative_return = cumulative_return + reward_vec
        reward_vecs.append(reward_vec.detach().cpu())
        scalar_rewards.append(float(scalar_reward.detach().item()))
        actions_applied.append(action)

        transitions.append(
            PPOTransition(
                observation=obs,
                action=action,
                reward=float(scalar_reward.detach().item()),
                next_observation=next_obs,
                done=bool(state.is_terminal()),
                old_log_prob=old_log_prob.detach(),
                old_value=value.reshape(-1)[0].detach(),
                next_value=next_value.reshape(-1)[0].detach(),
            )
        )

    final_obs = Observation.from_state(state, available_actions=available_actions)
    final_size = int(final_obs.obs_tensor[OBS_SIZE_IDX])
    final_depth = int(final_obs.obs_tensor[OBS_DEPTH_IDX])
    if reward_vecs:
        return_vec = torch.stack(reward_vecs, dim=0).sum(dim=0)
    else:
        return_vec = torch.zeros(int(reward_func.objective_dim), dtype=torch.float32)

    return MODRLPPOTrajectory(
        file_path=file_path,
        preference=preference,
        transitions=transitions,
        initial_size=initial_size,
        initial_depth=initial_depth,
        final_size=final_size,
        final_depth=final_depth,
        return_vec=return_vec.detach().cpu(),
        scalar_return=float(sum(scalar_rewards)),
        actions_applied=actions_applied,
    )

