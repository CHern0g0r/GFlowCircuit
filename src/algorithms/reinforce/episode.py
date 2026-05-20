from __future__ import annotations

import os
import time
from typing import Any

import pyspiel
from torch.distributions import Categorical

from src.algorithms.reinforce.policy import Policy
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX, get_depth, get_size
from src.utils import Observation, StepSample


def run_reinforce_episode(
    *,
    file_path: str,
    num_steps: int,
    policy: Policy,
    reward_class: type,
    sample_actions: bool,
    resyn2_baseline: dict[str, Any],
    baseline: str | None = None,
) -> dict[str, Any]:
    timing_enabled = os.getenv("GFC_TIMING", "0") == "1"
    timing: dict[str, Any] = {}
    t_episode_start = time.perf_counter()

    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    state = game.new_initial_state()
    trajectory: list[StepSample] = []

    obs0 = Observation.from_state(state, timing=timing if timing_enabled else None)
    initial_size = int(obs0.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(obs0.obs_tensor[OBS_DEPTH_IDX])

    reward_func = reward_class(initial_size, initial_depth)
    zhu_step_baseline = float(resyn2_baseline.get("zhu_reward_baseline_per_step", 0.0))
    if baseline == "zhu_resyn2":
        if hasattr(reward_func, "set_baseline"):
            reward_func.set_baseline(zhu_step_baseline)
        if hasattr(reward_func, "set_baseline_scale"):
            reward_func.set_baseline_scale(float(resyn2_baseline.get("zhu_reward_baseline_scale", 1.0)))

    best_size = initial_size
    best_depth = initial_depth
    best_qor = initial_size * initial_depth
    reward_raw_gain_sum = 0.0

    while not state.is_terminal():
        obs = Observation.from_state(state, timing=timing if timing_enabled else None)

        t_policy_start = time.perf_counter()
        logits = policy(obs)
        probs = policy.masked_action_distribution(logits, obs.legal_actions)
        t_policy_end = time.perf_counter()
        if timing_enabled:
            timing["policy_total_s"] = timing.get("policy_total_s", 0.0) + (t_policy_end - t_policy_start)

        prev_size = get_size(obs)
        prev_depth = get_depth(obs)

        if sample_actions:
            dist = Categorical(probs=probs)
            action = int(dist.sample().item())
        else:
            if probs.dim() == 2:
                action = int(probs.argmax(dim=-1).item())
            else:
                action = int(probs.argmax().item())

        t_apply_start = time.perf_counter()
        state.apply_action(action)
        t_apply_end = time.perf_counter()
        if timing_enabled:
            timing["apply_action_s"] = timing.get("apply_action_s", 0.0) + (t_apply_end - t_apply_start)

        next_obs = Observation.from_state(state, timing=timing if timing_enabled else None)
        sz = get_size(next_obs)
        dp = get_depth(next_obs)

        best_size = min(best_size, sz)
        best_depth = min(best_depth, dp)
        best_qor = min(best_qor, sz * dp)

        reward = reward_func(sz, dp, prev_size, prev_depth)
        reward_raw_gain = float(getattr(reward_func, "last_gain", reward))
        reward_raw_gain_sum += reward_raw_gain

        trajectory.append(
            StepSample(
                observation=obs,
                action=action,
                probs=probs,
                reward=float(reward),
            )
        )

    final_obs = Observation.from_state(state, timing=timing if timing_enabled else None)
    total_reward = float(sum(step.reward for step in trajectory))
    trajectory_reward = float(reward_func(get_size(final_obs), get_depth(final_obs), initial_size, initial_depth))

    out: dict[str, Any] = {
        "trajectory": trajectory,
        "initial_size": initial_size,
        "initial_depth": initial_depth,
        "final_size": int(final_obs.obs_tensor[OBS_SIZE_IDX]),
        "final_depth": int(final_obs.obs_tensor[OBS_DEPTH_IDX]),
        "final_return": total_reward,
        "num_steps_taken": len(trajectory),
        "terminal": bool(state.is_terminal()),
        "best_size": best_size,
        "best_depth": best_depth,
        "best_qor": best_qor,
        "actions_applied": [int(step.action) for step in trajectory],
        "final_step_reward": trajectory_reward,
        "baseline": baseline or "none",
    }

    r2_total = float(resyn2_baseline["resyn2_baseline_total_reward"])
    r2_final_step = float(resyn2_baseline["resyn2_baseline_final_step_reward"])
    out["resyn2_baseline_total_reward"] = r2_total
    out["resyn2_baseline_final_step_reward"] = r2_final_step

    if baseline == "resyn2":
        out["final_return"] = total_reward - r2_total
        out["final_step_reward"] = trajectory_reward - r2_final_step
    elif baseline == "zhu_resyn2":
        out["zhu_reward_baseline_per_step"] = zhu_step_baseline

    out["resyn2_variants"] = resyn2_baseline["resyn2_variants"]
    out["reward_raw_gain_mean"] = reward_raw_gain_sum / max(1, len(trajectory))
    out["reward_adjusted_mean"] = total_reward / max(1, len(trajectory))
    out["reward_baseline_per_step"] = zhu_step_baseline if baseline == "zhu_resyn2" else 0.0

    if timing_enabled:
        episode_s = time.perf_counter() - t_episode_start
        from_state_s = float(timing.get("obs_from_state_total_s", 0.0))
        circuit_graph_s = float(timing.get("circuit_graph_s", 0.0))
        apply_s = float(timing.get("apply_action_s", 0.0))
        policy_s = float(timing.get("policy_total_s", 0.0))
        steps = max(1, len(trajectory))
        print(
            "[timing] circuit={c} steps={n} total={tot:.3f}s "
            "from_state={fs:.3f}s(graph={cg:.3f}s) apply={ap:.3f}s policy={pl:.3f}s "
            "from_state_calls={calls} per_step={pps:.3f}s".format(
                c=file_path,
                n=len(trajectory),
                tot=episode_s,
                fs=from_state_s,
                cg=circuit_graph_s,
                ap=apply_s,
                pl=policy_s,
                calls=int(timing.get("obs_from_state_calls", 0)),
                pps=episode_s / steps,
            )
        )

    return out

