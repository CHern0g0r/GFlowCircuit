from __future__ import annotations

from src.models.policy import Policy
from src.utils import StepSample, Observation
import pyspiel
from torch.distributions import Categorical
from typing import Any
import os
import time


OBS_STEP_IDX = 0
OBS_NUM_STEPS_IDX = 1
OBS_SIZE_IDX = 2
OBS_DEPTH_IDX = 3
OBS_REWARD_IDX = 4


RESYN2_ACTION_SEQUENCE = [0, 1, 2, 0, 1, 3, 0, 4, 3, 0]


def _play_resyn2_reference(
    file_path: str,
    num_steps: int,
    reward_class: type,
    action_sequence: list[int] | None = None,
) -> tuple[float, float]:
    """Run fixed resyn2 actions; return (sum of per-step custom rewards, custom terminal reward)."""
    seq = RESYN2_ACTION_SEQUENCE if action_sequence is None else action_sequence
    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    state = game.new_initial_state()
    obs0 = Observation.from_state(state)
    initial_size = int(obs0.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(obs0.obs_tensor[OBS_DEPTH_IDX])
    reward_func = reward_class(initial_size, initial_depth)

    total_custom = 0.0
    for idx, action in enumerate(seq):
        if state.is_terminal():
            break
        obs = Observation.from_state(state)
        prev_size = get_size(obs)
        prev_depth = get_depth(obs)
        legal = list(state.legal_actions())
        if action not in legal:
            raise RuntimeError(
                f"Resyn2 baseline: action {action} at index {idx} is not legal. Legal: {legal}"
            )
        state.apply_action(action)
        next_obs = Observation.from_state(state)
        sz = get_size(next_obs)
        dp = get_depth(next_obs)
        total_custom += float(reward_func(sz, dp, prev_size, prev_depth))

    final_obs = Observation.from_state(state)
    final_step = float(
        reward_func(
            get_size(final_obs),
            get_depth(final_obs),
            initial_size,
            initial_depth,
        )
    )
    return float(total_custom), final_step


def _resyn2_average_gain_per_step(
    file_path: str,
    objective: str = "size",
    total_ops: int = 20,
) -> float:
    """Compute Zhu baseline: average gain per operation over 2x resyn2 (20 ops)."""
    if total_ops <= 0:
        raise ValueError("total_ops must be positive")

    game = pyspiel.load_game("circuit", {"num_steps": int(total_ops), "file_path": file_path})
    state = game.new_initial_state()
    obs0 = Observation.from_state(state)
    initial_size = get_size(obs0)
    initial_depth = get_depth(obs0)

    seq = (RESYN2_ACTION_SEQUENCE * ((total_ops + len(RESYN2_ACTION_SEQUENCE) - 1) // len(RESYN2_ACTION_SEQUENCE)))[
        :total_ops
    ]
    for action in seq:
        if state.is_terminal():
            break
        legal = list(state.legal_actions())
        if action not in legal:
            break
        state.apply_action(action)

    final_obs = Observation.from_state(state)
    final_size = get_size(final_obs)
    final_depth = get_depth(final_obs)
    if objective == "size":
        total_gain = (initial_size - final_size) / max(1, initial_size)
    elif objective == "depth":
        total_gain = (initial_depth - final_depth) / max(1, initial_depth)
    else:
        raise ValueError(f"Unsupported objective for Zhu baseline: {objective}")

    return float(total_gain / total_ops)


def evaluate_resyn2_variants(file_path: str) -> dict[str, dict[str, int | float]]:
    """Return paper-style baseline metrics: resyn2-1, resyn2-2, resyn2-inf."""
    variants = {
        "resyn2_1": len(RESYN2_ACTION_SEQUENCE),
        "resyn2_2": len(RESYN2_ACTION_SEQUENCE) * 2,
    }
    out: dict[str, dict[str, int | float]] = {}
    for name, steps in variants.items():
        game = pyspiel.load_game("circuit", {"num_steps": int(steps), "file_path": file_path})
        state = game.new_initial_state()
        obs0 = Observation.from_state(state)
        initial_size = get_size(obs0)
        initial_depth = get_depth(obs0)
        seq = (RESYN2_ACTION_SEQUENCE * ((steps + len(RESYN2_ACTION_SEQUENCE) - 1) // len(RESYN2_ACTION_SEQUENCE)))[:steps]
        for action in seq:
            if state.is_terminal():
                break
            if action not in state.legal_actions():
                break
            state.apply_action(action)
        obsf = Observation.from_state(state)
        out[name] = {
            "initial_size": initial_size,
            "initial_depth": initial_depth,
            "final_size": get_size(obsf),
            "final_depth": get_depth(obsf),
        }

    # resyn2-inf: stop after 5 consecutive unchanged recipe executions.
    max_recipes = 100
    game = pyspiel.load_game(
        "circuit",
        {"num_steps": int(max_recipes * len(RESYN2_ACTION_SEQUENCE)), "file_path": file_path},
    )
    state = game.new_initial_state()
    obs0 = Observation.from_state(state)
    initial_size = get_size(obs0)
    initial_depth = get_depth(obs0)
    unchanged_runs = 0
    for _ in range(max_recipes):
        before = Observation.from_state(state)
        before_pair = (get_size(before), get_depth(before))
        for action in RESYN2_ACTION_SEQUENCE:
            if state.is_terminal():
                break
            if action not in state.legal_actions():
                break
            state.apply_action(action)
        after = Observation.from_state(state)
        after_pair = (get_size(after), get_depth(after))
        if after_pair == before_pair:
            unchanged_runs += 1
        else:
            unchanged_runs = 0
        if unchanged_runs >= 5 or state.is_terminal():
            break
    obsf = Observation.from_state(state)
    out["resyn2_inf"] = {
        "initial_size": initial_size,
        "initial_depth": initial_depth,
        "final_size": get_size(obsf),
        "final_depth": get_depth(obsf),
    }
    return out


def build_resyn2_cache(
    circuits: list[str],
    num_steps: int,
    reward_class: type,
    baseline: str | None = None,
    baseline_scale: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Precompute Resyn2-related reference metrics once per circuit."""
    cache: dict[str, dict[str, Any]] = {}
    for file_path in circuits:
        r2_total, r2_final_step = _play_resyn2_reference(file_path, num_steps, reward_class)
        variants = evaluate_resyn2_variants(file_path)
        entry: dict[str, Any] = {
            "resyn2_baseline_total_reward": r2_total,
            "resyn2_baseline_final_step_reward": r2_final_step,
            "resyn2_variants": variants,
            "zhu_reward_baseline_per_step": 0.0,
        }
        if baseline == "zhu_resyn2":
            entry["zhu_reward_baseline_per_step"] = _resyn2_average_gain_per_step(
                file_path=file_path,
                objective="size",
                total_ops=20,
            )
            entry["zhu_reward_baseline_scale"] = float(baseline_scale)
        cache[file_path] = entry
    return cache


def get_obs_dim_and_num_actions(num_steps: int, sample_circuit: str) -> tuple[int, int, int, int]:
    probe_game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": sample_circuit})
    probe_state = probe_game.new_initial_state()
    obs_dim = len(probe_state.observation_tensor(0))
    num_actions = int(probe_game.num_distinct_actions())
    x, e, ea = pyspiel.circuit_graph(probe_state)
    node_dim = x.shape[1]
    edge_dim = ea.shape[1]
    return obs_dim, num_actions, node_dim, edge_dim


def get_size(obs: Observation) -> int:
    return int(obs.obs_tensor[OBS_SIZE_IDX])


def get_depth(obs: Observation) -> int:
    return int(obs.obs_tensor[OBS_DEPTH_IDX])


def run_episode(
    file_path: str,
    num_steps: int,
    policy: Policy,
    reward_class: type,
    sample_actions: bool,
    resyn2_baseline: dict[str, Any],
    baseline: str | None = None,
) -> dict:
    timing_enabled = os.getenv("GFC_TIMING", "0") == "1"
    timing: dict[str, Any] = {}
    t_episode_start = time.perf_counter()
    game = pyspiel.load_game("circuit", {"num_steps": num_steps, "file_path": file_path})
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

        reward = reward_func(
            sz,
            dp,
            prev_size,
            prev_depth)
        reward_raw_gain = float(getattr(reward_func, "last_gain", reward))
        reward_raw_gain_sum += reward_raw_gain
        trajectory.append(StepSample(
            observation=obs,
            action=action,
            probs=probs,
            reward=reward,
        ))

    final_obs = Observation.from_state(state, timing=timing if timing_enabled else None)
    total_reward = float(sum(step.reward for step in trajectory))
    trajectory_reward = float(
        reward_func(
            get_size(final_obs),
            get_depth(final_obs),
            initial_size,
            initial_depth,
        )
    )

    out: dict = {
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