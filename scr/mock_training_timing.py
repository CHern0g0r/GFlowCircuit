from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import hydra
import numpy as np
import pyspiel
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical

from src.algorithms.reinforce.policy import Policy
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX, build_resyn2_cache, get_depth, get_size
from src.models import REWARD_TYPES
from src.run import _build_models, _get_algorithm_name, _save_run_checkpoint
from src.utils import (
    Observation,
    StepSample,
    ZhuVectorState,
    discounted_returns,
    get_obs_dim_and_num_actions,
    load_circuits,
    normalize_available_actions,
    resolve_vector_action_ids,
    train_test_split,
)


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class TimingRecorder:
    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self.rows: list[dict[str, Any]] = []

    @contextmanager
    def timed(
        self,
        phase: str,
        *,
        sync_cuda: bool = False,
        **meta: Any,
    ) -> Iterator[None]:
        if sync_cuda:
            _sync_cuda(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                _sync_cuda(self.device)
            seconds = time.perf_counter() - start
            self.add(phase, seconds, **meta)

    def add(self, phase: str, seconds: float, **meta: Any) -> None:
        row = {"phase": phase, "seconds": float(seconds)}
        row.update({k: _jsonable(v) for k, v in meta.items()})
        self.rows.append(row)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_phase: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_phase[str(row["phase"])].append(float(row["seconds"]))

    out: dict[str, dict[str, float]] = {}
    for phase, values in sorted(by_phase.items()):
        arr = np.asarray(values, dtype=np.float64)
        out[phase] = {
            "calls": int(arr.size),
            "total_s": float(arr.sum()),
            "mean_s": float(arr.mean()) if arr.size else 0.0,
            "median_s": float(np.median(arr)) if arr.size else 0.0,
            "p90_s": _percentile(values, 90),
            "p95_s": _percentile(values, 95),
            "min_s": float(arr.min()) if arr.size else 0.0,
            "max_s": float(arr.max()) if arr.size else 0.0,
        }
    return out


def _phase_mean(rows: list[dict[str, Any]], phase: str) -> float:
    values = [float(row["seconds"]) for row in rows if row["phase"] == phase]
    return float(np.mean(values)) if values else 0.0


def _phase_total(rows: list[dict[str, Any]], phase: str) -> float:
    return float(sum(float(row["seconds"]) for row in rows if row["phase"] == phase))


def _event_count(rows: list[dict[str, Any]], phase: str) -> int:
    return sum(1 for row in rows if row["phase"] == phase)


def _load_hydra_config(config_name: str, overrides: list[str]) -> DictConfig:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "cfg"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides)


def _timed_reinforce_episode(
    *,
    recorder: TimingRecorder,
    file_path: str,
    num_steps: int,
    policy: Policy,
    reward_class: type,
    sample_actions: bool,
    resyn2_baseline: dict[str, Any],
    baseline: str | None,
    available_actions: list[int] | None,
    repeat_idx: int,
    run_idx: int,
    episode_idx: int,
    mode: str,
) -> dict[str, Any]:
    episode_start = time.perf_counter()
    timing: dict[str, Any] = {}
    meta = {
        "repeat_idx": repeat_idx,
        "run_idx": run_idx,
        "episode_idx": episode_idx,
        "mode": mode,
        "circuit": file_path,
    }

    with recorder.timed("episode_load_game", **meta):
        game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    with recorder.timed("episode_initial_state", **meta):
        state = game.new_initial_state()

    trajectory: list[StepSample] = []
    obs0 = Observation.from_state(state, available_actions=available_actions, timing=timing)
    initial_size = int(obs0.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(obs0.obs_tensor[OBS_DEPTH_IDX])

    reward_func = reward_class(initial_size, initial_depth)
    vector_state = ZhuVectorState(
        initial_size=initial_size,
        initial_depth=initial_depth,
        num_steps=int(num_steps),
        action_ids=resolve_vector_action_ids(policy.num_actions, available_actions),
    )
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
    action_sample_s = 0.0
    vector_s = 0.0
    reward_s = 0.0
    trajectory_s = 0.0
    policy_s = 0.0
    apply_s = 0.0

    while not state.is_terminal():
        obs = Observation.from_state(state, available_actions=available_actions, timing=timing)
        step_idx = len(trajectory)

        t0 = time.perf_counter()
        obs = obs.with_vector(
            vector_state.vector(
                current_size=get_size(obs),
                current_depth=get_depth(obs),
                step=step_idx,
            )
        )
        vector_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        _sync_cuda(recorder.device)
        logits = policy(obs)
        probs = policy.masked_action_distribution(logits, obs.legal_actions)
        _sync_cuda(recorder.device)
        policy_s += time.perf_counter() - t0

        prev_size = get_size(obs)
        prev_depth = get_depth(obs)

        t0 = time.perf_counter()
        if sample_actions:
            dist = Categorical(probs=probs)
            action = int(dist.sample().item())
        else:
            if probs.dim() == 2:
                action = int(probs.argmax(dim=-1).item())
            else:
                action = int(probs.argmax().item())
        action_sample_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        state.apply_action(action)
        apply_s += time.perf_counter() - t0
        vector_state.record_action(action=action, previous_size=prev_size, previous_depth=prev_depth)

        next_obs = Observation.from_state(state, available_actions=available_actions, timing=timing)
        sz = get_size(next_obs)
        dp = get_depth(next_obs)

        best_size = min(best_size, sz)
        best_depth = min(best_depth, dp)
        best_qor = min(best_qor, sz * dp)

        t0 = time.perf_counter()
        reward = reward_func(sz, dp, prev_size, prev_depth)
        reward_raw_gain = float(getattr(reward_func, "last_gain", reward))
        reward_raw_gain_sum += reward_raw_gain
        reward_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        trajectory.append(
            StepSample(
                observation=obs,
                action=action,
                probs=probs,
                reward=float(reward),
            )
        )
        trajectory_s += time.perf_counter() - t0

    final_obs = Observation.from_state(state, available_actions=available_actions, timing=timing)
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
        "resyn2_baseline_total_reward": float(resyn2_baseline["resyn2_baseline_total_reward"]),
        "resyn2_baseline_final_step_reward": float(resyn2_baseline["resyn2_baseline_final_step_reward"]),
        "resyn2_variants": resyn2_baseline["resyn2_variants"],
        "reward_raw_gain_mean": reward_raw_gain_sum / max(1, len(trajectory)),
        "reward_adjusted_mean": total_reward / max(1, len(trajectory)),
        "reward_baseline_per_step": zhu_step_baseline if baseline == "zhu_resyn2" else 0.0,
    }
    if baseline == "resyn2":
        out["final_return"] = total_reward - float(resyn2_baseline["resyn2_baseline_total_reward"])
        out["final_step_reward"] = trajectory_reward - float(resyn2_baseline["resyn2_baseline_final_step_reward"])
    elif baseline == "zhu_resyn2":
        out["zhu_reward_baseline_per_step"] = zhu_step_baseline

    common = {
        **meta,
        "steps": len(trajectory),
        "obs_from_state_calls": int(timing.get("obs_from_state_calls", 0)),
    }
    episode_s = time.perf_counter() - episode_start
    recorder.add("episode_total", episode_s, **common)
    recorder.add("episode_observation_from_state", float(timing.get("obs_from_state_total_s", 0.0)), **common)
    recorder.add("episode_observation_tensor", float(timing.get("obs_tensor_s", 0.0)), **common)
    recorder.add("episode_circuit_graph", float(timing.get("circuit_graph_s", 0.0)), **common)
    recorder.add("episode_tensor_wrap", float(timing.get("tensor_wrap_s", 0.0)), **common)
    recorder.add("episode_legal_actions", float(timing.get("legal_actions_s", 0.0)), **common)
    recorder.add("episode_policy_forward_mask", policy_s, **common)
    recorder.add("episode_action_sample", action_sample_s, **common)
    recorder.add("episode_apply_action", apply_s, **common)
    recorder.add("episode_vector_features", vector_s, **common)
    recorder.add("episode_reward", reward_s, **common)
    recorder.add("episode_trajectory_append", trajectory_s, **common)
    return out


def _timed_train_update(
    *,
    recorder: TimingRecorder,
    policy: Policy,
    value_network: torch.nn.Module | None,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer | None,
    episode: dict[str, Any],
    device: torch.device,
    gamma: float,
    entropy_beta: float,
    clip_grad_norm_policy: float | None,
    clip_grad_norm_value: float | None,
    normalize_returns: bool,
    repeat_idx: int,
    run_idx: int,
    episode_idx: int,
    circuit: str,
) -> dict[str, float]:
    steps: list[StepSample] = episode["trajectory"]
    meta = {
        "repeat_idx": repeat_idx,
        "run_idx": run_idx,
        "episode_idx": episode_idx,
        "mode": "train",
        "circuit": circuit,
        "steps": len(steps),
    }
    if not steps:
        recorder.add("train_update_total", 0.0, **meta)
        return {"policy_loss": 0.0, "value_loss": 0.0}

    update_start = time.perf_counter()
    with recorder.timed("train_returns", sync_cuda=True, **meta):
        rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32, device=device)
        returns = discounted_returns(rewards, gamma=float(gamma)).to(device)
        if normalize_returns and len(returns) > 0:
            mean_r = returns.mean()
            std_r = returns.std(unbiased=False)
            returns = (returns - mean_r) / (std_r + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    policy_update_s = 0.0
    value_update_s = 0.0

    for t, step in enumerate(steps):
        step_dev = step.to(device)
        ret_t = returns[t]
        advantage = ret_t
        if value_network is not None:
            _sync_cuda(device)
            value_pred = value_network(step_dev.observation).squeeze(0)
            _sync_cuda(device)
            advantage = (ret_t - value_pred).detach()

        _sync_cuda(device)
        t0 = time.perf_counter()
        policy_optimizer.zero_grad(set_to_none=True)
        loss = policy.reinforce_loss(step_dev, advantage)
        logits = policy(step_dev.observation)
        probs = policy.masked_action_distribution(logits, step_dev.observation.legal_actions).squeeze(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        loss = loss - float(entropy_beta) * entropy
        loss.backward()
        if clip_grad_norm_policy is not None and float(clip_grad_norm_policy) > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(clip_grad_norm_policy))
        policy_optimizer.step()
        _sync_cuda(device)
        policy_update_s += time.perf_counter() - t0
        policy_losses.append(float(loss.item()))

        if value_network is not None and value_optimizer is not None:
            _sync_cuda(device)
            t0 = time.perf_counter()
            value_optimizer.zero_grad(set_to_none=True)
            value_pred = value_network(step_dev.observation).squeeze(0)
            value_loss = torch.nn.functional.mse_loss(value_pred, ret_t.detach())
            value_loss.backward()
            if clip_grad_norm_value is not None and float(clip_grad_norm_value) > 0:
                torch.nn.utils.clip_grad_norm_(value_network.parameters(), float(clip_grad_norm_value))
            value_optimizer.step()
            _sync_cuda(device)
            value_update_s += time.perf_counter() - t0
            value_losses.append(float(value_loss.item()))

    recorder.add("train_policy_update", policy_update_s, **meta)
    recorder.add("train_value_update", value_update_s, **meta)
    recorder.add("train_update_total", time.perf_counter() - update_start, **meta)
    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
    }


def _timed_evaluate(
    *,
    recorder: TimingRecorder,
    test_circuits: list[str],
    policy: Policy,
    reward_class: type,
    resyn2_baselines: dict[str, dict[str, Any]],
    baseline: str | None,
    available_actions: list[int] | None,
    num_steps: int,
    best_of_rollouts: int,
    repeat_idx: int,
    run_idx: int,
    episode_idx: int,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    per_circuit = []
    for circuit in test_circuits:
        candidates = []
        for rollout_idx in range(max(1, int(best_of_rollouts))):
            ep = _timed_reinforce_episode(
                recorder=recorder,
                file_path=circuit,
                num_steps=num_steps,
                policy=policy,
                reward_class=reward_class,
                sample_actions=best_of_rollouts > 1,
                resyn2_baseline=resyn2_baselines[circuit],
                baseline=baseline,
                available_actions=available_actions,
                repeat_idx=repeat_idx,
                run_idx=run_idx,
                episode_idx=episode_idx,
                mode=f"eval_rollout_{rollout_idx}",
            )
            candidates.append(ep)
        best = max(candidates, key=lambda r: float(r["final_return"]))
        initial_size = max(1, int(best["initial_size"]))
        per_circuit.append(
            {
                "file_path": circuit,
                "final_return": best["final_return"],
                "size_reduction_pct": 100.0 * (initial_size - int(best["final_size"])) / initial_size,
                "num_steps_taken": best["num_steps_taken"],
            }
        )
    recorder.add(
        "eval_total",
        time.perf_counter() - eval_start,
        repeat_idx=repeat_idx,
        run_idx=run_idx,
        episode_idx=episode_idx,
        mode="eval",
        circuits=len(test_circuits),
        rollouts=max(1, int(best_of_rollouts)),
    )
    return {
        "num_circuits": len(per_circuit),
        "mean_final_return": float(np.mean([r["final_return"] for r in per_circuit])) if per_circuit else 0.0,
        "mean_size_reduction_pct": float(np.mean([r["size_reduction_pct"] for r in per_circuit])) if per_circuit else 0.0,
    }


def _projection(
    *,
    rows: list[dict[str, Any]],
    cfg: DictConfig,
    sample_episodes: int,
    test_circuits: int,
    infer_rollouts: int,
    target_num_runs: int,
) -> dict[str, float]:
    train_episode_rows = [
        row for row in rows if row["phase"] == "episode_total" and str(row.get("mode")) == "train"
    ]
    eval_episode_rows = [
        row for row in rows if row["phase"] == "episode_total" and str(row.get("mode", "")).startswith("eval_rollout")
    ]
    train_episode_mean = float(np.mean([row["seconds"] for row in train_episode_rows])) if train_episode_rows else 0.0
    eval_episode_mean = float(np.mean([row["seconds"] for row in eval_episode_rows])) if eval_episode_rows else train_episode_mean
    train_update_mean = _phase_mean(rows, "train_update_total")
    baseline_cache_total = _phase_total(rows, "baseline_cache")
    dimension_probe_total = _phase_total(rows, "dimension_probe")
    checkpoint_mean = _phase_mean(rows, "checkpoint_save")
    model_build_mean = _phase_mean(rows, "model_build")

    target_episodes = int(cfg.episodes)
    target_eval_every = int(cfg.eval_every)
    eval_points = {1, target_episodes}
    eval_points.update(range(target_eval_every, target_episodes + 1, target_eval_every))
    eval_events = len(eval_points)
    final_eval_events = 1

    projected_train = target_num_runs * target_episodes * (train_episode_mean + train_update_mean)
    projected_eval = (
        target_num_runs
        * (eval_events + final_eval_events)
        * int(test_circuits)
        * int(infer_rollouts)
        * eval_episode_mean
    )
    projected_model = target_num_runs * model_build_mean
    projected_checkpoint = target_num_runs * checkpoint_mean
    projected_total = dimension_probe_total + baseline_cache_total + projected_model + projected_train + projected_eval + projected_checkpoint

    measured_train_episode_total = sum(float(row["seconds"]) for row in train_episode_rows)
    measured_steps = sum(int(row.get("steps", 0)) for row in train_episode_rows)
    return {
        "sample_episodes": int(sample_episodes),
        "target_num_runs": int(target_num_runs),
        "target_episodes_per_run": int(target_episodes),
        "target_eval_events_per_run": int(eval_events),
        "target_final_eval_events_per_run": int(final_eval_events),
        "target_test_circuits": int(test_circuits),
        "target_infer_rollouts": int(infer_rollouts),
        "train_episode_mean_s": train_episode_mean,
        "train_update_mean_s": train_update_mean,
        "eval_rollout_episode_mean_s": eval_episode_mean,
        "baseline_cache_total_s": baseline_cache_total,
        "dimension_probe_total_s": dimension_probe_total,
        "model_build_mean_s": model_build_mean,
        "checkpoint_mean_s": checkpoint_mean,
        "projected_train_s": projected_train,
        "projected_eval_s": projected_eval,
        "projected_model_build_s": projected_model,
        "projected_checkpoint_s": projected_checkpoint,
        "projected_total_s": projected_total,
        "projected_total_min": projected_total / 60.0,
        "projected_total_hours": projected_total / 3600.0,
        "measured_train_seconds_per_env_step": measured_train_episode_total / max(1, measured_steps),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect structured timing data for GFlowCircuit training.")
    parser.add_argument("--config-name", default="zhu2020_size_mcnc")
    parser.add_argument("--dataset-cfg", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--infer-rollouts", type=int, default=None)
    parser.add_argument("--warmup-episodes", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json-out", default="output/mock_training_timing.json")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides, e.g. seed=1 available_actions='[0,1,2,3,4]'")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = list(args.overrides)
    if args.dataset_cfg is not None:
        overrides.append(f"dataset_cfg={args.dataset_cfg}")
    overrides.append("logging.tensorboard=false")
    cfg = _load_hydra_config(args.config_name, overrides)
    algorithm_name = _get_algorithm_name(cfg)
    if algorithm_name != "reinforce":
        raise NotImplementedError("mock_training_timing.py currently profiles the REINFORCE path only")

    if args.num_steps is not None:
        cfg.num_steps = int(args.num_steps)
    if args.eval_every is not None:
        cfg.eval_every = int(args.eval_every)
    if args.infer_rollouts is not None:
        OmegaConf.update(cfg, "paper_mode.infer_rollouts", int(args.infer_rollouts), merge=True)

    dataset_cfg = Path(to_absolute_path(cfg.dataset_cfg))
    circuits = load_circuits(dataset_cfg)
    if bool(OmegaConf.select(cfg, "paper_mode.per_circuit_mode")):
        train_circuits, test_circuits = circuits, circuits
    else:
        train_circuits, test_circuits = train_test_split(circuits, cfg.train_ratio, cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recorder = TimingRecorder(device=device)

    with recorder.timed("dimension_probe"):
        obs_dim, num_actions, node_dim, edge_dim = get_obs_dim_and_num_actions(int(cfg.num_steps), train_circuits[0])
    available_actions = normalize_available_actions(OmegaConf.select(cfg, "available_actions"), num_actions)
    reward_class = REWARD_TYPES[cfg.reward.type]
    baseline = OmegaConf.select(cfg, "baseline")
    baseline_scale = float(OmegaConf.select(cfg, "baseline_scale") or 1.0)
    with recorder.timed("baseline_cache"):
        resyn2_baselines = build_resyn2_cache(
            circuits=circuits,
            num_steps=int(cfg.num_steps),
            reward_class=reward_class,
            baseline=baseline,
            baseline_scale=baseline_scale,
        )

    infer_rollouts = int(OmegaConf.select(cfg, "paper_mode.infer_rollouts") or 1)
    policy_lr = float(OmegaConf.select(cfg, "policy_learning_rate") or cfg.learning_rate)
    value_lr = float(OmegaConf.select(cfg, "value_learning_rate") or cfg.learning_rate)
    entropy_beta = float(OmegaConf.select(cfg, "entropy_beta") or 0.0)
    clip_grad_norm_policy = OmegaConf.select(cfg, "clip_grad_norm_policy")
    clip_grad_norm_value = OmegaConf.select(cfg, "clip_grad_norm_value")
    normalize_returns = bool(OmegaConf.select(cfg, "normalize_returns") or False)

    rng = np.random.default_rng(int(cfg.seed))
    measured_training_episodes = 0
    last_policy: Policy | None = None
    last_value: torch.nn.Module | None = None

    for repeat_idx in range(int(args.repeat)):
        for run_idx in range(int(args.num_runs)):
            torch.manual_seed(int(cfg.seed) + run_idx + repeat_idx * 1000)
            with recorder.timed("model_build", repeat_idx=repeat_idx, run_idx=run_idx, sync_cuda=True):
                policy, value_net = _build_models(
                    cfg,
                    obs_dim=obs_dim,
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    num_actions=num_actions,
                    available_actions=available_actions,
                )
                policy = policy.to(device)
                if value_net is not None:
                    value_net = value_net.to(device)
            last_policy = policy
            last_value = value_net
            policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
            value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr) if value_net is not None else None

            for warmup_idx in range(int(args.warmup_episodes)):
                circuit = train_circuits[int(rng.integers(0, len(train_circuits)))]
                episode = _timed_reinforce_episode(
                    recorder=TimingRecorder(device=device),
                    file_path=circuit,
                    num_steps=int(cfg.num_steps),
                    policy=policy,
                    reward_class=reward_class,
                    sample_actions=True,
                    resyn2_baseline=resyn2_baselines[circuit],
                    baseline=baseline,
                    available_actions=available_actions,
                    repeat_idx=repeat_idx,
                    run_idx=run_idx,
                    episode_idx=-(warmup_idx + 1),
                    mode="warmup",
                )
                _timed_train_update(
                    recorder=TimingRecorder(device=device),
                    policy=policy,
                    value_network=value_net,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    episode=episode,
                    device=device,
                    gamma=float(cfg.gamma),
                    entropy_beta=entropy_beta,
                    clip_grad_norm_policy=float(clip_grad_norm_policy) if clip_grad_norm_policy is not None else None,
                    clip_grad_norm_value=float(clip_grad_norm_value) if clip_grad_norm_value is not None else None,
                    normalize_returns=normalize_returns,
                    repeat_idx=repeat_idx,
                    run_idx=run_idx,
                    episode_idx=-(warmup_idx + 1),
                    circuit=circuit,
                )

            for episode_idx in range(1, int(args.episodes) + 1):
                circuit = train_circuits[int(rng.integers(0, len(train_circuits)))]
                episode = _timed_reinforce_episode(
                    recorder=recorder,
                    file_path=circuit,
                    num_steps=int(cfg.num_steps),
                    policy=policy,
                    reward_class=reward_class,
                    sample_actions=True,
                    resyn2_baseline=resyn2_baselines[circuit],
                    baseline=baseline,
                    available_actions=available_actions,
                    repeat_idx=repeat_idx,
                    run_idx=run_idx,
                    episode_idx=episode_idx,
                    mode="train",
                )
                _timed_train_update(
                    recorder=recorder,
                    policy=policy,
                    value_network=value_net,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    episode=episode,
                    device=device,
                    gamma=float(cfg.gamma),
                    entropy_beta=entropy_beta,
                    clip_grad_norm_policy=float(clip_grad_norm_policy) if clip_grad_norm_policy is not None else None,
                    clip_grad_norm_value=float(clip_grad_norm_value) if clip_grad_norm_value is not None else None,
                    normalize_returns=normalize_returns,
                    repeat_idx=repeat_idx,
                    run_idx=run_idx,
                    episode_idx=episode_idx,
                    circuit=circuit,
                )
                measured_training_episodes += 1

                if episode_idx % int(cfg.eval_every) == 0 or episode_idx == 1 or episode_idx == int(args.episodes):
                    _timed_evaluate(
                        recorder=recorder,
                        test_circuits=test_circuits,
                        policy=policy,
                        reward_class=reward_class,
                        resyn2_baselines=resyn2_baselines,
                        baseline=baseline,
                        available_actions=available_actions,
                        num_steps=int(cfg.num_steps),
                        best_of_rollouts=infer_rollouts,
                        repeat_idx=repeat_idx,
                        run_idx=run_idx,
                        episode_idx=episode_idx,
                    )

    if last_policy is not None:
        with tempfile.TemporaryDirectory(prefix="gfc_timing_ckpt_") as tmpdir:
            with recorder.timed("checkpoint_save", sync_cuda=True):
                _save_run_checkpoint(
                    output_dir=Path(tmpdir),
                    run_idx=0,
                    seed=int(cfg.seed),
                    policy=last_policy,
                    value_net=last_value,
                )

    aggregate = _aggregate(recorder.rows)
    projection = _projection(
        rows=recorder.rows,
        cfg=cfg,
        sample_episodes=measured_training_episodes,
        test_circuits=len(test_circuits),
        infer_rollouts=infer_rollouts,
        target_num_runs=int(OmegaConf.select(cfg, "paper_mode.num_runs") or args.num_runs),
    )
    report = {
        "config_name": args.config_name,
        "dataset_cfg": str(dataset_cfg),
        "device": str(device),
        "num_circuits": len(circuits),
        "num_train_circuits": len(train_circuits),
        "num_test_circuits": len(test_circuits),
        "sample": {
            "episodes": int(args.episodes),
            "num_runs": int(args.num_runs),
            "repeat": int(args.repeat),
            "warmup_episodes": int(args.warmup_episodes),
            "num_steps": int(cfg.num_steps),
            "eval_every": int(cfg.eval_every),
            "infer_rollouts": int(infer_rollouts),
        },
        "aggregate": aggregate,
        "projection": projection,
        "rows": recorder.rows,
    }

    json_path = Path(to_absolute_path(args.json_out))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"json_out": str(json_path), "projection": projection}, indent=2))


if __name__ == "__main__":
    main()
