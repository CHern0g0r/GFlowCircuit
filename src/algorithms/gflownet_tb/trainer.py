from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from src.algorithms.gflownet_tb.eval import evaluate_tb
from src.algorithms.gflownet_tb.loss import trajectory_balance_loss
from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.sampler import sample_tb_trajectories
from src.discovery_metrics import (
    build_training_discovery_tracker,
    finalize_training_discovery,
    record_training_trajectory,
)
from src.metrics import TensorBoardLogger
from src.algorithms.gflownet_tb.optim import build_tb_optimizer


# Backwards compatibility for callers that imported the former private helper.
_build_tb_optimizer = build_tb_optimizer


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _tb_exploration_epsilon(
    *,
    episode: int,
    episodes: int,
    enabled: bool,
    epsilon_start: float,
    epsilon_end: float,
    warmup_episodes: int,
    decay_episodes: int | None,
) -> float:
    epsilon_start = _validate_probability("exploration_epsilon_start", epsilon_start)
    epsilon_end = _validate_probability("exploration_epsilon_end", epsilon_end)
    warmup_episodes = int(warmup_episodes)
    if warmup_episodes < 0:
        raise ValueError(f"exploration_warmup_episodes must be >= 0, got {warmup_episodes}")
    if decay_episodes is not None:
        decay_episodes = int(decay_episodes)
        if decay_episodes <= 0:
            raise ValueError(f"exploration_decay_episodes must be positive, got {decay_episodes}")

    if not bool(enabled):
        return 0.0

    episode = int(episode)
    episodes = int(episodes)
    if episode <= warmup_episodes:
        return epsilon_start

    resolved_decay_episodes = decay_episodes
    if resolved_decay_episodes is None:
        resolved_decay_episodes = max(1, episodes - warmup_episodes)
    progress = min(1.0, max(0.0, float(episode - warmup_episodes) / float(resolved_decay_episodes)))
    return epsilon_start + progress * (epsilon_end - epsilon_start)


class TBGFlowNetTrainer:
    def __init__(
        self,
        *,
        policy: TBGFlowNetPolicy,
        reward_class: type,
        train_circuits: list[str],
        test_circuits: list[str],
        resyn2_baselines: dict[str, dict[str, Any]],
        device: torch.device,
        seed: int,
        log_dir: Path | None = None,
        available_actions: list[int] | None = None,
    ) -> None:
        self.policy = policy
        self.reward_class = reward_class
        self.train_circuits = train_circuits
        self.test_circuits = test_circuits
        self.resyn2_baselines = resyn2_baselines
        self.available_actions = available_actions
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._tb = TensorBoardLogger(log_dir) if log_dir is not None else None

    def train(
        self,
        *,
        episodes: int,
        num_steps: int,
        eval_every: int,
        learning_rate: float,
        log_z_learning_rate: float,
        trajectories_per_episode: int,
        reward_alpha: float,
        reward_eps: float,
        reward_improvement_clip: float,
        exploration_epsilon_enabled: bool,
        exploration_epsilon_start: float,
        exploration_epsilon_end: float,
        exploration_warmup_episodes: int,
        exploration_decay_episodes: int | None,
        best_of_eval_rollouts: int,
        discovery_metrics_enabled: bool = True,
        discovery_emit_every_trajectories: int = 50,
    ) -> dict[str, Any]:
        optimizer = build_tb_optimizer(
            self.policy,
            learning_rate=learning_rate,
            log_z_learning_rate=log_z_learning_rate,
        )
        history: list[dict[str, Any]] = []
        discovery = build_training_discovery_tracker(
            enabled=discovery_metrics_enabled,
            circuits=self.train_circuits,
            resyn2_baselines=self.resyn2_baselines,
            emit_every_trajectories=discovery_emit_every_trajectories,
            tensorboard_logger=self._tb,
        )

        for ep in trange(1, episodes + 1, desc="Training TB"):
            exploration_epsilon = _tb_exploration_epsilon(
                episode=ep,
                episodes=episodes,
                enabled=exploration_epsilon_enabled,
                epsilon_start=exploration_epsilon_start,
                epsilon_end=exploration_epsilon_end,
                warmup_episodes=exploration_warmup_episodes,
                decay_episodes=exploration_decay_episodes,
            )
            batch_size = max(1, int(trajectories_per_episode))
            circuits = [
                self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))] for _ in range(batch_size)
            ]
            trajectories = sample_tb_trajectories(
                file_paths=circuits,
                num_steps=num_steps,
                policy=self.policy,
                reward_class=self.reward_class,
                reward_alpha=reward_alpha,
                reward_eps=reward_eps,
                reward_improvement_clip=reward_improvement_clip,
                sample_actions=True,
                available_actions=self.available_actions,
                epsilon_uniform=exploration_epsilon,
            )
            for trajectory in trajectories:
                record_training_trajectory(discovery, trajectory)

            optimizer.zero_grad(set_to_none=True)
            log_pf = torch.stack([t.log_pf_sum for t in trajectories])
            log_pb = torch.stack([t.log_pb_sum for t in trajectories])
            log_r = torch.tensor([t.log_reward for t in trajectories], dtype=torch.float32, device=self.device)
            loss = trajectory_balance_loss(
                log_z=self.policy.log_z,
                log_pf_sums=log_pf,
                log_rewards=log_r,
                log_pb_sums=log_pb,
            )
            loss.backward()
            optimizer.step()

            mean_final_return = float(np.mean([t.final_return for t in trajectories])) if trajectories else 0.0
            mean_terminal_reward = float(np.mean([t.terminal_reward for t in trajectories])) if trajectories else 0.0
            mean_traj_len = float(np.mean([len(t.steps) for t in trajectories])) if trajectories else 0.0

            if self._tb is not None:
                self._tb.add_scalars(
                    ep,
                    {
                        "train/policy_loss": float(loss.item()),
                        "train/log_z": float(self.policy.log_z.detach().item()),
                        "train/final_return": mean_final_return,
                        "train/terminal_reward": mean_terminal_reward,
                        "train/trajectory_len": mean_traj_len,
                        "train/exploration_epsilon": exploration_epsilon,
                    },
                )

            if ep % eval_every == 0 or ep == 1 or ep == episodes:
                eval_summary = self.evaluate(
                    num_steps=num_steps,
                    reward_alpha=reward_alpha,
                    reward_eps=reward_eps,
                    reward_improvement_clip=reward_improvement_clip,
                    best_of_rollouts=best_of_eval_rollouts,
                )
                row = {
                    "episode": ep,
                    "train_policy_loss": float(loss.item()),
                    "train_log_z": float(self.policy.log_z.detach().item()),
                    "train_final_return": mean_final_return,
                    "train_exploration_epsilon": exploration_epsilon,
                    "test_mean_final_return": eval_summary["mean_final_return"],
                    "test_mean_comparable_return": eval_summary["mean_comparable_return"],
                    "test_mean_size_reduction": eval_summary["mean_size_reduction"],
                    "test_mean_depth_reduction": eval_summary["mean_depth_reduction"],
                    "test_mean_size_reduction_pct": eval_summary["mean_size_reduction_pct"],
                    "test_win_rate_vs_resyn2_1": eval_summary["win_rate_vs_resyn2_1"],
                    "test_win_rate_vs_resyn2_2": eval_summary["win_rate_vs_resyn2_2"],
                    "test_mean_normalized_improvement_vs_resyn2_2": eval_summary[
                        "mean_normalized_improvement_vs_resyn2_2"
                    ],
                }
                history.append(row)
                print(json.dumps(row))
                if self._tb is not None:
                    self._tb.add_scalars(
                        ep,
                        {
                            "eval/mean_final_return": float(eval_summary["mean_final_return"]),
                            "eval/mean_comparable_return": float(eval_summary["mean_comparable_return"]),
                            "eval/mean_size_reduction": float(eval_summary["mean_size_reduction"]),
                            "eval/mean_depth_reduction": float(eval_summary["mean_depth_reduction"]),
                            "eval/mean_terminal_reward": float(eval_summary["mean_terminal_reward"]),
                            "eval/mean_final_size": float(eval_summary["mean_final_size"]),
                            "eval/mean_final_depth": float(eval_summary["mean_final_depth"]),
                            "eval/mean_final_qor": float(eval_summary["mean_final_qor"]),
                            "eval/best_final_return": float(eval_summary["best_final_return"]),
                            "eval/best_comparable_return": float(eval_summary["best_comparable_return"]),
                            "eval/win_rate_vs_resyn2_1": float(eval_summary["win_rate_vs_resyn2_1"]),
                            "eval/win_rate_vs_resyn2_2": float(eval_summary["win_rate_vs_resyn2_2"]),
                            "eval/mean_normalized_improvement_vs_resyn2_2": float(
                                eval_summary["mean_normalized_improvement_vs_resyn2_2"]
                            ),
                        },
                    )

        discovery_out = finalize_training_discovery(discovery)
        if self._tb is not None:
            self._tb.close()
        return {"history": history, **discovery_out}

    def evaluate(
        self,
        *,
        num_steps: int,
        reward_alpha: float,
        reward_eps: float,
        reward_improvement_clip: float,
        best_of_rollouts: int = 1,
    ) -> dict[str, Any]:
        return evaluate_tb(
            circuits=self.test_circuits,
            policy=self.policy,
            reward_class=self.reward_class,
            resyn2_baselines=self.resyn2_baselines,
            num_steps=num_steps,
            reward_alpha=reward_alpha,
            reward_eps=reward_eps,
            reward_improvement_clip=reward_improvement_clip,
            best_of_rollouts=best_of_rollouts,
            available_actions=self.available_actions,
        )
