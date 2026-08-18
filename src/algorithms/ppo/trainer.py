from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from src.algorithms.ppo.eval import evaluate_ppo
from src.algorithms.ppo.loss import compute_gae, ppo_minibatch_loss
from src.algorithms.ppo.policy import PPOPolicy
from src.algorithms.ppo.sampler import sample_ppo_trajectory
from src.algorithms.ppo.types import PPORollout, PPOTransition
from src.discovery_metrics import (
    build_training_discovery_tracker,
    finalize_training_discovery,
    record_training_trajectory,
)
from src.metrics import TensorBoardLogger


class PPOTrainer:
    def __init__(
        self,
        *,
        policy: PPOPolicy,
        value_network: nn.Module,
        reward_class: type,
        train_circuits: list[str],
        test_circuits: list[str],
        resyn2_baselines: dict[str, dict[str, Any]],
        device: torch.device,
        seed: int,
        log_dir: Path | None = None,
        baseline: str | None = None,
        available_actions: list[int] | None = None,
    ) -> None:
        self.policy = policy
        self.value_network = value_network
        self.reward_class = reward_class
        self.train_circuits = train_circuits
        self.test_circuits = test_circuits
        self.resyn2_baselines = resyn2_baselines
        self.available_actions = available_actions
        self.baseline = baseline
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._tb = TensorBoardLogger(log_dir) if log_dir is not None else None

    def _collect_rollout(self, *, num_steps: int, rollout_steps: int) -> PPORollout:
        trajectories = []
        transitions: list[PPOTransition] = []
        while len(transitions) < int(rollout_steps):
            circuit = self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))]
            trajectory = sample_ppo_trajectory(
                file_path=circuit,
                num_steps=int(num_steps),
                policy=self.policy,
                value_network=self.value_network,
                reward_class=self.reward_class,
                sample_actions=True,
                baseline=self.baseline,
                resyn2_baseline=self.resyn2_baselines[circuit],
                available_actions=self.available_actions,
            )
            trajectories.append(trajectory)
            transitions.extend(trajectory.transitions)
            if not trajectory.transitions:
                break
        return PPORollout(trajectories=trajectories, transitions=transitions)

    def train(
        self,
        *,
        episodes: int,
        num_steps: int,
        eval_every: int,
        learning_rate: float,
        gamma: float,
        rollout_steps: int,
        ppo_epochs: int,
        minibatch_size: int,
        clip_eps: float,
        value_loss_coef: float,
        entropy_beta: float,
        normalize_advantages: bool,
        clip_grad_norm: float | None,
        gae_lambda: float,
        best_of_eval_rollouts: int,
        discovery_metrics_enabled: bool = True,
        discovery_emit_every_trajectories: int = 50,
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_network.parameters()),
            lr=float(learning_rate),
        )
        history: list[dict[str, Any]] = []
        discovery = build_training_discovery_tracker(
            enabled=discovery_metrics_enabled,
            circuits=self.train_circuits,
            resyn2_baselines=self.resyn2_baselines,
            emit_every_trajectories=discovery_emit_every_trajectories,
            tensorboard_logger=self._tb,
        )

        for ep in trange(1, int(episodes) + 1, desc="Training PPO"):
            rollout = self._collect_rollout(num_steps=int(num_steps), rollout_steps=int(rollout_steps))
            for trajectory in rollout.trajectories:
                record_training_trajectory(discovery, trajectory)
            transitions = rollout.transitions
            if transitions:
                rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
                dones = torch.tensor([t.done for t in transitions], dtype=torch.bool, device=self.device)
                values = torch.stack([t.old_value.to(self.device).reshape(()) for t in transitions])
                next_values = torch.stack([t.next_value.to(self.device).reshape(()) for t in transitions])
                advantages, returns = compute_gae(
                    rewards=rewards,
                    dones=dones,
                    values=values,
                    next_values=next_values,
                    gamma=float(gamma),
                    gae_lambda=float(gae_lambda),
                )
                if normalize_advantages and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            else:
                rewards = torch.empty((0,), dtype=torch.float32, device=self.device)
                advantages = torch.empty((0,), dtype=torch.float32, device=self.device)
                returns = torch.empty((0,), dtype=torch.float32, device=self.device)

            batch_stats = []
            grad_norms = []
            if transitions:
                batch_size = len(transitions)
                mb_size = max(1, min(int(minibatch_size), batch_size))
                for _ in range(max(1, int(ppo_epochs))):
                    perm = torch.as_tensor(self.rng.permutation(batch_size), dtype=torch.long, device=self.device)
                    for start in range(0, batch_size, mb_size):
                        idx = perm[start : start + mb_size]
                        optimizer.zero_grad(set_to_none=True)
                        stats = ppo_minibatch_loss(
                            policy=self.policy,
                            value_network=self.value_network,
                            transitions=transitions,
                            advantages=advantages,
                            returns=returns,
                            indices=idx,
                            clip_eps=float(clip_eps),
                            value_loss_coef=float(value_loss_coef),
                            entropy_beta=float(entropy_beta),
                        )
                        stats.total_loss.backward()
                        if clip_grad_norm is not None and float(clip_grad_norm) > 0:
                            params = list(self.policy.parameters()) + list(self.value_network.parameters())
                            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, float(clip_grad_norm)).item()))
                        optimizer.step()
                        batch_stats.append(stats)

            mean_final_return = (
                float(np.mean([trajectory.final_return for trajectory in rollout.trajectories]))
                if rollout.trajectories
                else 0.0
            )
            mean_traj_len = (
                float(np.mean([len(trajectory.transitions) for trajectory in rollout.trajectories]))
                if rollout.trajectories
                else 0.0
            )

            def _mean_stat(name: str) -> float:
                if not batch_stats:
                    return 0.0
                return float(np.mean([float(getattr(stat, name).detach().item()) for stat in batch_stats]))

            train_total_loss = _mean_stat("total_loss")
            train_actor_loss = _mean_stat("actor_loss")
            train_value_loss = _mean_stat("value_loss")
            train_entropy = _mean_stat("entropy")
            train_approx_kl = _mean_stat("approx_kl")
            train_clip_fraction = _mean_stat("clip_fraction")
            train_mean_advantage = float(advantages.mean().detach().item()) if advantages.numel() else 0.0
            train_mean_return = float(returns.mean().detach().item()) if returns.numel() else 0.0
            train_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

            if self._tb is not None:
                self._tb.add_scalars(
                    ep,
                    {
                        "train/total_loss": train_total_loss,
                        "train/actor_loss": train_actor_loss,
                        "train/value_loss": train_value_loss,
                        "train/policy_entropy": train_entropy,
                        "train/approx_kl": train_approx_kl,
                        "train/clip_fraction": train_clip_fraction,
                        "train/mean_advantage": train_mean_advantage,
                        "train/mean_return": train_mean_return,
                        "train/final_return": mean_final_return,
                        "train/trajectory_len": mean_traj_len,
                        "train/grad_norm": train_grad_norm,
                    },
                )

            if ep % int(eval_every) == 0 or ep == 1 or ep == int(episodes):
                eval_summary = self.evaluate(
                    num_steps=int(num_steps),
                    best_of_rollouts=int(best_of_eval_rollouts),
                )
                row = {
                    "episode": ep,
                    "train_total_loss": train_total_loss,
                    "train_actor_loss": train_actor_loss,
                    "train_value_loss": train_value_loss,
                    "train_entropy": train_entropy,
                    "train_approx_kl": train_approx_kl,
                    "train_clip_fraction": train_clip_fraction,
                    "train_final_return": mean_final_return,
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
                            "eval/mean_final_size": float(eval_summary["mean_final_size"]),
                            "eval/mean_final_depth": float(eval_summary["mean_final_depth"]),
                            "eval/mean_final_qor": float(eval_summary["mean_final_qor"]),
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

    def evaluate(self, *, num_steps: int, best_of_rollouts: int = 1) -> dict[str, Any]:
        return evaluate_ppo(
            circuits=self.test_circuits,
            policy=self.policy,
            value_network=self.value_network,
            reward_class=self.reward_class,
            resyn2_baselines=self.resyn2_baselines,
            num_steps=int(num_steps),
            best_of_rollouts=int(best_of_rollouts),
            baseline=self.baseline,
            available_actions=self.available_actions,
        )
