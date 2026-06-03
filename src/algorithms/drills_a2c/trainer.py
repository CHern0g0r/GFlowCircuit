from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from src.algorithms.drills_a2c.eval import evaluate_drills_a2c
from src.algorithms.drills_a2c.loss import drills_a2c_loss
from src.algorithms.drills_a2c.policy import DrillsA2CPolicy
from src.algorithms.drills_a2c.sampler import sample_drills_a2c_trajectory
from src.algorithms.drills_a2c.types import DrillsA2CStep
from src.metrics import TensorBoardLogger


class DrillsA2CTrainer:
    def __init__(
        self,
        *,
        policy: DrillsA2CPolicy,
        value_network: nn.Module,
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
        self.value_network = value_network
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
        gamma: float,
        trajectories_per_episode: int,
        value_loss_coef: float,
        entropy_beta: float,
        clip_grad_norm: float | None,
        normalize_advantages: bool,
        best_of_eval_rollouts: int,
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_network.parameters()),
            lr=float(learning_rate),
        )
        history: list[dict[str, Any]] = []

        for ep in trange(1, int(episodes) + 1, desc="Training DRiLLS-A2C"):
            trajectories = []
            all_steps: list[DrillsA2CStep] = []
            for _ in range(max(1, int(trajectories_per_episode))):
                circuit = self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))]
                tr = sample_drills_a2c_trajectory(
                    file_path=circuit,
                    num_steps=int(num_steps),
                    policy=self.policy,
                    reward_class=self.reward_class,
                    sample_actions=True,
                    available_actions=self.available_actions,
                )
                trajectories.append(tr)
                all_steps.extend(tr.steps)

            optimizer.zero_grad(set_to_none=True)
            loss = drills_a2c_loss(
                policy=self.policy,
                value_network=self.value_network,
                steps=all_steps,
                gamma=float(gamma),
                value_loss_coef=float(value_loss_coef),
                entropy_beta=float(entropy_beta),
                normalize_advantages=bool(normalize_advantages),
            )
            loss.total_loss.backward()
            grad_norm = 0.0
            if clip_grad_norm is not None and float(clip_grad_norm) > 0:
                params = list(self.policy.parameters()) + list(self.value_network.parameters())
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(clip_grad_norm)).item())
            optimizer.step()

            mean_final_return = float(np.mean([t.final_return for t in trajectories])) if trajectories else 0.0
            mean_traj_len = float(np.mean([len(t.steps) for t in trajectories])) if trajectories else 0.0
            feasible_depth_rate = (
                float(np.mean([1.0 if t.feasible_depth else 0.0 for t in trajectories])) if trajectories else 0.0
            )

            if self._tb is not None:
                self._tb.add_scalars(
                    ep,
                    {
                        "train/total_loss": float(loss.total_loss.detach().item()),
                        "train/actor_loss": float(loss.actor_loss.detach().item()),
                        "train/critic_loss": float(loss.critic_loss.detach().item()),
                        "train/policy_entropy": float(loss.entropy.detach().item()),
                        "train/mean_advantage": float(loss.mean_advantage.detach().item()),
                        "train/mean_td_target": float(loss.mean_target.detach().item()),
                        "train/final_return": mean_final_return,
                        "train/trajectory_len": mean_traj_len,
                        "train/feasible_depth_rate": feasible_depth_rate,
                        "train/grad_norm": grad_norm,
                    },
                )

            if ep % int(eval_every) == 0 or ep == 1 or ep == int(episodes):
                eval_summary = self.evaluate(
                    num_steps=int(num_steps),
                    best_of_rollouts=int(best_of_eval_rollouts),
                )
                row = {
                    "episode": ep,
                    "train_total_loss": float(loss.total_loss.detach().item()),
                    "train_actor_loss": float(loss.actor_loss.detach().item()),
                    "train_critic_loss": float(loss.critic_loss.detach().item()),
                    "train_entropy": float(loss.entropy.detach().item()),
                    "train_final_return": mean_final_return,
                    "train_feasible_depth_rate": feasible_depth_rate,
                    "test_mean_final_return": eval_summary["mean_final_return"],
                    "test_mean_comparable_return": eval_summary["mean_comparable_return"],
                    "test_mean_size_reduction": eval_summary["mean_size_reduction"],
                    "test_mean_depth_reduction": eval_summary["mean_depth_reduction"],
                    "test_mean_size_reduction_pct": eval_summary["mean_size_reduction_pct"],
                    "test_mean_feasible_depth_rate": eval_summary["mean_feasible_depth_rate"],
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
                            "eval/mean_feasible_depth_rate": float(eval_summary["mean_feasible_depth_rate"]),
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

        if self._tb is not None:
            self._tb.close()
        return {"history": history}

    def evaluate(self, *, num_steps: int, best_of_rollouts: int = 1) -> dict[str, Any]:
        return evaluate_drills_a2c(
            circuits=self.test_circuits,
            policy=self.policy,
            reward_class=self.reward_class,
            resyn2_baselines=self.resyn2_baselines,
            num_steps=int(num_steps),
            best_of_rollouts=int(best_of_rollouts),
            available_actions=self.available_actions,
        )

