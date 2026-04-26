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
from src.algorithms.gflownet_tb.sampler import sample_tb_trajectory
from src.train.metrics import TensorBoardLogger


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
    ) -> None:
        self.policy = policy
        self.reward_class = reward_class
        self.train_circuits = train_circuits
        self.test_circuits = test_circuits
        self.resyn2_baselines = resyn2_baselines
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
        trajectories_per_episode: int,
        reward_alpha: float,
        reward_eps: float,
        reward_improvement_clip: float,
        best_of_eval_rollouts: int,
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        history: list[dict[str, Any]] = []

        for ep in trange(1, episodes + 1, desc="Training TB"):
            trajectories = []
            for _ in range(max(1, int(trajectories_per_episode))):
                circuit = self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))]
                tr = sample_tb_trajectory(
                    file_path=circuit,
                    num_steps=num_steps,
                    policy=self.policy,
                    reward_class=self.reward_class,
                    reward_alpha=reward_alpha,
                    reward_eps=reward_eps,
                    reward_improvement_clip=reward_improvement_clip,
                    sample_actions=True,
                )
                trajectories.append(tr)

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
                    "test_mean_final_return": eval_summary["mean_final_return"],
                    "test_td_mean_final_return": eval_summary["td_mean_final_return"],
                    "test_mean_size_reduction": eval_summary["mean_size_reduction"],
                    "test_mean_depth_reduction": eval_summary["mean_depth_reduction"],
                    "test_mean_size_reduction_pct": eval_summary["mean_size_reduction_pct"],
                    "test_win_rate_vs_resyn2_1": eval_summary["win_rate_vs_resyn2_1"],
                }
                history.append(row)
                print(json.dumps(row))
                if self._tb is not None:
                    self._tb.add_scalars(
                        ep,
                        {
                            "eval/mean_final_return": float(eval_summary["mean_final_return"]),
                            "eval/td_mean_final_return": float(eval_summary["td_mean_final_return"]),
                            "eval/mean_size_reduction": float(eval_summary["mean_size_reduction"]),
                            "eval/mean_depth_reduction": float(eval_summary["mean_depth_reduction"]),
                            "eval/mean_terminal_reward": float(eval_summary["mean_terminal_reward"]),
                            "eval/best_final_return": float(eval_summary["best_final_return"]),
                            "eval/td_best_final_return": float(eval_summary["td_best_final_return"]),
                            "eval/win_rate_vs_resyn2_1": float(eval_summary["win_rate_vs_resyn2_1"]),
                        },
                    )

        if self._tb is not None:
            self._tb.close()
        return {"history": history}

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
        )
