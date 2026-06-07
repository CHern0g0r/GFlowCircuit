from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from src.algorithms.pcn.archive import PCNArchive
from src.algorithms.pcn.eval import evaluate_pcn
from src.algorithms.pcn.loss import pcn_cross_entropy_loss
from src.algorithms.pcn.policy import PCNPolicy
from src.algorithms.pcn.sampler import sample_pcn_trajectory
from src.metrics import TensorBoardLogger
from src.models.mo_rewards import MultiObjectiveReward


class PCNTrainer:
    def __init__(
        self,
        *,
        policy: PCNPolicy,
        mo_reward_class: type[MultiObjectiveReward],
        train_circuits: list[str],
        test_circuits: list[str],
        resyn2_baselines: dict[str, dict[str, Any]],
        device: torch.device,
        seed: int,
        log_dir: Path | None = None,
        available_actions: list[int] | None = None,
        archive_capacity: int = 256,
        gamma: float = 1.0,
        crowding_threshold: float = 0.2,
        duplicate_penalty: float = 1e-5,
    ) -> None:
        self.policy = policy
        self.mo_reward_class = mo_reward_class
        self.train_circuits = train_circuits
        self.test_circuits = test_circuits
        self.resyn2_baselines = resyn2_baselines
        self.available_actions = available_actions
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.archive = PCNArchive(
            capacity=int(archive_capacity),
            gamma=float(gamma),
            crowding_threshold=float(crowding_threshold),
            duplicate_penalty=float(duplicate_penalty),
        )
        self._tb = TensorBoardLogger(log_dir) if log_dir is not None else None

    def train(
        self,
        *,
        episodes: int,
        num_steps: int,
        eval_every: int,
        learning_rate: float,
        random_seed_episodes: int,
        collect_episodes_per_iter: int,
        train_updates_per_iter: int,
        batch_size: int,
        desired_return_clip: bool,
        eval_target_limit: int | None,
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(learning_rate))
        history: list[dict[str, Any]] = []
        total_collected = 0

        seed_count = min(int(random_seed_episodes), int(episodes))
        for _ in trange(seed_count, desc="Seeding PCN archive"):
            circuit = self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))]
            trajectory = sample_pcn_trajectory(
                file_path=circuit,
                num_steps=int(num_steps),
                policy=self.policy,
                mo_reward_class=self.mo_reward_class,
                sample_actions=True,
                rng=self.rng,
                available_actions=self.available_actions,
                gamma=self.archive.gamma,
            )
            self.archive.add(trajectory)
            total_collected += 1

        progress = trange(total_collected, int(episodes), desc="Training PCN")
        while total_collected < int(episodes):
            losses: list[float] = []
            for _ in range(max(1, int(train_updates_per_iter))):
                batch = self.archive.sample_datapoints(batch_size=int(batch_size), rng=self.rng)
                optimizer.zero_grad(set_to_none=True)
                loss = pcn_cross_entropy_loss(self.policy, batch)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))

            collected_this_iter = 0
            returns_this_iter = []
            for _ in range(max(1, int(collect_episodes_per_iter))):
                if total_collected >= int(episodes):
                    break
                target = self.archive.sample_exploration_target(rng=self.rng)
                circuit = self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))]
                trajectory = sample_pcn_trajectory(
                    file_path=circuit,
                    num_steps=int(num_steps),
                    policy=self.policy,
                    mo_reward_class=self.mo_reward_class,
                    sample_actions=True,
                    rng=self.rng,
                    available_actions=self.available_actions,
                    desired_return=target.desired_return if target is not None else None,
                    desired_horizon=target.desired_horizon if target is not None else None,
                    desired_return_clip=bool(desired_return_clip),
                    gamma=self.archive.gamma,
                )
                self.archive.add(trajectory)
                returns_this_iter.append(float(trajectory.return_vec.sum().item()))
                total_collected += 1
                collected_this_iter += 1
                progress.update(1)

            mean_loss = float(np.mean(losses)) if losses else 0.0
            mean_iter_return = float(np.mean(returns_this_iter)) if returns_this_iter else 0.0
            nd_count = len(self.archive.non_dominated_trajectories())
            if self._tb is not None:
                self._tb.add_scalars(
                    total_collected,
                    {
                        "train/loss": mean_loss,
                        "train/archive_size": float(len(self.archive)),
                        "train/nondominated_count": float(nd_count),
                        "train/mean_return": mean_iter_return,
                    },
                )

            if (
                total_collected == seed_count + collected_this_iter
                or total_collected % int(eval_every) == 0
                or total_collected >= int(episodes)
            ):
                eval_summary = self.evaluate(
                    num_steps=int(num_steps),
                    eval_target_limit=eval_target_limit,
                    desired_return_clip=bool(desired_return_clip),
                )
                row = {
                    "episode": total_collected,
                    "train_loss": mean_loss,
                    "train_mean_return": mean_iter_return,
                    "archive_size": len(self.archive),
                    "nondominated_count": nd_count,
                    "test_mean_final_return": eval_summary["mean_final_return"],
                    "test_mean_comparable_return": eval_summary["mean_comparable_return"],
                    "test_mean_size_reduction": eval_summary["mean_size_reduction"],
                    "test_mean_depth_reduction": eval_summary["mean_depth_reduction"],
                    "test_mean_size_reduction_pct": eval_summary["mean_size_reduction_pct"],
                    "test_hypervolume": eval_summary["hypervolume"],
                    "test_nondominated_count": eval_summary["nondominated_count"],
                    "test_win_rate_vs_resyn2_1": eval_summary["win_rate_vs_resyn2_1"],
                    "test_win_rate_vs_resyn2_2": eval_summary["win_rate_vs_resyn2_2"],
                }
                history.append(row)
                print(json.dumps(row))
                if self._tb is not None:
                    self._tb.add_scalars(
                        total_collected,
                        {
                            "eval/mean_final_return": float(eval_summary["mean_final_return"]),
                            "eval/hypervolume": float(eval_summary["hypervolume"]),
                            "eval/nondominated_count": float(eval_summary["nondominated_count"]),
                            "eval/mean_final_size": float(eval_summary["mean_final_size"]),
                            "eval/mean_final_depth": float(eval_summary["mean_final_depth"]),
                            "eval/mean_final_qor": float(eval_summary["mean_final_qor"]),
                        },
                    )

        progress.close()
        if self._tb is not None:
            self._tb.close()
        return {"history": history}

    def evaluate(
        self,
        *,
        num_steps: int,
        eval_target_limit: int | None = None,
        desired_return_clip: bool = False,
    ) -> dict[str, Any]:
        return evaluate_pcn(
            circuits=self.test_circuits,
            policy=self.policy,
            mo_reward_class=self.mo_reward_class,
            resyn2_baselines=self.resyn2_baselines,
            archive=self.archive,
            num_steps=int(num_steps),
            seed=self.seed,
            available_actions=self.available_actions,
            eval_target_limit=eval_target_limit,
            desired_return_clip=desired_return_clip,
            gamma=self.archive.gamma,
        )

    def checkpoint_metadata(self, *, limit: int | None = None) -> dict[str, Any]:
        meta = self.archive.metadata(limit=limit)
        reward_type = getattr(self.mo_reward_class, "func", self.mo_reward_class)
        meta["objective_names"] = list(getattr(reward_type, "objective_names", ()))
        meta["objective_dim"] = int(getattr(reward_type, "objective_dim", 0))
        return meta
