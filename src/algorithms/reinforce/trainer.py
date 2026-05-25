from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from src.algorithms.reinforce.episode import run_reinforce_episode
from src.algorithms.reinforce.policy import Policy
from src.eval_metrics import (
    aggregate_common_eval_metrics,
    final_qor,
    normalized_improvement_vs_resyn2_2,
    size_reduction_pct,
)
from src.metrics import TensorBoardLogger
from src.utils import StepSample, discounted_returns


class ReinforceTrainer:
    def __init__(
        self,
        *,
        policy: Policy,
        value_network: nn.Module | None,
        reward_class: type,
        terminal_reward: bool,
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
        self.terminal_reward = bool(terminal_reward)
        self.train_circuits = train_circuits
        self.test_circuits = test_circuits
        self.baseline = baseline
        self.available_actions = available_actions
        self.resyn2_baselines = resyn2_baselines
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._tb = TensorBoardLogger(log_dir) if log_dir is not None else None

    def train(
        self,
        *,
        num_steps: int,
        episodes: int,
        eval_every: int,
        policy_learning_rate: float,
        value_learning_rate: float,
        gamma: float,
        baseline_alpha: float,
        best_of_eval_rollouts: int = 1,
        entropy_beta: float = 0.0,
        clip_grad_norm_policy: float | None = None,
        clip_grad_norm_value: float | None = None,
        normalize_returns: bool = False,
    ) -> dict[str, Any]:
        baseline_ema = torch.tensor(0.0, device=self.device)
        history: list[dict[str, Any]] = []

        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(policy_learning_rate))
        value_optimizer = None
        if self.value_network is not None:
            value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=float(value_learning_rate))

        for ep in trange(1, int(episodes) + 1, desc="Training REINFORCE"):
            circuit = self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))]
            episode = run_reinforce_episode(
                file_path=circuit,
                num_steps=int(num_steps),
                policy=self.policy,
                sample_actions=True,
                reward_class=self.reward_class,
                resyn2_baseline=self.resyn2_baselines[circuit],
                baseline=self.baseline,
                available_actions=self.available_actions,
            )

            steps: list[StepSample] = episode["trajectory"]
            if not self.terminal_reward:
                rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32, device=self.device)
                returns = discounted_returns(rewards, gamma=float(gamma)).to(self.device)
            else:
                terminal_return = torch.tensor(
                    float(episode["final_step_reward"]), dtype=torch.float32, device=self.device
                )
                if len(steps) == 0:
                    returns = torch.empty((0,), dtype=torch.float32, device=self.device)
                else:
                    exponents = torch.arange(len(steps) - 1, -1, -1, device=self.device, dtype=torch.float32)
                    returns = (float(gamma) ** exponents) * terminal_return

            if normalize_returns and len(returns) > 0:
                mean_r = returns.mean()
                std_r = returns.std(unbiased=False)
                returns = (returns - mean_r) / (std_r + 1e-8)

            policy_losses: list[float] = []
            value_losses: list[float] = []
            policy_entropies: list[float] = []
            policy_grad_norms: list[float] = []
            value_grad_norms: list[float] = []

            for t, step in enumerate(steps):
                step_dev = step.to(self.device)
                ret_t = returns[t]
                advantage = ret_t
                if self.value_network is not None:
                    value_pred = self.value_network(step_dev.observation).squeeze(0)
                    advantage = (ret_t - value_pred).detach()

                policy_optimizer.zero_grad(set_to_none=True)
                loss = self.policy.reinforce_loss(step_dev, advantage)
                logits = self.policy(step_dev.observation)
                probs = self.policy.masked_action_distribution(logits, step_dev.observation.legal_actions).squeeze(0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))
                loss = loss - float(entropy_beta) * entropy
                loss.backward()
                if clip_grad_norm_policy is not None and float(clip_grad_norm_policy) > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(clip_grad_norm_policy)).item()
                    )
                    policy_grad_norms.append(grad_norm)
                policy_optimizer.step()
                policy_losses.append(float(loss.item()))
                policy_entropies.append(float(entropy.item()))

                if self.value_network is not None and value_optimizer is not None:
                    value_optimizer.zero_grad(set_to_none=True)
                    value_pred = self.value_network(step_dev.observation).squeeze(0)
                    value_loss = torch.nn.functional.mse_loss(value_pred, ret_t.detach())
                    value_loss.backward()
                    if clip_grad_norm_value is not None and float(clip_grad_norm_value) > 0:
                        grad_norm_v = float(
                            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), float(clip_grad_norm_value)).item()
                        )
                        value_grad_norms.append(grad_norm_v)
                    value_optimizer.step()
                    value_losses.append(float(value_loss.item()))

            if len(returns) > 0:
                baseline_ema = (1.0 - float(baseline_alpha)) * baseline_ema + float(baseline_alpha) * returns.mean()

            train_return = float(returns[0].item()) if len(returns) > 0 else 0.0
            mean_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
            mean_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
            mean_policy_entropy = float(np.mean(policy_entropies)) if policy_entropies else 0.0
            mean_policy_grad_norm = float(np.mean(policy_grad_norms)) if policy_grad_norms else 0.0
            mean_value_grad_norm = float(np.mean(value_grad_norms)) if value_grad_norms else 0.0

            _, action_counts = np.unique(episode["actions_applied"], return_counts=True)
            action_prob = action_counts / action_counts.sum()
            action_entropy = -np.sum(action_prob * np.log(action_prob))

            if self._tb is not None:
                self._tb.add_scalars(
                    ep,
                    {
                        "train/return": train_return,
                        "train/policy_loss": mean_policy_loss,
                        "train/value_loss": mean_value_loss,
                        "train/best_size": float(episode["best_size"]),
                        "train/best_depth": float(episode["best_depth"]),
                        "train/best_qor": float(episode["best_qor"]),
                        "train/final_return": float(episode["final_return"]),
                        "train/action_entropy": float(action_entropy),
                        "train/policy_entropy": float(mean_policy_entropy),
                        "train/policy_grad_norm": float(mean_policy_grad_norm),
                        "train/value_grad_norm": float(mean_value_grad_norm),
                        "train/resyn2_baseline_total_reward": float(episode["resyn2_baseline_total_reward"]),
                        "train/resyn2_baseline_final_step_reward": float(episode["resyn2_baseline_final_step_reward"]),
                        "train/reward_raw_gain_mean": float(episode["reward_raw_gain_mean"]),
                        "train/reward_adjusted_mean": float(episode["reward_adjusted_mean"]),
                        "train/reward_baseline_per_step": float(episode["reward_baseline_per_step"]),
                    },
                )

            if ep % int(eval_every) == 0 or ep == 1 or ep == int(episodes):
                eval_summary = self.evaluate(num_steps=int(num_steps), best_of_rollouts=int(best_of_eval_rollouts))
                row = {
                    "episode": ep,
                    "train_circuit": circuit,
                    "train_final_return": episode["final_return"],
                    "train_final_size": episode["final_size"],
                    "train_steps": episode["num_steps_taken"],
                    "baseline": float(baseline_ema.item()),
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
                    "test_mean_resyn2_baseline_total_reward": eval_summary["mean_resyn2_baseline_total_reward"],
                    "test_mean_resyn2_baseline_final_step_reward": eval_summary["mean_resyn2_baseline_final_step_reward"],
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
                            "eval/mean_resyn2_baseline_total_reward": float(
                                eval_summary["mean_resyn2_baseline_total_reward"]
                            ),
                            "eval/mean_resyn2_baseline_final_step_reward": float(
                                eval_summary["mean_resyn2_baseline_final_step_reward"]
                            ),
                        },
                    )

        if self._tb is not None:
            self._tb.close()
        return {"history": history}

    def evaluate(self, *, num_steps: int, best_of_rollouts: int = 1) -> dict[str, Any]:
        per_circuit = []
        for c in tqdm(self.test_circuits, desc="Evaluating REINFORCE"):
            candidates = []
            for _ in range(max(1, int(best_of_rollouts))):
                sample_actions = best_of_rollouts > 1
                ep = run_reinforce_episode(
                    file_path=c,
                    num_steps=int(num_steps),
                    policy=self.policy,
                    sample_actions=sample_actions,
                    reward_class=self.reward_class,
                    resyn2_baseline=self.resyn2_baselines[c],
                    baseline=self.baseline,
                    available_actions=self.available_actions,
                )
                candidates.append(ep)
            ep = max(candidates, key=lambda r: float(r["final_return"]))
            resyn2_1_size = int(ep["resyn2_variants"]["resyn2_1"]["final_size"])
            resyn2_2_size = int(ep["resyn2_variants"]["resyn2_2"]["final_size"])
            per_circuit.append(
                {
                    "file_path": c,
                    "initial_size": ep["initial_size"],
                    "final_size": ep["final_size"],
                    "initial_depth": ep["initial_depth"],
                    "final_depth": ep["final_depth"],
                    "final_qor": final_qor(final_size=ep["final_size"], final_depth=ep["final_depth"]),
                    "final_return": ep["final_return"],
                    "comparable_return": ep["comparable_return"],
                    "num_steps_taken": ep["num_steps_taken"],
                    "size_reduction_pct": size_reduction_pct(
                        initial_size=ep["initial_size"],
                        final_size=ep["final_size"],
                    ),
                    "resyn2_baseline_total_reward": ep["resyn2_baseline_total_reward"],
                    "resyn2_baseline_final_step_reward": ep["resyn2_baseline_final_step_reward"],
                    "resyn2_1_size": resyn2_1_size,
                    "resyn2_2_size": resyn2_2_size,
                    "resyn2_inf_size": ep["resyn2_variants"]["resyn2_inf"]["final_size"],
                    "normalized_improvement_vs_resyn2_2": normalized_improvement_vs_resyn2_2(
                        initial_size=ep["initial_size"],
                        final_size=ep["final_size"],
                        resyn2_2_size=resyn2_2_size,
                    ),
                }
            )

        mean_return = float(np.mean([r["final_return"] for r in per_circuit])) if per_circuit else 0.0
        mean_comparable_return = (
            float(np.mean([r["comparable_return"] for r in per_circuit])) if per_circuit else 0.0
        )
        mean_reduction = float(np.mean([r["size_reduction_pct"] for r in per_circuit])) if per_circuit else 0.0
        mean_size_reduction = (
            float(np.mean([float(r["initial_size"] - r["final_size"]) for r in per_circuit])) if per_circuit else 0.0
        )
        mean_depth_reduction = (
            float(np.mean([float(r["initial_depth"] - r["final_depth"]) for r in per_circuit])) if per_circuit else 0.0
        )
        mean_resyn2_baseline_total_reward = (
            float(np.mean([r["resyn2_baseline_total_reward"] for r in per_circuit])) if per_circuit else 0.0
        )
        mean_resyn2_baseline_final_step_reward = (
            float(np.mean([r["resyn2_baseline_final_step_reward"] for r in per_circuit])) if per_circuit else 0.0
        )
        common = aggregate_common_eval_metrics(per_circuit)
        return {
            "num_circuits": len(per_circuit),
            "mean_final_return": mean_return,
            "mean_comparable_return": mean_comparable_return,
            "mean_size_reduction": mean_size_reduction,
            "mean_depth_reduction": mean_depth_reduction,
            "mean_size_reduction_pct": mean_reduction,
            "mean_final_size": common["mean_final_size"],
            "mean_final_depth": common["mean_final_depth"],
            "mean_final_qor": common["mean_final_qor"],
            "win_rate_vs_resyn2_1": common["win_rate_vs_resyn2_1"],
            "win_rate_vs_resyn2_2": common["win_rate_vs_resyn2_2"],
            "mean_normalized_improvement_vs_resyn2_2": common["mean_normalized_improvement_vs_resyn2_2"],
            "per_circuit": per_circuit,
            "mean_resyn2_baseline_total_reward": mean_resyn2_baseline_total_reward,
            "mean_resyn2_baseline_final_step_reward": mean_resyn2_baseline_final_step_reward,
        }
