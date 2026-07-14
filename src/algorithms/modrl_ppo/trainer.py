from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from src.algorithms.modrl_ppo.eval import evaluate_modrl_ppo
from src.algorithms.modrl_ppo.sampler import sample_modrl_ppo_trajectory
from src.algorithms.modrl_ppo.scalarization import preference_to_dict
from src.algorithms.modrl_ppo.types import MODRLPPORollout, PreferenceSpec
from src.algorithms.ppo.loss import compute_gae, ppo_minibatch_loss
from src.algorithms.ppo.policy import PPOPolicy
from src.algorithms.ppo.types import PPOTransition
from src.metrics import TensorBoardLogger
from src.models.mo_rewards import MultiObjectiveReward


class MODRLPPOTrainer:
    def __init__(
        self,
        *,
        policies: dict[str, PPOPolicy],
        value_networks: dict[str, torch.nn.Module],
        preferences: list[PreferenceSpec],
        mo_reward_class: type[MultiObjectiveReward],
        train_circuits: list[str],
        test_circuits: list[str],
        resyn2_baselines: dict[str, dict[str, Any]],
        device: torch.device,
        seed: int,
        log_dir: Path | None = None,
        available_actions: list[int] | None = None,
    ) -> None:
        self.policies = policies
        self.value_networks = value_networks
        self.preferences = preferences
        self.mo_reward_class = mo_reward_class
        self.train_circuits = train_circuits
        self.test_circuits = test_circuits
        self.resyn2_baselines = resyn2_baselines
        self.available_actions = available_actions
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self._tb = TensorBoardLogger(log_dir) if log_dir is not None else None
        missing_policies = [pref.id for pref in preferences if pref.id not in policies]
        missing_values = [pref.id for pref in preferences if pref.id not in value_networks]
        if missing_policies:
            raise ValueError(f"Missing MODRL policies for preferences: {missing_policies}")
        if missing_values:
            raise ValueError(f"Missing MODRL value networks for preferences: {missing_values}")

    def _collect_rollout(
        self,
        *,
        preference: PreferenceSpec,
        num_steps: int,
        rollout_steps: int,
    ) -> MODRLPPORollout:
        policy = self.policies[preference.id]
        value_network = self.value_networks[preference.id]
        trajectories = []
        transitions: list[PPOTransition] = []
        while len(transitions) < int(rollout_steps):
            circuit = self.train_circuits[int(self.rng.integers(0, len(self.train_circuits)))]
            trajectory = sample_modrl_ppo_trajectory(
                file_path=circuit,
                num_steps=int(num_steps),
                policy=policy,
                value_network=value_network,
                mo_reward_class=self.mo_reward_class,
                preference=preference,
                sample_actions=True,
                available_actions=self.available_actions,
            )
            trajectories.append(trajectory)
            transitions.extend(trajectory.transitions)
            if not trajectory.transitions:
                break
        return MODRLPPORollout(trajectories=trajectories, transitions=transitions)

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
    ) -> dict[str, Any]:
        optimizers = {
            preference.id: torch.optim.Adam(
                list(self.policies[preference.id].parameters())
                + list(self.value_networks[preference.id].parameters()),
                lr=float(learning_rate),
            )
            for preference in self.preferences
        }
        history: list[dict[str, Any]] = []

        for ep in trange(1, int(episodes) + 1, desc="Training MODRL-PPO"):
            preference_rows: list[dict[str, float | str]] = []
            for preference in self.preferences:
                policy = self.policies[preference.id]
                value_network = self.value_networks[preference.id]
                optimizer = optimizers[preference.id]
                rollout = self._collect_rollout(
                    preference=preference,
                    num_steps=int(num_steps),
                    rollout_steps=int(rollout_steps),
                )
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
                                policy=policy,
                                value_network=value_network,
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
                                params = list(policy.parameters()) + list(value_network.parameters())
                                grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, float(clip_grad_norm)).item()))
                            optimizer.step()
                            batch_stats.append(stats)

                def _mean_stat(name: str) -> float:
                    if not batch_stats:
                        return 0.0
                    return float(np.mean([float(getattr(stat, name).detach().item()) for stat in batch_stats]))

                row = {
                    "preference_id": preference.id,
                    "total_loss": _mean_stat("total_loss"),
                    "actor_loss": _mean_stat("actor_loss"),
                    "value_loss": _mean_stat("value_loss"),
                    "entropy": _mean_stat("entropy"),
                    "approx_kl": _mean_stat("approx_kl"),
                    "clip_fraction": _mean_stat("clip_fraction"),
                    "mean_return": float(returns.mean().detach().item()) if returns.numel() else 0.0,
                    "mean_scalar_return": float(np.mean([t.scalar_return for t in rollout.trajectories]))
                    if rollout.trajectories
                    else 0.0,
                    "mean_trajectory_len": float(np.mean([len(t.transitions) for t in rollout.trajectories]))
                    if rollout.trajectories
                    else 0.0,
                    "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
                }
                preference_rows.append(row)
                if self._tb is not None:
                    prefix = f"train/{preference.id}"
                    self._tb.add_scalars(
                        ep,
                        {
                            f"{prefix}/total_loss": float(row["total_loss"]),
                            f"{prefix}/actor_loss": float(row["actor_loss"]),
                            f"{prefix}/value_loss": float(row["value_loss"]),
                            f"{prefix}/mean_return": float(row["mean_return"]),
                            f"{prefix}/mean_scalar_return": float(row["mean_scalar_return"]),
                            f"{prefix}/trajectory_len": float(row["mean_trajectory_len"]),
                            f"{prefix}/entropy": float(row["entropy"]),
                            f"{prefix}/approx_kl": float(row["approx_kl"]),
                            f"{prefix}/clip_fraction": float(row["clip_fraction"]),
                            f"{prefix}/grad_norm": float(row["grad_norm"]),
                        },
                    )

            if ep % int(eval_every) == 0 or ep == 1 or ep == int(episodes):
                eval_summary = self.evaluate(num_steps=int(num_steps), best_of_rollouts=int(best_of_eval_rollouts))
                row = {
                    "episode": ep,
                    "train_mean_scalar_return": float(
                        np.mean([float(item["mean_scalar_return"]) for item in preference_rows])
                    )
                    if preference_rows
                    else 0.0,
                    "train_preferences": preference_rows,
                    "test_mean_front_size": eval_summary["mean_front_size"],
                    "test_mean_hypervolume": eval_summary["mean_hypervolume"],
                    "test_mean_size_span": eval_summary["mean_size_span"],
                    "test_mean_depth_span": eval_summary["mean_depth_span"],
                    "test_mean_best_final_size": eval_summary["mean_best_final_size"],
                    "test_mean_best_final_depth": eval_summary["mean_best_final_depth"],
                    "test_mean_best_final_qor": eval_summary["mean_best_final_qor"],
                    "test_win_rate_vs_resyn2_1": eval_summary["win_rate_vs_resyn2_1"],
                    "test_win_rate_vs_resyn2_2": eval_summary["win_rate_vs_resyn2_2"],
                }
                history.append(row)
                print(json.dumps(row))
                if self._tb is not None:
                    self._tb.add_scalars(
                        ep,
                        {
                            "eval/mean_front_size": float(eval_summary["mean_front_size"]),
                            "eval/mean_hypervolume": float(eval_summary["mean_hypervolume"]),
                            "eval/mean_size_span": float(eval_summary["mean_size_span"]),
                            "eval/mean_depth_span": float(eval_summary["mean_depth_span"]),
                            "eval/mean_best_final_size": float(eval_summary["mean_best_final_size"]),
                            "eval/mean_best_final_depth": float(eval_summary["mean_best_final_depth"]),
                            "eval/mean_best_final_qor": float(eval_summary["mean_best_final_qor"]),
                            "eval/win_rate_vs_resyn2_1": float(eval_summary["win_rate_vs_resyn2_1"]),
                            "eval/win_rate_vs_resyn2_2": float(eval_summary["win_rate_vs_resyn2_2"]),
                        },
                    )

        if self._tb is not None:
            self._tb.close()
        return {"history": history}

    def evaluate(self, *, num_steps: int, best_of_rollouts: int = 1) -> dict[str, Any]:
        return evaluate_modrl_ppo(
            circuits=self.test_circuits,
            policies=self.policies,
            value_networks=self.value_networks,
            preferences=self.preferences,
            mo_reward_class=self.mo_reward_class,
            resyn2_baselines=self.resyn2_baselines,
            num_steps=int(num_steps),
            best_of_rollouts=int(best_of_rollouts),
            available_actions=self.available_actions,
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "preferences": [preference_to_dict(preference) for preference in self.preferences],
            "policy_state_dicts": {
                preference.id: self.policies[preference.id].state_dict() for preference in self.preferences
            },
            "value_state_dicts": {
                preference.id: self.value_networks[preference.id].state_dict() for preference in self.preferences
            },
        }

