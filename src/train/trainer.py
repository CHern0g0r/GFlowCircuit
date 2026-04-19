import json
import numpy as np
import pyspiel
import torch
from typing import Any
from src.models.policy import Policy
from src.utils import StepSample, discounted_returns
from src.train.utils import run_episode


class Trainer:
    def __init__(self, policy: Policy, device: torch.device, seed: int):
        self.policy = policy
        self.device = device
        self.rng = np.random.default_rng(seed)

    def train(
        self,
        train_circuits: list[str],
        test_circuits: list[str],
        num_steps: int,
        episodes: int,
        eval_every: int,
        learning_rate: float,
        gamma: float,
        baseline_alpha: float,
    ) -> dict[str, Any]:
        baseline = torch.tensor(0.0)
        history: list[dict[str, Any]] = []

        for ep in range(1, episodes + 1):
            circuit = train_circuits[int(self.rng.integers(0, len(train_circuits)))]
            episode = run_episode(
                file_path=circuit,
                num_steps=num_steps,
                policy=self.policy,
                sample_actions=True,
            )
            steps: list[StepSample] = episode["trajectory"]
            rewards = torch.tensor([s.reward for s in steps], dtype=torch.float32, device=self.device)
            rewards = rewards.to(self.device)
            returns = discounted_returns(rewards, gamma=gamma)

            for t, step in enumerate(steps):
                self.policy.update(step, returns[t], learning_rate=learning_rate)

            if len(returns) > 0:
                baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * returns.mean()

            if ep % eval_every == 0 or ep == 1 or ep == episodes:
                eval_summary = self.evaluate(
                    circuits=test_circuits,
                    num_steps=num_steps,
                )
                row = {
                    "episode": ep,
                    "train_circuit": circuit,
                    "train_final_return": episode["final_return"],
                    "train_total_reward": episode["total_reward"],
                    "train_final_size": episode["final_size"],
                    "train_steps": episode["num_steps_taken"],
                    "baseline": baseline.item(),
                    "test_mean_return": eval_summary["mean_final_return"],
                    "test_mean_size_reduction_pct": eval_summary["mean_size_reduction_pct"],
                }
                history.append(row)
                print(json.dumps(row))

        return {
            "history": history,
        }


    def evaluate(
        self,
        circuits: list[str],
        num_steps: int,
    ) -> dict[str, Any]:
        per_circuit = []
        for c in circuits:
            ep = run_episode(
                file_path=c,
                num_steps=num_steps,
                policy=self.policy,
                sample_actions=False,
            )
            initial_size = max(1, int(ep["initial_size"]))
            reduction_pct = 100.0 * (initial_size - int(ep["final_size"])) / initial_size
            per_circuit.append(
                {
                    "file_path": c,
                    "initial_size": ep["initial_size"],
                    "final_size": ep["final_size"],
                    "initial_depth": ep["initial_depth"],
                    "final_depth": ep["final_depth"],
                    "final_return": ep["final_return"],
                    "total_reward": ep["total_reward"],
                    "num_steps_taken": ep["num_steps_taken"],
                    "size_reduction_pct": reduction_pct,
                }
            )

        mean_return = float(np.mean([r["final_return"] for r in per_circuit])) if per_circuit else 0.0
        mean_reduction = (
            float(np.mean([r["size_reduction_pct"] for r in per_circuit])) if per_circuit else 0.0
        )
        return {
            "num_circuits": len(per_circuit),
            "mean_final_return": mean_return,
            "mean_size_reduction_pct": mean_reduction,
            "per_circuit": per_circuit,
        }