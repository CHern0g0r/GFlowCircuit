from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.sampler import sample_tb_trajectories
from src.eval_metrics import (
    aggregate_common_eval_metrics,
    final_qor,
    normalized_improvement_vs_resyn2_2,
    size_reduction_pct,
)


def evaluate_tb(
    *,
    circuits: list[str],
    policy: TBGFlowNetPolicy,
    reward_class: type,
    resyn2_baselines: dict[str, dict[str, Any]],
    num_steps: int,
    reward_alpha: float,
    reward_eps: float,
    reward_improvement_clip: float,
    best_of_rollouts: int,
    available_actions: list[int] | None = None,
) -> dict[str, Any]:
    per_circuit = []
    for circuit in circuits:
        num_rollouts = max(1, int(best_of_rollouts))
        with torch.no_grad():
            candidates = sample_tb_trajectories(
                file_paths=[circuit for _ in range(num_rollouts)],
                num_steps=num_steps,
                policy=policy,
                reward_class=reward_class,
                reward_alpha=reward_alpha,
                reward_eps=reward_eps,
                reward_improvement_clip=reward_improvement_clip,
                sample_actions=num_rollouts > 1,
                available_actions=available_actions,
            )
        best = max(candidates, key=lambda t: float(t.final_return))
        variants = resyn2_baselines[circuit]["resyn2_variants"]
        resyn2_1_size = int(variants["resyn2_1"]["final_size"])
        resyn2_2_size = int(variants["resyn2_2"]["final_size"])
        per_circuit.append(
            {
                "file_path": circuit,
                "initial_size": best.initial_size,
                "initial_depth": best.initial_depth,
                "final_size": best.final_size,
                "final_depth": best.final_depth,
                "final_qor": final_qor(final_size=best.final_size, final_depth=best.final_depth),
                "final_return": best.final_return,
                "comparable_return": best.comparable_return,
                "terminal_reward": best.terminal_reward,
                "log_reward": best.log_reward,
                "trajectory_len": len(best.steps),
                "size_reduction_pct": size_reduction_pct(
                    initial_size=best.initial_size,
                    final_size=best.final_size,
                ),
                "resyn2_1_size": resyn2_1_size,
                "resyn2_2_size": resyn2_2_size,
                "normalized_improvement_vs_resyn2_2": normalized_improvement_vs_resyn2_2(
                    initial_size=best.initial_size,
                    final_size=best.final_size,
                    resyn2_2_size=resyn2_2_size,
                ),
            }
        )

    common = aggregate_common_eval_metrics(per_circuit)
    return {
        "num_circuits": len(per_circuit),
        "mean_final_return": float(np.mean([x["final_return"] for x in per_circuit])) if per_circuit else 0.0,
        "mean_comparable_return": float(np.mean([x["comparable_return"] for x in per_circuit])) if per_circuit else 0.0,
        "mean_terminal_reward": float(np.mean([x["terminal_reward"] for x in per_circuit])) if per_circuit else 0.0,
        "mean_size_reduction": float(
            np.mean([float(x["initial_size"] - x["final_size"]) for x in per_circuit])
        )
        if per_circuit
        else 0.0,
        "mean_depth_reduction": float(
            np.mean([float(x["initial_depth"] - x["final_depth"]) for x in per_circuit])
        )
        if per_circuit
        else 0.0,
        "mean_size_reduction_pct": float(np.mean([x["size_reduction_pct"] for x in per_circuit])) if per_circuit else 0.0,
        "mean_final_size": common["mean_final_size"],
        "mean_final_depth": common["mean_final_depth"],
        "mean_final_qor": common["mean_final_qor"],
        "best_final_return": max((x["final_return"] for x in per_circuit), default=0.0),
        "best_comparable_return": max((x["comparable_return"] for x in per_circuit), default=0.0),
        "win_rate_vs_resyn2_1": common["win_rate_vs_resyn2_1"],
        "win_rate_vs_resyn2_2": common["win_rate_vs_resyn2_2"],
        "mean_normalized_improvement_vs_resyn2_2": common["mean_normalized_improvement_vs_resyn2_2"],
        "per_circuit": per_circuit,
    }
