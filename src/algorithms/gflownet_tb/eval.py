from __future__ import annotations

from typing import Any

import numpy as np

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.sampler import sample_tb_trajectory


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
) -> dict[str, Any]:
    per_circuit = []
    for circuit in circuits:
        candidates = []
        for _ in range(max(1, int(best_of_rollouts))):
            candidates.append(
                sample_tb_trajectory(
                    file_path=circuit,
                    num_steps=num_steps,
                    policy=policy,
                    reward_class=reward_class,
                    reward_alpha=reward_alpha,
                    reward_eps=reward_eps,
                    reward_improvement_clip=reward_improvement_clip,
                    sample_actions=best_of_rollouts > 1,
                )
            )
        best = max(candidates, key=lambda t: float(t.final_return))
        init_size = max(1, int(best.initial_size))
        size_reduction_pct = 100.0 * (init_size - int(best.final_size)) / init_size
        per_circuit.append(
            {
                "file_path": circuit,
                "initial_size": best.initial_size,
                "initial_depth": best.initial_depth,
                "final_size": best.final_size,
                "final_depth": best.final_depth,
                "final_return": best.final_return,
                "td_final_return": best.td_final_return,
                "terminal_reward": best.terminal_reward,
                "log_reward": best.log_reward,
                "trajectory_len": len(best.steps),
                "size_reduction_pct": size_reduction_pct,
                "resyn2_1_size": int(resyn2_baselines[circuit]["resyn2_variants"]["resyn2_1"]["final_size"]),
            }
        )

    return {
        "num_circuits": len(per_circuit),
        "mean_final_return": float(np.mean([x["final_return"] for x in per_circuit])) if per_circuit else 0.0,
        "td_mean_final_return": float(np.mean([x["td_final_return"] for x in per_circuit])) if per_circuit else 0.0,
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
        "best_final_size": min((x["final_size"] for x in per_circuit), default=0),
        "best_final_depth": min((x["final_depth"] for x in per_circuit), default=0),
        "best_final_return": max((x["final_return"] for x in per_circuit), default=0.0),
        "td_best_final_return": max((x["td_final_return"] for x in per_circuit), default=0.0),
        "win_rate_vs_resyn2_1": (
            float(np.mean([1.0 if x["final_size"] < x["resyn2_1_size"] else 0.0 for x in per_circuit]))
            if per_circuit
            else 0.0
        ),
        "per_circuit": per_circuit,
    }
