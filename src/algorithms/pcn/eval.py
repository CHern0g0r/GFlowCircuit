from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.algorithms.pcn.archive import PCNArchive
from src.algorithms.pcn.policy import PCNPolicy
from src.algorithms.pcn.sampler import sample_pcn_trajectory
from src.eval_metrics import (
    aggregate_common_eval_metrics,
    final_qor,
    normalized_improvement_vs_resyn2_2,
    size_reduction_pct,
)
from src.models.mo_rewards import MultiObjectiveReward, crowding_distance, hypervolume_2d_max, non_dominated_mask_max


def evaluate_pcn(
    *,
    circuits: list[str],
    policy: PCNPolicy,
    mo_reward_class: type[MultiObjectiveReward],
    resyn2_baselines: dict[str, dict[str, Any]],
    archive: PCNArchive | None,
    num_steps: int,
    seed: int,
    available_actions: list[int] | None = None,
    eval_target_limit: int | None = None,
    desired_return_clip: bool = False,
    gamma: float = 1.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    archive_targets = archive.non_dominated_trajectories() if archive is not None else []
    if eval_target_limit is not None and int(eval_target_limit) > 0:
        if len(archive_targets) > int(eval_target_limit):
            returns = torch.stack([target.return_vec.detach().to(dtype=torch.float32, device="cpu") for target in archive_targets])
            keep = torch.argsort(crowding_distance(returns), descending=True)[: int(eval_target_limit)]
            archive_targets = [archive_targets[int(idx)] for idx in keep.tolist()]

    per_circuit: list[dict[str, Any]] = []
    all_return_vecs: list[torch.Tensor] = []
    all_points: list[dict[str, Any]] = []
    for circuit in circuits:
        if archive_targets:
            trajectories = [
                sample_pcn_trajectory(
                    file_path=circuit,
                    num_steps=int(num_steps),
                    policy=policy,
                    mo_reward_class=mo_reward_class,
                    sample_actions=False,
                    rng=rng,
                    available_actions=available_actions,
                    desired_return=target.return_vec,
                    desired_horizon=float(target.horizon),
                    desired_return_clip=desired_return_clip,
                    gamma=float(gamma),
                )
                for target in archive_targets
            ]
        else:
            trajectories = [
                sample_pcn_trajectory(
                    file_path=circuit,
                    num_steps=int(num_steps),
                    policy=policy,
                    mo_reward_class=mo_reward_class,
                    sample_actions=False,
                    rng=rng,
                    available_actions=available_actions,
                    gamma=float(gamma),
                )
            ]

        best = min(trajectories, key=lambda t: int(t.final_size) * int(t.final_depth))
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
                "final_return": float(best.return_vec.sum().item()),
                "comparable_return": float(best.return_vec.sum().item()),
                "trajectory_len": len(best.steps),
                "size_reduction_pct": size_reduction_pct(initial_size=best.initial_size, final_size=best.final_size),
                "resyn2_1_size": resyn2_1_size,
                "resyn2_2_size": resyn2_2_size,
                "normalized_improvement_vs_resyn2_2": normalized_improvement_vs_resyn2_2(
                    initial_size=best.initial_size,
                    final_size=best.final_size,
                    resyn2_2_size=resyn2_2_size,
                ),
            }
        )
        for trajectory in trajectories:
            all_return_vecs.append(trajectory.return_vec)
            all_points.append(
                {
                    "circuit": circuit,
                    "size": int(trajectory.final_size),
                    "depth": int(trajectory.final_depth),
                    "return_vec": trajectory.return_vec.detach().cpu().tolist(),
                }
            )

    common = aggregate_common_eval_metrics(per_circuit)
    if all_return_vecs:
        returns_tensor = torch.stack(all_return_vecs)
        nd_mask = non_dominated_mask_max(returns_tensor)
        nd_returns = returns_tensor[nd_mask]
        hypervolume = hypervolume_2d_max(nd_returns) if nd_returns.shape[1] == 2 else 0.0
    else:
        nd_returns = torch.empty((0, 2), dtype=torch.float32)
        hypervolume = 0.0

    return {
        "num_circuits": len(per_circuit),
        "num_eval_points": len(all_points),
        "nondominated_count": int(nd_returns.shape[0]),
        "hypervolume": float(hypervolume),
        "mean_final_return": float(np.mean([x["final_return"] for x in per_circuit])) if per_circuit else 0.0,
        "mean_comparable_return": float(np.mean([x["comparable_return"] for x in per_circuit])) if per_circuit else 0.0,
        "mean_size_reduction": float(np.mean([float(x["initial_size"] - x["final_size"]) for x in per_circuit]))
        if per_circuit
        else 0.0,
        "mean_depth_reduction": float(np.mean([float(x["initial_depth"] - x["final_depth"]) for x in per_circuit]))
        if per_circuit
        else 0.0,
        "mean_size_reduction_pct": float(np.mean([x["size_reduction_pct"] for x in per_circuit])) if per_circuit else 0.0,
        "mean_final_size": common["mean_final_size"],
        "mean_final_depth": common["mean_final_depth"],
        "mean_final_qor": common["mean_final_qor"],
        "win_rate_vs_resyn2_1": common["win_rate_vs_resyn2_1"],
        "win_rate_vs_resyn2_2": common["win_rate_vs_resyn2_2"],
        "mean_normalized_improvement_vs_resyn2_2": common["mean_normalized_improvement_vs_resyn2_2"],
        "points": all_points,
        "per_circuit": per_circuit,
    }
