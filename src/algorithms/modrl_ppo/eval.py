from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.algorithms.modrl_ppo.sampler import sample_modrl_ppo_trajectory
from src.algorithms.modrl_ppo.scalarization import preference_to_dict
from src.algorithms.modrl_ppo.types import PreferenceSpec
from src.algorithms.ppo.policy import PPOPolicy
from src.eval_metrics import final_qor, normalized_improvement_vs_resyn2_2, size_reduction_pct
from src.models.mo_rewards import MultiObjectiveReward, crowding_distance, hypervolume_2d_max


def _dominates_min(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        int(a["final_size"]) <= int(b["final_size"])
        and int(a["final_depth"]) <= int(b["final_depth"])
        and (int(a["final_size"]) < int(b["final_size"]) or int(a["final_depth"]) < int(b["final_depth"]))
    )


def pareto_front_min(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (int(item["final_size"]), int(item["final_depth"]), -float(item["scalar_return"])),
    ):
        key = (int(candidate["final_size"]), int(candidate["final_depth"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    front = []
    for candidate in deduped:
        if any(_dominates_min(other, candidate) for other in deduped):
            continue
        front.append(candidate)
    return sorted(front, key=lambda item: (int(item["final_size"]), int(item["final_depth"])))


def _candidate_from_trajectory(trajectory) -> dict[str, Any]:
    return {
        "preference_id": trajectory.preference.id,
        "preference": preference_to_dict(trajectory.preference),
        "initial_size": int(trajectory.initial_size),
        "initial_depth": int(trajectory.initial_depth),
        "final_size": int(trajectory.final_size),
        "final_depth": int(trajectory.final_depth),
        "final_qor": final_qor(final_size=trajectory.final_size, final_depth=trajectory.final_depth),
        "return_vec": [float(v) for v in trajectory.return_vec.detach().cpu().tolist()],
        "scalar_return": float(trajectory.scalar_return),
        "trajectory_len": len(trajectory.transitions),
        "actions": [int(action) for action in trajectory.actions_applied],
    }


def _span(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _front_metrics(front: list[dict[str, Any]]) -> dict[str, float]:
    if not front:
        return {
            "front_size": 0.0,
            "hypervolume": 0.0,
            "size_span": 0.0,
            "depth_span": 0.0,
            "mean_crowding": 0.0,
            "best_final_size": 0.0,
            "best_final_depth": 0.0,
            "best_final_qor": 0.0,
        }
    returns = torch.tensor([item["return_vec"] for item in front], dtype=torch.float32)
    crowding = crowding_distance(returns) if returns.shape[0] > 0 else torch.empty((0,), dtype=torch.float32)
    return {
        "front_size": float(len(front)),
        "hypervolume": float(hypervolume_2d_max(returns)),
        "size_span": _span([float(item["return_vec"][0]) for item in front]),
        "depth_span": _span([float(item["return_vec"][1]) for item in front]),
        "mean_crowding": float(crowding.mean().item()) if crowding.numel() else 0.0,
        "best_final_size": float(min(int(item["final_size"]) for item in front)),
        "best_final_depth": float(min(int(item["final_depth"]) for item in front)),
        "best_final_qor": float(min(int(item["final_qor"]) for item in front)),
    }


def evaluate_modrl_ppo(
    *,
    circuits: list[str],
    policies: dict[str, PPOPolicy],
    value_networks: dict[str, torch.nn.Module],
    preferences: list[PreferenceSpec],
    mo_reward_class: type[MultiObjectiveReward],
    resyn2_baselines: dict[str, dict[str, Any]],
    num_steps: int,
    best_of_rollouts: int,
    available_actions: list[int] | None = None,
) -> dict[str, Any]:
    per_circuit: list[dict[str, Any]] = []
    for circuit in circuits:
        candidates = []
        for preference in preferences:
            policy = policies[preference.id]
            value_network = value_networks[preference.id]
            pref_candidates = []
            for _ in range(max(1, int(best_of_rollouts))):
                trajectory = sample_modrl_ppo_trajectory(
                    file_path=circuit,
                    num_steps=int(num_steps),
                    policy=policy,
                    value_network=value_network,
                    mo_reward_class=mo_reward_class,
                    preference=preference,
                    sample_actions=best_of_rollouts > 1,
                    available_actions=available_actions,
                )
                pref_candidates.append(_candidate_from_trajectory(trajectory))
            candidates.append(max(pref_candidates, key=lambda item: float(item["scalar_return"])))

        front = pareto_front_min(candidates)
        metrics = _front_metrics(front)
        variants = resyn2_baselines[circuit]["resyn2_variants"]
        resyn2_1_size = int(variants["resyn2_1"]["final_size"])
        resyn2_2_size = int(variants["resyn2_2"]["final_size"])
        best_qor = min(front, key=lambda item: int(item["final_qor"])) if front else None
        best_size = min(front, key=lambda item: int(item["final_size"])) if front else None
        per_circuit.append(
            {
                "file_path": circuit,
                "initial_size": int(candidates[0]["initial_size"]) if candidates else 0,
                "initial_depth": int(candidates[0]["initial_depth"]) if candidates else 0,
                "candidates": candidates,
                "pareto_front": front,
                "resyn2_1_size": resyn2_1_size,
                "resyn2_2_size": resyn2_2_size,
                "resyn2_inf_size": int(variants["resyn2_inf"]["final_size"]),
                "front_size": metrics["front_size"],
                "hypervolume": metrics["hypervolume"],
                "size_span": metrics["size_span"],
                "depth_span": metrics["depth_span"],
                "mean_crowding": metrics["mean_crowding"],
                "best_final_size": metrics["best_final_size"],
                "best_final_depth": metrics["best_final_depth"],
                "best_final_qor": metrics["best_final_qor"],
                "best_size_reduction_pct": size_reduction_pct(
                    initial_size=int(candidates[0]["initial_size"]) if candidates else 0,
                    final_size=int(best_size["final_size"]) if best_size is not None else 0,
                ),
                "normalized_improvement_vs_resyn2_2": normalized_improvement_vs_resyn2_2(
                    initial_size=int(candidates[0]["initial_size"]) if candidates else 0,
                    final_size=int(best_size["final_size"]) if best_size is not None else 0,
                    resyn2_2_size=resyn2_2_size,
                ),
                "win_vs_resyn2_1": 1.0 if best_size is not None and int(best_size["final_size"]) < resyn2_1_size else 0.0,
                "win_vs_resyn2_2": 1.0 if best_size is not None and int(best_size["final_size"]) < resyn2_2_size else 0.0,
                "best_qor_candidate": best_qor,
            }
        )

    def _mean(name: str) -> float:
        return float(np.mean([float(item[name]) for item in per_circuit])) if per_circuit else 0.0

    return {
        "num_circuits": len(per_circuit),
        "num_preferences": len(preferences),
        "mean_front_size": _mean("front_size"),
        "mean_hypervolume": _mean("hypervolume"),
        "mean_size_span": _mean("size_span"),
        "mean_depth_span": _mean("depth_span"),
        "mean_crowding": _mean("mean_crowding"),
        "mean_best_final_size": _mean("best_final_size"),
        "mean_best_final_depth": _mean("best_final_depth"),
        "mean_best_final_qor": _mean("best_final_qor"),
        "mean_size_reduction_pct": _mean("best_size_reduction_pct"),
        "win_rate_vs_resyn2_1": _mean("win_vs_resyn2_1"),
        "win_rate_vs_resyn2_2": _mean("win_vs_resyn2_2"),
        "mean_normalized_improvement_vs_resyn2_2": _mean("normalized_improvement_vs_resyn2_2"),
        "per_circuit": per_circuit,
    }

