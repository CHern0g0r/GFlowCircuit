from __future__ import annotations

from statistics import fmean
from typing import Any


def comparable_return(
    *,
    reward_class: type,
    initial_size: int,
    initial_depth: int,
    final_size: int,
    final_depth: int,
) -> float:
    reward_func = reward_class(int(initial_size), int(initial_depth))
    return float(
        reward_func(
            int(final_size),
            int(final_depth),
            int(initial_size),
            int(initial_depth),
        )
    )


def final_qor(*, final_size: int, final_depth: int) -> int:
    return int(final_size) * int(final_depth)


def size_reduction_pct(*, initial_size: int, final_size: int) -> float:
    denom = max(1, int(initial_size))
    return 100.0 * (int(initial_size) - int(final_size)) / denom


def normalized_improvement_vs_resyn2_2(
    *,
    initial_size: int,
    final_size: int,
    resyn2_2_size: int,
) -> float:
    denom = max(1, int(initial_size))
    method_improvement = (int(initial_size) - int(final_size)) / denom
    resyn2_improvement = (int(initial_size) - int(resyn2_2_size)) / denom
    return float(method_improvement - resyn2_improvement)


def aggregate_common_eval_metrics(per_circuit: list[dict[str, Any]]) -> dict[str, float]:
    if not per_circuit:
        return {
            "mean_final_size": 0.0,
            "mean_final_depth": 0.0,
            "mean_final_qor": 0.0,
            "win_rate_vs_resyn2_1": 0.0,
            "win_rate_vs_resyn2_2": 0.0,
            "mean_normalized_improvement_vs_resyn2_2": 0.0,
        }

    out = {
        "mean_final_size": float(fmean(float(r["final_size"]) for r in per_circuit)),
        "mean_final_depth": float(fmean(float(r["final_depth"]) for r in per_circuit)),
        "mean_final_qor": float(fmean(float(r["final_qor"]) for r in per_circuit)),
        "win_rate_vs_resyn2_2": float(
            fmean(1.0 if int(r["final_size"]) < int(r["resyn2_2_size"]) else 0.0 for r in per_circuit)
        ),
        "mean_normalized_improvement_vs_resyn2_2": float(
            fmean(float(r["normalized_improvement_vs_resyn2_2"]) for r in per_circuit)
        ),
    }
    if all("resyn2_1_size" in r for r in per_circuit):
        out["win_rate_vs_resyn2_1"] = float(
            fmean(1.0 if int(r["final_size"]) < int(r["resyn2_1_size"]) else 0.0 for r in per_circuit)
        )
    else:
        out["win_rate_vs_resyn2_1"] = 0.0
    return out
