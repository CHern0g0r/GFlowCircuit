from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.algorithms.modrl_ppo.types import PreferenceSpec


OBJECTIVE_INDEX = {
    "size": 0,
    "depth": 1,
}


def objective_index(name: str) -> int:
    try:
        return OBJECTIVE_INDEX[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown MODRL objective: {name}") from exc


def parse_preference_specs(raw: Any) -> list[PreferenceSpec]:
    if isinstance(raw, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, dict)):
        raise TypeError("modrl_ppo.preferences must be a list of preference mappings")

    specs: list[PreferenceSpec] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("Each MODRL preference must be a mapping")
        mode = str(item.get("mode", "")).lower()
        pref_id = str(item.get("id") or _default_preference_id(item, mode))
        if pref_id in seen:
            raise ValueError(f"Duplicate MODRL preference id: {pref_id}")
        seen.add(pref_id)
        if mode == "linear":
            weights_raw = item.get("weights")
            if not isinstance(weights_raw, Iterable) or isinstance(weights_raw, (str, bytes, dict)):
                raise ValueError(f"Linear preference {pref_id} requires list weights")
            weights = tuple(float(v) for v in weights_raw)
            if len(weights) != len(OBJECTIVE_INDEX):
                raise ValueError(f"Linear preference {pref_id} requires {len(OBJECTIVE_INDEX)} weights")
            specs.append(PreferenceSpec(id=pref_id, mode=mode, weights=weights))
        elif mode == "tlo":
            constraint = str(item.get("constraint_objective"))
            optimize = str(item.get("optimize_objective"))
            if constraint == optimize:
                raise ValueError(f"TLO preference {pref_id} must use different objectives")
            objective_index(constraint)
            objective_index(optimize)
            specs.append(
                PreferenceSpec(
                    id=pref_id,
                    mode=mode,
                    constraint_objective=constraint,
                    optimize_objective=optimize,
                    threshold=float(item.get("threshold", 0.0)),
                    lex_scale=float(item.get("lex_scale", 10.0)),
                )
            )
        else:
            raise ValueError(f"Unknown MODRL preference mode: {mode}")
    if not specs:
        raise ValueError("At least one MODRL preference is required")
    return specs


def _default_preference_id(item: dict[str, Any], mode: str) -> str:
    if mode == "linear":
        weights = "_".join(str(v) for v in item.get("weights", ()))
        return f"linear_{weights}"
    if mode == "tlo":
        constraint = item.get("constraint_objective", "constraint")
        optimize = item.get("optimize_objective", "optimize")
        threshold = item.get("threshold", 0.0)
        return f"tlo_{constraint}_{threshold}_{optimize}"
    return "unknown"


def scalarize_linear(reward_vec: torch.Tensor, weights: tuple[float, ...]) -> torch.Tensor:
    reward_vec = reward_vec.to(dtype=torch.float32)
    weight_tensor = torch.tensor(weights, dtype=reward_vec.dtype, device=reward_vec.device)
    if reward_vec.shape[-1] != weight_tensor.numel():
        raise ValueError(f"Reward vector dim {reward_vec.shape[-1]} does not match weights {weight_tensor.numel()}")
    return torch.sum(reward_vec * weight_tensor, dim=-1)


def tlo_utility(return_vec: torch.Tensor, preference: PreferenceSpec) -> torch.Tensor:
    if preference.mode != "tlo":
        raise ValueError("tlo_utility requires a TLO PreferenceSpec")
    if preference.constraint_objective is None or preference.optimize_objective is None:
        raise ValueError("TLO preference is missing objectives")
    if preference.threshold is None:
        raise ValueError("TLO preference is missing threshold")
    return_vec = return_vec.to(dtype=torch.float32)
    c_idx = objective_index(preference.constraint_objective)
    o_idx = objective_index(preference.optimize_objective)
    constrained = torch.minimum(
        return_vec[..., c_idx],
        torch.tensor(float(preference.threshold), dtype=return_vec.dtype, device=return_vec.device),
    )
    return float(preference.lex_scale) * constrained + return_vec[..., o_idx]


def scalar_reward_from_step(
    *,
    preference: PreferenceSpec,
    reward_vec: torch.Tensor,
    cumulative_before: torch.Tensor,
) -> torch.Tensor:
    reward_vec = reward_vec.to(dtype=torch.float32)
    cumulative_before = cumulative_before.to(dtype=torch.float32)
    if preference.mode == "linear":
        if preference.weights is None:
            raise ValueError("Linear preference is missing weights")
        return scalarize_linear(reward_vec, preference.weights)
    if preference.mode == "tlo":
        cumulative_after = cumulative_before + reward_vec
        return tlo_utility(cumulative_after, preference) - tlo_utility(cumulative_before, preference)
    raise ValueError(f"Unknown MODRL preference mode: {preference.mode}")


def preference_to_dict(preference: PreferenceSpec) -> dict[str, object]:
    return {
        "id": preference.id,
        "mode": preference.mode,
        "weights": list(preference.weights) if preference.weights is not None else None,
        "constraint_objective": preference.constraint_objective,
        "optimize_objective": preference.optimize_objective,
        "threshold": preference.threshold,
        "lex_scale": preference.lex_scale,
    }

