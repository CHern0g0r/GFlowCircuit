from __future__ import annotations

from functools import partial
from typing import Any

import torch


class MultiObjectiveReward:
    objective_names: tuple[str, ...] = ()
    objective_dim: int = 0

    def __call__(
        self,
        size: int,
        depth: int,
        prev_size: int,
        prev_depth: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def max_return_vector(self) -> torch.Tensor:
        return torch.ones(self.objective_dim, dtype=torch.float32)


class SizeDepthImprovementReward(MultiObjectiveReward):
    objective_names = ("size", "depth")
    objective_dim = 2

    def __init__(
        self,
        initial_size: int,
        initial_depth: int,
        normalize: bool = True,
        objectives: list[str] | tuple[str, ...] = ("size", "depth"),
    ) -> None:
        objective_tuple = tuple(str(item) for item in objectives)
        if objective_tuple != self.objective_names:
            raise ValueError("SizeDepthImprovementReward currently supports objectives=['size', 'depth']")
        self.initial_size = int(initial_size)
        self.initial_depth = int(initial_depth)
        self.normalize = bool(normalize)
        self._size_denom = float(max(1, self.initial_size)) if self.normalize else 1.0
        self._depth_denom = float(max(1, self.initial_depth)) if self.normalize else 1.0

    def __call__(
        self,
        size: int,
        depth: int,
        prev_size: int,
        prev_depth: int,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                float(int(prev_size) - int(size)) / self._size_denom,
                float(int(prev_depth) - int(depth)) / self._depth_denom,
            ],
            dtype=torch.float32,
        )


MO_REWARD_TYPES = {
    "size_depth_improvement": SizeDepthImprovementReward,
}


def mo_reward_factory(mo_reward_cfg: dict[str, Any]) -> type[MultiObjectiveReward] | partial:
    reward_type = mo_reward_cfg.get("type")
    if reward_type not in MO_REWARD_TYPES:
        raise ValueError(f"Unknown multi-objective reward type: {reward_type}")
    reward_kwargs = {k: v for k, v in mo_reward_cfg.items() if k not in {"type", "objective_dim"}}
    if not reward_kwargs:
        return MO_REWARD_TYPES[reward_type]
    return partial(MO_REWARD_TYPES[reward_type], **reward_kwargs)


def discounted_vector_returns(rewards: list[torch.Tensor] | torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    if isinstance(rewards, list):
        if not rewards:
            return torch.empty((0, 0), dtype=torch.float32)
        reward_tensor = torch.stack([r.detach().to(dtype=torch.float32, device="cpu") for r in rewards], dim=0)
    else:
        reward_tensor = rewards.detach().to(dtype=torch.float32)
    if reward_tensor.dim() != 2:
        raise ValueError(f"Expected rewards with shape [T, D], got {tuple(reward_tensor.shape)}")

    out = torch.zeros_like(reward_tensor)
    running = torch.zeros(reward_tensor.shape[1], dtype=reward_tensor.dtype, device=reward_tensor.device)
    for idx in range(reward_tensor.shape[0] - 1, -1, -1):
        running = reward_tensor[idx] + float(gamma) * running
        out[idx] = running
    return out


def dominates_max(a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    return bool(torch.all(a >= b) and torch.any(a > b))


def non_dominated_mask_max(points: torch.Tensor) -> torch.Tensor:
    points = points.to(dtype=torch.float32)
    if points.dim() != 2:
        raise ValueError(f"Expected points with shape [N, D], got {tuple(points.shape)}")
    n = int(points.shape[0])
    mask = torch.ones(n, dtype=torch.bool, device=points.device)
    for i in range(n):
        if not bool(mask[i]):
            continue
        dominates_i = torch.all(points >= points[i], dim=1) & torch.any(points > points[i], dim=1)
        if bool(torch.any(dominates_i)):
            mask[i] = False
    return mask


def crowding_distance(points: torch.Tensor) -> torch.Tensor:
    points = points.to(dtype=torch.float32)
    if points.dim() != 2:
        raise ValueError(f"Expected points with shape [N, D], got {tuple(points.shape)}")
    n, d = int(points.shape[0]), int(points.shape[1])
    if n == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    if n <= 2:
        return torch.ones(n, dtype=torch.float32, device=points.device) * float(d)

    out = torch.zeros(n, dtype=torch.float32, device=points.device)
    ranges = torch.clamp(points.max(dim=0).values - points.min(dim=0).values, min=1e-8)
    normalized = (points - points.min(dim=0).values) / ranges
    for objective_idx in range(d):
        order = torch.argsort(normalized[:, objective_idx])
        out[order[0]] += 1.0
        out[order[-1]] += 1.0
        lower = normalized[order[:-2], objective_idx]
        upper = normalized[order[2:], objective_idx]
        out[order[1:-1]] += torch.abs(upper - lower)
    return out


def hypervolume_2d_max(points: torch.Tensor, reference: tuple[float, float] = (0.0, 0.0)) -> float:
    points = points.detach().to(dtype=torch.float32, device="cpu")
    if points.numel() == 0:
        return 0.0
    if points.dim() != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape [N, 2], got {tuple(points.shape)}")

    ref_x, ref_y = float(reference[0]), float(reference[1])
    valid = points[(points[:, 0] > ref_x) & (points[:, 1] > ref_y)]
    if valid.numel() == 0:
        return 0.0
    valid = valid[non_dominated_mask_max(valid)]
    order = torch.argsort(valid[:, 0], descending=True)
    sorted_points = valid[order]

    hv = 0.0
    covered_y = ref_y
    for point in sorted_points:
        x = float(point[0].item())
        y = float(point[1].item())
        if y <= covered_y:
            continue
        hv += max(0.0, x - ref_x) * (y - covered_y)
        covered_y = y
    return float(hv)


__all__ = [
    "MultiObjectiveReward",
    "SizeDepthImprovementReward",
    "MO_REWARD_TYPES",
    "mo_reward_factory",
    "discounted_vector_returns",
    "dominates_max",
    "non_dominated_mask_max",
    "crowding_distance",
    "hypervolume_2d_max",
]
