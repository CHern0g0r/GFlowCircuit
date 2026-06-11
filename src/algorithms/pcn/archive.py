from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.algorithms.pcn.types import PCNDatapoint, PCNTarget, PCNTrajectory
from src.models.mo_rewards import crowding_distance, discounted_vector_returns, non_dominated_mask_max


class PCNArchive:
    def __init__(
        self,
        *,
        capacity: int,
        gamma: float = 1.0,
        crowding_threshold: float = 0.2,
        duplicate_penalty: float = 1e-5,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("PCN archive capacity must be positive")
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.crowding_threshold = float(crowding_threshold)
        self.duplicate_penalty = float(duplicate_penalty)
        self.trajectories: list[PCNTrajectory] = []

    @staticmethod
    def _return_key(return_vec: torch.Tensor) -> tuple[float, ...]:
        return tuple(round(float(v), 8) for v in return_vec.detach().to(dtype=torch.float32, device="cpu").tolist())

    def __len__(self) -> int:
        return len(self.trajectories)

    def add(self, trajectory: PCNTrajectory) -> None:
        if not trajectory.steps:
            return
        self.trajectories.append(trajectory)
        self.prune()

    def add_many(self, trajectories: list[PCNTrajectory]) -> None:
        for trajectory in trajectories:
            if trajectory.steps:
                self.trajectories.append(trajectory)
        self.prune()

    def returns_tensor(self) -> torch.Tensor:
        if not self.trajectories:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack([t.return_vec.detach().to(dtype=torch.float32, device="cpu") for t in self.trajectories])

    def non_dominated_indices(self) -> list[int]:
        returns = self.returns_tensor()
        if returns.numel() == 0:
            return []
        mask = non_dominated_mask_max(returns)
        return [int(i) for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()]

    def non_dominated_trajectories(self) -> list[PCNTrajectory]:
        return [self.trajectories[i] for i in self.non_dominated_indices()]

    def _scores(self) -> torch.Tensor:
        returns = self.returns_tensor()
        if returns.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)
        nd_mask = non_dominated_mask_max(returns)
        nd = returns[nd_mask]
        if nd.numel() == 0:
            return torch.zeros(returns.shape[0], dtype=torch.float32)
        dists = torch.cdist(returns, nd).min(dim=1).values * -1.0
        crowding = crowding_distance(returns)
        scores = dists.clone()
        crowded = crowding <= self.crowding_threshold
        scores[crowded] = 2.0 * (scores[crowded] - self.duplicate_penalty)

        nd_indices = torch.nonzero(nd_mask, as_tuple=False).flatten()
        seen: set[tuple[float, ...]] = set()
        for idx in nd_indices.tolist():
            key = self._return_key(returns[int(idx)])
            if key in seen:
                scores[int(idx)] -= 1.0 + self.duplicate_penalty
            else:
                seen.add(key)
        return scores

    @staticmethod
    def _unique_by_return(trajectories: list[PCNTrajectory]) -> list[PCNTrajectory]:
        out: list[PCNTrajectory] = []
        seen: set[tuple[float, ...]] = set()
        for trajectory in trajectories:
            key = PCNArchive._return_key(trajectory.return_vec)
            if key in seen:
                continue
            seen.add(key)
            out.append(trajectory)
        return out

    def prune(self) -> None:
        if len(self.trajectories) <= self.capacity:
            return
        scores = self._scores()
        keep = torch.argsort(scores, descending=True)[: self.capacity]
        keep_set = {int(i) for i in keep.tolist()}
        self.trajectories = [trajectory for idx, trajectory in enumerate(self.trajectories) if idx in keep_set]

    def sample_datapoints(self, *, batch_size: int, rng: np.random.Generator) -> list[PCNDatapoint]:
        if not self.trajectories:
            raise ValueError("Cannot sample from empty PCN archive")
        out: list[PCNDatapoint] = []
        for _ in range(max(1, int(batch_size))):
            traj = self.trajectories[int(rng.integers(0, len(self.trajectories)))]
            rewards = [step.reward_vec for step in traj.steps]
            returns_to_go = discounted_vector_returns(rewards, gamma=self.gamma)
            step_idx = int(rng.integers(0, len(traj.steps)))
            step = traj.steps[step_idx]
            out.append(
                PCNDatapoint(
                    observation=step.observation,
                    action=int(step.action),
                    legal_actions=list(step.legal_actions),
                    desired_return=returns_to_go[step_idx].detach().cpu(),
                    desired_horizon=float(len(traj.steps) - step_idx),
                )
            )
        return out

    def sample_exploration_target(self, *, rng: np.random.Generator) -> PCNTarget | None:
        nd = self.non_dominated_trajectories()
        if not nd:
            return None
        returns = torch.stack([traj.return_vec.detach().to(dtype=torch.float32, device="cpu") for traj in nd])
        base_idx = int(rng.integers(0, len(nd)))
        target_return = returns[base_idx].clone()
        objective_idx = int(rng.integers(0, target_return.numel()))
        sigma = float(torch.std(returns[:, objective_idx], unbiased=False).item())
        target_return[objective_idx] += float(rng.uniform(0.0, sigma)) if sigma > 0.0 else 0.0
        return PCNTarget(desired_return=target_return, desired_horizon=float(nd[base_idx].horizon))

    def metadata(self, *, limit: int | None = None) -> dict[str, Any]:
        nd = self._unique_by_return(self.non_dominated_trajectories())
        if limit is not None and int(limit) > 0 and len(nd) > int(limit):
            returns = torch.stack([t.return_vec.detach().to(dtype=torch.float32, device="cpu") for t in nd])
            keep = torch.argsort(crowding_distance(returns), descending=True)[: int(limit)]
            nd = [nd[int(idx)] for idx in keep.tolist()]
        return {
            "archive_size": len(self.trajectories),
            "nondominated_size": len(self.non_dominated_trajectories()),
            "unique_nondominated_size": len(self._unique_by_return(self.non_dominated_trajectories())),
            "archive_target_returns": [t.return_vec.detach().cpu().tolist() for t in nd],
            "archive_target_horizons": [int(t.horizon) for t in nd],
        }
