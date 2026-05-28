import numpy as np
import torch
import yaml
import pyspiel
import time

from pathlib import Path
from torch_geometric.data import Data
from collections import deque
from dataclasses import dataclass, replace
from typing import Any


@dataclass
class Observation:
    obs_tensor: torch.Tensor
    graph: Data
    legal_actions: list[int]
    vector_tensor: torch.Tensor | None = None

    @classmethod
    def from_state(
        cls,
        state: pyspiel.State,
        available_actions: list[int] | None = None,
        timing: dict[str, Any] | None = None,
    ) -> "Observation":
        t0 = time.perf_counter()
        t_obs = time.perf_counter()
        obs_tensor = torch.as_tensor(state.observation_tensor(0), dtype=torch.float32)
        t_graph = time.perf_counter()
        x, e, ea = pyspiel.circuit_graph(state)
        t_tensor = time.perf_counter()
        graph = Data(
            x=torch.as_tensor(x),
            edge_index=torch.as_tensor(e).T,
            edge_attr=torch.as_tensor(ea)
        )
        t_legal = time.perf_counter()
        legal_actions = filter_legal_actions(state.legal_actions(), available_actions)
        if available_actions is not None and not legal_actions and not state.is_terminal():
            raise ValueError(
                f"available_actions={available_actions} removes all legal actions "
                f"from state legal actions {list(state.legal_actions())}"
            )
        t1 = time.perf_counter()
        if timing is not None:
            timing["obs_from_state_calls"] = timing.get("obs_from_state_calls", 0) + 1
            timing["obs_from_state_total_s"] = timing.get("obs_from_state_total_s", 0.0) + (t1 - t0)
            timing["obs_tensor_s"] = timing.get("obs_tensor_s", 0.0) + (t_graph - t_obs)
            timing["circuit_graph_s"] = timing.get("circuit_graph_s", 0.0) + (t_tensor - t_graph)
            timing["tensor_wrap_s"] = timing.get("tensor_wrap_s", 0.0) + (t_legal - t_tensor)
            timing["legal_actions_s"] = timing.get("legal_actions_s", 0.0) + (t1 - t_legal)
        return cls(obs_tensor, graph, legal_actions)


    def observation_to_device(self, device: torch.device) -> "Observation":
        """Move observation tensors to ``device`` (OpenSpiel observations default to CPU)."""
        vector_tensor = self.vector_tensor
        if (
            self.obs_tensor.device == device
            and self.graph.x.device == device
            and (vector_tensor is None or vector_tensor.device == device)
        ):
            return self
        return Observation(
            obs_tensor=self.obs_tensor.to(device, non_blocking=device.type == "cuda"),
            graph=self.graph.to(device),
            legal_actions=list(self.legal_actions),
            vector_tensor=vector_tensor.to(device, non_blocking=device.type == "cuda") if vector_tensor is not None else None,
        )

    def with_vector(self, vector_tensor: torch.Tensor) -> "Observation":
        return replace(self, vector_tensor=vector_tensor)


@dataclass
class StepSample:
    observation: Observation
    action: int
    probs: torch.Tensor
    reward: float

    def to(self, device: torch.device) -> "StepSample":
        """Copy with observation (and probs if needed) moved to ``device`` for model forward passes."""
        obs_dev = self.observation.observation_to_device(device)
        probs = self.probs
        if probs.device != device:
            probs = probs.to(device)
        return StepSample(observation=obs_dev, action=self.action, probs=probs, reward=self.reward)


def train_test_split(items: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if len(items) == 1:
        return items[:], items[:]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))
    n_train = int(round(len(items) * train_ratio))
    n_train = max(1, min(len(items) - 1, n_train))
    train_ids = perm[:n_train]
    test_ids = perm[n_train:]
    train = [items[i] for i in train_ids]
    test = [items[i] for i in test_ids]
    return train, test


def load_circuits(dataset_cfg: Path) -> list[str]:
    cfg = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8")) or {}
    root = Path(cfg["path"])
    fmt = str(cfg.get("format", "aig"))
    names = cfg.get("files", [])
    if not isinstance(names, list) or not names:
        raise ValueError(f"No circuits found in config: {dataset_cfg}")

    circuits = []
    for name in names:
        p = root / f"{name}.{fmt}"
        print(p.absolute())
        if p.exists():
            circuits.append(str(p))
        else:
            print(f"[warn] skipping missing circuit: {p}")

    if not circuits:
        raise FileNotFoundError("No existing circuit files found from dataset config.")
    return circuits


def get_obs_dim_and_num_actions(num_steps: int, sample_circuit: str) -> tuple[int, int, int, int]:
    probe_game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": sample_circuit})
    probe_state = probe_game.new_initial_state()
    obs_dim = len(probe_state.observation_tensor(0))
    num_actions = int(probe_game.num_distinct_actions())
    x, _, ea = pyspiel.circuit_graph(probe_state)
    node_dim = x.shape[1]
    edge_dim = ea.shape[1]
    return obs_dim, num_actions, node_dim, edge_dim


def normalize_available_actions(value: Any, num_actions: int) -> list[int] | None:
    """Normalize an optional action whitelist from config.

    ``None`` means every action exposed by the environment is available.
    """
    if value is None:
        return None
    actions = [int(action) for action in value]
    if not actions:
        raise ValueError("available_actions must be null or a non-empty list of action ids")
    if len(set(actions)) != len(actions):
        raise ValueError(f"available_actions contains duplicates: {actions}")
    invalid = [action for action in actions if action < 0 or action >= int(num_actions)]
    if invalid:
        raise ValueError(f"available_actions contains ids outside [0, {int(num_actions) - 1}]: {invalid}")
    return actions


def filter_legal_actions(
    legal_actions: Any,
    available_actions: list[int] | None,
) -> list[int]:
    legal = [int(action) for action in legal_actions]
    if available_actions is None:
        return legal
    available = set(available_actions)
    return [action for action in legal if action in available]


def resolve_vector_action_ids(
    num_actions: int,
    available_actions: list[int] | None,
) -> list[int]:
    if available_actions is not None:
        return list(available_actions)
    return list(range(int(num_actions)))


def zhu_vector_dim(
    num_actions: int,
    available_actions: list[int] | None = None,
) -> int:
    return 5 + len(resolve_vector_action_ids(num_actions, available_actions))


@dataclass
class ZhuVectorState:
    initial_size: int
    initial_depth: int
    num_steps: int
    action_ids: list[int]
    history_window: int = 3

    def __post_init__(self) -> None:
        if not self.action_ids:
            raise ValueError("ZhuVectorState requires at least one action id")
        self._initial_size_denom = float(max(1, int(self.initial_size)))
        self._initial_depth_denom = float(max(1, int(self.initial_depth)))
        self._num_steps_denom = float(max(1, int(self.num_steps)))
        self.previous_size = int(self.initial_size)
        self.previous_depth = int(self.initial_depth)
        self._action_to_idx = {int(action): idx for idx, action in enumerate(self.action_ids)}
        self._recent_actions: deque[int] = deque(maxlen=int(self.history_window))

    def vector(self, *, current_size: int, current_depth: int, step: int) -> torch.Tensor:
        counts = torch.zeros(len(self.action_ids), dtype=torch.float32)
        for action in self._recent_actions:
            idx = self._action_to_idx.get(int(action))
            if idx is not None:
                counts[idx] += 1.0
        counts = counts / float(max(1, int(self.history_window)))
        prefix = torch.tensor(
            [
                float(current_size) / self._initial_size_denom,
                float(current_depth) / self._initial_depth_denom,
                float(self.previous_size) / self._initial_size_denom,
                float(self.previous_depth) / self._initial_depth_denom,
            ],
            dtype=torch.float32,
        )
        suffix = torch.tensor([float(step) / self._num_steps_denom], dtype=torch.float32)
        return torch.cat([prefix, counts, suffix], dim=0)

    def record_action(self, *, action: int, previous_size: int, previous_depth: int) -> None:
        self.previous_size = int(previous_size)
        self.previous_depth = int(previous_depth)
        self._recent_actions.append(int(action))


def discounted_returns(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    out = torch.zeros(len(rewards), dtype=rewards.dtype, device=rewards.device)
    g = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out
