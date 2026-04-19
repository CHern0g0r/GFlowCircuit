import numpy as np
import torch
import yaml
import pyspiel

from pathlib import Path
from torch_geometric.data import Data
from dataclasses import dataclass


@dataclass
class Observation:
    obs_tensor: torch.Tensor
    graph: Data
    legal_actions: list[int]

    @classmethod
    def from_state(cls, state: pyspiel.State) -> "Observation":
        obs_tensor = torch.as_tensor(state.observation_tensor(0), dtype=torch.float32)
        x, e, ea = pyspiel.circuit_graph(state)
        graph = Data(
            x=torch.as_tensor(x),
            edge_index=torch.as_tensor(e).T,
            edge_attr=torch.as_tensor(ea)
        )
        legal_actions = list(state.legal_actions())
        return cls(obs_tensor, graph, legal_actions)


@dataclass
class StepSample:
    observation: Observation
    action: int
    probs: torch.Tensor
    reward: float


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
        if p.exists():
            circuits.append(str(p))
        else:
            print(f"[warn] skipping missing circuit: {p}")

    if not circuits:
        raise FileNotFoundError("No existing circuit files found from dataset config.")
    return circuits


def discounted_returns(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    out = torch.zeros(len(rewards), dtype=rewards.dtype)
    g = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out
