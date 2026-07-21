from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyspiel
import torch

from src.algorithms.gflownet_tb.exact import ExactNode, ExactTree, Prefix, ProbabilityFn
from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.reward import transform_terminal_reward
from src.baselines.resyn2 import OBS_DEPTH_IDX, OBS_SIZE_IDX
from src.eval_metrics import comparable_return
from src.utils import Observation, ZhuVectorState, filter_legal_actions, resolve_vector_action_ids


@dataclass(frozen=True)
class CircuitEnumeration:
    tree: ExactTree
    circuit_path: str
    initial_size: int
    initial_depth: int
    terminal_metadata: dict[Prefix, dict[str, float | int]]


def enumerate_circuit_tree(
    *,
    circuit_path: str | Path,
    horizon: int,
    reward_class: type | Callable[..., Any],
    reward_alpha: float,
    reward_eps: float,
    reward_improvement_clip: float,
    available_actions: list[int] | None,
    require_constant_branching: int | None = None,
) -> CircuitEnumeration:
    """Enumerate a deterministic circuit action-prefix tree with cached observations."""
    circuit_path = str(Path(circuit_path).resolve())
    horizon = int(horizon)
    game = pyspiel.load_game("circuit", {"num_steps": horizon, "file_path": circuit_path})
    root_state = game.new_initial_state()
    root_obs = Observation.from_state(root_state, available_actions=available_actions)
    initial_size = int(root_obs.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(root_obs.obs_tensor[OBS_DEPTH_IDX])
    num_actions = int(game.num_distinct_actions())
    action_ids = resolve_vector_action_ids(num_actions, available_actions)
    root_vector = ZhuVectorState(
        initial_size=initial_size,
        initial_depth=initial_depth,
        num_steps=horizon,
        action_ids=action_ids,
    )

    nodes: dict[Prefix, ExactNode] = {}
    terminal_log_rewards: dict[Prefix, float] = {}
    terminal_metadata: dict[Prefix, dict[str, float | int]] = {}

    def visit(state: pyspiel.State, prefix: Prefix, vector_state: ZhuVectorState) -> None:
        if state.is_terminal():
            obs_tensor = torch.as_tensor(state.observation_tensor(0), dtype=torch.float32)
            final_size = int(obs_tensor[OBS_SIZE_IDX])
            final_depth = int(obs_tensor[OBS_DEPTH_IDX])
            improvement_raw = comparable_return(
                reward_class=reward_class,
                initial_size=initial_size,
                initial_depth=initial_depth,
                final_size=final_size,
                final_depth=final_depth,
            )
            transformed_reward = transform_terminal_reward(
                improvement=improvement_raw,
                reward_alpha=reward_alpha,
                reward_eps=reward_eps,
                reward_improvement_clip=reward_improvement_clip,
            )
            terminal_log_rewards[prefix] = transformed_reward.log_reward
            terminal_metadata[prefix] = {
                "final_size": final_size,
                "final_depth": final_depth,
                "improvement_raw": transformed_reward.raw_improvement,
                "improvement_clipped": transformed_reward.clipped_improvement,
                "terminal_reward": transformed_reward.reward,
                "log_reward": transformed_reward.log_reward,
                "log_pb": 0.0,
            }
            return

        obs = Observation.from_state(state, available_actions=available_actions)
        current_size = int(obs.obs_tensor[OBS_SIZE_IDX])
        current_depth = int(obs.obs_tensor[OBS_DEPTH_IDX])
        legal_actions = tuple(int(action) for action in obs.legal_actions)
        if require_constant_branching is not None and len(legal_actions) != int(require_constant_branching):
            raise ValueError(
                f"prefix {prefix} has {len(legal_actions)} legal actions; "
                f"expected {int(require_constant_branching)}"
            )
        vector_tensor = vector_state.vector(
            current_size=current_size,
            current_depth=current_depth,
            step=len(prefix),
        )
        nodes[prefix] = ExactNode(
            prefix=prefix,
            legal_actions=legal_actions,
            payload=obs.with_vector(vector_tensor),
        )

        for action in legal_actions:
            child_state = state.clone()
            child_vector = copy.deepcopy(vector_state)
            child_vector.record_action(
                action=action,
                previous_size=current_size,
                previous_depth=current_depth,
            )
            child_state.apply_action(action)
            visit(child_state, prefix + (action,), child_vector)

    visit(root_state, (), root_vector)
    tree = ExactTree(
        depth=horizon,
        num_actions=num_actions,
        nodes=nodes,
        terminal_log_rewards=terminal_log_rewards,
        name=f"circuit_{Path(circuit_path).stem}_depth_{horizon}",
    )
    if require_constant_branching is not None:
        expected_terminals = int(require_constant_branching) ** horizon
        if len(tree.terminal_log_rewards) != expected_terminals:
            raise ValueError(
                f"enumerated {len(tree.terminal_log_rewards)} terminal sequences; "
                f"expected {expected_terminals}"
            )
    return CircuitEnumeration(
        tree=tree,
        circuit_path=circuit_path,
        initial_size=initial_size,
        initial_depth=initial_depth,
        terminal_metadata=terminal_metadata,
    )


def neural_probability_fn(*, tree: ExactTree, policy: TBGFlowNetPolicy) -> ProbabilityFn:
    def probabilities(prefixes: list[Prefix]) -> torch.Tensor:
        observations = [tree.nodes[prefix].payload for prefix in prefixes]
        logits = policy(observations)
        legal_rows = [list(tree.nodes[prefix].legal_actions) for prefix in prefixes]
        return policy.masked_probs(logits, legal_rows)

    return probabilities


def _update_tensor_hash(hasher: Any, name: str, tensor: torch.Tensor | None) -> None:
    hasher.update(name.encode("utf-8"))
    if tensor is None:
        hasher.update(b"none")
        return
    value = tensor.detach().to(device="cpu").contiguous()
    hasher.update(str(value.dtype).encode("utf-8"))
    hasher.update(str(tuple(value.shape)).encode("utf-8"))
    hasher.update(value.numpy().tobytes())


def observation_signature(observation: Observation, legal_actions: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    _update_tensor_hash(hasher, "obs", observation.obs_tensor)
    _update_tensor_hash(hasher, "vector", observation.vector_tensor)
    _update_tensor_hash(hasher, "graph_x", observation.graph.x)
    _update_tensor_hash(hasher, "graph_edge_index", observation.graph.edge_index)
    _update_tensor_hash(hasher, "graph_edge_attr", observation.graph.edge_attr)
    hasher.update(repr(tuple(legal_actions)).encode("utf-8"))
    return hasher.hexdigest()


def analyze_observation_aliases(
    tree: ExactTree,
    *,
    target_tolerance: float = 1e-8,
) -> dict[str, Any]:
    groups: dict[str, list[Prefix]] = {}
    for prefix in tree.nonterminal_prefixes:
        node = tree.nodes[prefix]
        if not isinstance(node.payload, Observation):
            raise TypeError(f"prefix {prefix} does not contain an Observation payload")
        signature = observation_signature(node.payload, node.legal_actions)
        groups.setdefault(signature, []).append(prefix)

    collision_groups: list[dict[str, Any]] = []
    conflicting_groups = 0
    for signature, prefixes in sorted(groups.items()):
        if len(prefixes) < 2:
            continue
        target_rows: list[list[float]] = []
        for prefix in prefixes:
            target = tree.target_action_probs(prefix)
            target_rows.append([float(target.get(action, 0.0)) for action in range(tree.num_actions)])
        max_spread = max(
            max(row[action] for row in target_rows) - min(row[action] for row in target_rows)
            for action in range(tree.num_actions)
        )
        conflicting = bool(max_spread > float(target_tolerance))
        conflicting_groups += int(conflicting)
        collision_groups.append(
            {
                "signature": signature,
                "prefixes": [list(prefix) for prefix in prefixes],
                "target_action_probabilities": target_rows,
                "max_target_probability_spread": float(max_spread),
                "conflicting": conflicting,
            }
        )

    return {
        "num_prefixes": len(tree.nodes),
        "num_unique_observations": len(groups),
        "num_collision_groups": len(collision_groups),
        "num_conflicting_groups": conflicting_groups,
        "target_tolerance": float(target_tolerance),
        "groups": collision_groups,
    }


__all__ = [
    "CircuitEnumeration",
    "analyze_observation_aliases",
    "enumerate_circuit_tree",
    "neural_probability_fn",
    "observation_signature",
]
