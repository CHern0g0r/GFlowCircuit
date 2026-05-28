from __future__ import annotations

import pyspiel

from src.utils import Observation

OBS_STEP_IDX = 0
OBS_NUM_STEPS_IDX = 1
OBS_SIZE_IDX = 2
OBS_DEPTH_IDX = 3
OBS_REWARD_IDX = 4

RESYN2_ACTION_SEQUENCE = [0, 1, 2, 0, 1, 3, 0, 4, 3, 0]


def _repeat_resyn2_sequence(total_ops: int) -> list[int]:
    if total_ops <= 0:
        raise ValueError("total_ops must be positive")
    repeats = (int(total_ops) + len(RESYN2_ACTION_SEQUENCE) - 1) // len(RESYN2_ACTION_SEQUENCE)
    return (RESYN2_ACTION_SEQUENCE * repeats)[: int(total_ops)]


def get_size(obs: Observation) -> int:
    return int(obs.obs_tensor[OBS_SIZE_IDX])


def get_depth(obs: Observation) -> int:
    return int(obs.obs_tensor[OBS_DEPTH_IDX])


def _play_resyn2_reference(
    file_path: str,
    num_steps: int,
    reward_class: type,
    action_sequence: list[int] | None = None,
) -> tuple[float, float]:
    """Run fixed resyn2 actions; return (sum of per-step custom rewards, custom terminal reward)."""
    seq = _repeat_resyn2_sequence(num_steps) if action_sequence is None else action_sequence
    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    state = game.new_initial_state()
    obs0 = Observation.from_state(state)
    initial_size = get_size(obs0)
    initial_depth = get_depth(obs0)
    reward_func = reward_class(initial_size, initial_depth)

    total_custom = 0.0
    for idx, action in enumerate(seq):
        if state.is_terminal():
            break
        obs = Observation.from_state(state)
        prev_size = get_size(obs)
        prev_depth = get_depth(obs)
        legal = list(state.legal_actions())
        if action not in legal:
            raise RuntimeError(
                f"Resyn2 baseline: action {action} at index {idx} is not legal. Legal: {legal}"
            )
        state.apply_action(action)
        next_obs = Observation.from_state(state)
        total_custom += float(reward_func(get_size(next_obs), get_depth(next_obs), prev_size, prev_depth))

    final_obs = Observation.from_state(state)
    final_step = float(reward_func(get_size(final_obs), get_depth(final_obs), initial_size, initial_depth))
    return float(total_custom), float(final_step)


def _resyn2_average_reward_per_step(file_path: str, reward_class: type, total_ops: int = 20) -> float:
    """Compute Zhu baseline: average custom reward per operation over 2x resyn2."""
    if total_ops <= 0:
        raise ValueError("total_ops must be positive")

    game = pyspiel.load_game("circuit", {"num_steps": int(total_ops), "file_path": file_path})
    state = game.new_initial_state()
    obs0 = Observation.from_state(state)
    initial_size = get_size(obs0)
    initial_depth = get_depth(obs0)
    reward_func = reward_class(initial_size, initial_depth)

    total_reward = 0.0
    seq = _repeat_resyn2_sequence(total_ops)
    for action in seq:
        if state.is_terminal():
            break
        if action not in state.legal_actions():
            break
        obs = Observation.from_state(state)
        prev_size = get_size(obs)
        prev_depth = get_depth(obs)
        state.apply_action(action)
        next_obs = Observation.from_state(state)
        total_reward += float(reward_func(get_size(next_obs), get_depth(next_obs), prev_size, prev_depth))

    return float(total_reward / total_ops)


def evaluate_resyn2_variants(file_path: str) -> dict[str, dict[str, int | float]]:
    """Return paper-style baseline metrics: resyn2-1, resyn2-2, resyn2-inf."""
    variants = {
        "resyn2_1": len(RESYN2_ACTION_SEQUENCE),
        "resyn2_2": len(RESYN2_ACTION_SEQUENCE) * 2,
    }
    out: dict[str, dict[str, int | float]] = {}
    for name, steps in variants.items():
        game = pyspiel.load_game("circuit", {"num_steps": int(steps), "file_path": file_path})
        state = game.new_initial_state()
        obs0 = Observation.from_state(state)
        initial_size = get_size(obs0)
        initial_depth = get_depth(obs0)
        seq = _repeat_resyn2_sequence(steps)
        for action in seq:
            if state.is_terminal():
                break
            if action not in state.legal_actions():
                break
            state.apply_action(action)
        obsf = Observation.from_state(state)
        out[name] = {
            "initial_size": initial_size,
            "initial_depth": initial_depth,
            "final_size": get_size(obsf),
            "final_depth": get_depth(obsf),
        }

    # resyn2-inf: stop after 5 consecutive unchanged recipe executions.
    max_recipes = 100
    game = pyspiel.load_game(
        "circuit",
        {"num_steps": int(max_recipes * len(RESYN2_ACTION_SEQUENCE)), "file_path": file_path},
    )
    state = game.new_initial_state()
    obs0 = Observation.from_state(state)
    initial_size = get_size(obs0)
    initial_depth = get_depth(obs0)
    unchanged_runs = 0
    for _ in range(max_recipes):
        before = Observation.from_state(state)
        before_pair = (get_size(before), get_depth(before))
        for action in RESYN2_ACTION_SEQUENCE:
            if state.is_terminal():
                break
            if action not in state.legal_actions():
                break
            state.apply_action(action)
        after = Observation.from_state(state)
        after_pair = (get_size(after), get_depth(after))
        if after_pair == before_pair:
            unchanged_runs += 1
        else:
            unchanged_runs = 0
        if unchanged_runs >= 5 or state.is_terminal():
            break
    obsf = Observation.from_state(state)
    out["resyn2_inf"] = {
        "initial_size": initial_size,
        "initial_depth": initial_depth,
        "final_size": get_size(obsf),
        "final_depth": get_depth(obsf),
    }
    return out


def build_resyn2_cache(
    *,
    circuits: list[str],
    num_steps: int,
    reward_class: type,
    baseline: str | None = None,
    baseline_scale: float = 1.0,
) -> dict[str, dict[str, object]]:
    """Precompute Resyn2-related reference metrics once per circuit."""
    cache: dict[str, dict[str, object]] = {}
    for file_path in circuits:
        r2_total, r2_final_step = _play_resyn2_reference(file_path, num_steps, reward_class)
        variants = evaluate_resyn2_variants(file_path)
        entry: dict[str, object] = {
            "resyn2_baseline_total_reward": float(r2_total),
            "resyn2_baseline_final_step_reward": float(r2_final_step),
            "resyn2_variants": variants,
            "zhu_reward_baseline_per_step": 0.0,
        }
        if baseline == "zhu_resyn2":
            entry["zhu_reward_baseline_per_step"] = _resyn2_average_reward_per_step(
                file_path=file_path,
                reward_class=reward_class,
                total_ops=20,
            )
            entry["zhu_reward_baseline_scale"] = float(baseline_scale)
        cache[file_path] = entry
    return cache


__all__ = [
    "OBS_STEP_IDX",
    "OBS_NUM_STEPS_IDX",
    "OBS_SIZE_IDX",
    "OBS_DEPTH_IDX",
    "OBS_REWARD_IDX",
    "RESYN2_ACTION_SEQUENCE",
    "get_size",
    "get_depth",
    "evaluate_resyn2_variants",
    "build_resyn2_cache",
]

"""Resyn2 baseline evaluator for OpenSpiel circuit optimization.

This script evaluates a fixed action sequence (resyn2 by default) on the
OpenSpiel `circuit` game and reports episode statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyspiel
import yaml


ACTION_NAMES = {
    0: "balance",
    1: "rewrite",
    2: "refactor",
    3: "rewrite -z",
    4: "refactor -z",
    5: "resub fast",
    6: "resub strong",
}

SEQUENCE = [0, 1, 2, 0, 1, 3, 0, 4, 3, 0]

OBS_STEP_IDX = 0
OBS_NUM_STEPS_IDX = 1
OBS_SIZE_IDX = 2
OBS_DEPTH_IDX = 3
OBS_REWARD_IDX = 4


def _load_action_sequence(config_path: Path) -> list[int]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "action_sequence" not in cfg:
        raise ValueError(f"Missing 'action_sequence' key in config: {config_path}")

    sequence = cfg["action_sequence"]
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("'action_sequence' must be a non-empty list")

    try:
        return [int(a) for a in sequence]
    except (TypeError, ValueError) as exc:
        raise ValueError("'action_sequence' must contain integer action ids") from exc


def _state_metrics(state: pyspiel.State) -> dict[str, Any]:
    obs = np.array(state.observation_tensor(0))
    return {
        "step": int(obs[OBS_STEP_IDX]),
        "num_steps": int(obs[OBS_NUM_STEPS_IDX]),
        "size": int(obs[OBS_SIZE_IDX]),
        "depth": int(obs[OBS_DEPTH_IDX]),
        "reward": float(obs[OBS_REWARD_IDX]),
        "return": float(state.returns()[0]),
    }


def evaluate_resyn2(
    file_path: str,
    action_sequence: list[int],
    num_steps: int | None = None,
    output_path: str | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    if num_steps is None:
        num_steps = len(action_sequence)
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")

    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": file_path})
    state = game.new_initial_state()

    best_state = state.clone()
    best_return = float(state.returns()[0])
    initial = _state_metrics(state)
    transitions = []

    if not quiet:
        print(f"Game: circuit")
        print(f"File: {file_path}")
        print(f"Configured steps: {num_steps}")
        print(f"Action sequence: {action_sequence}")
        print(f"Initial state: {state}")

    for idx, action in enumerate(action_sequence):
        if state.is_terminal():
            break

        legal = state.legal_actions()
        if action not in legal:
            raise RuntimeError(
                f"Action {action} at sequence index {idx} is not legal. "
                f"Legal actions: {legal}"
            )

        before = _state_metrics(state)
        state.apply_action(action)
        after = _state_metrics(state)
        current_return = after["return"]
        if current_return > best_return:
            best_return = current_return
            best_state = state.clone()

        transition = {
            "seq_idx": idx,
            "action_id": int(action),
            "action_name": ACTION_NAMES.get(action, str(action)),
            "size_before": before["size"],
            "size_after": after["size"],
            "depth_before": before["depth"],
            "depth_after": after["depth"],
            "reward": after["reward"],
            "return": after["return"],
            "step": after["step"],
        }
        transitions.append(transition)

        if not quiet:
            print(
                "  step={step:3d}  action={name:<12s} ({aid})  "
                "size: {sb} -> {sa}  depth: {db} -> {da}  reward={rw:+.3f}".format(
                    step=transition["step"],
                    name=transition["action_name"],
                    aid=transition["action_id"],
                    sb=transition["size_before"],
                    sa=transition["size_after"],
                    db=transition["depth_before"],
                    da=transition["depth_after"],
                    rw=transition["reward"],
                )
            )

    final = _state_metrics(state)
    total_reward = float(sum(t["reward"] for t in transitions))
    actions_applied = [t["action_id"] for t in transitions]
    action_names = [t["action_name"] for t in transitions]

    result = {
        "file_path": file_path,
        "num_steps": num_steps,
        "sequence_length": len(action_sequence),
        "actions_applied": actions_applied,
        "action_names": action_names,
        "initial_size": initial["size"],
        "initial_depth": initial["depth"],
        "final_size": final["size"],
        "final_depth": final["depth"],
        "total_reward": total_reward,
        "final_return": final["return"],
        "best_return": best_return,
        "terminal": bool(state.is_terminal()),
        "transitions": transitions,
    }

    if output_path:
        save_ok = pyspiel.save_circuit(best_state, output_path)
        if save_ok <= 0:
            raise RuntimeError(f"Failed to save best state to: {output_path}")
        result["saved_best_to"] = output_path

    if not quiet:
        print("\n=== Summary ===")
        print(f"Applied actions: {len(actions_applied)}")
        print(f"Initial size:    {result['initial_size']}")
        print(f"Final size:      {result['final_size']}")
        print(f"Initial depth:   {result['initial_depth']}")
        print(f"Final depth:     {result['final_depth']}")
        if result["initial_size"] > 0:
            reduction = 100.0 * (result["initial_size"] - result["final_size"]) / result["initial_size"]
            print(f"Size reduction:  {reduction:.2f}%")
        print(f"Total reward:    {result['total_reward']:.3f}")
        print(f"Final return:    {result['final_return']:.3f}")
        print(f"Best return:     {result['best_return']:.3f}")
        if output_path:
            print(f"Saved best state to: {output_path}")

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed resyn2 sequence on OpenSpiel circuit game.")
    parser.add_argument("--file_path", required=True, help="Path to input circuit file (.aig or .blif).")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Environment num_steps. Defaults to sequence length from config.",
    )
    parser.add_argument("--output_path", default=None, help="Optional path to save best-return circuit.")
    parser.add_argument("--json_out", default=None, help="Optional path to write full result JSON.")
    parser.add_argument("--quiet", action="store_true", help="Only print final JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sequence = SEQUENCE

    result = evaluate_resyn2(
        file_path=args.file_path,
        action_sequence=sequence,
        num_steps=args.num_steps,
        output_path=args.output_path,
        quiet=args.quiet,
    )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k != "transitions"}, indent=2))


if __name__ == "__main__":
    main()
