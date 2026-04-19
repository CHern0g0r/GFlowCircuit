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
