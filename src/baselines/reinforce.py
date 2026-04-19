"""Minimal REINFORCE baseline for OpenSpiel circuit optimization.

This script trains a linear softmax policy on observation tensors from the
OpenSpiel `circuit` game (no graph features). It reads dataset metadata from
YAML, splits circuits into train/test sets, runs REINFORCE on train circuits,
and evaluates greedy policy performance on test circuits.
"""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from typing import Any

import numpy as np
import pyspiel

from src.models.policy import ReinforcePolicy, Policy
from src.utils import (
    train_test_split,
    StepSample,
    load_circuits,
    discounted_returns,
)
from src.models.Linear import IdEncoder, LinearHead
from src.train.utils import get_obs_dim_and_num_actions
from src.train import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic REINFORCE training for OpenSpiel circuit game.")
    parser.add_argument(
        "--dataset_cfg",
        type=Path,
        required=True,
        help="YAML config listing EPFL arithmetic circuits.",
    )
    parser.add_argument("--num_steps", type=int, default=10, help="OpenSpiel circuit episode horizon.")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Fraction of circuits for train split.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--eval_every", type=int, default=25, help="Evaluate every N episodes.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="REINFORCE learning rate.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Reward discount.")
    parser.add_argument(
        "--baseline_alpha",
        type=float,
        default=0.05,
        help="EMA coefficient for value baseline (variance reduction).",
    )
    parser.add_argument("--json_out", type=Path, default=None, help="Optional path to dump final report JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    circuits = load_circuits(args.dataset_cfg)
    train_circuits, test_circuits = train_test_split(circuits, args.train_ratio, args.seed)

    print("Train circuits:")
    for c in train_circuits:
        print(f"  - {c}")
    print("Test circuits:")
    for c in test_circuits:
        print(f"  - {c}")

    obs_dim, num_actions = get_obs_dim_and_num_actions(args.num_steps, train_circuits[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = IdEncoder()
    head = LinearHead(obs_dim, num_actions)
    policy = ReinforcePolicy(encoder=enc, head=head, num_actions=num_actions)

    trainer = Trainer(policy=policy, device=device, seed=args.seed)

    train_out = trainer.train(
        train_circuits=train_circuits,
        test_circuits=test_circuits,
        num_steps=args.num_steps,
        episodes=args.episodes,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        baseline_alpha=args.baseline_alpha,
    )

    final_eval = trainer.evaluate(
        circuits=test_circuits,
        num_steps=args.num_steps,
    )

    report = {
        "dataset_cfg": str(args.dataset_cfg),
        "seed": args.seed,
        "num_steps": args.num_steps,
        "episodes": args.episodes,
        "train_ratio": args.train_ratio,
        "train_circuits": train_circuits,
        "test_circuits": test_circuits,
        "history": train_out["history"],
        "final_eval": final_eval,
    }

    print("\nFinal test evaluation:")
    print(json.dumps(final_eval, indent=2))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {args.json_out}")


if __name__ == "__main__":
    main()
