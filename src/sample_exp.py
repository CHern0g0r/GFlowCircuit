from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyspiel
import torch
from omegaconf import DictConfig, OmegaConf

from src.baselines.resyn2 import get_depth, get_size
from src.test import _load_cfg, _load_policy, _tb_reward_params
from src.utils import Observation

from tqdm import tqdm


_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_experiment_dir(*, experiment: str, outputs_root: Path) -> Path:
    candidate = Path(experiment).expanduser()
    if candidate.is_absolute():
        experiment_dir = candidate.resolve()
    else:
        experiment_dir = (outputs_root / candidate).resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    return experiment_dir


def _discover_run_checkpoints(experiment_dir: Path) -> list[tuple[int, Path]]:
    saved_models = experiment_dir / "saved_models"
    if not saved_models.is_dir():
        raise FileNotFoundError(f"No saved_models directory in experiment: {saved_models}")

    runs: list[tuple[int, Path]] = []
    for run_dir in sorted(saved_models.iterdir()):
        if not run_dir.is_dir():
            continue
        match = _RUN_DIR_RE.match(run_dir.name)
        if match is None:
            continue
        checkpoint_path = run_dir / "last.pt"
        if checkpoint_path.exists():
            runs.append((int(match.group(1)), checkpoint_path.resolve()))

    if not runs:
        raise FileNotFoundError(f"No run_*/last.pt checkpoints found under: {saved_models}")
    return runs


def _initial_circuit_metrics(*, circuit_path: Path, num_steps: int) -> tuple[int, int]:
    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": str(circuit_path)})
    obs = Observation.from_state(game.new_initial_state())
    return int(get_size(obs)), int(get_depth(obs))


def _sample_trajectories(
    *,
    checkpoint_path: Path,
    cfg: DictConfig,
    circuit_path: Path,
    num_steps: int,
    num_samples: int,
    device: torch.device,
    seed: int,
) -> list[tuple[int, int]]:
    loaded = _load_policy(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        circuit_path=circuit_path,
        num_steps=num_steps,
        device=device,
    )
    algorithm_name = loaded["algorithm"]
    policy = loaded["policy"]
    reward_class = loaded["reward_class"]
    available_actions = loaded["available_actions"]

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    metrics: list[tuple[int, int]] = []
    if algorithm_name == "gflownet_tb":
        from src.algorithms.gflownet_tb.sampler import sample_tb_trajectory

        tb_params = _tb_reward_params(cfg)
        for _ in range(max(1, int(num_samples))):
            with torch.no_grad():
                trajectory = sample_tb_trajectory(
                    file_path=str(circuit_path),
                    num_steps=num_steps,
                    policy=policy,
                    reward_class=reward_class,
                    sample_actions=True,
                    available_actions=available_actions,
                    **tb_params,
                )
            metrics.append((int(trajectory.final_size), int(trajectory.final_depth)))
    elif algorithm_name == "reinforce":
        from src.algorithms.reinforce.episode import run_reinforce_episode
        from src.baselines.resyn2 import build_resyn2_cache

        baseline = OmegaConf.select(cfg, "baseline")
        baseline_scale = float(OmegaConf.select(cfg, "baseline_scale") or 1.0)
        resyn2_baselines = build_resyn2_cache(
            circuits=[str(circuit_path)],
            num_steps=int(num_steps),
            reward_class=reward_class,
            baseline=baseline,
            baseline_scale=baseline_scale,
        )
        resyn2_baseline = resyn2_baselines[str(circuit_path)]
        for _ in range(max(1, int(num_samples))):
            with torch.no_grad():
                episode = run_reinforce_episode(
                    file_path=str(circuit_path),
                    num_steps=num_steps,
                    policy=policy,
                    reward_class=reward_class,
                    sample_actions=True,
                    resyn2_baseline=resyn2_baseline,
                    baseline=baseline,
                    available_actions=available_actions,
                )
            metrics.append((int(episode["final_size"]), int(episode["final_depth"])))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return metrics


def _sample_experiment(
    *,
    experiment_dir: Path,
    circuit_paths: list[Path],
    num_samples: int,
    num_steps: int | None,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    config_path = experiment_dir / ".hydra" / "config.yaml"
    cfg = _load_cfg(config_path)
    resolved_num_steps = int(num_steps if num_steps is not None else cfg["num_steps"])

    rows: list[dict[str, object]] = []
    for circuit_path in circuit_paths:
        initial_size, initial_depth = _initial_circuit_metrics(
            circuit_path=circuit_path,
            num_steps=resolved_num_steps,
        )
        rows.append(
            {
                "circuit": str(circuit_path),
                "run_id": None,
                "size": initial_size,
                "depth": initial_depth,
            }
        )

    run_checkpoints = _discover_run_checkpoints(experiment_dir)
    for run_idx, (run_id, checkpoint_path) in enumerate(run_checkpoints):
        for circuit_idx, circuit_path in enumerate(circuit_paths):
            sample_seed = int(seed) + run_idx * 10_000 + circuit_idx * 1_000
            for final_size, final_depth in tqdm(_sample_trajectories(
                checkpoint_path=checkpoint_path,
                cfg=cfg,
                circuit_path=circuit_path,
                num_steps=resolved_num_steps,
                num_samples=num_samples,
                device=device,
                seed=sample_seed,
            ), desc=f"Sampling circuit {circuit_path.name} for run {run_id}"):
                rows.append(
                    {
                        "circuit": str(circuit_path),
                        "run_id": int(run_id),
                        "size": int(final_size),
                        "depth": int(final_depth),
                    }
                )
    df = pd.DataFrame(rows, columns=["circuit", "run_id", "size", "depth"])
    df["run_id"] = df["run_id"].astype("Int64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample trajectories for all training runs in a Hydra experiment directory.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help='Experiment path relative to outputs/, e.g. "2026-05-26/12:06_tb_zhu_i10_79035".',
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="Root outputs directory. Defaults to <repo>/outputs.",
    )
    parser.add_argument(
        "--circuit",
        action="append",
        required=True,
        help="Path to a circuit file (.blif/.aig). Repeat for multiple circuits.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of stochastic rollouts per run and circuit.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_steps from the experiment config.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for sampling.")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu. Defaults to cuda if available.")
    args = parser.parse_args()

    outputs_root = (
        Path(args.outputs_root).expanduser().resolve()
        if args.outputs_root is not None
        else _repo_root() / "outputs"
    )
    experiment_dir = _resolve_experiment_dir(experiment=args.experiment, outputs_root=outputs_root)

    circuit_paths: list[Path] = []
    for circuit_arg in args.circuit:
        circuit_path = Path(circuit_arg).expanduser().resolve()
        if not circuit_path.exists():
            raise FileNotFoundError(f"Circuit not found: {circuit_path}")
        circuit_paths.append(circuit_path)
    if not circuit_paths:
        raise ValueError("At least one --circuit is required")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    df = _sample_experiment(
        experiment_dir=experiment_dir,
        circuit_paths=circuit_paths,
        num_samples=int(args.num_samples),
        num_steps=args.num_steps,
        device=device,
        seed=int(args.seed),
    )

    out_path = experiment_dir / "points.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
