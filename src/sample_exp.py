from __future__ import annotations

import argparse
import numpy as np
import re
from dataclasses import dataclass
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
_PCN_TARGET_KEY_DECIMALS = 8


@dataclass(frozen=True)
class PCNTargetCommand:
    target_return: torch.Tensor
    target_horizon: float
    sample_actions: bool
    source: str


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


def _pcn_target_commands(
    *,
    target_returns: list,
    target_horizons: list,
    num_samples: int,
    rng: np.random.Generator,
    mode: str,
    zero_variance_jitter: float = 0.05,
) -> list[PCNTargetCommand]:
    anchors = _dedupe_pcn_target_commands(target_returns=target_returns, target_horizons=target_horizons)
    if not anchors:
        raise ValueError("PCN sampling requires archive_target_returns and archive_target_horizons in the checkpoint")

    if mode == "paper":
        return anchors
    if mode == "stochastic-actions":
        return [
            PCNTargetCommand(
                target_return=command.target_return,
                target_horizon=command.target_horizon,
                sample_actions=True,
                source="archive",
            )
            for command in anchors
        ]
    if mode == "broad-target":
        return _pcn_broad_target_commands(
            anchors=anchors,
            num_samples=num_samples,
            rng=rng,
            jitter_scale=zero_variance_jitter,
        )
    if mode != "target":
        raise ValueError(f"Unsupported PCN sampling mode: {mode}")

    sample_count = max(1, int(num_samples))
    if len(anchors) >= sample_count:
        return anchors[:sample_count]

    returns = torch.stack([command.target_return for command in anchors], dim=0)
    out = list(anchors)
    while len(out) < sample_count:
        base_idx = int(rng.integers(0, len(anchors)))
        objective_idx = int(rng.integers(0, returns.shape[1]))
        target_return = returns[base_idx].clone()
        sigma = float(torch.std(returns[:, objective_idx], unbiased=False).item())
        source = "perturbed"
        if sigma > 0.0:
            target_return[objective_idx] += float(rng.uniform(0.0, sigma))
        else:
            target_return[objective_idx] += _pcn_zero_variance_jitter(rng=rng, scale=zero_variance_jitter)
            source = "jittered"
        out.append(
            PCNTargetCommand(
                target_return=target_return,
                target_horizon=anchors[base_idx].target_horizon,
                sample_actions=False,
                source=source,
            )
        )
    return out


def _dedupe_pcn_target_commands(*, target_returns: list, target_horizons: list) -> list[PCNTargetCommand]:
    out: list[PCNTargetCommand] = []
    seen: set[tuple[float, ...]] = set()
    for target_return, target_horizon in zip(target_returns, target_horizons):
        return_tensor = torch.tensor(target_return, dtype=torch.float32)
        key = tuple(round(float(value), _PCN_TARGET_KEY_DECIMALS) for value in return_tensor.tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            PCNTargetCommand(
                target_return=return_tensor,
                target_horizon=float(target_horizon),
                sample_actions=False,
                source="archive",
            )
        )
    return out


def _pcn_zero_variance_jitter(*, rng: np.random.Generator, scale: float = 0.05) -> float:
    return float(rng.uniform(0.0, max(0.0, float(scale))))


def _pcn_broad_target_commands(
    *,
    anchors: list[PCNTargetCommand],
    num_samples: int,
    rng: np.random.Generator,
    jitter_scale: float = 0.05,
) -> list[PCNTargetCommand]:
    sample_count = max(1, int(num_samples))
    returns = torch.stack([command.target_return for command in anchors], dim=0)
    min_returns = returns.min(dim=0).values
    max_returns = returns.max(dim=0).values
    ranges = max_returns - min_returns
    fallback = torch.full_like(ranges, float(jitter_scale))
    ranges = torch.where(ranges > 0.0, ranges, fallback)
    low = min_returns
    high = max_returns + ranges

    out: list[PCNTargetCommand] = []
    if returns.shape[1] == 2 and sample_count > 1:
        weights = torch.linspace(0.0, 1.0, steps=sample_count, dtype=torch.float32)
        for idx, weight in enumerate(weights):
            target_return = low + (high - low) * weight
            horizon = anchors[idx % len(anchors)].target_horizon
            out.append(
                PCNTargetCommand(
                    target_return=target_return,
                    target_horizon=horizon,
                    sample_actions=False,
                    source="broad",
                )
            )
        return out

    for idx in range(sample_count):
        random_unit = torch.tensor(rng.uniform(0.0, 1.0, size=int(returns.shape[1])), dtype=torch.float32)
        target_return = low + (high - low) * random_unit
        out.append(
            PCNTargetCommand(
                target_return=target_return,
                target_horizon=anchors[idx % len(anchors)].target_horizon,
                sample_actions=False,
                source="broad",
            )
        )
    return out


def _sample_trajectories(
    *,
    checkpoint_path: Path,
    cfg: DictConfig,
    circuit_path: Path,
    num_steps: int,
    num_samples: int,
    device: torch.device,
    seed: int,
    pcn_sampling_mode: str,
    pcn_zero_variance_jitter: float,
) -> list[dict[str, object]]:
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
    mo_reward_class = loaded.get("mo_reward_class")
    pcn_meta = loaded.get("pcn", {})
    available_actions = loaded["available_actions"]

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    metrics: list[dict[str, object]] = []
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
            metrics.append({"size": int(trajectory.final_size), "depth": int(trajectory.final_depth)})
    elif algorithm_name == "drills_a2c":
        from src.algorithms.drills_a2c.sampler import sample_drills_a2c_trajectory

        for _ in range(max(1, int(num_samples))):
            with torch.no_grad():
                trajectory = sample_drills_a2c_trajectory(
                    file_path=str(circuit_path),
                    num_steps=num_steps,
                    policy=policy,
                    reward_class=reward_class,
                    sample_actions=True,
                    available_actions=available_actions,
                )
            metrics.append({"size": int(trajectory.final_size), "depth": int(trajectory.final_depth)})
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
            metrics.append({"size": int(episode["final_size"]), "depth": int(episode["final_depth"])})
    elif algorithm_name == "ppo":
        from src.algorithms.ppo.sampler import sample_ppo_trajectory
        from src.baselines.resyn2 import build_resyn2_cache

        value_net = loaded["value_net"]
        if value_net is None:
            raise ValueError("ppo sampling requires a value network")

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
                trajectory = sample_ppo_trajectory(
                    file_path=str(circuit_path),
                    num_steps=num_steps,
                    policy=policy,
                    value_network=value_net,
                    reward_class=reward_class,
                    sample_actions=True,
                    baseline=baseline,
                    resyn2_baseline=resyn2_baseline,
                    available_actions=available_actions,
                )
            metrics.append({"size": int(trajectory.final_size), "depth": int(trajectory.final_depth)})
    elif algorithm_name == "pcn":
        from src.algorithms.pcn.sampler import sample_pcn_trajectory

        pcn_cfg = OmegaConf.select(cfg, "pcn")
        if pcn_cfg is None:
            pcn_cfg = OmegaConf.select(cfg, "algorithm.pcn")
        if pcn_cfg is None:
            pcn_cfg = OmegaConf.create({})
        desired_return_clip = bool(OmegaConf.select(pcn_cfg, "desired_return_clip") or False)
        target_returns = pcn_meta.get("archive_target_returns", []) if isinstance(pcn_meta, dict) else []
        target_horizons = pcn_meta.get("archive_target_horizons", []) if isinstance(pcn_meta, dict) else []
        rng = np.random.default_rng(int(seed))
        for command in _pcn_target_commands(
            target_returns=target_returns,
            target_horizons=target_horizons,
            num_samples=num_samples,
            rng=rng,
            mode=pcn_sampling_mode,
            zero_variance_jitter=pcn_zero_variance_jitter,
        ):
            with torch.no_grad():
                trajectory = sample_pcn_trajectory(
                    file_path=str(circuit_path),
                    num_steps=num_steps,
                    policy=policy,
                    mo_reward_class=mo_reward_class,
                    sample_actions=command.sample_actions,
                    rng=rng,
                    available_actions=available_actions,
                    desired_return=command.target_return,
                    desired_horizon=command.target_horizon,
                    desired_return_clip=desired_return_clip,
                    gamma=float(cfg.get("gamma", 1.0)),
                )
            row: dict[str, object] = {
                "size": int(trajectory.final_size),
                "depth": int(trajectory.final_depth),
                "target_horizon": float(command.target_horizon),
                "pcn_sampling_mode": pcn_sampling_mode,
                "target_source": command.source,
            }
            for objective_idx, target_value in enumerate(command.target_return.tolist()):
                row[f"target_return_{objective_idx}"] = float(target_value)
            metrics.append(row)
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
    pcn_sampling_mode: str,
    pcn_zero_variance_jitter: float,
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
            for sample_row in tqdm(_sample_trajectories(
                checkpoint_path=checkpoint_path,
                cfg=cfg,
                circuit_path=circuit_path,
                num_steps=resolved_num_steps,
                num_samples=num_samples,
                device=device,
                seed=sample_seed,
                pcn_sampling_mode=pcn_sampling_mode,
                pcn_zero_variance_jitter=pcn_zero_variance_jitter,
            ), desc=f"Sampling circuit {circuit_path.name} for run {run_id}"):
                row = {
                    "circuit": str(circuit_path),
                    "run_id": int(run_id),
                    **sample_row,
                }
                rows.append(row)
    df = pd.DataFrame(rows)
    leading_columns = ["circuit", "run_id", "size", "depth"]
    other_columns = [column for column in df.columns if column not in leading_columns]
    df = df[leading_columns + other_columns]
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
    parser.add_argument("--num-samples", type=int, default=20, help="Number of sampled rollouts per run and circuit.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_steps from the experiment config.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for sampling.")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu. Defaults to cuda if available.")
    parser.add_argument(
        "--pcn-sampling-mode",
        choices=("paper", "target", "broad-target", "stochastic-actions"),
        default="target",
        help=(
            "PCN only: 'paper' rolls out archived targets deterministically; "
            "'target' samples desired-return targets and rolls out deterministically; "
            "'broad-target' sweeps a broader deterministic target range; "
            "'stochastic-actions' samples actions from archived targets for debugging."
        ),
    )
    parser.add_argument(
        "--pcn-zero-variance-jitter",
        type=float,
        default=0.05,
        help="PCN only: target-return jitter used when archive targets have zero variance.",
    )
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
        pcn_sampling_mode=str(args.pcn_sampling_mode),
        pcn_zero_variance_jitter=float(args.pcn_zero_variance_jitter),
    )

    out_path = experiment_dir / "points.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
