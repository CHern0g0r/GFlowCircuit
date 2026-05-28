from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from omegaconf import DictConfig, OmegaConf

from src.algorithms.gflownet_tb import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.sampler import sample_tb_trajectory
from src.algorithms.reinforce import ReinforcePolicy
from src.algorithms.reinforce.episode import run_reinforce_episode
from src.baselines.resyn2 import build_resyn2_cache
from src.models import (
    reward_class_factory,
    encoder_factory,
    head_factory,
    prepare_encoder_config,
    value_factory,
    value_input_dim,
)
from src.utils import get_obs_dim_and_num_actions, normalize_available_actions


def _default_config_from_checkpoint(checkpoint_path: Path) -> Path:
    # Expected checkpoint location:
    #   <run_output>/saved_models/run_i/last.pt
    # Expected hydra config location:
    #   <run_output>/.hydra/config.yaml
    run_output = checkpoint_path.resolve().parents[2]
    return run_output / ".hydra" / "config.yaml"


def _load_cfg(config_path: Path) -> DictConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {config_path}")
    return cfg


def _get_algorithm_name(cfg: DictConfig) -> str:
    algo_cfg = OmegaConf.select(cfg, "algorithm")
    if algo_cfg is None:
        return "reinforce"
    name = OmegaConf.select(algo_cfg, "name")
    if name is None:
        if isinstance(algo_cfg, str):
            return str(algo_cfg)
        return "reinforce"
    return str(name)


def _build_backbone(
    cfg: DictConfig,
    *,
    obs_dim: int,
    node_dim: int,
    edge_dim: int,
    num_actions: int,
    available_actions: list[int] | None,
):
    encoder_cfg = OmegaConf.to_container(cfg["encoder"], resolve=True)
    if not isinstance(encoder_cfg, dict):
        raise TypeError("encoder config must resolve to a mapping")
    encoder_cfg = prepare_encoder_config(
        encoder_cfg,
        obs_dim=obs_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_actions=num_actions,
        available_actions=available_actions,
    )

    enc = encoder_factory(encoder_cfg=encoder_cfg)
    head = head_factory(
        obs_dim=encoder_cfg["out_dim"],
        num_actions=num_actions,
        head_cfg=cfg["head"],
    )
    return enc, head


def _tb_reward_params(
    cfg: DictConfig,
    *,
    reward_alpha: float | None = None,
    reward_eps: float | None = None,
    reward_improvement_clip: float | None = None,
) -> dict[str, float]:
    tb_cfg = OmegaConf.select(cfg, "tb")
    if tb_cfg is None:
        tb_cfg = OmegaConf.select(cfg, "algorithm.tb")
    return {
        "reward_alpha": float(
            reward_alpha if reward_alpha is not None else (OmegaConf.select(tb_cfg, "reward_alpha") or 4.0)
        ),
        "reward_eps": float(reward_eps if reward_eps is not None else (OmegaConf.select(tb_cfg, "reward_eps") or 1e-8)),
        "reward_improvement_clip": float(
            reward_improvement_clip
            if reward_improvement_clip is not None
            else (OmegaConf.select(tb_cfg, "reward_improvement_clip") or 2.0)
        ),
    }


def _load_policy(
    *,
    checkpoint_path: Path,
    cfg: DictConfig,
    circuit_path: Path,
    num_steps: int,
    device: torch.device,
) -> dict[str, object]:
    obs_dim, num_actions, node_dim, edge_dim = get_obs_dim_and_num_actions(num_steps, str(circuit_path))
    available_actions = normalize_available_actions(OmegaConf.select(cfg, "available_actions"), num_actions)
    reward_type = str(cfg["reward"]["type"])
    reward_class = reward_class_factory(cfg["reward"])
    algorithm_name = _get_algorithm_name(cfg)

    enc, head = _build_backbone(
        cfg,
        obs_dim=obs_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_actions=num_actions,
        available_actions=available_actions,
    )
    value_net = None
    if algorithm_name == "gflownet_tb":
        policy = TBGFlowNetPolicy(encoder=enc, head=head, num_actions=num_actions).to(device)
    elif algorithm_name == "reinforce":
        policy = ReinforcePolicy(encoder=enc, head=head, num_actions=num_actions).to(device)
        value_cfg = OmegaConf.select(cfg, "value")
        if value_cfg is not None:
            value_cfg_dict = OmegaConf.to_container(value_cfg, resolve=True)
            if not isinstance(value_cfg_dict, dict):
                raise TypeError("value config must resolve to a mapping")
            value_net = value_factory(
                obs_dim=value_input_dim(
                    obs_dim=obs_dim,
                    num_actions=num_actions,
                    available_actions=available_actions,
                    value_cfg=value_cfg_dict,
                ),
                value_cfg=value_cfg_dict,
            ).to(device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("policy_state_dict")
    if state_dict is None:
        raise KeyError(f"'policy_state_dict' is missing in checkpoint: {checkpoint_path}")
    policy.load_state_dict(state_dict)
    policy.eval()

    if value_net is not None and ckpt.get("value_state_dict") is not None:
        value_net.load_state_dict(ckpt["value_state_dict"])
        value_net.eval()

    return {
        "algorithm": algorithm_name,
        "policy": policy,
        "available_actions": available_actions,
        "reward_class": reward_class,
        "reward_type": reward_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test/sampling entrypoint for trained models (single circuit).")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file (e.g. saved_models/run_0/last.pt).")
    parser.add_argument("--circuit", required=True, help="Path to a circuit file (e.g. .blif/.aig).")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Hydra config.yaml used for training. If omitted, inferred from checkpoint path.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of stochastic rollouts to run.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_steps from config.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu. Defaults to cuda if available.")
    parser.add_argument("--json-out", default=None, help="Optional path to save sampling results as JSON.")

    # TB-specific overrides (ignored for REINFORCE)
    parser.add_argument("--reward-alpha", type=float, default=None)
    parser.add_argument("--reward-eps", type=float, default=None)
    parser.add_argument("--reward-improvement-clip", type=float, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    circuit_path = Path(args.circuit).expanduser().resolve()
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit not found: {circuit_path}")

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config is not None
        else _default_config_from_checkpoint(checkpoint_path)
    )
    cfg = _load_cfg(config_path)

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    num_steps = int(args.num_steps if args.num_steps is not None else cfg["num_steps"])
    loaded = _load_policy(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        circuit_path=circuit_path,
        num_steps=num_steps,
        device=device,
    )
    algorithm_name = str(loaded["algorithm"])
    policy = loaded["policy"]
    available_actions = loaded["available_actions"]
    reward_type = str(loaded["reward_type"])
    reward_class = loaded["reward_class"]

    if algorithm_name == "gflownet_tb":
        tb_params = _tb_reward_params(
            cfg,
            reward_alpha=args.reward_alpha,
            reward_eps=args.reward_eps,
            reward_improvement_clip=args.reward_improvement_clip,
        )

        trajectories = []
        for _ in range(max(1, int(args.num_samples))):
            trajectories.append(
                sample_tb_trajectory(
                    file_path=str(circuit_path),
                    num_steps=num_steps,
                    policy=policy,
                    reward_class=reward_class,
                    sample_actions=True,
                    available_actions=available_actions,
                    **tb_params,
                )
            )

        sizes = [int(t.final_size) for t in trajectories]
        depths = [int(t.final_depth) for t in trajectories]
        qors = [int(t.final_size) * int(t.final_depth) for t in trajectories]
        returns = [float(t.final_return) for t in trajectories]
        comparable_returns = [float(t.comparable_return) for t in trajectories]
        initial_size = int(trajectories[0].initial_size) if trajectories else 0
        initial_depth = int(trajectories[0].initial_depth) if trajectories else 0

        result = {
            "algorithm": algorithm_name,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "circuit": str(circuit_path),
            "num_samples": len(trajectories),
            "num_steps": num_steps,
            "reward_type": reward_type,
            "available_actions": available_actions,
            "reward_alpha": tb_params["reward_alpha"],
            "reward_eps": tb_params["reward_eps"],
            "reward_improvement_clip": tb_params["reward_improvement_clip"],
            "initial_size": initial_size,
            "initial_depth": initial_depth,
            "final_sizes": sizes,
            "final_depths": depths,
            "final_returns": returns,
            "comparable_returns": comparable_returns,
            "mean_final_size": mean(sizes) if sizes else None,
            "mean_final_depth": mean(depths) if depths else None,
            "mean_final_qor": mean(qors) if qors else None,
            "mean_final_return": mean(returns) if returns else None,
            "mean_comparable_return": mean(comparable_returns) if comparable_returns else None,
        }
    elif algorithm_name == "reinforce":
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

        episodes = []
        for _ in range(max(1, int(args.num_samples))):
            episodes.append(
                run_reinforce_episode(
                    file_path=str(circuit_path),
                    num_steps=num_steps,
                    policy=policy,
                    reward_class=reward_class,
                    sample_actions=True,
                    resyn2_baseline=resyn2_baseline,
                    baseline=baseline,
                    available_actions=available_actions,
                )
            )

        sizes = [int(ep["final_size"]) for ep in episodes]
        depths = [int(ep["final_depth"]) for ep in episodes]
        qors = [int(ep["final_size"]) * int(ep["final_depth"]) for ep in episodes]
        returns = [float(ep["final_return"]) for ep in episodes]
        comparable_returns = [float(ep["comparable_return"]) for ep in episodes]
        initial_size = int(episodes[0]["initial_size"]) if episodes else 0
        initial_depth = int(episodes[0]["initial_depth"]) if episodes else 0

        result = {
            "algorithm": algorithm_name,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "circuit": str(circuit_path),
            "num_samples": len(episodes),
            "num_steps": num_steps,
            "reward_type": reward_type,
            "baseline": baseline,
            "available_actions": available_actions,
            "initial_size": initial_size,
            "initial_depth": initial_depth,
            "final_sizes": sizes,
            "final_depths": depths,
            "final_returns": returns,
            "comparable_returns": comparable_returns,
            "mean_final_size": mean(sizes) if sizes else None,
            "mean_final_depth": mean(depths) if depths else None,
            "mean_final_qor": mean(qors) if qors else None,
            "mean_final_return": mean(returns) if returns else None,
            "mean_comparable_return": mean(comparable_returns) if comparable_returns else None,
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    print(json.dumps(result, indent=2))
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved sampling results to: {out_path}")


if __name__ == "__main__":
    main()
