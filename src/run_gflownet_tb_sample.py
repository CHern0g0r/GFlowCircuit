from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from omegaconf import DictConfig, OmegaConf

from src.algorithms.gflownet_tb import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.sampler import sample_tb_trajectory
from src.models import REWARD_TYPES, encoder_factory, head_factory
from src.utils import get_obs_dim_and_num_actions, normalize_available_actions


def _build_tb_policy(cfg: DictConfig, obs_dim: int, node_dim: int, num_actions: int) -> TBGFlowNetPolicy:
    encoder_cfg = OmegaConf.to_container(cfg["encoder"], resolve=True)
    if not isinstance(encoder_cfg, dict):
        raise TypeError("encoder config must resolve to a mapping")
    if "in_dim" not in encoder_cfg:
        encoder_cfg["in_dim"] = node_dim if encoder_cfg.get("input_graph", False) else obs_dim
    if "out_dim" not in encoder_cfg:
        encoder_cfg["out_dim"] = obs_dim

    enc = encoder_factory(encoder_cfg=encoder_cfg)
    head = head_factory(
        obs_dim=encoder_cfg["out_dim"],
        num_actions=num_actions,
        head_cfg=cfg["head"],
    )
    return TBGFlowNetPolicy(encoder=enc, head=head, num_actions=num_actions)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a trained GFlowNet-TB model on a single circuit multiple times."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file (e.g. saved_models/run_0/last.pt).")
    parser.add_argument("--circuit", required=True, help="Path to a circuit file (e.g. .blif/.aig).")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Hydra config.yaml used for training. If omitted, inferred from checkpoint path.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of stochastic rollouts to run.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_steps from config.")
    parser.add_argument(
        "--reward-alpha",
        type=float,
        default=None,
        help="Override TB reward_alpha used by trajectory sampler.",
    )
    parser.add_argument(
        "--reward-eps",
        type=float,
        default=None,
        help="Override TB reward_eps used by trajectory sampler.",
    )
    parser.add_argument(
        "--reward-improvement-clip",
        type=float,
        default=None,
        help="Override TB reward_improvement_clip used by trajectory sampler.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu. Defaults to cuda if available.")
    parser.add_argument("--json-out", default=None, help="Optional path to save sampling results as JSON.")
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
    obs_dim, num_actions, node_dim, _ = get_obs_dim_and_num_actions(num_steps, str(circuit_path))
    available_actions = normalize_available_actions(OmegaConf.select(cfg, "available_actions"), num_actions)

    policy = _build_tb_policy(cfg, obs_dim=obs_dim, node_dim=node_dim, num_actions=num_actions).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("policy_state_dict")
    if state_dict is None:
        raise KeyError(f"'policy_state_dict' is missing in checkpoint: {checkpoint_path}")
    policy.load_state_dict(state_dict)
    policy.eval()

    reward_type = str(cfg["reward"]["type"])
    reward_class = REWARD_TYPES[reward_type]
    tb_cfg = OmegaConf.select(cfg, "tb")
    if tb_cfg is None:
        tb_cfg = OmegaConf.select(cfg, "algorithm.tb")

    reward_alpha = float(args.reward_alpha if args.reward_alpha is not None else (OmegaConf.select(tb_cfg, "reward_alpha") or 4.0))
    reward_eps = float(args.reward_eps if args.reward_eps is not None else (OmegaConf.select(tb_cfg, "reward_eps") or 1e-8))
    reward_improvement_clip = float(
        args.reward_improvement_clip
        if args.reward_improvement_clip is not None
        else (OmegaConf.select(tb_cfg, "reward_improvement_clip") or 2.0)
    )

    trajectories = []
    for _ in range(max(1, int(args.num_samples))):
        tr = sample_tb_trajectory(
            file_path=str(circuit_path),
            num_steps=num_steps,
            policy=policy,
            reward_class=reward_class,
            reward_alpha=reward_alpha,
            reward_eps=reward_eps,
            reward_improvement_clip=reward_improvement_clip,
            sample_actions=True,
            available_actions=available_actions,
        )
        trajectories.append(tr)

    sizes = [int(t.final_size) for t in trajectories]
    depths = [int(t.final_depth) for t in trajectories]
    returns = [float(t.final_return) for t in trajectories]
    initial_size = int(trajectories[0].initial_size) if trajectories else 0
    initial_depth = int(trajectories[0].initial_depth) if trajectories else 0

    result = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "circuit": str(circuit_path),
        "num_samples": len(trajectories),
        "num_steps": num_steps,
        "reward_type": reward_type,
        "available_actions": available_actions,
        "reward_alpha": reward_alpha,
        "reward_eps": reward_eps,
        "reward_improvement_clip": reward_improvement_clip,
        "initial_size": initial_size,
        "initial_depth": initial_depth,
        "final_sizes": sizes,
        "final_depths": depths,
        "final_returns": returns,
        "best_final_size": min(sizes) if sizes else None,
        "best_final_depth": min(depths) if depths else None,
        "mean_final_size": mean(sizes) if sizes else None,
        "mean_final_depth": mean(depths) if depths else None,
        "mean_final_return": mean(returns) if returns else None,
    }

    print(json.dumps(result, indent=2))
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved sampling results to: {out_path}")


if __name__ == "__main__":
    main()
