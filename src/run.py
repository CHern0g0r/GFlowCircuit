
from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    train_test_split,
    load_circuits,
)
from src.models import (
    policy_factory,
    REWARD_TYPES,
    encoder_factory,
    head_factory,
    value_factory,
)
from src.train.utils import get_obs_dim_and_num_actions, build_resyn2_cache
from src.train import Trainer


def _build_models(cfg: DictConfig, obs_dim: int, node_dim: int, num_actions: int):
    encoder_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
    if not isinstance(encoder_cfg, dict):
        raise TypeError("encoder config must resolve to a mapping")
    if "in_dim" not in encoder_cfg:
        if encoder_cfg.get("input_graph", False):
            encoder_cfg["in_dim"] = node_dim
        else:
            encoder_cfg["in_dim"] = obs_dim
    if "out_dim" not in encoder_cfg:
        encoder_cfg["out_dim"] = obs_dim

    enc = encoder_factory(encoder_cfg=encoder_cfg)
    head = head_factory(
        obs_dim=encoder_cfg["out_dim"],
        num_actions=num_actions,
        head_cfg=cfg.head,
    )
    policy = policy_factory(
        encoder=enc,
        head=head,
        num_actions=num_actions,
        policy_cfg=cfg.policy,
    )
    value_cfg = OmegaConf.select(cfg, "value")
    value_net = None
    if value_cfg is not None:
        value_net = value_factory(obs_dim=obs_dim, value_cfg=value_cfg)
    return policy, value_net


@hydra.main(version_base=None, config_path="../cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    dataset_cfg = Path(to_absolute_path(cfg.dataset_cfg))
    output_root = Path(to_absolute_path(cfg.output_dir))
    output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    circuits = load_circuits(dataset_cfg)
    print(f"Loaded {len(circuits)} circuits")
    if bool(OmegaConf.select(cfg, "paper_mode.per_circuit_mode")):
        train_circuits, test_circuits = circuits, circuits
    else:
        train_circuits, test_circuits = train_test_split(circuits, cfg.train_ratio, cfg.seed)
        print(f"Split into {len(train_circuits)} train and {len(test_circuits)} test circuits")

    obs_dim, num_actions, node_dim, edge_dim = get_obs_dim_and_num_actions(cfg.num_steps, train_circuits[0])
    print(f"Obs dim: {obs_dim}, num actions: {num_actions}, node dim: {node_dim}, edge dim: {edge_dim}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_class = REWARD_TYPES[cfg.reward.type]

    use_tb = OmegaConf.select(cfg, "logging.tensorboard")
    if use_tb is None:
        use_tb = True
    if use_tb:
        tb_log_dir = output_dir / "tensorboard"
    else:
        tb_log_dir = None

    baseline = OmegaConf.select(cfg, "baseline")
    baseline_scale = float(OmegaConf.select(cfg, "baseline_scale") or 1.0)

    num_runs = int(OmegaConf.select(cfg, "paper_mode.num_runs") or 1)
    best_of_rollouts = int(OmegaConf.select(cfg, "paper_mode.infer_rollouts") or 1)
    policy_lr = float(OmegaConf.select(cfg, "policy_learning_rate") or cfg.learning_rate)
    value_lr = float(OmegaConf.select(cfg, "value_learning_rate") or cfg.learning_rate)
    entropy_beta = float(OmegaConf.select(cfg, "entropy_beta") or 0.0)
    clip_grad_norm_policy = OmegaConf.select(cfg, "clip_grad_norm_policy")
    clip_grad_norm_value = OmegaConf.select(cfg, "clip_grad_norm_value")
    normalize_returns = bool(OmegaConf.select(cfg, "normalize_returns") or False)

    resyn2_baselines = build_resyn2_cache(
        circuits=circuits,
        num_steps=int(cfg.num_steps),
        reward_class=reward_class,
        baseline=baseline,
        baseline_scale=baseline_scale,
    )

    runs: list[dict] = []
    for run_idx in range(num_runs):
        print("Starting run", run_idx)
        policy, value_net = _build_models(cfg, obs_dim=obs_dim, node_dim=node_dim, num_actions=num_actions)
        trainer = Trainer(
            train_circuits=train_circuits,
            test_circuits=test_circuits,
            policy=policy,
            value_network=value_net,
            reward_class=reward_class,
            terminal_reward=bool(cfg.terminal_reward),
            baseline=baseline,
            resyn2_baselines=resyn2_baselines,
            device=device,
            seed=int(cfg.seed + run_idx),
            log_dir=(tb_log_dir / f"run_{run_idx}") if tb_log_dir is not None else None,
        )

        train_out = trainer.train(
            num_steps=int(cfg.num_steps),
            episodes=int(cfg.episodes),
            eval_every=int(cfg.eval_every),
            policy_learning_rate=policy_lr,
            value_learning_rate=value_lr,
            gamma=float(cfg.gamma),
            baseline_alpha=float(cfg.baseline_alpha),
            best_of_eval_rollouts=best_of_rollouts,
            entropy_beta=entropy_beta,
            clip_grad_norm_policy=float(clip_grad_norm_policy) if clip_grad_norm_policy is not None else None,
            clip_grad_norm_value=float(clip_grad_norm_value) if clip_grad_norm_value is not None else None,
            normalize_returns=normalize_returns,
        )

        final_eval = trainer.evaluate(num_steps=int(cfg.num_steps), best_of_rollouts=best_of_rollouts)
        runs.append(
            {
                "run_idx": run_idx,
                "seed": int(cfg.seed + run_idx),
                "history": train_out["history"],
                "final_eval": final_eval,
            }
        )

    report = {
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        "dataset_cfg": str(dataset_cfg),
        "seed": int(cfg.seed),
        "num_steps": int(cfg.num_steps),
        "episodes": int(cfg.episodes),
        "train_ratio": float(cfg.train_ratio),
        "baseline": baseline,
        "train_circuits": train_circuits,
        "test_circuits": test_circuits,
        "runs": runs,
        "mean_final_return_across_runs": float(
            sum(r["final_eval"]["mean_final_return"] for r in runs) / max(1, len(runs))
        ),
    }

    print("\nFinal test evaluation:")
    print(json.dumps(runs[-1]["final_eval"], indent=2))

    json_out = OmegaConf.select(cfg, "json_out")
    if json_out not in (None, "", "~"):
        json_path = Path(to_absolute_path(str(json_out)))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {json_path}")

    if tb_log_dir is not None:
        print(f"\nTensorBoard logs: {tb_log_dir}")
        print(f"View with: tensorboard --logdir {tb_log_dir}")


if __name__ == "__main__":
    main()
