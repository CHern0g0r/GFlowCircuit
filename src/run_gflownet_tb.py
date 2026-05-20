from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.algorithms.gflownet_tb import TBGFlowNetPolicy, TBGFlowNetTrainer
from src.baselines.resyn2 import build_resyn2_cache
from src.models import REWARD_TYPES, encoder_factory, head_factory
from src.utils import get_obs_dim_and_num_actions, load_circuits, normalize_available_actions, train_test_split


def _save_tb_run_checkpoint(
    *,
    output_dir: Path,
    run_idx: int,
    seed: int,
    policy: TBGFlowNetPolicy,
) -> Path:
    run_dir = output_dir / "saved_models" / f"run_{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    torch.save(
        {
            "run_idx": int(run_idx),
            "seed": int(seed),
            "policy_state_dict": policy.state_dict(),
            "log_z": float(policy.log_z.detach().cpu().item()),
        },
        ckpt_path,
    )
    return ckpt_path


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


@hydra.main(version_base=None, config_path="../cfg", config_name="gflownet_tb_mcnc")
def main(cfg: DictConfig) -> None:
    dataset_cfg = Path(to_absolute_path(cfg["dataset_cfg"]))
    output_root = Path(to_absolute_path(cfg["output_dir"]))
    output_root.mkdir(parents=True, exist_ok=True)

    circuits = load_circuits(dataset_cfg)
    if bool(cfg["paper_mode"]["per_circuit_mode"]):
        train_circuits, test_circuits = circuits, circuits
    else:
        train_circuits, test_circuits = train_test_split(circuits, cfg["train_ratio"], cfg["seed"])

    obs_dim, num_actions, node_dim, _ = get_obs_dim_and_num_actions(cfg["num_steps"], train_circuits[0])
    available_actions = normalize_available_actions(OmegaConf.select(cfg, "available_actions"), num_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_class = REWARD_TYPES[cfg["reward"]["type"]]
    tb_enabled = bool(cfg["logging"]["tensorboard"])
    tb_root = output_root / "tensorboard" if tb_enabled else None
    resyn2_baselines = build_resyn2_cache(
        circuits=circuits,
        num_steps=int(cfg["num_steps"]),
        reward_class=reward_class,
    )

    tb_cfg = OmegaConf.select(cfg, "tb")
    if tb_cfg is None:
        tb_cfg = OmegaConf.select(cfg, "algorithm.tb")
    tb_trajectories_per_episode = int(OmegaConf.select(tb_cfg, "trajectories_per_episode") or 4)
    tb_reward_alpha = float(OmegaConf.select(tb_cfg, "reward_alpha") or 4.0)
    tb_reward_eps = float(OmegaConf.select(tb_cfg, "reward_eps") or 1e-8)
    tb_reward_improvement_clip = float(OmegaConf.select(tb_cfg, "reward_improvement_clip") or 2.0)

    num_runs = int(OmegaConf.select(cfg, "paper_mode.num_runs") or 1)
    base_seed = int(cfg["seed"])
    runs: list[dict] = []
    for run_idx in range(num_runs):
        print("Starting run", run_idx)
        run_seed = base_seed + run_idx
        policy = _build_tb_policy(cfg, obs_dim=obs_dim, node_dim=node_dim, num_actions=num_actions).to(device)
        trainer = TBGFlowNetTrainer(
            policy=policy,
            reward_class=reward_class,
            train_circuits=train_circuits,
            test_circuits=test_circuits,
            resyn2_baselines=resyn2_baselines,
            device=device,
            seed=run_seed,
            log_dir=(tb_root / f"run_{run_idx}") if tb_root is not None else None,
            available_actions=available_actions,
        )
        train_out = trainer.train(
            episodes=int(cfg["episodes"]),
            num_steps=int(cfg["num_steps"]),
            eval_every=int(cfg["eval_every"]),
            learning_rate=float(cfg["learning_rate"]),
            trajectories_per_episode=tb_trajectories_per_episode,
            reward_alpha=tb_reward_alpha,
            reward_eps=tb_reward_eps,
            reward_improvement_clip=tb_reward_improvement_clip,
            best_of_eval_rollouts=int(cfg["paper_mode"]["infer_rollouts"]),
        )
        final_eval = trainer.evaluate(
            num_steps=int(cfg["num_steps"]),
            reward_alpha=tb_reward_alpha,
            reward_eps=tb_reward_eps,
            reward_improvement_clip=tb_reward_improvement_clip,
            best_of_rollouts=int(cfg["paper_mode"]["infer_rollouts"]),
        )
        ckpt_path = _save_tb_run_checkpoint(
            output_dir=output_root,
            run_idx=run_idx,
            seed=run_seed,
            policy=policy,
        )
        print(f"Saved checkpoint: {ckpt_path}")
        runs.append(
            {
                "run_idx": run_idx,
                "seed": run_seed,
                "history": train_out["history"],
                "final_eval": final_eval,
                "checkpoint_path": str(ckpt_path),
            }
        )

    report = {
        "algorithm": "gflownet_tb",
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        "dataset_cfg": str(dataset_cfg),
        "seed": int(cfg["seed"]),
        "num_steps": int(cfg["num_steps"]),
        "episodes": int(cfg["episodes"]),
        "train_ratio": float(cfg["train_ratio"]),
        "available_actions": available_actions,
        "tb": {
            "trajectories_per_episode": tb_trajectories_per_episode,
            "reward_alpha": tb_reward_alpha,
            "reward_eps": tb_reward_eps,
            "reward_improvement_clip": tb_reward_improvement_clip,
        },
        "train_circuits": train_circuits,
        "test_circuits": test_circuits,
        "runs": runs,
        "history": runs[-1]["history"] if runs else [],
        "final_eval": runs[-1]["final_eval"] if runs else {},
        "mean_final_return_across_runs": float(
            sum(r["final_eval"]["mean_final_return"] for r in runs) / max(1, len(runs))
        ),
    }

    print("\nFinal test evaluation:")
    print(json.dumps(runs[-1]["final_eval"] if runs else {}, indent=2))
    json_out = cfg["json_out"]
    if json_out not in (None, "", "~"):
        json_path = Path(to_absolute_path(str(json_out)))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {json_path}")


if __name__ == "__main__":
    main()
