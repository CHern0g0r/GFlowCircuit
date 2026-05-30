
from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.algorithms.drills_a2c import DrillsA2CPolicy, DrillsA2CTrainer
from src.algorithms.gflownet_tb import TBGFlowNetPolicy, TBGFlowNetTrainer
from src.algorithms.reinforce import ReinforcePolicy, ReinforceTrainer
from src.baselines.resyn2 import build_resyn2_cache
from src.utils import (
    get_obs_dim_and_num_actions,
    load_circuits,
    normalize_available_actions,
    train_test_split,
)
from src.models import (
    reward_class_factory,
    encoder_factory,
    head_factory,
    prepare_encoder_config,
    value_input_dim,
    value_factory,
)


def _save_run_checkpoint(
    *,
    output_dir: Path,
    run_idx: int,
    seed: int,
    policy: torch.nn.Module,
    value_net: torch.nn.Module | None,
) -> Path:
    run_dir = output_dir / "saved_models" / f"run_{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    payload: dict[str, object] = {
        "run_idx": int(run_idx),
        "seed": int(seed),
        "policy_state_dict": policy.state_dict(),
    }
    if value_net is not None:
        payload["value_state_dict"] = value_net.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


def _build_models(
    cfg: DictConfig,
    obs_dim: int,
    node_dim: int,
    edge_dim: int,
    num_actions: int,
    available_actions: list[int] | None,
):
    encoder_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
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
        head_cfg=cfg.head,
    )
    policy = ReinforcePolicy(encoder=enc, head=head, num_actions=num_actions)
    value_cfg = OmegaConf.select(cfg, "value")
    value_net = None
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
        )
    return policy, value_net


def _build_drills_a2c_models(
    cfg: DictConfig,
    obs_dim: int,
    node_dim: int,
    edge_dim: int,
    num_actions: int,
    available_actions: list[int] | None,
):
    encoder_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
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
        head_cfg=cfg.head,
    )
    policy = DrillsA2CPolicy(encoder=enc, head=head, num_actions=num_actions)

    value_cfg = OmegaConf.select(cfg, "value")
    if value_cfg is None:
        raise ValueError("drills_a2c requires a value network config")
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
    )
    return policy, value_net


def _build_tb_policy(
    cfg: DictConfig,
    obs_dim: int,
    node_dim: int,
    edge_dim: int,
    num_actions: int,
    available_actions: list[int] | None,
) -> TBGFlowNetPolicy:
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
    return TBGFlowNetPolicy(encoder=enc, head=head, num_actions=num_actions)


def _get_algorithm_name(cfg: DictConfig) -> str:
    algo_cfg = OmegaConf.select(cfg, "algorithm")
    if algo_cfg is None:
        return "reinforce"
    name = OmegaConf.select(algo_cfg, "name")
    if name is None:
        # Backwards-compat: some configs might set algorithm: gflownet_tb (string)
        if isinstance(algo_cfg, str):
            return str(algo_cfg)
        return "reinforce"
    return str(name)


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
    available_actions = normalize_available_actions(OmegaConf.select(cfg, "available_actions"), num_actions)
    available_actions_msg = "all" if available_actions is None else str(available_actions)
    print(
        f"Obs dim: {obs_dim}, num actions: {num_actions}, node dim: {node_dim}, "
        f"edge dim: {edge_dim}, available actions: {available_actions_msg}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_class = reward_class_factory(cfg.reward)
    algorithm_name = _get_algorithm_name(cfg)

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
        run_seed = int(cfg.seed + run_idx)
        if algorithm_name == "gflownet_tb":
            policy = _build_tb_policy(
                cfg,
                obs_dim=obs_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_actions=num_actions,
                available_actions=available_actions,
            ).to(device)
            trainer = TBGFlowNetTrainer(
                policy=policy,
                reward_class=reward_class,
                train_circuits=train_circuits,
                test_circuits=test_circuits,
                resyn2_baselines=resyn2_baselines,
                device=device,
                seed=run_seed,
                log_dir=(tb_log_dir / f"run_{run_idx}") if tb_log_dir is not None else None,
                available_actions=available_actions,
            )

            tb_cfg = OmegaConf.select(cfg, "tb")
            if tb_cfg is None:
                tb_cfg = OmegaConf.select(cfg, "algorithm.tb")
            tb_trajectories_per_episode = int(OmegaConf.select(tb_cfg, "trajectories_per_episode") or 4)
            tb_reward_alpha = float(OmegaConf.select(tb_cfg, "reward_alpha") or 4.0)
            tb_reward_eps = float(OmegaConf.select(tb_cfg, "reward_eps") or 1e-8)
            tb_reward_improvement_clip = float(OmegaConf.select(tb_cfg, "reward_improvement_clip") or 2.0)

            train_out = trainer.train(
                episodes=int(cfg.episodes),
                num_steps=int(cfg.num_steps),
                eval_every=int(cfg.eval_every),
                learning_rate=float(cfg.learning_rate),
                trajectories_per_episode=tb_trajectories_per_episode,
                reward_alpha=tb_reward_alpha,
                reward_eps=tb_reward_eps,
                reward_improvement_clip=tb_reward_improvement_clip,
                best_of_eval_rollouts=best_of_rollouts,
            )
            final_eval = trainer.evaluate(
                num_steps=int(cfg.num_steps),
                reward_alpha=tb_reward_alpha,
                reward_eps=tb_reward_eps,
                reward_improvement_clip=tb_reward_improvement_clip,
                best_of_rollouts=best_of_rollouts,
            )

            ckpt_path = _save_run_checkpoint(
                output_dir=output_dir,
                run_idx=run_idx,
                seed=run_seed,
                policy=policy,
                value_net=None,
            )
            runs.append(
                {
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "history": train_out["history"],
                    "final_eval": final_eval,
                    "checkpoint_path": str(ckpt_path),
                }
            )
        elif algorithm_name == "drills_a2c":
            policy, value_net = _build_drills_a2c_models(
                cfg,
                obs_dim=obs_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_actions=num_actions,
                available_actions=available_actions,
            )
            policy = policy.to(device)
            value_net = value_net.to(device)

            trainer = DrillsA2CTrainer(
                policy=policy,
                value_network=value_net,
                reward_class=reward_class,
                train_circuits=train_circuits,
                test_circuits=test_circuits,
                resyn2_baselines=resyn2_baselines,
                device=device,
                seed=run_seed,
                log_dir=(tb_log_dir / f"run_{run_idx}") if tb_log_dir is not None else None,
                available_actions=available_actions,
            )

            drills_cfg = OmegaConf.select(cfg, "algorithm.drills")
            if drills_cfg is None:
                drills_cfg = OmegaConf.select(cfg, "drills")
            if drills_cfg is None:
                drills_cfg = OmegaConf.create({})
            trajectories_per_episode = int(OmegaConf.select(drills_cfg, "trajectories_per_episode") or 1)
            value_loss_coef = float(OmegaConf.select(drills_cfg, "value_loss_coef") or 0.5)
            drills_entropy_beta = float(OmegaConf.select(drills_cfg, "entropy_beta") or 0.0)
            drills_clip_grad_norm = OmegaConf.select(drills_cfg, "clip_grad_norm")
            normalize_advantages = bool(OmegaConf.select(drills_cfg, "normalize_advantages") or False)

            train_out = trainer.train(
                episodes=int(cfg.episodes),
                num_steps=int(cfg.num_steps),
                eval_every=int(cfg.eval_every),
                learning_rate=float(cfg.learning_rate),
                gamma=float(cfg.gamma),
                trajectories_per_episode=trajectories_per_episode,
                value_loss_coef=value_loss_coef,
                entropy_beta=drills_entropy_beta,
                clip_grad_norm=float(drills_clip_grad_norm) if drills_clip_grad_norm is not None else None,
                normalize_advantages=normalize_advantages,
                best_of_eval_rollouts=best_of_rollouts,
            )
            final_eval = trainer.evaluate(num_steps=int(cfg.num_steps), best_of_rollouts=best_of_rollouts)
            ckpt_path = _save_run_checkpoint(
                output_dir=output_dir,
                run_idx=run_idx,
                seed=run_seed,
                policy=policy,
                value_net=value_net,
            )
            runs.append(
                {
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "history": train_out["history"],
                    "final_eval": final_eval,
                    "checkpoint_path": str(ckpt_path),
                }
            )
        elif algorithm_name == "reinforce":
            policy, value_net = _build_models(
                cfg,
                obs_dim=obs_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_actions=num_actions,
                available_actions=available_actions,
            )
            policy = policy.to(device)
            if value_net is not None:
                value_net = value_net.to(device)

            trainer = ReinforceTrainer(
                train_circuits=train_circuits,
                test_circuits=test_circuits,
                policy=policy,
                value_network=value_net,
                reward_class=reward_class,
                terminal_reward=bool(cfg.terminal_reward),
                baseline=baseline,
                resyn2_baselines=resyn2_baselines,
                device=device,
                seed=run_seed,
                log_dir=(tb_log_dir / f"run_{run_idx}") if tb_log_dir is not None else None,
                available_actions=available_actions,
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
            ckpt_path = _save_run_checkpoint(
                output_dir=output_dir,
                run_idx=run_idx,
                seed=run_seed,
                policy=policy,
                value_net=value_net,
            )
            runs.append(
                {
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "history": train_out["history"],
                    "final_eval": final_eval,
                    "checkpoint_path": str(ckpt_path),
                }
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        print(f"Saved checkpoint: {runs[-1]['checkpoint_path']}")

    report = {
        "algorithm": algorithm_name,
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        "dataset_cfg": str(dataset_cfg),
        "seed": int(cfg.seed),
        "num_steps": int(cfg.num_steps),
        "episodes": int(cfg.episodes),
        "train_ratio": float(cfg.train_ratio),
        "baseline": baseline,
        "available_actions": available_actions,
        "train_circuits": train_circuits,
        "test_circuits": test_circuits,
        "runs": runs,
        "mean_final_return_across_runs": float(
            sum(r["final_eval"]["mean_final_return"] for r in runs) / max(1, len(runs))
        ),
    }

    print("\nFinal test evaluation:")
    print(json.dumps(runs[-1]["final_eval"] if runs else {}, indent=2))

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
