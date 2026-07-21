
from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.algorithms.drills_a2c import DrillsA2CPolicy, DrillsA2CTrainer
from src.algorithms.gflownet_tb import TBGFlowNetPolicy, TBGFlowNetTrainer, build_tb_policy
from src.algorithms.ppo import PPOPolicy, PPOTrainer
from src.algorithms.pcn import PCNPolicy, PCNTrainer
from src.algorithms.reinforce import ReinforcePolicy, ReinforceTrainer
from src.baselines.resyn2 import build_resyn2_cache
from src.discovery_metrics import write_discovery_artifacts
from src.utils import (
    get_obs_dim_and_num_actions,
    load_circuits,
    normalize_available_actions,
    train_test_split,
)
from src.models import (
    reward_class_factory,
    mo_reward_factory,
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
    extra_payload: dict[str, object] | None = None,
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
    if extra_payload is not None:
        payload.update(extra_payload)
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


def _build_ppo_models(
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
    policy = PPOPolicy(encoder=enc, head=head, num_actions=num_actions)

    value_cfg = OmegaConf.select(cfg, "value")
    if value_cfg is None:
        raise ValueError("ppo requires a value network config")
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


# Backwards compatibility for code that imported the former private builder.
_build_tb_policy = build_tb_policy


def _build_pcn_policy(
    cfg: DictConfig,
    obs_dim: int,
    node_dim: int,
    edge_dim: int,
    num_actions: int,
    available_actions: list[int] | None,
) -> PCNPolicy:
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

    pcn_cfg = OmegaConf.select(cfg, "pcn")
    if pcn_cfg is None:
        pcn_cfg = OmegaConf.select(cfg, "algorithm.pcn")
    if pcn_cfg is None:
        pcn_cfg = OmegaConf.create({})
    embedding_dim = int(OmegaConf.select(pcn_cfg, "embedding_dim") or 64)
    objective_dim = int(OmegaConf.select(cfg, "mo_reward.objective_dim") or 2)

    enc = encoder_factory(encoder_cfg=encoder_cfg)
    head = head_factory(
        obs_dim=embedding_dim,
        num_actions=num_actions,
        head_cfg=cfg["head"],
    )
    return PCNPolicy(
        encoder=enc,
        head=head,
        num_actions=num_actions,
        encoder_out_dim=int(encoder_cfg["out_dim"]),
        objective_dim=objective_dim,
        num_steps=int(cfg.num_steps),
        embedding_dim=embedding_dim,
        condition_scale=float(OmegaConf.select(pcn_cfg, "condition_scale") or 1.0),
    )


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
    mo_reward_cfg = OmegaConf.select(cfg, "mo_reward")
    if mo_reward_cfg is None:
        mo_reward_cfg = OmegaConf.create({"type": "size_depth_improvement", "normalize": True, "objectives": ["size", "depth"]})
    mo_reward_cfg_dict = OmegaConf.to_container(mo_reward_cfg, resolve=True)
    if not isinstance(mo_reward_cfg_dict, dict):
        raise TypeError("mo_reward config must resolve to a mapping")
    mo_reward_class = mo_reward_factory(mo_reward_cfg_dict)
    algorithm_name = _get_algorithm_name(cfg)

    use_tb = OmegaConf.select(cfg, "logging.tensorboard")
    if use_tb is None:
        use_tb = True
    if use_tb:
        tb_log_dir = output_dir / "tensorboard"
    else:
        tb_log_dir = None

    discovery_cfg = OmegaConf.select(cfg, "discovery_metrics")
    discovery_enabled_raw = OmegaConf.select(discovery_cfg, "enabled") if discovery_cfg is not None else None
    discovery_enabled = True if discovery_enabled_raw is None else bool(discovery_enabled_raw)
    discovery_emit_raw = (
        OmegaConf.select(discovery_cfg, "emit_every_trajectories") if discovery_cfg is not None else None
    )
    discovery_emit_every = 50 if discovery_emit_raw is None else int(discovery_emit_raw)

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
            tb_batch_size = OmegaConf.select(tb_cfg, "batch_size")
            if tb_batch_size is None:
                tb_batch_size = OmegaConf.select(tb_cfg, "trajectories_per_episode")
            if tb_batch_size is None:
                tb_batch_size = 4
            tb_trajectories_per_episode = int(tb_batch_size)
            tb_reward_alpha_cfg = OmegaConf.select(tb_cfg, "reward_alpha")
            tb_reward_eps_cfg = OmegaConf.select(tb_cfg, "reward_eps")
            tb_reward_improvement_clip_cfg = OmegaConf.select(tb_cfg, "reward_improvement_clip")
            tb_reward_alpha = float(4.0 if tb_reward_alpha_cfg is None else tb_reward_alpha_cfg)
            tb_reward_eps = float(1e-8 if tb_reward_eps_cfg is None else tb_reward_eps_cfg)
            tb_reward_improvement_clip = float(
                2.0 if tb_reward_improvement_clip_cfg is None else tb_reward_improvement_clip_cfg
            )
            tb_log_z_learning_rate_cfg = OmegaConf.select(tb_cfg, "log_z_learning_rate")
            if tb_log_z_learning_rate_cfg is None:
                tb_log_z_learning_rate = 10.0 * float(cfg.learning_rate)
            else:
                tb_log_z_learning_rate = float(tb_log_z_learning_rate_cfg)
            tb_exploration_epsilon_enabled_cfg = OmegaConf.select(tb_cfg, "exploration_epsilon_enabled")
            tb_exploration_epsilon_enabled = (
                True if tb_exploration_epsilon_enabled_cfg is None else bool(tb_exploration_epsilon_enabled_cfg)
            )
            tb_exploration_epsilon_start_cfg = OmegaConf.select(tb_cfg, "exploration_epsilon_start")
            tb_exploration_epsilon_end_cfg = OmegaConf.select(tb_cfg, "exploration_epsilon_end")
            tb_exploration_warmup_episodes_cfg = OmegaConf.select(tb_cfg, "exploration_warmup_episodes")
            tb_exploration_decay_episodes_cfg = OmegaConf.select(tb_cfg, "exploration_decay_episodes")
            tb_exploration_epsilon_start = float(
                0.5 if tb_exploration_epsilon_start_cfg is None else tb_exploration_epsilon_start_cfg
            )
            tb_exploration_epsilon_end = float(
                0.01 if tb_exploration_epsilon_end_cfg is None else tb_exploration_epsilon_end_cfg
            )
            tb_exploration_warmup_episodes = int(
                20 if tb_exploration_warmup_episodes_cfg is None else tb_exploration_warmup_episodes_cfg
            )
            tb_exploration_decay_episodes = (
                None
                if tb_exploration_decay_episodes_cfg is None
                else int(tb_exploration_decay_episodes_cfg)
            )

            train_out = trainer.train(
                episodes=int(cfg.episodes),
                num_steps=int(cfg.num_steps),
                eval_every=int(cfg.eval_every),
                learning_rate=float(cfg.learning_rate),
                log_z_learning_rate=tb_log_z_learning_rate,
                trajectories_per_episode=tb_trajectories_per_episode,
                reward_alpha=tb_reward_alpha,
                reward_eps=tb_reward_eps,
                reward_improvement_clip=tb_reward_improvement_clip,
                exploration_epsilon_enabled=tb_exploration_epsilon_enabled,
                exploration_epsilon_start=tb_exploration_epsilon_start,
                exploration_epsilon_end=tb_exploration_epsilon_end,
                exploration_warmup_episodes=tb_exploration_warmup_episodes,
                exploration_decay_episodes=tb_exploration_decay_episodes,
                best_of_eval_rollouts=best_of_rollouts,
                discovery_metrics_enabled=discovery_enabled,
                discovery_emit_every_trajectories=discovery_emit_every,
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
                    "discovery_front": train_out["discovery_front"],
                    "discovery_metrics": train_out["discovery_metrics"],
                    "final_eval": final_eval,
                    "checkpoint_path": str(ckpt_path),
                }
            )
        elif algorithm_name == "pcn":
            policy = _build_pcn_policy(
                cfg,
                obs_dim=obs_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_actions=num_actions,
                available_actions=available_actions,
            ).to(device)
            pcn_cfg = OmegaConf.select(cfg, "pcn")
            if pcn_cfg is None:
                pcn_cfg = OmegaConf.select(cfg, "algorithm.pcn")
            if pcn_cfg is None:
                pcn_cfg = OmegaConf.create({})

            archive_capacity = int(OmegaConf.select(pcn_cfg, "archive_capacity") or 256)
            random_seed_episodes = int(OmegaConf.select(pcn_cfg, "random_seed_episodes") or 32)
            collect_episodes_per_iter = int(OmegaConf.select(pcn_cfg, "collect_episodes_per_iter") or 8)
            train_updates_per_iter = int(OmegaConf.select(pcn_cfg, "train_updates_per_iter") or 64)
            batch_size = int(OmegaConf.select(pcn_cfg, "batch_size") or 128)
            crowding_threshold = float(OmegaConf.select(pcn_cfg, "crowding_threshold") or 0.2)
            duplicate_penalty = float(OmegaConf.select(pcn_cfg, "duplicate_penalty") or 1e-5)
            target_noise_scale = float(OmegaConf.select(pcn_cfg, "target_noise_scale") or 0.0)
            target_min_sigma = float(OmegaConf.select(pcn_cfg, "target_min_sigma") or 0.0)
            desired_return_clip = bool(OmegaConf.select(pcn_cfg, "desired_return_clip") or False)
            eval_target_limit_raw = OmegaConf.select(pcn_cfg, "eval_target_limit")
            eval_target_limit = int(eval_target_limit_raw) if eval_target_limit_raw is not None else None

            trainer = PCNTrainer(
                policy=policy,
                mo_reward_class=mo_reward_class,
                train_circuits=train_circuits,
                test_circuits=test_circuits,
                resyn2_baselines=resyn2_baselines,
                device=device,
                seed=run_seed,
                log_dir=(tb_log_dir / f"run_{run_idx}") if tb_log_dir is not None else None,
                available_actions=available_actions,
                archive_capacity=archive_capacity,
                gamma=float(cfg.gamma),
                crowding_threshold=crowding_threshold,
                duplicate_penalty=duplicate_penalty,
                target_noise_scale=target_noise_scale,
                target_min_sigma=target_min_sigma,
            )
            train_out = trainer.train(
                episodes=int(cfg.episodes),
                num_steps=int(cfg.num_steps),
                eval_every=int(cfg.eval_every),
                learning_rate=float(cfg.learning_rate),
                random_seed_episodes=random_seed_episodes,
                collect_episodes_per_iter=collect_episodes_per_iter,
                train_updates_per_iter=train_updates_per_iter,
                batch_size=batch_size,
                desired_return_clip=desired_return_clip,
                eval_target_limit=eval_target_limit,
                discovery_metrics_enabled=discovery_enabled,
                discovery_emit_every_trajectories=discovery_emit_every,
            )
            final_eval = trainer.evaluate(
                num_steps=int(cfg.num_steps),
                eval_target_limit=eval_target_limit,
                desired_return_clip=desired_return_clip,
            )
            ckpt_path = _save_run_checkpoint(
                output_dir=output_dir,
                run_idx=run_idx,
                seed=run_seed,
                policy=policy,
                value_net=None,
                extra_payload={"pcn": trainer.checkpoint_metadata(limit=eval_target_limit)},
            )
            runs.append(
                {
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "history": train_out["history"],
                    "discovery_front": train_out["discovery_front"],
                    "discovery_metrics": train_out["discovery_metrics"],
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
                discovery_metrics_enabled=discovery_enabled,
                discovery_emit_every_trajectories=discovery_emit_every,
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
                    "discovery_front": train_out["discovery_front"],
                    "discovery_metrics": train_out["discovery_metrics"],
                    "final_eval": final_eval,
                    "checkpoint_path": str(ckpt_path),
                }
            )
        elif algorithm_name == "ppo":
            policy, value_net = _build_ppo_models(
                cfg,
                obs_dim=obs_dim,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_actions=num_actions,
                available_actions=available_actions,
            )
            policy = policy.to(device)
            value_net = value_net.to(device)

            trainer = PPOTrainer(
                policy=policy,
                value_network=value_net,
                reward_class=reward_class,
                train_circuits=train_circuits,
                test_circuits=test_circuits,
                resyn2_baselines=resyn2_baselines,
                device=device,
                seed=run_seed,
                log_dir=(tb_log_dir / f"run_{run_idx}") if tb_log_dir is not None else None,
                baseline=baseline,
                available_actions=available_actions,
            )

            ppo_cfg = OmegaConf.select(cfg, "algorithm.ppo")
            if ppo_cfg is None:
                ppo_cfg = OmegaConf.select(cfg, "ppo")
            if ppo_cfg is None:
                ppo_cfg = OmegaConf.create({})
            rollout_steps = int(OmegaConf.select(ppo_cfg, "rollout_steps") or 200)
            ppo_epochs = int(OmegaConf.select(ppo_cfg, "ppo_epochs") or 20)
            minibatch_size = int(OmegaConf.select(ppo_cfg, "minibatch_size") or 64)
            clip_eps = float(OmegaConf.select(ppo_cfg, "clip_eps") or 0.2)
            value_loss_coef = float(OmegaConf.select(ppo_cfg, "value_loss_coef") or 0.5)
            ppo_entropy_beta = float(OmegaConf.select(ppo_cfg, "entropy_beta") or 0.0)
            normalize_advantages = bool(OmegaConf.select(ppo_cfg, "normalize_advantages") or False)
            ppo_clip_grad_norm = OmegaConf.select(ppo_cfg, "clip_grad_norm")
            gae_lambda_cfg = OmegaConf.select(ppo_cfg, "gae_lambda")
            gae_lambda = float(gae_lambda_cfg) if gae_lambda_cfg is not None else 0.95

            train_out = trainer.train(
                episodes=int(cfg.episodes),
                num_steps=int(cfg.num_steps),
                eval_every=int(cfg.eval_every),
                learning_rate=float(cfg.learning_rate),
                gamma=float(cfg.gamma),
                rollout_steps=rollout_steps,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                clip_eps=clip_eps,
                value_loss_coef=value_loss_coef,
                entropy_beta=ppo_entropy_beta,
                normalize_advantages=normalize_advantages,
                clip_grad_norm=float(ppo_clip_grad_norm) if ppo_clip_grad_norm is not None else None,
                gae_lambda=gae_lambda,
                best_of_eval_rollouts=best_of_rollouts,
                discovery_metrics_enabled=discovery_enabled,
                discovery_emit_every_trajectories=discovery_emit_every,
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
                    "discovery_front": train_out["discovery_front"],
                    "discovery_metrics": train_out["discovery_metrics"],
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
                discovery_metrics_enabled=discovery_enabled,
                discovery_emit_every_trajectories=discovery_emit_every,
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
                    "discovery_front": train_out["discovery_front"],
                    "discovery_metrics": train_out["discovery_metrics"],
                    "final_eval": final_eval,
                    "checkpoint_path": str(ckpt_path),
                }
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        print(f"Saved checkpoint: {runs[-1]['checkpoint_path']}")

    discovery_artifacts = write_discovery_artifacts(
        output_dir=output_dir,
        algorithm=algorithm_name,
        runs=runs,
    )
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
        "discovery_artifacts": discovery_artifacts,
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
