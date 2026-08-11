"""Experiment 3: paired calibrated-logZ initialization for the TB GFlowNet."""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.gflownet_tb.diagnostics import (
    SequenceArchive,
    canonical_sha256,
    derive_seed,
    distribution_summary,
    module_is_finite,
    optimizer_is_finite,
    parameter_gradient_norm,
    parameter_snapshot,
    parameter_update_norm,
    state_dict_checksum,
)
from src.experiments.tb_experiment_common import (
    _append_jsonl,
    _capture_global_rng,
    _compose_project_config,
    _detach_trajectories,
    _environment_versions,
    _epsilon_for_update,
    _evaluation_milestone,
    _file_sha256,
    _git_commit,
    _json_safe,
    _optimizer_to_device,
    _parameter_groups,
    _policy_state_cpu,
    _repo_root,
    _resolve_circuit,
    _resolve_device,
    _restore_global_rng,
    _score_trajectory_set,
    _source_tree_sha256,
    _tb_value,
    _torch_generator,
    _trajectory_row,
    _write_json,
    _write_milestone_tables,
    calibration_target,
    differentiable_trajectory_scores,
    initialize_log_z,
)


RUN_SCHEMA_VERSION = 2
CALIBRATION_TRAJECTORIES = 64
CALIBRATION_EPSILON = 0.5
EXPECTED_VARIANTS = ("z0", "zcal")
EXPERIMENT_4_RATES = (0.003, 0.01, 0.03, 0.1)
BATCH_SIZE_INFLUENCE_BATCHES = (1, 4, 8, 16, 32)
BATCH_SIZE_SCHEDULE_UNIT = 4


def _resolved_configuration(args: argparse.Namespace, cfg: Any, circuit_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    explicit_log_z_lr = getattr(args, "log_z_learning_rate", None)
    configured_log_z_lr = (
        float(explicit_log_z_lr)
        if explicit_log_z_lr is not None
        else _tb_value(cfg, "log_z_learning_rate", None, preserve_none=True)
    )
    policy_lr = float(cfg.learning_rate)
    resolved_log_z_lr = 10.0 * policy_lr if configured_log_z_lr is None else float(configured_log_z_lr)
    configured_batch_size = _tb_value(cfg, "batch_size", None, preserve_none=True)
    if configured_batch_size is None:
        configured_batch_size = _tb_value(cfg, "trajectories_per_episode", 4)
    actions_raw = OmegaConf.select(cfg, "available_actions")
    actions = None if actions_raw is None else [int(action) for action in actions_raw]
    experiment = str(getattr(args, "experiment_name", "calibrated_log_z_initialization"))
    configured_optimizer_updates = int(cfg.episodes)
    if experiment in {"trajectory_budget_selection", "batch_size_influence"}:
        configured_optimizer_updates = int(args.max_trajectories) // int(configured_batch_size)
    trajectory_indexed_epsilon = experiment == "batch_size_influence"
    values = {
        "schema_version": int(getattr(args, "run_schema_version", RUN_SCHEMA_VERSION)),
        "experiment": experiment,
        "variant": str(args.variant).lower(),
        "config_name": args.config_name,
        "circuit": circuit_path.stem,
        "circuit_path": str(circuit_path),
        "seed": int(args.seed),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(args.device),
        "max_trajectories": int(args.max_trajectories),
        "schedule_trajectories": int(args.schedule_trajectories),
        "configured_optimizer_updates": configured_optimizer_updates,
        "milestones": [int(value) for value in args.milestones],
        "num_steps": int(cfg.num_steps),
        "available_actions": actions,
        "trajectories_per_update": int(configured_batch_size),
        "policy_learning_rate": policy_lr,
        "log_z_learning_rate_configured": configured_log_z_lr,
        "log_z_learning_rate_resolved": resolved_log_z_lr,
        "reward_alpha": float(_tb_value(cfg, "reward_alpha", 4.0)),
        "reward_eps": float(_tb_value(cfg, "reward_eps", 1e-8)),
        "reward_improvement_clip": float(_tb_value(cfg, "reward_improvement_clip", 2.0)),
        "exploration_epsilon_enabled": bool(_tb_value(cfg, "exploration_epsilon_enabled", True)),
        "exploration_epsilon_start": float(_tb_value(cfg, "exploration_epsilon_start", 0.5)),
        "exploration_epsilon_end": float(_tb_value(cfg, "exploration_epsilon_end", 0.01)),
        "exploration_warmup_updates": int(_tb_value(cfg, "exploration_warmup_episodes", 20)),
        "exploration_decay_updates": _tb_value(cfg, "exploration_decay_episodes", None, preserve_none=True),
        "epsilon_schedule_indexing": (
            "canonical_trajectory_groups" if trajectory_indexed_epsilon else "optimizer_updates"
        ),
        "epsilon_schedule_unit_trajectories": (
            BATCH_SIZE_SCHEDULE_UNIT if trajectory_indexed_epsilon else int(configured_batch_size)
        ),
        "calibration_trajectories": CALIBRATION_TRAJECTORIES,
        "calibration_epsilon": CALIBRATION_EPSILON,
        "calibration_batch_order": "collection_order",
        "gradient_clipping": None,
        "fixed_validation_trajectories": 256,
        "fresh_validation_trajectories": 128,
        "search_trajectories": 50,
        "search_budgets": [1, 2, 5, 10, 20, 50],
        "project_config": OmegaConf.to_container(cfg, resolve=True),
    }
    scientific_keys = (
        "schema_version", "experiment", "variant", "config_name", "num_steps",
        "available_actions", "trajectories_per_update", "policy_learning_rate",
        "log_z_learning_rate_configured", "log_z_learning_rate_resolved", "reward_alpha",
        "reward_eps", "reward_improvement_clip", "exploration_epsilon_enabled",
        "exploration_epsilon_start", "exploration_epsilon_end", "exploration_warmup_updates",
        "exploration_decay_updates", "schedule_trajectories", "configured_optimizer_updates",
        "calibration_trajectories", "calibration_epsilon", "calibration_batch_order",
        "gradient_clipping", "fixed_validation_trajectories", "fresh_validation_trajectories",
        "search_trajectories", "search_budgets",
    )
    scientific = {key: values[key] for key in scientific_keys}
    if experiment == "batch_size_influence":
        scientific.update({
            "epsilon_schedule_indexing": values["epsilon_schedule_indexing"],
            "epsilon_schedule_unit_trajectories": values["epsilon_schedule_unit_trajectories"],
        })
    if experiment in {"trajectory_budget_selection", "batch_size_influence"}:
        scientific.update({
            "max_trajectories": values["max_trajectories"],
            "milestones": values["milestones"],
        })
    paired = {key: value for key, value in scientific.items() if key != "variant"}
    values["scientific_configuration"] = scientific
    values["scientific_configuration_fingerprint"] = canonical_sha256(scientific)
    values["paired_configuration_fingerprint"] = canonical_sha256(paired)
    if values["experiment"] == "log_z_learning_rate":
        pairing = {
            key: value for key, value in scientific.items()
            if key not in {"variant", "log_z_learning_rate_configured", "log_z_learning_rate_resolved"}
        }
        values["pairing_configuration_fingerprint"] = canonical_sha256(pairing)
    if values["experiment"] == "batch_size_influence":
        pairing = {
            key: value for key, value in scientific.items()
            if key not in {"trajectories_per_update", "configured_optimizer_updates"}
        }
        values["batch_pairing_configuration_fingerprint"] = canonical_sha256(pairing)
    return values


def _validate_configuration(resolved: Mapping[str, Any]) -> None:
    experiment = str(resolved["experiment"])
    expected_rate = 0.01
    if experiment == "log_z_learning_rate":
        rate = float(resolved["log_z_learning_rate_resolved"])
        if rate not in EXPERIMENT_4_RATES:
            raise ValueError(f"Experiment 4 logZ rate must be one of {EXPERIMENT_4_RATES}, got {rate}")
        expected_rate = rate
    elif experiment in {"trajectory_budget_selection", "batch_size_influence"}:
        expected_rate = 0.01
    expected_batch_size = 4
    if experiment == "batch_size_influence":
        expected_batch_size = int(resolved["trajectories_per_update"])
        if expected_batch_size not in BATCH_SIZE_INFLUENCE_BATCHES:
            raise ValueError(
                "Experiment 5.5 batch size must be one of "
                f"{BATCH_SIZE_INFLUENCE_BATCHES}, got {expected_batch_size}"
            )
    required = {
        "variant": resolved["variant"],
        "num_steps": 20,
        "available_actions": list(range(7)),
        "trajectories_per_update": expected_batch_size,
        "policy_learning_rate": 0.001,
        "log_z_learning_rate_resolved": expected_rate,
        "reward_alpha": 4.0,
        "reward_eps": 1e-8,
        "reward_improvement_clip": 2.0,
        "exploration_epsilon_start": 0.5,
        "exploration_epsilon_end": 0.01,
        "exploration_warmup_updates": 20,
        "configured_optimizer_updates": (
            int(resolved["max_trajectories"]) // int(resolved["trajectories_per_update"])
            if experiment in {"trajectory_budget_selection", "batch_size_influence"} else 200
        ),
        "calibration_trajectories": 64,
        "calibration_epsilon": 0.5,
    }
    if resolved["variant"] not in EXPECTED_VARIANTS:
        raise ValueError(f"variant must be one of {EXPECTED_VARIANTS}")
    if experiment == "trajectory_budget_selection" and resolved["variant"] != "zcal":
        raise ValueError("Experiment 5 requires calibrated logZ initialization (zcal)")
    if experiment == "batch_size_influence" and resolved["variant"] != "zcal":
        raise ValueError("Experiment 5.5 requires calibrated logZ initialization (zcal)")
    failures = {
        key: {"expected": expected, "actual": resolved.get(key)}
        for key, expected in required.items() if resolved.get(key) != expected
    }
    if failures:
        label = "Experiment 4" if experiment == "log_z_learning_rate" else "Experiment 3"
        raise ValueError(f"configuration no longer matches {label}: {failures}")
    if not resolved["exploration_epsilon_enabled"]:
        raise ValueError("active epsilon schedule is disabled")
    if int(resolved["schedule_trajectories"]) != 800:
        raise ValueError(f"{experiment} retains the 800-trajectory epsilon horizon")


def _calibration_payload(
    trajectories: Sequence[Any], *, seed: int, initial_score: Mapping[str, Any], target: float,
    schema_version: int = RUN_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": int(schema_version),
        "rng_seed": int(seed),
        "epsilon_uniform": CALIBRATION_EPSILON,
        "batch_order": "collection_order",
        "analytic_log_z_target": float(target),
        "initial_log_pf": list(initial_score["log_pf_values"]),
        "log_pb": list(initial_score["log_pb_values"]),
        "log_r": list(initial_score["log_r_values"]),
        "actions": [[int(step.action) for step in trajectory.steps] for trajectory in trajectories],
        "trajectories": list(trajectories),
    }


def _trajectory_set_checksum(trajectories: Sequence[Any]) -> str:
    """Content checksum independent of torch serialization details."""
    return canonical_sha256([
        {
            "file_path": str(trajectory.file_path),
            "actions": [int(step.action) for step in trajectory.steps],
            "legal_actions": [list(map(int, step.legal_actions)) for step in trajectory.steps],
            "initial_size": int(trajectory.initial_size),
            "initial_depth": int(trajectory.initial_depth),
            "final_size": int(trajectory.final_size),
            "final_depth": int(trajectory.final_depth),
            "log_reward": float(trajectory.log_reward),
            "log_pb": float(trajectory.log_pb_sum.detach().cpu()),
        }
        for trajectory in trajectories
    ])


def _checkpoint_payload(
    *, policy: Any, optimizer: torch.optim.Optimizer, archive: SequenceArchive,
    counters: Mapping[str, int], gradient_norms: Sequence[float],
    milestone_rows: Sequence[Mapping[str, Any]], train_generator: torch.Generator,
    circuit_rng: np.random.Generator, replay_rng: np.random.Generator,
    fixed_checksum: str, calibration_checksum: str, resolved: Mapping[str, Any],
    metadata: Mapping[str, Any], numerical_failure: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": int(resolved["schema_version"]),
        "experiment": str(resolved["experiment"]),
        "variant": resolved["variant"],
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "replay": None,
        "archive": archive.state_dict(),
        "counters": dict(counters),
        "gradient_norms": list(gradient_norms),
        "milestone_rows": list(milestone_rows),
        "global_rng": _capture_global_rng(),
        "train_action_generator_state": train_generator.get_state(),
        "circuit_rng_state": circuit_rng.bit_generator.state,
        "replay_rng_state": replay_rng.bit_generator.state,
        "fixed_cache_checksum": fixed_checksum,
        "calibration_cache_checksum": calibration_checksum,
        "calibration_phase_complete": int(counters["optimizer_updates"]) >= 16,
        "resolved_config": dict(resolved),
        "run_metadata": dict(metadata),
        "git_commit": metadata.get("git_commit"),
        "source_tree_sha256": metadata.get("source_tree_sha256"),
        "numerical_failure": numerical_failure,
    }


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_checkpoint_payload(**kwargs), temporary)
    temporary.replace(path)


def _training_tensors(policy: Any, trajectories: Sequence[Any], *, cached: bool, device: torch.device):
    if cached:
        return differentiable_trajectory_scores(policy, trajectories)
    log_pf = torch.stack([trajectory.log_pf_sum for trajectory in trajectories])
    log_pb = torch.stack([trajectory.log_pb_sum for trajectory in trajectories])
    log_r = torch.tensor([trajectory.log_reward for trajectory in trajectories], dtype=log_pf.dtype, device=device)
    return log_pf, log_pb, log_r


def canonical_trajectory_epsilon_values(
    *,
    first_trajectory: int,
    count: int,
    schedule_trajectories: int = 800,
    enabled: bool = True,
    start: float = 0.5,
    end: float = 0.01,
    warmup_updates: int = 20,
    decay_updates: int | None = None,
) -> list[float]:
    """Return the original batch-four epsilon schedule indexed by trajectory."""
    if first_trajectory <= 0 or count <= 0:
        raise ValueError("first_trajectory and count must be positive")
    if schedule_trajectories % BATCH_SIZE_SCHEDULE_UNIT:
        raise ValueError("schedule trajectories must be divisible by four")
    return [
        _epsilon_for_update(
            (trajectory + BATCH_SIZE_SCHEDULE_UNIT - 1) // BATCH_SIZE_SCHEDULE_UNIT,
            schedule_updates=schedule_trajectories // BATCH_SIZE_SCHEDULE_UNIT,
            enabled=enabled,
            start=start,
            end=end,
            warmup_updates=warmup_updates,
            decay_updates=decay_updates,
        )
        for trajectory in range(first_trajectory, first_trajectory + count)
    ]


def _peak_resource_usage(device: torch.device) -> dict[str, int | None]:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak_rss //= 1024
    if device.type != "cuda":
        return {
            "peak_host_rss_kib": peak_rss,
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_reserved_bytes": None,
        }
    torch.cuda.synchronize(device)
    return {
        "peak_host_rss_kib": peak_rss,
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_experiment(args: argparse.Namespace) -> int:
    from omegaconf import OmegaConf

    from src.algorithms.gflownet_tb.factory import build_tb_policy
    from src.algorithms.gflownet_tb.loss import trajectory_balance_loss
    from src.algorithms.gflownet_tb.optim import build_tb_optimizer
    from src.algorithms.gflownet_tb.sampler import sample_tb_trajectories
    from src.models import reward_class_factory
    from src.utils import get_obs_dim_and_num_actions, normalize_available_actions

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    milestones_path = output_dir / "milestones.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    fixed_path = output_dir / "fixed_validation.pt"
    calibration_path = output_dir / "calibration.pt"

    circuit_path = _resolve_circuit(args.circuit)
    device = _resolve_device(args.device)
    cfg = _compose_project_config(args.config_name)
    if getattr(args, "batch_size", None) is not None:
        cfg.tb.batch_size = int(args.batch_size)
        cfg.tb.trajectories_per_episode = int(args.batch_size)
    if getattr(args, "log_z_learning_rate", None) is not None:
        cfg.tb.log_z_learning_rate = float(args.log_z_learning_rate)
    cfg.output_dir = str(output_dir)
    resolved = _resolved_configuration(args, cfg, circuit_path)
    _validate_configuration(resolved)
    batch_size = int(resolved["trajectories_per_update"])
    if args.max_trajectories < CALIBRATION_TRAJECTORIES or args.max_trajectories % batch_size:
        raise ValueError(
            "--max-trajectories must be at least 64 and divisible by the configured batch size"
        )
    if args.max_trajectories > args.schedule_trajectories and resolved["experiment"] not in {
        "trajectory_budget_selection", "batch_size_influence",
    }:
        raise ValueError("max trajectories cannot exceed the epsilon schedule budget")
    milestones = sorted(set(int(value) for value in args.milestones))
    if (
        not milestones
        or milestones[-1] > args.max_trajectories
        or any(value < 64 or value % batch_size for value in milestones)
    ):
        raise ValueError(
            "milestones must be divisible by the configured batch size and lie in "
            "[64, max-trajectories]"
        )

    reward_cfg = OmegaConf.to_container(cfg.reward, resolve=True)
    if not isinstance(reward_cfg, dict):
        raise TypeError("reward config must resolve to a mapping")
    reward_class = reward_class_factory(reward_cfg)
    obs_dim, num_actions, node_dim, edge_dim = get_obs_dim_and_num_actions(int(resolved["num_steps"]), str(circuit_path))
    available_actions = normalize_available_actions(resolved["available_actions"], num_actions)
    seeds = {name: derive_seed(args.seed, name) for name in (
        "initialization", "training_actions", "circuit_selection", "replay",
        "evaluation_fixed_validation", "evaluation_fresh_policy", "evaluation_search",
    )}
    source_checksum = _source_tree_sha256(_repo_root())
    metadata: dict[str, Any] = {
        "schema_version": int(resolved["schema_version"]),
        "experiment": str(resolved["experiment"]),
        "variant": resolved["variant"],
        "circuit_sha256": _file_sha256(circuit_path),
        "source_tree_sha256": source_checksum,
        "git_commit": _git_commit(_repo_root()),
        "environment": _environment_versions(),
        "rng_seeds": seeds,
        "rng_derivation": "sha256(gflowcircuit-active-baseline-v1, base_seed, stream_name)",
    }
    initialization_seed = seeds["initialization"]
    random.seed(initialization_seed)
    np.random.seed(initialization_seed % (2**32))
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    policy = build_tb_policy(cfg, obs_dim=obs_dim, node_dim=node_dim, edge_dim=edge_dim,
                             num_actions=num_actions, available_actions=available_actions).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pre_calibration_checksum = state_dict_checksum(_policy_state_cpu(policy))
    metadata["pre_calibration_parameter_checksum"] = pre_calibration_checksum
    train_generator = _torch_generator(device, seeds["training_actions"])
    circuit_rng = np.random.default_rng(seeds["circuit_selection"])
    replay_rng = np.random.default_rng(seeds["replay"])
    counters = {
        "training_trajectories": 0, "training_transitions": 0,
        "training_presentations": 0, "optimizer_updates": 0,
        "calibration_trajectories": 0, "calibration_target_presentations": 0,
        "calibration_training_presentations": 0, "new_training_trajectories": 0,
        "validation_rollouts": 0, "validation_transitions": 0,
        "validation_rescoring_presentations": 0,
    }
    gradient_norms: list[float] = []
    milestone_rows: list[dict[str, Any]] = []
    fixed_trajectories: list[Any]
    calibration_trajectories: list[Any]
    archive: SequenceArchive
    fixed_checksum = ""
    calibration_checksum = ""
    resume_path = args.resume_checkpoint.resolve() if args.resume_checkpoint else None

    if resume_path is None:
        for path in (metrics_path, milestones_path, trajectories_path):
            if path.exists() and path.stat().st_size:
                raise FileExistsError(f"refusing to overwrite existing run artifact: {path}")
            path.write_text("", encoding="utf-8")
        with torch.no_grad():
            fixed_generator = _torch_generator(device, seeds["evaluation_fixed_validation"])
            fixed_trajectories = _detach_trajectories(sample_tb_trajectories(
                file_paths=[str(circuit_path)] * 256, num_steps=int(resolved["num_steps"]),
                policy=policy, reward_class=reward_class, reward_alpha=float(resolved["reward_alpha"]),
                reward_eps=float(resolved["reward_eps"]),
                reward_improvement_clip=float(resolved["reward_improvement_clip"]),
                sample_actions=True, available_actions=available_actions, epsilon_uniform=1.0,
                action_generator=fixed_generator,
            ))
            calibration_trajectories = _detach_trajectories(sample_tb_trajectories(
                file_paths=[str(circuit_path)] * CALIBRATION_TRAJECTORIES,
                num_steps=int(resolved["num_steps"]), policy=policy, reward_class=reward_class,
                reward_alpha=float(resolved["reward_alpha"]), reward_eps=float(resolved["reward_eps"]),
                reward_improvement_clip=float(resolved["reward_improvement_clip"]),
                sample_actions=True, available_actions=available_actions,
                epsilon_uniform=CALIBRATION_EPSILON, action_generator=train_generator,
            ))
        torch.save({"schema_version": int(resolved["schema_version"]), "rng_seed": seeds["evaluation_fixed_validation"],
                    "trajectories": fixed_trajectories}, fixed_path)
        fixed_checksum = _file_sha256(fixed_path)
        initial_fixed = _score_trajectory_set(policy, fixed_trajectories,
                                              reward_eps=float(resolved["reward_eps"]),
                                              reward_improvement_clip=float(resolved["reward_improvement_clip"]))
        initial_calibration = _score_trajectory_set(policy, calibration_trajectories,
                                                    reward_eps=float(resolved["reward_eps"]),
                                                    reward_improvement_clip=float(resolved["reward_improvement_clip"]))
        target = calibration_target(policy, calibration_trajectories)
        torch.save(_calibration_payload(
            calibration_trajectories, seed=seeds["training_actions"],
            initial_score=initial_calibration, target=target,
            schema_version=int(resolved["schema_version"]),
        ), calibration_path)
        calibration_checksum = _file_sha256(calibration_path)
        assigned = initialize_log_z(policy, resolved["variant"], target)
        metadata.update({
            "fixed_validation_checksum": fixed_checksum,
            "calibration_cache_checksum": calibration_checksum,
            "fixed_sequence_checksum": _trajectory_set_checksum(fixed_trajectories),
            "calibration_sequence_checksum": _trajectory_set_checksum(calibration_trajectories),
            "calibration_analytic_target": target,
            "assigned_initial_log_z": assigned,
            "post_initialization_parameter_checksum": state_dict_checksum(_policy_state_cpu(policy)),
        })
        _write_json(output_dir / "resolved_config.json", resolved)
        _write_json(output_dir / "run_metadata.json", metadata)
        _append_jsonl(metrics_path, {"row_type": "initial_fixed_validation", "trajectory_budget": 0,
                                     "optimizer_update": 0, "fixed_uniform": initial_fixed,
                                     "wall_time_seconds": time.perf_counter() - started})
        post_calibration_score = _score_trajectory_set(policy, calibration_trajectories,
                                                       reward_eps=float(resolved["reward_eps"]),
                                                       reward_improvement_clip=float(resolved["reward_improvement_clip"]))
        _append_jsonl(metrics_path, {"row_type": "calibration_initialization", "trajectory_budget": 64,
                                     "optimizer_update": 0, "variant": resolved["variant"],
                                     "analytic_log_z_target": target, "assigned_log_z": assigned,
                                     "calibration": post_calibration_score,
                                     "wall_time_seconds": time.perf_counter() - started})
        for index, trajectory in enumerate(fixed_trajectories, 1):
            _append_jsonl(trajectories_path, _trajectory_row(trajectory, source="fixed_uniform", milestone=0, local_index=index))
        for index, trajectory in enumerate(calibration_trajectories, 1):
            _append_jsonl(trajectories_path, _trajectory_row(trajectory, source="calibration", milestone=64, local_index=index))
        first = calibration_trajectories[0]
        archive = SequenceArchive(circuit=circuit_path.stem, initial_size=int(first.initial_size), initial_depth=int(first.initial_depth))
        for trajectory in calibration_trajectories:
            archive.record(actions=[int(step.action) for step in trajectory.steps],
                           initial_size=int(trajectory.initial_size), initial_depth=int(trajectory.initial_depth),
                           final_size=int(trajectory.final_size), final_depth=int(trajectory.final_depth),
                           comparable_return=float(trajectory.comparable_return))
        counters.update({
            "training_trajectories": 64,
            "training_transitions": sum(len(item.steps) for item in calibration_trajectories),
            "calibration_trajectories": 64,
            "calibration_target_presentations": 64,
            "validation_rollouts": len(fixed_trajectories),
            "validation_transitions": sum(len(item.steps) for item in fixed_trajectories),
        })
        optimizer = build_tb_optimizer(policy, learning_rate=float(resolved["policy_learning_rate"]),
                                       log_z_learning_rate=float(resolved["log_z_learning_rate_resolved"]))
    else:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        prior = checkpoint["resolved_config"]
        if prior["scientific_configuration_fingerprint"] != resolved["scientific_configuration_fingerprint"]:
            raise ValueError("resume checkpoint scientific configuration does not match")
        if checkpoint.get("source_tree_sha256") != source_checksum:
            raise ValueError("resume checkpoint source tree does not match the running code")
        optimizer = build_tb_optimizer(policy, learning_rate=float(resolved["policy_learning_rate"]),
                                       log_z_learning_rate=float(resolved["log_z_learning_rate_resolved"]))
        policy.load_state_dict(checkpoint["policy"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        _optimizer_to_device(optimizer, device)
        archive = SequenceArchive.from_state_dict(checkpoint["archive"])
        counters = {key: int(value) for key, value in checkpoint["counters"].items()}
        gradient_norms = [float(value) for value in checkpoint["gradient_norms"]]
        milestone_rows = list(checkpoint["milestone_rows"])
        _restore_global_rng(checkpoint["global_rng"])
        train_generator.set_state(checkpoint["train_action_generator_state"])
        circuit_rng.bit_generator.state = checkpoint["circuit_rng_state"]
        replay_rng.bit_generator.state = checkpoint["replay_rng_state"]
        fixed_checksum = str(checkpoint["fixed_cache_checksum"])
        calibration_checksum = str(checkpoint["calibration_cache_checksum"])
        if _file_sha256(fixed_path) != fixed_checksum or _file_sha256(calibration_path) != calibration_checksum:
            raise ValueError("fixed or calibration cache checksum mismatch on resume")
        fixed_trajectories = torch.load(fixed_path, map_location="cpu", weights_only=False)["trajectories"]
        calibration_trajectories = torch.load(calibration_path, map_location="cpu", weights_only=False)["trajectories"]
        metadata = checkpoint["run_metadata"]

    completed_milestones = {int(row["trajectory_budget"]) for row in milestone_rows}
    expected_updates = args.max_trajectories // batch_size
    numerical_failure: str | None = None
    try:
        while counters["optimizer_updates"] < expected_updates:
            update = counters["optimizer_updates"] + 1
            cached = update <= CALIBRATION_TRAJECTORIES // batch_size
            epsilon_values: list[float]
            if cached:
                epsilon_values = [CALIBRATION_EPSILON] * batch_size
            elif resolved["experiment"] == "batch_size_influence":
                epsilon_values = canonical_trajectory_epsilon_values(
                    first_trajectory=counters["training_trajectories"] + 1,
                    count=batch_size,
                    schedule_trajectories=int(args.schedule_trajectories),
                    enabled=bool(resolved["exploration_epsilon_enabled"]),
                    start=float(resolved["exploration_epsilon_start"]),
                    end=float(resolved["exploration_epsilon_end"]),
                    warmup_updates=int(resolved["exploration_warmup_updates"]),
                    decay_updates=resolved["exploration_decay_updates"],
                )
            else:
                epsilon = _epsilon_for_update(
                    update,
                    schedule_updates=args.schedule_trajectories // batch_size,
                    enabled=bool(resolved["exploration_epsilon_enabled"]),
                    start=float(resolved["exploration_epsilon_start"]),
                    end=float(resolved["exploration_epsilon_end"]),
                    warmup_updates=int(resolved["exploration_warmup_updates"]),
                    decay_updates=resolved["exploration_decay_updates"],
                )
                epsilon_values = [epsilon] * batch_size
            policy.train()
            if cached:
                start = (update - 1) * batch_size
                trajectories = calibration_trajectories[start:start + batch_size]
            else:
                selected_paths = [str(circuit_path) for _ in range(batch_size) if int(circuit_rng.integers(0, 1)) == 0]
                if len(selected_paths) != batch_size:
                    raise RuntimeError("circuit selection stream returned invalid batch")
                trajectories = sample_tb_trajectories(
                    file_paths=selected_paths, num_steps=int(resolved["num_steps"]), policy=policy,
                    reward_class=reward_class, reward_alpha=float(resolved["reward_alpha"]),
                    reward_eps=float(resolved["reward_eps"]),
                    reward_improvement_clip=float(resolved["reward_improvement_clip"]),
                    sample_actions=True, available_actions=available_actions,
                    epsilon_uniform=epsilon_values,
                    action_generator=train_generator,
                )
            log_pf, log_pb, log_r = _training_tensors(policy, trajectories, cached=cached, device=device)
            training_score = _score_trajectory_set(policy, trajectories, reward_eps=float(resolved["reward_eps"]),
                                                   reward_improvement_clip=float(resolved["reward_improvement_clip"]))
            policy_parameters, log_z_parameter = _parameter_groups(policy)
            before = parameter_snapshot(policy_parameters)
            log_z_before = float(log_z_parameter.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss = trajectory_balance_loss(log_z=policy.log_z, log_pf_sums=log_pf, log_rewards=log_r, log_pb_sums=log_pb)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at update {update}")
            loss.backward()
            policy_grad_norm = parameter_gradient_norm(policy_parameters)
            log_z_grad_norm = parameter_gradient_norm([log_z_parameter])
            gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                                   for parameter in [*policy_parameters, log_z_parameter])
            if not gradients_finite:
                raise FloatingPointError(f"non-finite gradient at update {update}")
            optimizer.step()
            policy_update_norm = parameter_update_norm(before, policy_parameters)
            log_z_update = abs(float(log_z_parameter.detach().cpu()) - log_z_before)
            if not module_is_finite(policy) or not optimizer_is_finite(optimizer):
                raise FloatingPointError(f"non-finite parameters or optimizer state at update {update}")
            counters["optimizer_updates"] = update
            counters["training_presentations"] += len(trajectories)
            if cached:
                counters["calibration_training_presentations"] += len(trajectories)
            else:
                counters["training_trajectories"] += len(trajectories)
                counters["new_training_trajectories"] += len(trajectories)
                counters["training_transitions"] += sum(len(item.steps) for item in trajectories)
                for local_index, trajectory in enumerate(trajectories, 1):
                    archive.record(actions=[int(step.action) for step in trajectory.steps],
                                   initial_size=int(trajectory.initial_size), initial_depth=int(trajectory.initial_depth),
                                   final_size=int(trajectory.final_size), final_depth=int(trajectory.final_depth),
                                   comparable_return=float(trajectory.comparable_return))
                    _append_jsonl(trajectories_path, _trajectory_row(
                        trajectory, source="training", milestone=counters["training_trajectories"], local_index=local_index))
            gradient_norms.append(policy_grad_norm)
            raw = [float(item.comparable_return) for item in trajectories]
            train_residual = training_score["residual"]
            _append_jsonl(metrics_path, {
                "row_type": "training_update", "optimizer_update": update,
                "trajectory_budget": counters["training_trajectories"],
                "training_source": "calibration" if cached else "new_on_policy",
                "calibration_batch_start": (update - 1) * batch_size if cached else None,
                "epsilon_uniform": float(np.mean(epsilon_values)),
                "epsilon_uniform_min": float(min(epsilon_values)),
                "epsilon_uniform_max": float(max(epsilon_values)),
                "epsilon_uniform_values": epsilon_values,
                "loss": float(loss.detach().cpu()), "residual": train_residual,
                "target_gap": float(train_residual["log_z_target_gap"]),
                "regression": training_score["regression"], "policy": training_score["policy"],
                "raw_improvement": distribution_summary(raw),
                "policy_gradient_norm": policy_grad_norm, "log_z_gradient_norm": log_z_grad_norm,
                "policy_parameter_update_norm": policy_update_norm, "absolute_log_z_update": log_z_update,
                "policy_learning_rate": optimizer.param_groups[0]["lr"],
                "log_z_learning_rate": optimizer.param_groups[1]["lr"],
                "gradient_clipping_threshold": None, "gradient_clipping_occurred": False,
                "finite_loss": True, "finite_gradients": gradients_finite,
                "finite_parameters": module_is_finite(policy),
                "finite_optimizer_state": optimizer_is_finite(optimizer),
                "archive": archive.snapshot(), "counters": dict(counters),
                "wall_time_seconds": time.perf_counter() - started,
            })
            budget = counters["training_trajectories"]
            if budget in milestones and budget not in completed_milestones:
                train_state = train_generator.get_state().clone()
                row = _evaluation_milestone(
                    policy=policy, circuit_path=circuit_path, fixed_trajectories=fixed_trajectories,
                    trajectory_budget=budget, update=update, device=device,
                    num_steps=int(resolved["num_steps"]), reward_class=reward_class,
                    reward_alpha=float(resolved["reward_alpha"]), reward_eps=float(resolved["reward_eps"]),
                    reward_improvement_clip=float(resolved["reward_improvement_clip"]),
                    available_actions=available_actions, fresh_seed=seeds["evaluation_fresh_policy"],
                    search_seed=seeds["evaluation_search"], archive=archive, counters=counters,
                    gradient_norms=gradient_norms, trajectories_path=trajectories_path,
                )
                if not torch.equal(train_state, train_generator.get_state()):
                    raise RuntimeError("evaluation advanced training-action generator")
                row["wall_time_seconds"] = time.perf_counter() - started
                row["resource_usage"] = _peak_resource_usage(device)
                milestone_rows.append(row)
                completed_milestones.add(budget)
                _append_jsonl(milestones_path, row)
                _append_jsonl(metrics_path, {"row_type": "milestone", "trajectory_budget": budget,
                                             "optimizer_update": update, "fixed_uniform": row["fixed_uniform"],
                                             "fresh_on_policy": row["fresh_on_policy"], "search": row["search"],
                                             "training_archive": row["training_archive"],
                                             "wall_time_seconds": row["wall_time_seconds"]})
                _write_milestone_tables(output_dir, row)
                _save_checkpoint(checkpoints_dir / f"trajectory_{budget}.pt", policy=policy, optimizer=optimizer,
                                 archive=archive, counters=counters, gradient_norms=gradient_norms,
                                 milestone_rows=milestone_rows, train_generator=train_generator,
                                 circuit_rng=circuit_rng, replay_rng=replay_rng, fixed_checksum=fixed_checksum,
                                 calibration_checksum=calibration_checksum, resolved=resolved, metadata=metadata,
                                 numerical_failure=None)
        missing = [value for value in milestones if value not in completed_milestones]
        if missing:
            raise RuntimeError(f"run completed without required milestones: {missing}")
    except Exception as exc:
        numerical_failure = f"{type(exc).__name__}: {exc}"
        emergency = checkpoints_dir / f"emergency_trajectory_{counters['training_trajectories']}.pt"
        _save_checkpoint(emergency, policy=policy, optimizer=optimizer, archive=archive, counters=counters,
                         gradient_norms=gradient_norms, milestone_rows=milestone_rows,
                         train_generator=train_generator, circuit_rng=circuit_rng, replay_rng=replay_rng,
                         fixed_checksum=fixed_checksum, calibration_checksum=calibration_checksum,
                         resolved=resolved, metadata=metadata, numerical_failure=numerical_failure)
        _write_json(output_dir / "run_summary.json", {
            "schema_version": int(resolved["schema_version"]), "complete": False, "variant": resolved["variant"],
            "experiment": str(resolved["experiment"]),
            "numerical_failure": numerical_failure, "traceback": traceback.format_exc(),
            "emergency_checkpoint": str(emergency), "counters": counters, "milestones": milestone_rows,
            "scientific_configuration_fingerprint": resolved["scientific_configuration_fingerprint"],
            "paired_configuration_fingerprint": resolved["paired_configuration_fingerprint"],
            "batch_pairing_configuration_fingerprint": resolved.get(
                "batch_pairing_configuration_fingerprint"
            ),
            "wall_time_seconds": time.perf_counter() - started,
            "resource_usage": _peak_resource_usage(device),
        })
        raise
    summary = {
        "schema_version": int(resolved["schema_version"]), "experiment": str(resolved["experiment"]),
        "complete": True, "numerical_failure": None, "variant": resolved["variant"],
        "circuit": circuit_path.stem, "seed": int(args.seed),
        "scientific_configuration_fingerprint": resolved["scientific_configuration_fingerprint"],
        "paired_configuration_fingerprint": resolved["paired_configuration_fingerprint"],
        "batch_pairing_configuration_fingerprint": resolved.get(
            "batch_pairing_configuration_fingerprint"
        ),
        "pre_calibration_parameter_checksum": pre_calibration_checksum,
        "post_initialization_parameter_checksum": metadata["post_initialization_parameter_checksum"],
        "fixed_validation_checksum": fixed_checksum, "calibration_cache_checksum": calibration_checksum,
        "fixed_sequence_checksum": metadata["fixed_sequence_checksum"],
        "calibration_sequence_checksum": metadata["calibration_sequence_checksum"],
        "source_tree_sha256": source_checksum, "git_commit": metadata.get("git_commit"),
        "counters": counters, "milestones": milestone_rows, "final_archive": archive.snapshot(),
        "final_checkpoint": str(checkpoints_dir / f"trajectory_{args.max_trajectories}.pt"),
        "wall_time_seconds": time.perf_counter() - started,
        "resource_usage": _peak_resource_usage(device),
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


def _add_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("run", help="run one Experiment 3 variant/circuit/seed")
    parser.add_argument("--variant", choices=EXPECTED_VARIANTS, required=True)
    parser.add_argument("--config-name", default="tb_zhuDOP")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=800)
    parser.add_argument("--schedule-trajectories", type=int, default=800)
    parser.add_argument("--milestones", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.set_defaults(handler=run_experiment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    from src.experiments.tb_logz_calibration_report import add_report_parser
    add_report_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"logZ-calibration {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
