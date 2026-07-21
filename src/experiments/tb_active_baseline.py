"""Instrumented reproduction and diagnosis of the active TB baseline.

This entry point is intentionally isolated from ``src.run`` and the production
trainer.  It composes the same configuration and calls the same policy, reward,
sampler, loss, and optimizer components while adding deterministic RNG streams,
validation caches, checkpoints, and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.gflownet_tb.diagnostics import (
    SequenceArchive,
    SequenceRecord,
    canonical_sha256,
    derive_seed,
    distribution_summary,
    module_is_finite,
    nested_search_metrics,
    optimizer_is_finite,
    parameter_gradient_norm,
    parameter_snapshot,
    parameter_update_norm,
    regression_diagnostics,
    residual_diagnostics,
    state_dict_checksum,
)


RUN_SCHEMA_VERSION = 1
DEFAULT_CIRCUIT_ROOT = Path("data/hdl-benchmarks/mcnc/Combinational/blif")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else _json_safe(value)
                    for key, value in row.items()
                }
            )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    hasher = hashlib.sha256()
    candidates: list[Path] = []
    for directory in (root / "src", root / "cfg"):
        candidates.extend(
            path for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    for filename in ("requirements.txt", "pyproject.toml"):
        path = root / filename
        if path.is_file():
            candidates.append(path)
    for path in sorted(candidates):
        relative = path.relative_to(root).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _git_commit(root: Path) -> str | None:
    explicit = os.environ.get("GFLOWCIRCUIT_GIT_COMMIT")
    if explicit:
        return explicit
    sidecar = root / ".myhpc-source-commit"
    if sidecar.is_file():
        value = sidecar.read_text(encoding="utf-8").strip()
        if len(value) == 40 and all(character in "0123456789abcdefABCDEF" for character in value):
            return value.lower()
        raise ValueError(f"invalid myhpc source-commit sidecar: {sidecar}")
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _compose_project_config(config_name: str) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str((_repo_root() / "cfg").resolve())):
        return compose(config_name=config_name)


def _tb_value(cfg: Any, key: str, default: Any, *, preserve_none: bool = False) -> Any:
    from omegaconf import OmegaConf

    tb_cfg = OmegaConf.select(cfg, "tb")
    if tb_cfg is None:
        tb_cfg = OmegaConf.select(cfg, "algorithm.tb")
    value = OmegaConf.select(tb_cfg, key) if tb_cfg is not None else None
    if value is None and not preserve_none:
        return default
    return value


def _resolve_circuit(value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix == "":
        candidate = DEFAULT_CIRCUIT_ROOT / f"{value}.blif"
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"circuit not found: {candidate}")
    return candidate


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _epsilon_for_update(
    update: int,
    *,
    schedule_updates: int,
    enabled: bool,
    start: float,
    end: float,
    warmup_updates: int,
    decay_updates: int | None,
) -> float:
    if not enabled:
        return 0.0
    if update <= warmup_updates:
        return float(start)
    resolved_decay = max(1, schedule_updates - warmup_updates) if decay_updates is None else decay_updates
    progress = min(1.0, max(0.0, (update - warmup_updates) / float(resolved_decay)))
    return float(start + progress * (end - start))


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _capture_global_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_global_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _environment_versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }
    for name in ("torch_geometric", "omegaconf", "hydra", "pyspiel"):
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic only
            result[name] = f"unavailable: {exc}"
    return result


def _policy_state_cpu(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _detach_trajectories(trajectories: Sequence[Any]) -> list[Any]:
    from src.algorithms.gflownet_tb.types import TBStep, TBTrajectory

    detached: list[TBTrajectory] = []
    for trajectory in trajectories:
        steps = [
            TBStep(
                observation=step.observation.observation_to_device(torch.device("cpu")),
                action=int(step.action),
                legal_actions=list(step.legal_actions),
                log_pf=step.log_pf.detach().cpu(),
            )
            for step in trajectory.steps
        ]
        detached.append(
            TBTrajectory(
                file_path=trajectory.file_path,
                steps=steps,
                initial_size=int(trajectory.initial_size),
                initial_depth=int(trajectory.initial_depth),
                final_size=int(trajectory.final_size),
                final_depth=int(trajectory.final_depth),
                final_return=float(trajectory.final_return),
                comparable_return=float(trajectory.comparable_return),
                log_pf_sum=trajectory.log_pf_sum.detach().cpu(),
                log_pb_sum=trajectory.log_pb_sum.detach().cpu(),
                log_reward=float(trajectory.log_reward),
                terminal_reward=float(trajectory.terminal_reward),
            )
        )
    return detached


def _trajectory_row(
    trajectory: Any,
    *,
    source: str,
    milestone: int,
    local_index: int,
) -> dict[str, Any]:
    return {
        "source": source,
        "milestone_trajectories": int(milestone),
        "local_index": int(local_index),
        "circuit": Path(trajectory.file_path).stem,
        "actions": [int(step.action) for step in trajectory.steps],
        "trajectory_length": len(trajectory.steps),
        "initial_size": int(trajectory.initial_size),
        "initial_depth": int(trajectory.initial_depth),
        "final_size": int(trajectory.final_size),
        "final_depth": int(trajectory.final_depth),
        "raw_improvement": float(trajectory.comparable_return),
        "log_reward": float(trajectory.log_reward),
        "terminal_reward": float(trajectory.terminal_reward),
        "log_pb": float(trajectory.log_pb_sum.detach().cpu()),
    }


def _records_from_trajectories(trajectories: Sequence[Any]) -> list[SequenceRecord]:
    return [
        SequenceRecord(
            index=index,
            actions=tuple(int(step.action) for step in trajectory.steps),
            initial_size=int(trajectory.initial_size),
            initial_depth=int(trajectory.initial_depth),
            final_size=int(trajectory.final_size),
            final_depth=int(trajectory.final_depth),
            comparable_return=float(trajectory.comparable_return),
        )
        for index, trajectory in enumerate(trajectories, start=1)
    ]


def _score_trajectory_set(
    policy: Any,
    trajectories: Sequence[Any],
    *,
    reward_eps: float,
    reward_improvement_clip: float,
    chunk_size: int = 256,
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("cannot score an empty trajectory set")
    observations: list[Any] = []
    legal_rows: list[list[int]] = []
    actions: list[int] = []
    depths: list[int] = []
    trajectory_ids: list[int] = []
    for trajectory_id, trajectory in enumerate(trajectories):
        for depth, step in enumerate(trajectory.steps):
            observations.append(step.observation)
            legal_rows.append(list(step.legal_actions))
            actions.append(int(step.action))
            depths.append(depth)
            trajectory_ids.append(trajectory_id)

    trajectory_log_pf = torch.zeros(len(trajectories), dtype=torch.float64)
    policy_rows: list[dict[str, Any]] = []
    finite = True
    policy_was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(observations), chunk_size):
            stop = min(len(observations), start + chunk_size)
            obs_chunk = observations[start:stop]
            legal_chunk = legal_rows[start:stop]
            action_chunk = actions[start:stop]
            logits = policy(obs_chunk)
            probs = policy.masked_probs(logits, legal_chunk)
            selected = policy.log_prob_legal_batch(logits, legal_chunk, action_chunk)
            finite = finite and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(probs).all())
            for row_index in range(stop - start):
                global_index = start + row_index
                legal = legal_chunk[row_index]
                legal_idx = torch.tensor(legal, dtype=torch.long, device=probs.device)
                legal_probs = probs[row_index].index_select(0, legal_idx)
                illegal_mask = torch.ones(policy.num_actions, dtype=torch.bool, device=probs.device)
                illegal_mask[legal_idx] = False
                illegal = probs[row_index][illegal_mask]
                max_probability = float(legal_probs.max().detach().cpu())
                policy_rows.append(
                    {
                        "depth": depths[global_index],
                        "normalization_error": abs(float(legal_probs.sum().detach().cpu()) - 1.0),
                        "illegal_probability": float(illegal.max().detach().cpu()) if illegal.numel() else 0.0,
                        "entropy": float((-(legal_probs * legal_probs.clamp_min(1e-30).log()).sum()).detach().cpu()),
                        "max_action_probability": max_probability,
                        "legal_action_count": len(legal),
                        "logit_values": [
                            float(value) for value in logits[row_index].detach().cpu().tolist()
                        ],
                        "collapse": max_probability > 0.999,
                    }
                )
                trajectory_log_pf[trajectory_ids[global_index]] += selected[row_index].detach().cpu().double()
    policy.train(policy_was_training)

    def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        return {
            "state_count": len(rows),
            "max_normalization_error": max(float(row["normalization_error"]) for row in rows),
            "max_illegal_probability": max(float(row["illegal_probability"]) for row in rows),
            "entropy": distribution_summary([float(row["entropy"]) for row in rows]),
            "max_action_probability": distribution_summary(
                [float(row["max_action_probability"]) for row in rows]
            ),
            "legal_action_count": distribution_summary(
                [float(row["legal_action_count"]) for row in rows]
            ),
            "logits": distribution_summary(
                [float(value) for row in rows for value in row["logit_values"]]
            ),
            "collapse_fraction": float(np.mean([bool(row["collapse"]) for row in rows])),
            "collapse_probability_threshold": 0.999,
        }

    by_depth = {
        str(depth): summarize_rows([row for row in policy_rows if int(row["depth"]) == depth])
        for depth in sorted(set(depths))
    }
    policy_summary = {**summarize_rows(policy_rows), "by_depth": by_depth}
    log_r = torch.tensor([trajectory.log_reward for trajectory in trajectories], dtype=torch.float64)
    log_pb = torch.tensor(
        [float(trajectory.log_pb_sum.detach().cpu()) for trajectory in trajectories], dtype=torch.float64
    )
    residual = residual_diagnostics(
        log_z=policy.log_z,
        log_pf=trajectory_log_pf,
        log_r=log_r,
        log_pb=log_pb,
    )
    raw = np.asarray([trajectory.comparable_return for trajectory in trajectories], dtype=np.float64)
    clipped = np.clip(raw, -reward_improvement_clip, reward_improvement_clip)
    rewards = np.asarray([trajectory.terminal_reward for trajectory in trajectories], dtype=np.float64)
    regression = regression_diagnostics(trajectory_log_pf, log_r + log_pb)
    return {
        "trajectory_count": len(trajectories),
        "transition_count": len(observations),
        "finite": finite and bool(residual["finite"]),
        "residual": residual,
        "regression": regression,
        "policy": policy_summary,
        "reward": {
            "raw_improvement": distribution_summary(raw.tolist()),
            "clipped_improvement": distribution_summary(clipped.tolist()),
            "reward_floor_rate": float(np.mean(rewards <= reward_eps * (1.0 + 1e-9))),
            "upper_clip_rate": float(np.mean(raw >= reward_improvement_clip)),
            "lower_clip_rate": float(np.mean(raw <= -reward_improvement_clip)),
        },
        "log_pf_values": trajectory_log_pf.tolist(),
        "log_pb_values": log_pb.tolist(),
        "log_r_values": log_r.tolist(),
    }


def _parameter_groups(policy: Any) -> tuple[list[torch.nn.Parameter], torch.nn.Parameter]:
    from src.algorithms.gflownet_tb.optim import policy_parameters_excluding_log_z

    return policy_parameters_excluding_log_z(policy), policy.log_z


def _gradient_ratio(values: Sequence[float]) -> float:
    if not values:
        return math.inf
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    p99 = float(np.quantile(array, 0.99))
    if median == 0.0:
        return 1.0 if p99 == 0.0 else 1e300
    return p99 / median


def _checkpoint_payload(
    *,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    archive: SequenceArchive,
    counters: Mapping[str, int],
    gradient_norms: Sequence[float],
    milestone_rows: Sequence[Mapping[str, Any]],
    train_action_generator: torch.Generator,
    circuit_rng: np.random.Generator,
    replay_rng: np.random.Generator,
    fixed_cache_checksum: str,
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    numerical_failure: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "replay": None,
        "archive": archive.state_dict(),
        "counters": dict(counters),
        "gradient_norms": list(gradient_norms),
        "milestone_rows": list(milestone_rows),
        "global_rng": _capture_global_rng(),
        "train_action_generator_state": train_action_generator.get_state(),
        "circuit_rng_state": circuit_rng.bit_generator.state,
        "replay_rng_state": replay_rng.bit_generator.state,
        "fixed_cache_checksum": fixed_cache_checksum,
        "resolved_config": dict(resolved_config),
        "run_metadata": dict(run_metadata),
        "git_commit": run_metadata.get("git_commit"),
        "source_tree_sha256": run_metadata.get("source_tree_sha256"),
        "numerical_failure": numerical_failure,
    }


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_checkpoint_payload(**kwargs), temporary)
    temporary.replace(path)


def _evaluation_milestone(
    *,
    policy: Any,
    circuit_path: Path,
    fixed_trajectories: Sequence[Any],
    trajectory_budget: int,
    update: int,
    device: torch.device,
    num_steps: int,
    reward_class: type,
    reward_alpha: float,
    reward_eps: float,
    reward_improvement_clip: float,
    available_actions: list[int] | None,
    fresh_seed: int,
    search_seed: int,
    archive: SequenceArchive,
    counters: dict[str, int],
    gradient_norms: Sequence[float],
    trajectories_path: Path,
) -> dict[str, Any]:
    from src.algorithms.gflownet_tb.sampler import sample_tb_trajectories

    training_rng_before = _capture_global_rng()
    policy_checksum_before = state_dict_checksum(_policy_state_cpu(policy))
    train_mode = policy.training
    try:
        policy.eval()
        with torch.no_grad():
            fresh_generator = _torch_generator(device, fresh_seed)
            fresh = sample_tb_trajectories(
                file_paths=[str(circuit_path)] * 128,
                num_steps=num_steps,
                policy=policy,
                reward_class=reward_class,
                reward_alpha=reward_alpha,
                reward_eps=reward_eps,
                reward_improvement_clip=reward_improvement_clip,
                sample_actions=True,
                available_actions=available_actions,
                epsilon_uniform=0.0,
                action_generator=fresh_generator,
            )
            search_generator = _torch_generator(device, search_seed)
            search = sample_tb_trajectories(
                file_paths=[str(circuit_path)] * 50,
                num_steps=num_steps,
                policy=policy,
                reward_class=reward_class,
                reward_alpha=reward_alpha,
                reward_eps=reward_eps,
                reward_improvement_clip=reward_improvement_clip,
                sample_actions=True,
                available_actions=available_actions,
                epsilon_uniform=0.0,
                action_generator=search_generator,
            )
        fixed_metrics = _score_trajectory_set(
            policy, fixed_trajectories,
            reward_eps=reward_eps,
            reward_improvement_clip=reward_improvement_clip,
        )
        fresh_metrics = _score_trajectory_set(
            policy, fresh,
            reward_eps=reward_eps,
            reward_improvement_clip=reward_improvement_clip,
        )
        search_records = _records_from_trajectories(search)
        search_metrics = nested_search_metrics(
            search_records,
            initial_size=archive.initial_size,
            initial_depth=archive.initial_depth,
        )
        for index, trajectory in enumerate(fresh, start=1):
            _append_jsonl(
                trajectories_path,
                _trajectory_row(
                    trajectory, source="fresh_on_policy", milestone=trajectory_budget, local_index=index
                ),
            )
        for index, trajectory in enumerate(search, start=1):
            _append_jsonl(
                trajectories_path,
                _trajectory_row(trajectory, source="search", milestone=trajectory_budget, local_index=index),
            )
        counters["validation_rollouts"] += len(fresh) + len(search)
        counters["validation_rescoring_presentations"] += sum(
            len(trajectory.steps) for trajectory in fixed_trajectories
        ) + sum(len(trajectory.steps) for trajectory in fresh)
        counters["validation_transitions"] += sum(len(trajectory.steps) for trajectory in fresh) + sum(
            len(trajectory.steps) for trajectory in search
        )
        finite = bool(fixed_metrics["finite"] and fresh_metrics["finite"])
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "trajectory_budget": int(trajectory_budget),
            "optimizer_update": int(update),
            "fixed_uniform": fixed_metrics,
            "fresh_on_policy": fresh_metrics,
            "search": search_metrics,
            "training_archive": archive.snapshot(),
            "optimizer_health": {
                "policy_gradient_p99_median_ratio": _gradient_ratio(gradient_norms),
                "gradient_clipping_enabled": False,
                "gradient_clipping_threshold": None,
                "gradient_clipping_rate": 0.0,
            },
            "counters": dict(counters),
            "finite": finite,
        }
    finally:
        policy.train(train_mode)
        policy_checksum_after = state_dict_checksum(_policy_state_cpu(policy))
        if policy_checksum_after != policy_checksum_before:
            raise RuntimeError("evaluation changed policy parameters")
        _restore_global_rng(training_rng_before)


def _write_milestone_tables(output_dir: Path, row: Mapping[str, Any]) -> None:
    budget = int(row["trajectory_budget"])
    table_dir = output_dir / "tables"
    validation_rows: list[dict[str, Any]] = []
    for stratum in ("fixed_uniform", "fresh_on_policy"):
        value = row[stratum]
        validation_rows.append(
            {
                "trajectory_budget": budget,
                "stratum": stratum,
                **value["residual"],
                **{f"regression_{key}": item for key, item in value["regression"].items()},
                "max_normalization_error": value["policy"]["max_normalization_error"],
                "max_illegal_probability": value["policy"]["max_illegal_probability"],
                "collapse_fraction": value["policy"]["collapse_fraction"],
            }
        )
    _write_csv(table_dir / f"trajectory_{budget}_validation.csv", validation_rows)
    _write_csv(table_dir / f"trajectory_{budget}_best_of_n.csv", row["search"]["budgets"])


def _resolved_configuration(args: argparse.Namespace, cfg: Any, circuit_path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    configured_log_z_lr = _tb_value(cfg, "log_z_learning_rate", None, preserve_none=True)
    policy_lr = float(cfg.learning_rate)
    resolved_log_z_lr = 10.0 * policy_lr if configured_log_z_lr is None else float(configured_log_z_lr)
    configured_batch_size = _tb_value(cfg, "batch_size", None, preserve_none=True)
    if configured_batch_size is None:
        configured_batch_size = _tb_value(cfg, "trajectories_per_episode", 4)
    trajectories_per_update = int(configured_batch_size)
    num_steps = int(cfg.num_steps)
    actions_raw = OmegaConf.select(cfg, "available_actions")
    actions = None if actions_raw is None else [int(action) for action in actions_raw]
    values = {
        "schema_version": RUN_SCHEMA_VERSION,
        "config_name": args.config_name,
        "circuit": circuit_path.stem,
        "circuit_path": str(circuit_path),
        "seed": int(args.seed),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(args.device),
        "max_trajectories": int(args.max_trajectories),
        "schedule_trajectories": int(args.schedule_trajectories),
        "configured_optimizer_updates": int(cfg.episodes),
        "milestones": [int(value) for value in args.milestones],
        "num_steps": num_steps,
        "available_actions": actions,
        "trajectories_per_update": trajectories_per_update,
        "policy_learning_rate": policy_lr,
        "log_z_learning_rate_configured": configured_log_z_lr,
        "log_z_learning_rate_resolved": resolved_log_z_lr,
        "reward_alpha": float(_tb_value(cfg, "reward_alpha", 4.0)),
        "reward_eps": float(_tb_value(cfg, "reward_eps", 1e-8)),
        "reward_improvement_clip": float(_tb_value(cfg, "reward_improvement_clip", 2.0)),
        "initial_log_z": 0.0,
        "exploration_epsilon_enabled": bool(_tb_value(cfg, "exploration_epsilon_enabled", True)),
        "exploration_epsilon_start": float(_tb_value(cfg, "exploration_epsilon_start", 0.5)),
        "exploration_epsilon_end": float(_tb_value(cfg, "exploration_epsilon_end", 0.01)),
        "exploration_warmup_updates": int(_tb_value(cfg, "exploration_warmup_episodes", 20)),
        "exploration_decay_updates": _tb_value(cfg, "exploration_decay_episodes", None, preserve_none=True),
        "gradient_clipping": None,
        "fixed_validation_trajectories": 256,
        "fresh_validation_trajectories": 128,
        "search_trajectories": 50,
        "search_budgets": [1, 2, 5, 10, 20, 50],
        "project_config": OmegaConf.to_container(cfg, resolve=True),
    }
    scientific = {
        key: values[key]
        for key in (
            "schema_version", "config_name", "num_steps", "available_actions",
            "trajectories_per_update", "policy_learning_rate",
            "log_z_learning_rate_configured", "log_z_learning_rate_resolved",
            "reward_alpha", "reward_eps", "reward_improvement_clip", "initial_log_z",
            "exploration_epsilon_enabled", "exploration_epsilon_start", "exploration_epsilon_end",
            "exploration_warmup_updates", "exploration_decay_updates", "schedule_trajectories",
            "configured_optimizer_updates",
            "gradient_clipping", "fixed_validation_trajectories",
            "fresh_validation_trajectories", "search_trajectories", "search_budgets",
        )
    }
    values["scientific_configuration"] = scientific
    values["scientific_configuration_fingerprint"] = canonical_sha256(scientific)
    return values


def _validate_active_config(resolved: Mapping[str, Any]) -> None:
    required = {
        "num_steps": 20,
        "available_actions": list(range(7)),
        "trajectories_per_update": 4,
        "policy_learning_rate": 0.001,
        "log_z_learning_rate_resolved": 0.01,
        "reward_alpha": 4.0,
        "reward_eps": 1e-8,
        "reward_improvement_clip": 2.0,
        "initial_log_z": 0.0,
        "exploration_epsilon_start": 0.5,
        "exploration_epsilon_end": 0.01,
        "exploration_warmup_updates": 20,
        "configured_optimizer_updates": 200,
    }
    failures = {
        key: {"expected": expected, "actual": resolved.get(key)}
        for key, expected in required.items()
        if resolved.get(key) != expected
    }
    if failures:
        raise ValueError(f"{resolved['config_name']} no longer matches the active baseline: {failures}")
    if not resolved["exploration_epsilon_enabled"]:
        raise ValueError("active epsilon schedule is disabled")
    if resolved["schedule_trajectories"] != 800:
        raise ValueError("the active epsilon schedule must retain an 800-trajectory horizon")
    if (
        int(resolved["configured_optimizer_updates"])
        * int(resolved["trajectories_per_update"])
        != int(resolved["schedule_trajectories"])
    ):
        raise ValueError("configured optimizer updates and batch size do not define 800 trajectories")


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

    circuit_path = _resolve_circuit(args.circuit)
    device = _resolve_device(args.device)
    cfg = _compose_project_config(args.config_name)
    cfg.output_dir = str(output_dir)
    resolved = _resolved_configuration(args, cfg, circuit_path)
    _validate_active_config(resolved)
    batch_size = int(resolved["trajectories_per_update"])
    if args.max_trajectories <= 0 or args.max_trajectories % batch_size:
        raise ValueError("--max-trajectories must be positive and divisible by four")
    milestones = sorted(set(int(value) for value in args.milestones))
    if not milestones or milestones[-1] > args.max_trajectories:
        raise ValueError("milestones must be non-empty and at most --max-trajectories")
    if any(value <= 0 or value % batch_size for value in milestones):
        raise ValueError("every milestone must be positive and divisible by four")
    if args.max_trajectories > args.schedule_trajectories:
        raise ValueError("max trajectories cannot exceed the epsilon schedule budget")

    reward_cfg = OmegaConf.to_container(cfg.reward, resolve=True)
    if not isinstance(reward_cfg, dict):
        raise TypeError("reward config must resolve to a mapping")
    reward_class = reward_class_factory(reward_cfg)
    obs_dim, num_actions, node_dim, edge_dim = get_obs_dim_and_num_actions(
        int(resolved["num_steps"]), str(circuit_path)
    )
    available_actions = normalize_available_actions(resolved["available_actions"], num_actions)

    seeds = {
        name: derive_seed(args.seed, name)
        for name in (
            "initialization", "training_actions", "circuit_selection", "replay",
            "evaluation_fixed_validation", "evaluation_fresh_policy", "evaluation_search",
        )
    }
    source_checksum = _source_tree_sha256(_repo_root())
    metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
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
    policy = build_tb_policy(
        cfg,
        obs_dim=obs_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_actions=num_actions,
        available_actions=available_actions,
    ).to(device)
    initial_checksum = state_dict_checksum(_policy_state_cpu(policy))
    metadata["initial_parameter_checksum"] = initial_checksum

    optimizer = build_tb_optimizer(
        policy,
        learning_rate=float(resolved["policy_learning_rate"]),
        log_z_learning_rate=float(resolved["log_z_learning_rate_resolved"]),
    )
    train_generator = _torch_generator(device, seeds["training_actions"])
    circuit_rng = np.random.default_rng(seeds["circuit_selection"])
    replay_rng = np.random.default_rng(seeds["replay"])

    counters = {
        "training_trajectories": 0,
        "training_transitions": 0,
        "training_presentations": 0,
        "optimizer_updates": 0,
        "validation_rollouts": 0,
        "validation_transitions": 0,
        "validation_rescoring_presentations": 0,
    }
    gradient_norms: list[float] = []
    milestone_rows: list[dict[str, Any]] = []
    archive: SequenceArchive | None = None
    fixed_cache_path = output_dir / "fixed_validation.pt"
    fixed_checksum = ""
    fixed_trajectories: list[Any]

    resume_path = args.resume_checkpoint.resolve() if args.resume_checkpoint else None
    if resume_path is None:
        for path in (metrics_path, milestones_path, trajectories_path):
            if path.exists() and path.stat().st_size:
                raise FileExistsError(f"refusing to overwrite existing run artifact: {path}")
            path.write_text("", encoding="utf-8")
        _write_json(output_dir / "resolved_config.json", resolved)
        with torch.no_grad():
            fixed_generator = _torch_generator(device, seeds["evaluation_fixed_validation"])
            fixed_trajectories = _detach_trajectories(
                sample_tb_trajectories(
                    file_paths=[str(circuit_path)] * 256,
                    num_steps=int(resolved["num_steps"]),
                    policy=policy,
                    reward_class=reward_class,
                    reward_alpha=float(resolved["reward_alpha"]),
                    reward_eps=float(resolved["reward_eps"]),
                    reward_improvement_clip=float(resolved["reward_improvement_clip"]),
                    sample_actions=True,
                    available_actions=available_actions,
                    epsilon_uniform=1.0,
                    action_generator=fixed_generator,
                )
            )
        torch.save(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "rng_seed": seeds["evaluation_fixed_validation"],
                "trajectories": fixed_trajectories,
            },
            fixed_cache_path,
        )
        fixed_checksum = _file_sha256(fixed_cache_path)
        metadata["fixed_validation_checksum"] = fixed_checksum
        _write_json(output_dir / "run_metadata.json", metadata)
        counters["validation_rollouts"] = len(fixed_trajectories)
        counters["validation_transitions"] = sum(
            len(trajectory.steps) for trajectory in fixed_trajectories
        )
        initial_metrics = _score_trajectory_set(
            policy,
            fixed_trajectories,
            reward_eps=float(resolved["reward_eps"]),
            reward_improvement_clip=float(resolved["reward_improvement_clip"]),
        )
        _append_jsonl(
            metrics_path,
            {
                "row_type": "initial_fixed_validation",
                "trajectory_budget": 0,
                "optimizer_update": 0,
                "fixed_uniform": initial_metrics,
                "wall_time_seconds": time.perf_counter() - started,
            },
        )
        for index, trajectory in enumerate(fixed_trajectories, start=1):
            _append_jsonl(
                trajectories_path,
                _trajectory_row(trajectory, source="fixed_uniform", milestone=0, local_index=index),
            )
        first = fixed_trajectories[0]
        archive = SequenceArchive(
            circuit=circuit_path.stem,
            initial_size=int(first.initial_size),
            initial_depth=int(first.initial_depth),
        )
    else:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        prior_resolved = checkpoint["resolved_config"]
        if prior_resolved["scientific_configuration_fingerprint"] != resolved["scientific_configuration_fingerprint"]:
            raise ValueError("resume checkpoint scientific configuration does not match")
        for key in ("circuit", "seed"):
            if prior_resolved[key] != resolved[key]:
                raise ValueError(f"resume checkpoint {key} does not match")
        if checkpoint.get("source_tree_sha256") != source_checksum:
            raise ValueError("resume checkpoint source tree does not match the running code")
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
        if _file_sha256(fixed_cache_path) != fixed_checksum:
            raise ValueError("fixed validation cache checksum mismatch on resume")
        fixed_payload = torch.load(fixed_cache_path, map_location="cpu", weights_only=False)
        fixed_trajectories = fixed_payload["trajectories"]
        metadata = checkpoint["run_metadata"]
        if state_dict_checksum(_policy_state_cpu(policy)) == initial_checksum and counters["optimizer_updates"]:
            raise RuntimeError("resumed trained policy unexpectedly equals initialization")

    assert archive is not None
    completed_milestones = {int(row["trajectory_budget"]) for row in milestone_rows}
    numerical_failure: str | None = None
    expected_updates = args.max_trajectories // batch_size

    try:
        while counters["optimizer_updates"] < expected_updates:
            update = counters["optimizer_updates"] + 1
            epsilon = _epsilon_for_update(
                update,
                schedule_updates=args.schedule_trajectories // batch_size,
                enabled=bool(resolved["exploration_epsilon_enabled"]),
                start=float(resolved["exploration_epsilon_start"]),
                end=float(resolved["exploration_epsilon_end"]),
                warmup_updates=int(resolved["exploration_warmup_updates"]),
                decay_updates=resolved["exploration_decay_updates"],
            )
            selected_paths = [
                str(circuit_path) for _ in range(batch_size)
                if int(circuit_rng.integers(0, 1)) == 0
            ]
            if len(selected_paths) != batch_size:
                raise RuntimeError("circuit selection stream returned invalid batch")
            policy.train()
            trajectories = sample_tb_trajectories(
                file_paths=selected_paths,
                num_steps=int(resolved["num_steps"]),
                policy=policy,
                reward_class=reward_class,
                reward_alpha=float(resolved["reward_alpha"]),
                reward_eps=float(resolved["reward_eps"]),
                reward_improvement_clip=float(resolved["reward_improvement_clip"]),
                sample_actions=True,
                available_actions=available_actions,
                epsilon_uniform=epsilon,
                action_generator=train_generator,
            )
            log_pf = torch.stack([trajectory.log_pf_sum for trajectory in trajectories])
            log_pb = torch.stack([trajectory.log_pb_sum for trajectory in trajectories])
            log_r = torch.tensor(
                [trajectory.log_reward for trajectory in trajectories], dtype=torch.float32, device=device
            )
            log_z_before = float(policy.log_z.detach().cpu())
            training_score = _score_trajectory_set(
                policy,
                trajectories,
                reward_eps=float(resolved["reward_eps"]),
                reward_improvement_clip=float(resolved["reward_improvement_clip"]),
            )
            train_residual = training_score["residual"]
            policy_parameters, log_z_parameter = _parameter_groups(policy)
            before = parameter_snapshot(policy_parameters)
            log_z_parameter_before = float(log_z_parameter.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss = trajectory_balance_loss(
                log_z=policy.log_z,
                log_pf_sums=log_pf,
                log_rewards=log_r,
                log_pb_sums=log_pb,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at update {update}")
            loss.backward()
            policy_grad_norm = parameter_gradient_norm(policy_parameters)
            log_z_grad_norm = parameter_gradient_norm([log_z_parameter])
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in [*policy_parameters, log_z_parameter]
            )
            if not gradients_finite:
                raise FloatingPointError(f"non-finite gradient at update {update}")
            optimizer.step()
            policy_update_norm = parameter_update_norm(before, policy_parameters)
            log_z_update = abs(float(log_z_parameter.detach().cpu()) - log_z_parameter_before)
            if not module_is_finite(policy) or not optimizer_is_finite(optimizer):
                raise FloatingPointError(f"non-finite parameters or optimizer state at update {update}")

            counters["optimizer_updates"] = update
            counters["training_trajectories"] += len(trajectories)
            transitions = sum(len(trajectory.steps) for trajectory in trajectories)
            counters["training_transitions"] += transitions
            counters["training_presentations"] += len(trajectories)
            gradient_norms.append(policy_grad_norm)
            for local_index, trajectory in enumerate(trajectories, start=1):
                actions = [int(step.action) for step in trajectory.steps]
                archive.record(
                    actions=actions,
                    initial_size=int(trajectory.initial_size),
                    initial_depth=int(trajectory.initial_depth),
                    final_size=int(trajectory.final_size),
                    final_depth=int(trajectory.final_depth),
                    comparable_return=float(trajectory.comparable_return),
                )
                _append_jsonl(
                    trajectories_path,
                    _trajectory_row(
                        trajectory,
                        source="training",
                        milestone=counters["training_trajectories"],
                        local_index=local_index,
                    ),
                )
            raw = [float(trajectory.comparable_return) for trajectory in trajectories]
            clipped = [
                max(-float(resolved["reward_improvement_clip"]), min(float(resolved["reward_improvement_clip"]), value))
                for value in raw
            ]
            row = {
                "row_type": "training_update",
                "optimizer_update": update,
                "trajectory_budget": counters["training_trajectories"],
                "epsilon_uniform": epsilon,
                "loss": float(loss.detach().cpu()),
                "residual": train_residual,
                "regression": training_score["regression"],
                "policy": training_score["policy"],
                "raw_improvement": distribution_summary(raw),
                "clipped_improvement": distribution_summary(clipped),
                "reward_floor_rate": float(np.mean([
                    trajectory.terminal_reward <= float(resolved["reward_eps"]) * (1.0 + 1e-9)
                    for trajectory in trajectories
                ])),
                "upper_reward_clip_rate": float(np.mean([
                    value >= float(resolved["reward_improvement_clip"]) for value in raw
                ])),
                "lower_reward_clip_rate": float(np.mean([
                    value <= -float(resolved["reward_improvement_clip"]) for value in raw
                ])),
                "policy_gradient_norm": policy_grad_norm,
                "log_z_gradient_norm": log_z_grad_norm,
                "policy_parameter_update_norm": policy_update_norm,
                "absolute_log_z_update": log_z_update,
                "policy_learning_rate": optimizer.param_groups[0]["lr"],
                "log_z_learning_rate": optimizer.param_groups[1]["lr"],
                "gradient_clipping_threshold": None,
                "gradient_clipping_occurred": False,
                "finite_loss": True,
                "finite_gradients": gradients_finite,
                "finite_parameters": module_is_finite(policy),
                "finite_optimizer_state": optimizer_is_finite(optimizer),
                "archive": archive.snapshot(),
                "counters": dict(counters),
                "wall_time_seconds": time.perf_counter() - started,
            }
            _append_jsonl(metrics_path, row)

            budget = counters["training_trajectories"]
            if budget in milestones and budget not in completed_milestones:
                training_action_state_before_evaluation = train_generator.get_state().clone()
                milestone_row = _evaluation_milestone(
                    policy=policy,
                    circuit_path=circuit_path,
                    fixed_trajectories=fixed_trajectories,
                    trajectory_budget=budget,
                    update=update,
                    device=device,
                    num_steps=int(resolved["num_steps"]),
                    reward_class=reward_class,
                    reward_alpha=float(resolved["reward_alpha"]),
                    reward_eps=float(resolved["reward_eps"]),
                    reward_improvement_clip=float(resolved["reward_improvement_clip"]),
                    available_actions=available_actions,
                    fresh_seed=seeds["evaluation_fresh_policy"],
                    search_seed=seeds["evaluation_search"],
                    archive=archive,
                    counters=counters,
                    gradient_norms=gradient_norms,
                    trajectories_path=trajectories_path,
                )
                if not torch.equal(
                    training_action_state_before_evaluation,
                    train_generator.get_state(),
                ):
                    raise RuntimeError("evaluation advanced the training-action generator")
                milestone_row["wall_time_seconds"] = time.perf_counter() - started
                milestone_rows.append(milestone_row)
                completed_milestones.add(budget)
                _append_jsonl(milestones_path, milestone_row)
                _append_jsonl(
                    metrics_path,
                    {
                        "row_type": "milestone",
                        "trajectory_budget": budget,
                        "optimizer_update": update,
                        "fixed_uniform": milestone_row["fixed_uniform"],
                        "fresh_on_policy": milestone_row["fresh_on_policy"],
                        "search": milestone_row["search"],
                        "training_archive": milestone_row["training_archive"],
                        "wall_time_seconds": milestone_row["wall_time_seconds"],
                    },
                )
                _write_milestone_tables(output_dir, milestone_row)
                _save_checkpoint(
                    checkpoints_dir / f"trajectory_{budget}.pt",
                    policy=policy,
                    optimizer=optimizer,
                    archive=archive,
                    counters=counters,
                    gradient_norms=gradient_norms,
                    milestone_rows=milestone_rows,
                    train_action_generator=train_generator,
                    circuit_rng=circuit_rng,
                    replay_rng=replay_rng,
                    fixed_cache_checksum=fixed_checksum,
                    resolved_config=resolved,
                    run_metadata=metadata,
                    numerical_failure=None,
                )

        missing = [milestone for milestone in milestones if milestone not in completed_milestones]
        if missing:
            raise RuntimeError(f"run completed without required milestones: {missing}")
    except Exception as exc:
        numerical_failure = f"{type(exc).__name__}: {exc}"
        emergency_path = checkpoints_dir / f"emergency_trajectory_{counters['training_trajectories']}.pt"
        _save_checkpoint(
            emergency_path,
            policy=policy,
            optimizer=optimizer,
            archive=archive,
            counters=counters,
            gradient_norms=gradient_norms,
            milestone_rows=milestone_rows,
            train_action_generator=train_generator,
            circuit_rng=circuit_rng,
            replay_rng=replay_rng,
            fixed_cache_checksum=fixed_checksum,
            resolved_config=resolved,
            run_metadata=metadata,
            numerical_failure=numerical_failure,
        )
        summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "complete": False,
            "numerical_failure": numerical_failure,
            "traceback": traceback.format_exc(),
            "emergency_checkpoint": str(emergency_path),
            "counters": counters,
            "milestones": milestone_rows,
            "scientific_configuration_fingerprint": resolved["scientific_configuration_fingerprint"],
            "initial_parameter_checksum": initial_checksum,
            "wall_time_seconds": time.perf_counter() - started,
        }
        _write_json(output_dir / "run_summary.json", summary)
        raise

    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "complete": True,
        "numerical_failure": None,
        "circuit": circuit_path.stem,
        "seed": int(args.seed),
        "scientific_configuration_fingerprint": resolved["scientific_configuration_fingerprint"],
        "initial_parameter_checksum": initial_checksum,
        "fixed_validation_checksum": fixed_checksum,
        "source_tree_sha256": source_checksum,
        "git_commit": metadata.get("git_commit"),
        "counters": counters,
        "milestones": milestone_rows,
        "final_archive": archive.snapshot(),
        "final_checkpoint": str(checkpoints_dir / f"trajectory_{args.max_trajectories}.pt"),
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


def _add_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("run", help="run one active-baseline circuit/seed")
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
    from src.experiments.tb_active_baseline_report import add_report_parser

    add_report_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"active-baseline {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
