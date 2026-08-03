"""Shared helpers for instrumented full-horizon TB experiments.

The active-baseline module remains the compatibility owner for its existing
artifact schema.  New experiments import the stable mechanics re-exported
here and add only experiment-specific state and decisions.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from src.experiments.tb_active_baseline import (  # re-exported compatibility layer
    _append_jsonl,
    _capture_global_rng,
    _compose_project_config,
    _detach_trajectories,
    _environment_versions,
    _epsilon_for_update,
    _evaluation_milestone,
    _file_sha256,
    _git_commit,
    _gradient_ratio,
    _json_safe,
    _optimizer_to_device,
    _parameter_groups,
    _policy_state_cpu,
    _records_from_trajectories,
    _repo_root,
    _resolve_circuit,
    _resolve_device,
    _restore_global_rng,
    _score_trajectory_set,
    _source_tree_sha256,
    _tb_value,
    _torch_generator,
    _trajectory_row,
    _write_csv,
    _write_json,
    _write_milestone_tables,
)


def differentiable_trajectory_scores(
    policy: Any,
    trajectories: Sequence[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rescore stored actions under the current policy with gradients enabled."""
    if not trajectories:
        raise ValueError("cannot score an empty trajectory set")
    values: list[torch.Tensor] = []
    for trajectory in trajectories:
        observations = [step.observation for step in trajectory.steps]
        legal_rows = [list(step.legal_actions) for step in trajectory.steps]
        actions = [int(step.action) for step in trajectory.steps]
        logits = policy(observations)
        selected = policy.log_prob_legal_batch(logits, legal_rows, actions)
        values.append(selected.sum())
    device = values[0].device
    log_pf = torch.stack(values)
    log_pb = torch.as_tensor(
        [float(trajectory.log_pb_sum.detach().cpu()) for trajectory in trajectories],
        dtype=log_pf.dtype,
        device=device,
    )
    log_r = torch.as_tensor(
        [float(trajectory.log_reward) for trajectory in trajectories],
        dtype=log_pf.dtype,
        device=device,
    )
    return log_pf, log_pb, log_r


def calibration_target(policy: Any, trajectories: Sequence[Any]) -> float:
    """Return mean(log R + log P_B - log P_F) at the current policy."""
    with torch.no_grad():
        log_pf, log_pb, log_r = differentiable_trajectory_scores(policy, trajectories)
        target = (log_r + log_pb - log_pf).mean()
    if not bool(torch.isfinite(target)):
        raise FloatingPointError("calibration target is non-finite")
    return float(target.detach().cpu())


def initialize_log_z(policy: Any, variant: str, target: float) -> float:
    """Apply the Experiment 3 initialization and return the assigned value."""
    normalized = variant.lower()
    if normalized not in {"z0", "zcal"}:
        raise ValueError(f"unknown logZ initialization variant: {variant}")
    value = 0.0 if normalized == "z0" else float(target)
    if not torch.isfinite(torch.tensor(value)):
        raise FloatingPointError("initial logZ is non-finite")
    with torch.no_grad():
        policy.log_z.fill_(value)
    return value


__all__ = [
    "calibration_target",
    "differentiable_trajectory_scores",
    "initialize_log_z",
]
