"""Train and compare the calibrated GFlowNet against existing baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

PROTOCOL_PATH = Path("cfg/exp/gfn_baseline_comparison/protocol.yaml")
EXPECTED_CIRCUITS = (
    "C1355",
    "C5315",
    "adder",
    "apex1",
    "bc0",
    "dalu",
    "k2",
    "max",
)
EXPECTED_BASELINES = ("reinforce", "drills", "ppo")
IDENTITY_COLUMNS = ["method", "circuit_name", "training_seed", "evaluation_seed", "sample_id"]
COMMIT_HASH = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
METRICS = ("hypervolume", "mean_product_improvement", "best_size_reduction", "best_depth_reduction")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class CircuitTask:
    circuit: str
    circuit_index: int
    dataset_cfg: str
    circuit_path: str

    @property
    def task_id(self) -> str:
        return f"gflownet__{self.circuit}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _select_path(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(dotted_path)
        current = current[component]
    return current


def _equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, bool) or isinstance(rhs, bool):
        return lhs is rhs
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=1e-12)
    return lhs == rhs


def _torch_load(path: Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return value


class ComparisonProtocol:
    def __init__(self, *, path: Path, data: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.repo_root = self.path.parents[3]
        self.data = dict(data)
        self.protocol_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._validate()

    @classmethod
    def load(cls, path: Path | str = PROTOCOL_PATH) -> ComparisonProtocol:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ProtocolError("protocol root must be a mapping")
        return cls(path=resolved, data=data)

    @property
    def circuits(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.data["circuits"])

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["common"]["training_seeds"])

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["common"]["evaluation_seeds"])

    @property
    def sample_budgets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["common"]["sample_budgets"])

    @property
    def max_samples(self) -> int:
        return int(self.data["common"]["max_samples_per_seed"])

    def task(self, circuit_index: int) -> CircuitTask:
        if circuit_index < 0 or circuit_index >= len(self.circuits):
            raise ProtocolError(f"circuit index must be in [0, {len(self.circuits) - 1}]")
        circuit = self.circuits[circuit_index]
        cfg = self.data["circuits"][circuit]
        return CircuitTask(
            circuit=circuit,
            circuit_index=int(circuit_index),
            dataset_cfg=str(cfg["dataset_cfg"]),
            circuit_path=str(cfg["circuit_path"]),
        )

    def tasks(self) -> list[CircuitTask]:
        return [self.task(index) for index in range(len(self.circuits))]

    def _validate(self) -> None:
        required = {
            "version", "campaign", "status", "source", "common", "gflownet",
            "expected_model", "circuits", "baseline_artifacts", "martin",
        }
        missing = required - set(self.data)
        if missing:
            raise ProtocolError(f"protocol missing keys: {sorted(missing)}")
        if int(self.data["version"]) != 1:
            raise ProtocolError("only protocol version 1 is supported")
        if self.circuits != EXPECTED_CIRCUITS:
            raise ProtocolError(f"circuits must be ordered as {EXPECTED_CIRCUITS}")
        if self.training_seeds != tuple(range(10)) or self.evaluation_seeds != tuple(range(10)):
            raise ProtocolError("training and evaluation seeds must both be 0 through 9")
        if self.sample_budgets != (10, 50, 100, 200) or self.max_samples != 200:
            raise ProtocolError("sample budgets must be [10, 50, 100, 200] with max 200")

        common = self.data["common"]
        gfn = self.data["gflownet"]
        expected = {
            "common.num_steps": (common.get("num_steps"), 20),
            "common.available_actions": (common.get("available_actions"), list(range(7))),
            "common.training_trajectories": (common.get("training_trajectories"), 800),
            "common.environment_transitions": (common.get("environment_transitions"), 16000),
            "gflownet.variant": (gfn.get("variant"), "zcal"),
            "gflownet.policy_learning_rate": (gfn.get("policy_learning_rate"), 0.001),
            "gflownet.log_z_learning_rate": (gfn.get("log_z_learning_rate"), 0.01),
            "gflownet.trajectories_per_update": (gfn.get("trajectories_per_update"), 4),
            "gflownet.optimizer_updates": (gfn.get("optimizer_updates"), 200),
            "gflownet.calibration_trajectories": (gfn.get("calibration_trajectories"), 64),
            "gflownet.subsequent_on_policy_trajectories": (
                gfn.get("subsequent_on_policy_trajectories"), 736,
            ),
            "gflownet.reward_alpha": (gfn.get("reward_alpha"), 4.0),
            "gflownet.exploration_epsilon_start": (gfn.get("exploration_epsilon_start"), 0.5),
            "gflownet.exploration_epsilon_end": (gfn.get("exploration_epsilon_end"), 0.01),
            "gflownet.exploration_warmup_updates": (gfn.get("exploration_warmup_updates"), 20),
            "gflownet.exploration_schedule_trajectories": (
                gfn.get("exploration_schedule_trajectories"), 800,
            ),
        }
        failures = {
            key: {"expected": wanted, "actual": actual}
            for key, (actual, wanted) in expected.items()
            if not _equal(actual, wanted)
        }
        if failures:
            raise ProtocolError(f"protocol changes the selected descriptive control: {failures}")
        if int(gfn["trajectories_per_update"]) * int(gfn["optimizer_updates"]) != 800:
            raise ProtocolError("GFlowNet update accounting must resolve to 800 trajectories")
        if int(gfn["calibration_trajectories"]) + int(gfn["subsequent_on_policy_trajectories"]) != 800:
            raise ProtocolError("calibration and subsequent trajectories must sum to 800")
        if tuple(self.data["baseline_artifacts"]) != EXPECTED_BASELINES:
            raise ProtocolError(f"baseline artifacts must be ordered as {EXPECTED_BASELINES}")
        if self.data["martin"].get("project") != "gflowcircuit-gfn-baseline-comparison":
            raise ProtocolError("unexpected Martin project")

        base_commit = str(self.data["source"].get("base_commit", ""))
        if not COMMIT_HASH.fullmatch(base_commit):
            raise ProtocolError("source.base_commit must be a full lowercase commit hash")
        for circuit, cfg in self.data["circuits"].items():
            for key in ("dataset_cfg", "circuit_path"):
                path = self.repo_root / str(cfg[key])
                if not path.is_file():
                    raise ProtocolError(f"missing {circuit} {key}: {path}")


def _resolved_health_config(protocol: ComparisonProtocol) -> DictConfig:
    from src.experiments.tb_active_baseline import _compose_project_config

    cfg = _compose_project_config(str(protocol.data["common"]["config_name"]))
    cfg.tb.log_z_learning_rate = float(protocol.data["gflownet"]["log_z_learning_rate"])
    cfg.tb.trajectories_per_episode = int(protocol.data["gflownet"]["trajectories_per_update"])
    # The health runner replaces this Hydra-runtime interpolation before it
    # resolves the config. Validation must do the same outside a Hydra job.
    cfg.output_dir = "/tmp/gfn-baseline-comparison-compose"
    return cfg


def _validate_project_config(protocol: ComparisonProtocol, config: Mapping[str, Any]) -> None:
    expected: dict[str, Any] = {
        **dict(protocol.data["expected_model"]),
        "num_steps": int(protocol.data["common"]["num_steps"]),
        "available_actions": list(protocol.data["common"]["available_actions"]),
        "learning_rate": float(protocol.data["gflownet"]["policy_learning_rate"]),
        "tb.trajectories_per_episode": int(protocol.data["gflownet"]["trajectories_per_update"]),
        "tb.log_z_learning_rate": float(protocol.data["gflownet"]["log_z_learning_rate"]),
        "tb.reward_alpha": float(protocol.data["gflownet"]["reward_alpha"]),
        "tb.reward_eps": float(protocol.data["gflownet"]["reward_eps"]),
        "tb.reward_improvement_clip": float(protocol.data["gflownet"]["reward_improvement_clip"]),
        "tb.exploration_epsilon_enabled": True,
        "tb.exploration_epsilon_start": float(protocol.data["gflownet"]["exploration_epsilon_start"]),
        "tb.exploration_epsilon_end": float(protocol.data["gflownet"]["exploration_epsilon_end"]),
        "tb.exploration_warmup_episodes": int(protocol.data["gflownet"]["exploration_warmup_updates"]),
    }
    failures: dict[str, dict[str, Any]] = {}
    for path, wanted in expected.items():
        try:
            actual = _select_path(config, path)
        except KeyError:
            failures[path] = {"expected": wanted, "actual": "<missing>"}
            continue
        if not _equal(actual, wanted):
            failures[path] = {"expected": wanted, "actual": actual}
    if failures:
        raise ValueError(f"resolved project configuration mismatch: {failures}")


def validate_protocol(protocol: ComparisonProtocol, *, compose: bool = True) -> dict[str, Any]:
    if compose:
        cfg = _resolved_health_config(protocol)
        value = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(value, Mapping):
            raise ProtocolError("resolved Hydra config is not a mapping")
        _validate_project_config(protocol, value)
    tasks = protocol.tasks()
    if len(tasks) != 8 or len({task.task_id for task in tasks}) != 8:
        raise ProtocolError("protocol must expand to eight unique circuit tasks")
    return {
        "campaign": protocol.data["campaign"],
        "status": protocol.data["status"],
        "caveat": protocol.data["caveat"],
        "protocol_hash": protocol.protocol_hash,
        "circuits": len(tasks),
        "tasks": len(tasks),
        "trained_models": len(tasks) * len(protocol.training_seeds),
        "training_trajectories_per_model": int(protocol.data["common"]["training_trajectories"]),
        "unique_evaluation_rollouts": (
            len(tasks) * len(protocol.training_seeds) * len(protocol.evaluation_seeds) * protocol.max_samples
        ),
        "sample_budgets": list(protocol.sample_budgets),
    }


def _next_attempt_dir(task_root: Path) -> Path:
    numbers: list[int] = []
    for path in task_root.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return task_root / f"attempt_{max(numbers, default=0) + 1:03d}"


def _training_command(
    protocol: ComparisonProtocol,
    task: CircuitTask,
    *,
    seed: int,
    output_dir: Path,
    python_executable: str,
    device_name: str,
) -> list[str]:
    gfn = protocol.data["gflownet"]
    return [
        python_executable,
        "-m",
        "src.experiments.tb_logz_calibration",
        "run",
        "--variant",
        str(gfn["variant"]),
        "--config-name",
        str(protocol.data["common"]["config_name"]),
        "--circuit",
        str((protocol.repo_root / task.circuit_path).resolve()),
        "--seed",
        str(int(seed)),
        "--output-dir",
        str(output_dir),
        "--max-trajectories",
        str(int(protocol.data["common"]["training_trajectories"])),
        "--schedule-trajectories",
        str(int(gfn["exploration_schedule_trajectories"])),
        "--milestones",
        *[str(int(value)) for value in gfn["milestones"]],
        "--device",
        device_name,
    ]


def _run_command(command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False)
    return int(completed.returncode)


def _validate_health_run(
    protocol: ComparisonProtocol,
    task: CircuitTask,
    *,
    seed: int,
    run_dir: Path,
) -> Path:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"run summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "complete": True,
        "numerical_failure": None,
        "variant": "zcal",
        "circuit": task.circuit,
        "seed": int(seed),
    }
    for key, wanted in expected_summary.items():
        if summary.get(key) != wanted:
            raise ValueError(f"{task.circuit}/seed {seed}: expected {key}={wanted!r}, got {summary.get(key)!r}")
    counters = summary.get("counters")
    if not isinstance(counters, Mapping):
        raise TypeError(f"{task.circuit}/seed {seed}: counters are missing")
    expected_counters = {
        "training_trajectories": 800,
        "training_transitions": 16000,
        "training_presentations": 800,
        "optimizer_updates": 200,
        "calibration_trajectories": 64,
        "calibration_target_presentations": 64,
        "calibration_training_presentations": 64,
        "new_training_trajectories": 736,
    }
    failures = {
        key: {"expected": wanted, "actual": counters.get(key)}
        for key, wanted in expected_counters.items()
        if int(counters.get(key, -1)) != wanted
    }
    if failures:
        raise ValueError(f"{task.circuit}/seed {seed}: trajectory accounting mismatch: {failures}")

    checkpoint_path = Path(str(summary.get("final_checkpoint", "")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    if not checkpoint_path.is_file():
        fallback = run_dir / "checkpoints" / "trajectory_800.pt"
        if not fallback.is_file():
            raise ValueError(f"final checkpoint is missing: {checkpoint_path}")
        checkpoint_path = fallback
    checkpoint = _torch_load(checkpoint_path)
    if checkpoint.get("numerical_failure") is not None:
        raise ValueError(f"checkpoint records numerical failure: {checkpoint.get('numerical_failure')}")
    if checkpoint.get("variant") != "zcal":
        raise ValueError("checkpoint does not use calibrated logZ initialization")
    checkpoint_counters = checkpoint.get("counters")
    if not isinstance(checkpoint_counters, Mapping):
        raise TypeError("checkpoint counters are missing")
    for key, wanted in expected_counters.items():
        if int(checkpoint_counters.get(key, -1)) != wanted:
            raise ValueError(f"checkpoint counter {key} is not {wanted}")
    if checkpoint.get("calibration_phase_complete") is not True:
        raise ValueError("checkpoint calibration phase is incomplete")
    archive = checkpoint.get("archive")
    if not isinstance(archive, Mapping) or not isinstance(archive.get("records"), list):
        raise TypeError("checkpoint training archive is missing")
    if len(archive["records"]) != int(protocol.data["common"]["training_trajectories"]):
        raise ValueError("checkpoint training archive does not contain 800 trajectories")
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise TypeError("checkpoint resolved configuration is missing")
    expected_resolved = {
        "trajectories_per_update": 4,
        "policy_learning_rate": 0.001,
        "log_z_learning_rate_resolved": 0.01,
        "reward_alpha": 4.0,
        "exploration_epsilon_start": 0.5,
        "exploration_epsilon_end": 0.01,
        "exploration_warmup_updates": 20,
        "schedule_trajectories": 800,
        "calibration_trajectories": 64,
    }
    for key, wanted in expected_resolved.items():
        if not _equal(resolved.get(key), wanted):
            raise ValueError(f"checkpoint config {key}: expected {wanted!r}, got {resolved.get(key)!r}")
    project_config = resolved.get("project_config")
    if not isinstance(project_config, Mapping):
        raise TypeError("checkpoint project configuration is missing")
    _validate_project_config(protocol, project_config)
    policy_state = checkpoint.get("policy")
    if not isinstance(policy_state, Mapping) or not policy_state:
        raise ValueError("checkpoint policy state is missing")
    if any(isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()) for value in policy_state.values()):
        raise ValueError("checkpoint contains non-finite policy parameters")
    return checkpoint_path.resolve()


def _load_health_policy(checkpoint_path: Path, *, circuit_path: Path, device: torch.device) -> dict[str, Any]:
    from src.algorithms.gflownet_tb.factory import build_tb_policy
    from src.models import reward_class_factory
    from src.utils import get_obs_dim_and_num_actions, normalize_available_actions

    checkpoint = _torch_load(checkpoint_path, map_location=device)
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, Mapping) or not isinstance(resolved.get("project_config"), Mapping):
        raise TypeError(f"checkpoint has no resolved project config: {checkpoint_path}")
    cfg = OmegaConf.create(resolved["project_config"])
    obs_dim, num_actions, node_dim, edge_dim = get_obs_dim_and_num_actions(
        int(cfg.num_steps), str(circuit_path)
    )
    available_actions = normalize_available_actions(OmegaConf.select(cfg, "available_actions"), num_actions)
    policy = build_tb_policy(
        cfg,
        obs_dim=obs_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_actions=num_actions,
        available_actions=available_actions,
    ).to(device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    reward_cfg = OmegaConf.to_container(cfg.reward, resolve=True)
    if not isinstance(reward_cfg, dict):
        raise TypeError("checkpoint reward config is not a mapping")
    return {
        "policy": policy,
        "cfg": cfg,
        "reward_class": reward_class_factory(reward_cfg),
        "available_actions": available_actions,
        "reward_alpha": float(resolved["reward_alpha"]),
        "reward_eps": float(resolved["reward_eps"]),
        "reward_improvement_clip": float(resolved["reward_improvement_clip"]),
    }


def sample_health_checkpoint(
    *,
    checkpoint_path: Path,
    circuit_path: Path,
    num_samples: int,
    evaluation_seed: int,
    device: torch.device,
) -> list[dict[str, int]]:
    from src.algorithms.gflownet_tb.sampler import sample_tb_trajectories

    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    loaded = _load_health_policy(checkpoint_path, circuit_path=circuit_path, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(evaluation_seed))
    cfg = loaded["cfg"]
    with torch.no_grad():
        trajectories = sample_tb_trajectories(
            file_paths=[str(circuit_path)] * int(num_samples),
            num_steps=int(cfg.num_steps),
            policy=loaded["policy"],
            reward_class=loaded["reward_class"],
            reward_alpha=float(loaded["reward_alpha"]),
            reward_eps=float(loaded["reward_eps"]),
            reward_improvement_clip=float(loaded["reward_improvement_clip"]),
            sample_actions=True,
            available_actions=loaded["available_actions"],
            epsilon_uniform=0.0,
            action_generator=generator,
        )
    return [{"size": int(item.final_size), "depth": int(item.final_depth)} for item in trajectories]


def _reference_row(task: CircuitTask, *, circuit_path: Path, num_steps: int) -> dict[str, Any]:
    import pyspiel

    from src.baselines.resyn2 import get_depth, get_size
    from src.utils import Observation

    game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": str(circuit_path)})
    observation = Observation.from_state(game.new_initial_state())
    return {
        "method": "gflownet",
        "circuit_name": task.circuit,
        "circuit": str(circuit_path),
        "run_id": None,
        "training_seed": None,
        "evaluation_seed": None,
        "sample_id": None,
        "size": int(get_size(observation)),
        "depth": int(get_depth(observation)),
        "source_checkpoint": None,
    }


def _evaluate_task(
    protocol: ComparisonProtocol,
    task: CircuitTask,
    *,
    attempt_dir: Path,
    checkpoints: Mapping[int, Path],
    device: torch.device,
) -> None:
    circuit_path = (protocol.repo_root / task.circuit_path).resolve()
    raw_dir = attempt_dir / "evaluation" / "raw"
    budget_dir = attempt_dir / "evaluation" / "budgets"
    raw_frames: list[pd.DataFrame] = []
    for evaluation_seed in protocol.evaluation_seeds:
        rows: list[dict[str, Any]] = []
        for run_id, training_seed in enumerate(protocol.training_seeds):
            checkpoint = checkpoints[training_seed]
            sampled = sample_health_checkpoint(
                checkpoint_path=checkpoint,
                circuit_path=circuit_path,
                num_samples=protocol.max_samples,
                evaluation_seed=evaluation_seed,
                device=device,
            )
            for sample_id, values in enumerate(sampled):
                rows.append({
                    "method": "gflownet",
                    "circuit_name": task.circuit,
                    "circuit": str(circuit_path),
                    "run_id": run_id,
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "sample_id": sample_id,
                    **values,
                    "source_checkpoint": str(checkpoint),
                })
        frame = pd.DataFrame(rows)
        _write_csv(raw_dir / f"seed_{evaluation_seed:02d}.csv", frame)
        raw_frames.append(frame)
    raw = pd.concat(raw_frames, ignore_index=True)
    reference = _reference_row(
        task,
        circuit_path=circuit_path,
        num_steps=int(protocol.data["common"]["num_steps"]),
    )
    _write_csv(attempt_dir / "points.csv", pd.concat([pd.DataFrame([reference]), raw], ignore_index=True))
    for budget in protocol.sample_budgets:
        view = raw.loc[raw["sample_id"] < budget].copy()
        view["sample_budget"] = budget
        budget_reference = {**reference, "sample_budget": budget}
        _write_csv(
            budget_dir / f"points_n{budget:03d}.csv",
            pd.concat([pd.DataFrame([budget_reference]), view], ignore_index=True),
        )


def validate_attempt(
    protocol: ComparisonProtocol,
    task: CircuitTask,
    *,
    attempt_dir: Path,
) -> dict[str, int]:
    checkpoints = {
        seed: _validate_health_run(
            protocol,
            task,
            seed=seed,
            run_dir=attempt_dir / "training" / f"seed_{seed:02d}",
        )
        for seed in protocol.training_seeds
    }
    raw_frames: list[pd.DataFrame] = []
    for evaluation_seed in protocol.evaluation_seeds:
        path = attempt_dir / "evaluation" / "raw" / f"seed_{evaluation_seed:02d}.csv"
        if not path.is_file():
            raise ValueError(f"raw evaluation file is missing: {path}")
        frame = pd.read_csv(path)
        if len(frame) != len(protocol.training_seeds) * protocol.max_samples:
            raise ValueError(f"unexpected row count in {path}")
        if set(frame["evaluation_seed"].astype(int)) != {evaluation_seed}:
            raise ValueError(f"wrong evaluation seed in {path}")
        for training_seed in protocol.training_seeds:
            group = frame.loc[frame["training_seed"].astype(int) == training_seed]
            if list(group["sample_id"].astype(int)) != list(range(protocol.max_samples)):
                raise ValueError(f"non-contiguous samples for training seed {training_seed} in {path}")
            if set(group["source_checkpoint"].astype(str)) != {str(checkpoints[training_seed])}:
                raise ValueError(f"wrong checkpoint for training seed {training_seed} in {path}")
        raw_frames.append(frame)
    raw = pd.concat(raw_frames, ignore_index=True)
    expected_samples = len(protocol.training_seeds) * len(protocol.evaluation_seeds) * protocol.max_samples
    if len(raw) != expected_samples or raw.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("raw evaluation identities are incomplete or duplicated")
    canonical = pd.read_csv(attempt_dir / "points.csv")
    if len(canonical.loc[canonical["run_id"].isna()]) != 1 or len(canonical) != expected_samples + 1:
        raise ValueError("canonical points.csv has the wrong reference/sample count")
    raw_payload = raw.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    canonical_payload = canonical.loc[canonical["run_id"].notna()].sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    compare_columns = [*IDENTITY_COLUMNS, "run_id", "size", "depth", "source_checkpoint"]
    if not raw_payload[compare_columns].equals(canonical_payload[compare_columns]):
        raise ValueError("canonical points.csv does not match raw evaluation")
    for budget in protocol.sample_budgets:
        path = attempt_dir / "evaluation" / "budgets" / f"points_n{budget:03d}.csv"
        frame = pd.read_csv(path)
        samples = frame.loc[frame["run_id"].notna()].sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
        expected = raw.loc[raw["sample_id"] < budget].sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
        if len(frame.loc[frame["run_id"].isna()]) != 1 or len(samples) != len(expected):
            raise ValueError(f"wrong reference/sample count in {path}")
        if not samples[compare_columns].equals(expected[compare_columns]):
            raise ValueError(f"{path} is not an exact nested prefix")
        if set(frame["sample_budget"].dropna().astype(int)) != {budget}:
            raise ValueError(f"wrong sample budget in {path}")
    return {
        "checkpoint_count": len(checkpoints),
        "evaluation_seed_count": len(protocol.evaluation_seeds),
        "unique_sample_count": expected_samples,
    }


def run_task(
    protocol: ComparisonProtocol,
    *,
    circuit_index: int,
    artifact_root: Path,
    python_executable: str,
    device_name: str,
    project_commit: str | None = None,
) -> dict[str, Any]:
    from src.experiments.tb_active_baseline import _source_tree_sha256

    if project_commit is not None and not COMMIT_HASH.fullmatch(project_commit):
        raise ProtocolError("project commit must be a full lowercase commit hash")
    task = protocol.task(circuit_index)
    task_root = artifact_root.resolve() / "tasks" / task.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(task_root)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    status_path = task_root / "task_status.json"
    metadata: dict[str, Any] = {
        "task_id": task.task_id,
        "method": "gflownet",
        "circuit": task.circuit,
        "circuit_index": task.circuit_index,
        "state": "running",
        "attempt_dir": str(attempt_dir),
        "protocol_hash": protocol.protocol_hash,
        "source_base_commit": protocol.data["source"]["base_commit"],
        "source_tree_sha256": _source_tree_sha256(protocol.repo_root),
        "project_commit": project_commit,
        "device": device_name,
        "started_at": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    _write_json(attempt_dir / "attempt.json", metadata)
    _write_json(status_path, metadata)
    started = time.monotonic()
    error: str | None = None
    checkpoints: dict[int, Path] = {}
    try:
        for seed in protocol.training_seeds:
            run_dir = attempt_dir / "training" / f"seed_{seed:02d}"
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            command = _training_command(
                protocol,
                task,
                seed=seed,
                output_dir=run_dir,
                python_executable=python_executable,
                device_name=device_name,
            )
            seed_started = time.monotonic()
            exit_code = _run_command(
                command,
                cwd=protocol.repo_root,
                stdout_path=attempt_dir / f"train_seed_{seed:02d}.stdout.log",
                stderr_path=attempt_dir / f"train_seed_{seed:02d}.stderr.log",
            )
            metadata.setdefault("training", {})[str(seed)] = {
                "command": command,
                "exit_code": exit_code,
                "wall_time_seconds": time.monotonic() - seed_started,
            }
            _write_json(attempt_dir / "attempt.json", metadata)
            if exit_code != 0:
                raise ValueError(f"training seed {seed} exited with code {exit_code}")
            checkpoints[seed] = _validate_health_run(protocol, task, seed=seed, run_dir=run_dir)
        sample_started = time.monotonic()
        _evaluate_task(protocol, task, attempt_dir=attempt_dir, checkpoints=checkpoints, device=torch.device(device_name))
        metadata["sample_wall_time_seconds"] = time.monotonic() - sample_started
        metadata.update(validate_attempt(protocol, task, attempt_dir=attempt_dir))
        metadata["state"] = "complete"
    except Exception as exc:  # noqa: BLE001 - task metadata must record every failure
        error = str(exc)
        metadata["state"] = "failed"
        metadata["error"] = error
    metadata["finished_at"] = utc_now()
    metadata["wall_time_seconds"] = time.monotonic() - started
    _write_json(attempt_dir / "attempt.json", metadata)
    _write_json(status_path, metadata)
    if error is not None:
        raise RuntimeError(f"{task.task_id}: {error}")
    return metadata


def strict_pareto_front(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted({
        (float(x), float(y))
        for x, y in points
        if math.isfinite(float(x)) and math.isfinite(float(y)) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    })
    return [
        point for point in unique
        if not any(
            other != point and other[0] <= point[0] and other[1] <= point[1]
            for other in unique
        )
    ]


def strict_hypervolume(points: Iterable[tuple[float, float]]) -> float:
    front = strict_pareto_front(points)
    if not front:
        return 0.0
    area = 0.0
    best_y = 1.0
    for index, (x, y) in enumerate(front):
        best_y = min(best_y, y)
        next_x = front[index + 1][0] if index + 1 < len(front) else 1.0
        area += max(0.0, next_x - x) * max(0.0, 1.0 - best_y)
    return float(area)


def _metric_record(
    samples: pd.DataFrame,
    *,
    initial_size: float,
    initial_depth: float,
) -> dict[str, float]:
    normalized_size = samples["size"].astype(float).to_numpy() / initial_size
    normalized_depth = samples["depth"].astype(float).to_numpy() / initial_depth
    points = list(zip(normalized_size.tolist(), normalized_depth.tolist()))
    product_improvement = 1.0 - normalized_size * normalized_depth
    return {
        "hypervolume": strict_hypervolume(points),
        "mean_product_improvement": float(np.mean(product_improvement)),
        "best_size_reduction": float(np.max(1.0 - normalized_size)),
        "best_depth_reduction": float(np.max(1.0 - normalized_depth)),
        "distinct_endpoints": float(len(set(zip(samples["size"].astype(int), samples["depth"].astype(int))))),
    }


def _validate_comparison_frame(
    frame: pd.DataFrame,
    *,
    method: str,
    circuit: str,
    budget: int,
    protocol: ComparisonProtocol,
) -> tuple[pd.DataFrame, float, float]:
    references = frame.loc[frame["run_id"].isna()]
    samples = frame.loc[frame["run_id"].notna()].copy()
    expected = len(protocol.training_seeds) * len(protocol.evaluation_seeds) * budget
    if len(references) != 1 or len(samples) != expected:
        raise ValueError(
            f"{method}/{circuit}/N={budget}: references={len(references)}, samples={len(samples)}, expected={expected}"
        )
    if set(samples["training_seed"].astype(int)) != set(protocol.training_seeds):
        raise ValueError(f"{method}/{circuit}/N={budget}: wrong training seeds")
    if set(samples["evaluation_seed"].astype(int)) != set(protocol.evaluation_seeds):
        raise ValueError(f"{method}/{circuit}/N={budget}: wrong evaluation seeds")
    if samples.duplicated(["training_seed", "evaluation_seed", "sample_id"]).any():
        raise ValueError(f"{method}/{circuit}/N={budget}: duplicate sample identities")
    for (_, _), group in samples.groupby(["training_seed", "evaluation_seed"], sort=True):
        if list(group.sort_values("sample_id")["sample_id"].astype(int)) != list(range(budget)):
            raise ValueError(f"{method}/{circuit}/N={budget}: samples are not an exact 0..N-1 prefix")
    initial_size = float(references.iloc[0]["size"])
    initial_depth = float(references.iloc[0]["depth"])
    if initial_size <= 0 or initial_depth <= 0:
        raise ValueError(f"{method}/{circuit}: invalid reference dimensions")
    return samples, initial_size, initial_depth


def _complete_attempt(root: Path, *, method: str, circuit: str) -> Path:
    task_root = root / "tasks" / f"{method}__{circuit}"
    candidates: list[tuple[int, Path]] = []
    for attempt in task_root.glob("attempt_*"):
        metadata_path = attempt / "attempt.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("state") == "complete":
            try:
                number = int(attempt.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((number, attempt))
    if not candidates:
        raise ValueError(f"no complete artifact for {method}/{circuit} below {root}")
    return max(candidates)[1]


def _bootstrap_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def hierarchical_bootstrap_ci(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("hierarchical bootstrap values must be a non-empty 2-D matrix")
    rng = np.random.default_rng(int(seed))
    train_count, evaluation_count = matrix.shape
    train_indices = rng.integers(0, train_count, size=(repetitions, train_count))
    evaluation_indices = rng.integers(
        0, evaluation_count, size=(repetitions, train_count, evaluation_count)
    )
    sampled = matrix[train_indices[:, :, None], evaluation_indices]
    means = sampled.mean(axis=(1, 2))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _matrix(rows: pd.DataFrame, metric: str, protocol: ComparisonProtocol) -> np.ndarray:
    pivot = rows.pivot(index="training_seed", columns="evaluation_seed", values=metric)
    pivot = pivot.reindex(index=protocol.training_seeds, columns=protocol.evaluation_seeds)
    if pivot.isna().any().any():
        raise ValueError(f"incomplete metric matrix for {metric}")
    return pivot.to_numpy(dtype=float)


def _pooled_front_rows(
    samples: pd.DataFrame,
    *,
    method: str,
    circuit: str,
    budget: int,
    initial_size: float,
    initial_depth: float,
) -> list[dict[str, Any]]:
    raw_points = [
        (float(row.size) / initial_size, float(row.depth) / initial_depth)
        for row in samples[["size", "depth"]].itertuples(index=False)
    ]
    return [
        {
            "method": method,
            "circuit": circuit,
            "sample_budget": budget,
            "normalized_size": x,
            "normalized_depth": y,
        }
        for x, y in strict_pareto_front(raw_points)
    ]


def compare_artifacts(
    protocol: ComparisonProtocol,
    *,
    gfn_root: Path,
    baseline_roots: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    if tuple(baseline_roots) != EXPECTED_BASELINES:
        raise ProtocolError(f"baseline roots must be ordered as {EXPECTED_BASELINES}")
    method_roots = {"gflownet": gfn_root.resolve(), **{key: value.resolve() for key, value in baseline_roots.items()}}
    metric_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    pooled_hv: dict[tuple[str, str, int], float] = {}
    artifact_manifest: dict[str, dict[str, str]] = {method: {} for method in method_roots}

    for method, root in method_roots.items():
        for circuit in protocol.circuits:
            attempt = _complete_attempt(root, method=method, circuit=circuit)
            artifact_manifest[method][circuit] = str(attempt)
            for budget in protocol.sample_budgets:
                frame = pd.read_csv(attempt / "evaluation" / "budgets" / f"points_n{budget:03d}.csv")
                samples, initial_size, initial_depth = _validate_comparison_frame(
                    frame,
                    method=method,
                    circuit=circuit,
                    budget=budget,
                    protocol=protocol,
                )
                for (training_seed, evaluation_seed), group in samples.groupby(
                    ["training_seed", "evaluation_seed"], sort=True
                ):
                    metric_rows.append({
                        "method": method,
                        "circuit": circuit,
                        "sample_budget": budget,
                        "training_seed": int(training_seed),
                        "evaluation_seed": int(evaluation_seed),
                        **_metric_record(group, initial_size=initial_size, initial_depth=initial_depth),
                    })
                front = _pooled_front_rows(
                    samples,
                    method=method,
                    circuit=circuit,
                    budget=budget,
                    initial_size=initial_size,
                    initial_depth=initial_depth,
                )
                pooled_rows.extend(front)
                pooled_hv[(method, circuit, budget)] = strict_hypervolume(
                    (row["normalized_size"], row["normalized_depth"]) for row in front
                )

    raw_metrics = pd.DataFrame(metric_rows)
    front_frame = pd.DataFrame(pooled_rows)
    repetitions = int(protocol.data["common"]["bootstrap_repetitions"])
    base_seed = int(protocol.data["common"]["bootstrap_seed"])
    summary_rows: list[dict[str, Any]] = []

    for method in method_roots:
        for budget in protocol.sample_budgets:
            circuit_groups: list[pd.DataFrame] = []
            for circuit in protocol.circuits:
                group = raw_metrics.loc[
                    (raw_metrics["method"] == method)
                    & (raw_metrics["circuit"] == circuit)
                    & (raw_metrics["sample_budget"] == budget)
                ]
                circuit_groups.append(group)
                row: dict[str, Any] = {
                    "method": method,
                    "circuit": circuit,
                    "sample_budget": budget,
                    "pooled_hypervolume": pooled_hv[(method, circuit, budget)],
                }
                for metric in METRICS:
                    matrix = _matrix(group, metric, protocol)
                    low, high = hierarchical_bootstrap_ci(
                        matrix,
                        repetitions=repetitions,
                        seed=_bootstrap_seed(base_seed, "summary", method, circuit, budget, metric),
                    )
                    row[f"mean_{metric}"] = float(matrix.mean())
                    row[f"std_{metric}"] = float(matrix.std(ddof=0))
                    row[f"ci_low_{metric}"] = low
                    row[f"ci_high_{metric}"] = high
                summary_rows.append(row)

            averaged = pd.concat(circuit_groups).groupby(
                ["training_seed", "evaluation_seed"], as_index=False
            )[list(METRICS)].mean()
            overall: dict[str, Any] = {
                "method": method,
                "circuit": "__mean__",
                "sample_budget": budget,
                "pooled_hypervolume": float(np.mean([
                    pooled_hv[(method, circuit, budget)] for circuit in protocol.circuits
                ])),
            }
            for metric in METRICS:
                matrix = _matrix(averaged, metric, protocol)
                low, high = hierarchical_bootstrap_ci(
                    matrix,
                    repetitions=repetitions,
                    seed=_bootstrap_seed(base_seed, "summary", method, "__mean__", budget, metric),
                )
                overall[f"mean_{metric}"] = float(matrix.mean())
                overall[f"std_{metric}"] = float(matrix.std(ddof=0))
                overall[f"ci_low_{metric}"] = low
                overall[f"ci_high_{metric}"] = high
            summary_rows.append(overall)

    summary = pd.DataFrame(summary_rows)
    pairwise_rows: list[dict[str, Any]] = []
    for baseline in EXPECTED_BASELINES:
        for budget in protocol.sample_budgets:
            circuit_differences: list[pd.DataFrame] = []
            for circuit in protocol.circuits:
                gfn = raw_metrics.loc[
                    (raw_metrics["method"] == "gflownet")
                    & (raw_metrics["circuit"] == circuit)
                    & (raw_metrics["sample_budget"] == budget)
                ]
                other = raw_metrics.loc[
                    (raw_metrics["method"] == baseline)
                    & (raw_metrics["circuit"] == circuit)
                    & (raw_metrics["sample_budget"] == budget)
                ]
                paired = gfn.merge(
                    other,
                    on=["circuit", "sample_budget", "training_seed", "evaluation_seed"],
                    suffixes=("_gfn", "_baseline"),
                    validate="one_to_one",
                )
                differences = paired[["training_seed", "evaluation_seed"]].copy()
                row: dict[str, Any] = {
                    "baseline": baseline,
                    "circuit": circuit,
                    "sample_budget": budget,
                }
                for metric in METRICS:
                    differences[metric] = paired[f"{metric}_gfn"] - paired[f"{metric}_baseline"]
                    matrix = _matrix(differences, metric, protocol)
                    low, high = hierarchical_bootstrap_ci(
                        matrix,
                        repetitions=repetitions,
                        seed=_bootstrap_seed(base_seed, "pair", baseline, circuit, budget, metric),
                    )
                    row[f"mean_difference_{metric}"] = float(matrix.mean())
                    row[f"ci_low_difference_{metric}"] = low
                    row[f"ci_high_difference_{metric}"] = high
                pairwise_rows.append(row)
                circuit_differences.append(differences)

            averaged = pd.concat(circuit_differences).groupby(
                ["training_seed", "evaluation_seed"], as_index=False
            )[list(METRICS)].mean()
            overall = {"baseline": baseline, "circuit": "__mean__", "sample_budget": budget}
            for metric in METRICS:
                matrix = _matrix(averaged, metric, protocol)
                low, high = hierarchical_bootstrap_ci(
                    matrix,
                    repetitions=repetitions,
                    seed=_bootstrap_seed(base_seed, "pair", baseline, "__mean__", budget, metric),
                )
                overall[f"mean_difference_{metric}"] = float(matrix.mean())
                overall[f"ci_low_difference_{metric}"] = low
                overall[f"ci_high_difference_{metric}"] = high
            pairwise_rows.append(overall)
    pairwise = pd.DataFrame(pairwise_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "seed_metrics.csv", raw_metrics)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "pairwise.csv", pairwise)
    _write_csv(output_dir / "pooled_fronts.csv", front_frame)
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    report_path = output_dir / "comparison_report.md"
    _write_report(protocol, summary=summary, pairwise=pairwise, output_path=report_path)
    result = {
        "campaign": protocol.data["campaign"],
        "protocol_hash": protocol.protocol_hash,
        "methods": list(method_roots),
        "circuits": list(protocol.circuits),
        "sample_budgets": list(protocol.sample_budgets),
        "seed_metric_rows": len(raw_metrics),
        "summary_rows": len(summary),
        "pairwise_rows": len(pairwise),
        "report": str(report_path),
        "completed_at": utc_now(),
    }
    _write_json(output_dir / "comparison_summary.json", result)
    return result


def _write_report(
    protocol: ComparisonProtocol,
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# GFlowNet vs. policy-gradient baselines",
        "",
        str(protocol.data["caveat"]),
        "",
        (
            "All methods use 800 training trajectories per circuit and training seed. "
            "Intervals are deterministic 95% hierarchical-bootstrap intervals over training and evaluation seeds."
        ),
        "",
        "## Equal-weight cross-circuit hypervolume",
        "",
        "| Method | N | Mean HV | 95% CI | Mean pooled HV |",
        "|:--|--:|--:|:--|--:|",
    ]
    overall = summary.loc[summary["circuit"] == "__mean__"].sort_values(["sample_budget", "method"])
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.method} | {int(row.sample_budget)} | {row.mean_hypervolume:.6f} | "
            f"[{row.ci_low_hypervolume:.6f}, {row.ci_high_hypervolume:.6f}] | "
            f"{row.pooled_hypervolume:.6f} |"
        )
    lines.extend([
        "",
        "## GFlowNet paired hypervolume differences",
        "",
        "Positive values favor GFlowNet. Pairing uses the same circuit, training seed, and evaluation seed.",
        "",
        "| Baseline | Circuit | N | Mean difference | 95% CI |",
        "|:--|:--|--:|--:|:--|",
    ])
    selected = pairwise.loc[pairwise["sample_budget"] == max(protocol.sample_budgets)].sort_values(
        ["baseline", "circuit"]
    )
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.baseline} | {row.circuit} | {int(row.sample_budget)} | "
            f"{row.mean_difference_hypervolume:+.6f} | "
            f"[{row.ci_low_difference_hypervolume:+.6f}, {row.ci_high_difference_hypervolume:+.6f}] |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `seed_metrics.csv`: per training/evaluation seed metrics.",
        "- `summary.csv`: best-of-N summaries and confidence intervals.",
        "- `pairwise.csv`: paired GFlowNet-minus-baseline differences.",
        "- `pooled_fronts.csv`: normalized pooled Pareto-front coordinates.",
        "- `artifact_manifest.json`: exact completed attempts used in the report.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_baseline_roots(values: Sequence[str], protocol: ComparisonProtocol) -> dict[str, Path]:
    if not values:
        return {key: Path(value) for key, value in protocol.data["baseline_artifacts"].items()}
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ProtocolError("--baseline-root must use METHOD=PATH")
        method, raw_path = value.split("=", 1)
        if method not in EXPECTED_BASELINES or method in parsed:
            raise ProtocolError(f"invalid or duplicate baseline root method: {method}")
        parsed[method] = Path(raw_path)
    if tuple(parsed) != EXPECTED_BASELINES:
        raise ProtocolError(f"baseline roots must be supplied in order: {EXPECTED_BASELINES}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate protocol and resolved Hydra configuration")
    validate.add_argument("--skip-compose", action="store_true")

    run = subparsers.add_parser("run-task", help="train and evaluate one circuit")
    run.add_argument("--circuit-index", required=True, type=int)
    run.add_argument("--artifact-root", required=True, type=Path)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--device", default="cuda")
    run.add_argument("--project-commit")

    compare = subparsers.add_parser("compare", help="compare completed GFlowNet and baseline artifacts")
    compare.add_argument("--gfn-root", required=True, type=Path)
    compare.add_argument("--baseline-root", action="append", default=[])
    compare.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = ComparisonProtocol.load(args.protocol)
        if args.command == "validate":
            result = validate_protocol(protocol, compose=not bool(args.skip_compose))
        elif args.command == "run-task":
            result = run_task(
                protocol,
                circuit_index=int(args.circuit_index),
                artifact_root=args.artifact_root,
                python_executable=str(args.python),
                device_name=str(args.device),
                project_commit=args.project_commit,
            )
        elif args.command == "compare":
            result = compare_artifacts(
                protocol,
                gfn_root=args.gfn_root,
                baseline_roots=_parse_baseline_roots(args.baseline_root, protocol),
                output_dir=args.output_dir,
            )
        else:  # pragma: no cover
            raise ProtocolError(f"unknown command: {args.command}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (KeyError, OSError, ProtocolError, RuntimeError, TypeError, ValueError) as exc:
        print(f"gfn-baseline-comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
