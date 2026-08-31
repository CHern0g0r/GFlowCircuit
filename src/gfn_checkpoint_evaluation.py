"""Validate and evaluate the completed backbone-matched GFlowNet checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


PROTOCOL_PATH = Path("cfg/exp/gfn_checkpoint_evaluation/protocol.yaml")
EXPECTED_CIRCUITS = ("C1355", "C5315", "adder", "apex1", "bc0", "dalu", "k2", "max")
IDENTITY_COLUMNS = ["method", "circuit_name", "training_seed", "evaluation_seed", "sample_id"]
PAYLOAD_COLUMNS = [*IDENTITY_COLUMNS, "run_id", "size", "depth", "source_checkpoint"]
COMMIT_HASH = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationTask:
    circuit: str
    circuit_index: int
    dataset_cfg: str
    circuit_path: str
    provenance_task_index: int

    @property
    def task_id(self) -> str:
        return f"gflownet_tb__{self.circuit}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _select_path(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(dotted_path)
        current = current[component]
    return current


def _values_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, bool) or isinstance(rhs, bool):
        return lhs is rhs
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return abs(float(lhs) - float(rhs)) <= 1e-12
    return lhs == rhs


def _torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return value


def _all_tensors_finite(value: object) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    return True


def _parse_provenance(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"training provenance is missing: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


class GFNCheckpointEvaluationProtocol:
    def __init__(self, *, path: Path, data: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.repo_root = self.path.parents[3]
        self.data = dict(data)
        self.protocol_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._validate_structure()

    @classmethod
    def load(cls, path: Path | str = PROTOCOL_PATH) -> "GFNCheckpointEvaluationProtocol":
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

    @property
    def gflownet_batch_size(self) -> int:
        return int(self.data["common"]["gflownet_batch_size"])

    @property
    def training_artifact_root(self) -> Path:
        return Path(str(self.data["training"]["artifact_root"]))

    def task(self, circuit_index: int) -> EvaluationTask:
        if circuit_index < 0 or circuit_index >= len(self.circuits):
            raise ProtocolError(f"circuit index must be in [0, {len(self.circuits) - 1}]")
        circuit = self.circuits[circuit_index]
        cfg = self.data["circuits"][circuit]
        return EvaluationTask(
            circuit=circuit,
            circuit_index=int(circuit_index),
            dataset_cfg=str(cfg["dataset_cfg"]),
            circuit_path=str(cfg["circuit_path"]),
            provenance_task_index=int(cfg["provenance_task_index"]),
        )

    def tasks(self) -> list[EvaluationTask]:
        return [self.task(index) for index in range(len(self.circuits))]

    def _validate_structure(self) -> None:
        required = {"version", "campaign", "method", "common", "training", "circuits", "martin"}
        missing = required - set(self.data)
        if missing:
            raise ProtocolError(f"protocol missing keys: {sorted(missing)}")
        if int(self.data["version"]) != 1:
            raise ProtocolError("only checkpoint-evaluation protocol version 1 is supported")
        if self.data["method"] != "gflownet_tb":
            raise ProtocolError("method must be gflownet_tb")
        if self.circuits != EXPECTED_CIRCUITS:
            raise ProtocolError(f"circuits must be ordered as {EXPECTED_CIRCUITS}")
        expected_seeds = tuple(range(10))
        if self.training_seeds != expected_seeds or self.evaluation_seeds != expected_seeds:
            raise ProtocolError("training and evaluation seeds must both be 0 through 9")
        if self.sample_budgets != (10, 50, 100, 200):
            raise ProtocolError("sample budgets must be [10, 50, 100, 200]")
        if self.max_samples != 200 or self.gflownet_batch_size != 20:
            raise ProtocolError("evaluation must use 200 samples per seed in GFlowNet batches of 20")

        training = self.data["training"]
        if not COMMIT_HASH.fullmatch(str(training["project_commit"])):
            raise ProtocolError("training project_commit must be a full lowercase hash")
        if not SHA256.fullmatch(str(training["config_sha256"])):
            raise ProtocolError("training config_sha256 must be a lowercase SHA256")
        if str(training["slurm_array_job_id"]) != "22933":
            raise ProtocolError("training slurm_array_job_id must be 22933")
        if not str(training["artifact_root"]).startswith("/shared/home/fedor.chernogorskii/agent/art/"):
            raise ProtocolError("training artifact root must remain below the Martin agent root")

        for task in self.tasks():
            if not (self.repo_root / task.dataset_cfg).is_file():
                raise ProtocolError(f"missing dataset config for {task.circuit}: {task.dataset_cfg}")
            if not (self.repo_root / task.circuit_path).is_file():
                raise ProtocolError(f"missing circuit for {task.circuit}: {task.circuit_path}")
            if task.provenance_task_index not in range(4):
                raise ProtocolError(f"invalid provenance task index for {task.circuit}")

        martin = self.data["martin"]
        if martin.get("project") != "gflowcircuit-gfn-backbone-matched-eval":
            raise ProtocolError("unexpected Martin project")
        if martin.get("job_name") != "gfc-gfn-backbone-matched-eval-v1":
            raise ProtocolError("unexpected Martin job name")
        for component in (martin["project"], martin["job_name"]):
            if not SAFE_COMPONENT.fullmatch(str(component)):
                raise ProtocolError(f"unsafe Martin component: {component}")


def _circuit_artifact(training_artifact_root: Path, task: EvaluationTask) -> Path:
    return training_artifact_root.resolve() / "circuits" / task.circuit


def validate_training_artifacts(
    protocol: GFNCheckpointEvaluationProtocol,
    task: EvaluationTask,
    *,
    training_artifact_root: Path,
) -> dict[str, Any]:
    training = protocol.data["training"]
    provenance_path = training_artifact_root / f"provenance-{task.provenance_task_index}.txt"
    provenance = _parse_provenance(provenance_path)
    expected_provenance = {
        "project_commit": str(training["project_commit"]),
        "config": str(training["config"]),
        "config_sha256": str(training["config_sha256"]),
        "slurm_array_job_id": str(training["slurm_array_job_id"]),
        "slurm_array_task_id": str(task.provenance_task_index),
    }
    if provenance != expected_provenance:
        raise ValueError(f"training provenance mismatch in {provenance_path}")

    circuit_artifact = _circuit_artifact(training_artifact_root, task)
    config_path = circuit_artifact / "hydra" / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"resolved Hydra config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError(f"resolved Hydra config is invalid: {config_path}")
    if str(config.get("dataset_cfg")) != task.dataset_cfg:
        raise ValueError(f"dataset_cfg mismatch for {task.circuit}")
    for dotted_path, expected in training["expected"].items():
        try:
            actual = _select_path(config, str(dotted_path))
        except KeyError as exc:
            raise ValueError(f"resolved config is missing {dotted_path}") from exc
        if not _values_equal(actual, expected):
            raise ValueError(
                f"resolved config mismatch for {dotted_path}: expected {expected!r}, got {actual!r}"
            )

    checkpoints = sorted((circuit_artifact / "saved_models").glob("run_*/last.pt"))
    if len(checkpoints) != len(protocol.training_seeds):
        raise ValueError(f"expected 10 checkpoints for {task.circuit}, found {len(checkpoints)}")
    observed: list[tuple[int, int]] = []
    for checkpoint in checkpoints:
        payload = _torch_load(checkpoint)
        run_idx = int(payload.get("run_idx", -1))
        seed = int(payload.get("seed", -1))
        observed.append((run_idx, seed))
        policy_state = payload.get("policy_state_dict")
        if not isinstance(policy_state, Mapping) or not policy_state:
            raise ValueError(f"policy state is missing or empty: {checkpoint}")
        if not _all_tensors_finite(policy_state):
            raise ValueError(f"policy state contains non-finite tensors: {checkpoint}")
        summary = payload.get("tb_training")
        if not isinstance(summary, Mapping):
            raise ValueError(f"tb_training summary is missing: {checkpoint}")
        expected_summary = {
            "log_z_initialization": "calibrated",
            "calibration_trajectories": 64,
            "new_on_policy_trajectories": 736,
            "training_trajectories": 800,
            "training_presentations": 800,
            "optimizer_updates": 200,
        }
        for key, expected in expected_summary.items():
            if not _values_equal(summary.get(key), expected):
                raise ValueError(f"invalid tb_training.{key} in {checkpoint}")
        calibration_target = summary.get("calibration_target")
        if not isinstance(calibration_target, (int, float)) or not np.isfinite(calibration_target):
            raise ValueError(f"invalid calibration target in {checkpoint}")

    expected_pairs = list(enumerate(protocol.training_seeds))
    if sorted(observed) != expected_pairs:
        raise ValueError(f"checkpoint run/seed pairs differ: expected {expected_pairs}, got {sorted(observed)}")

    report_path = circuit_artifact / str(training["report_file"])
    if not report_path.is_file():
        raise ValueError(f"training report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("algorithm") != "gflownet_tb":
        raise ValueError(f"training report algorithm mismatch: {report_path}")
    report_pairs = sorted((int(row["run_idx"]), int(row["seed"])) for row in report.get("runs", []))
    if report_pairs != expected_pairs:
        raise ValueError(f"training report run/seed pairs differ: {report_path}")

    return {
        "circuit_artifact": circuit_artifact,
        "config_path": config_path,
        "saved_models_dir": circuit_artifact / "saved_models",
        "checkpoint_count": len(checkpoints),
    }


def _reference_frame(reference: Mapping[str, Any], *, sample_budget: int | None = None) -> pd.DataFrame:
    row = dict(reference)
    if sample_budget is not None:
        row["sample_budget"] = int(sample_budget)
    return pd.DataFrame([row])


def _sample_identity(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    samples = frame.loc[frame["run_id"].notna()]
    return set(samples[IDENTITY_COLUMNS].itertuples(index=False, name=None))


def _sample_payload(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    samples = frame.loc[frame["run_id"].notna()]
    return sorted(samples[PAYLOAD_COLUMNS].itertuples(index=False, name=None))


def _next_attempt_dir(evaluation_root: Path) -> Path:
    numbers: list[int] = []
    for path in evaluation_root.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return evaluation_root / f"attempt_{max(numbers, default=0) + 1:03d}"


def _sample_task(
    protocol: GFNCheckpointEvaluationProtocol,
    task: EvaluationTask,
    *,
    attempt_dir: Path,
    training_paths: Mapping[str, Any],
    device: Any,
) -> None:
    from src.sample_exp import evaluation_reference_row, sample_paired_evaluation_seed_from_paths

    circuit_path = (protocol.repo_root / task.circuit_path).resolve()
    frames: list[pd.DataFrame] = []
    for evaluation_seed in protocol.evaluation_seeds:
        frame = sample_paired_evaluation_seed_from_paths(
            config_path=Path(training_paths["config_path"]),
            saved_models_dir=Path(training_paths["saved_models_dir"]),
            circuit_path=circuit_path,
            method="gflownet_tb",
            circuit_name=task.circuit,
            num_samples=protocol.max_samples,
            evaluation_seed=evaluation_seed,
            device=device,
            num_steps=int(protocol.data["common"]["num_steps"]),
            gflownet_batch_size=protocol.gflownet_batch_size,
        )
        _write_csv(attempt_dir / "raw" / f"seed_{evaluation_seed:02d}.csv", frame)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    reference = evaluation_reference_row(
        circuit_path=circuit_path,
        method="gflownet_tb",
        circuit_name=task.circuit,
        num_steps=int(protocol.data["common"]["num_steps"]),
    )
    canonical = pd.concat([_reference_frame(reference), raw], ignore_index=True, sort=False)
    _write_csv(attempt_dir / "points.csv", canonical)
    for budget in protocol.sample_budgets:
        view = raw.loc[raw["sample_id"] < int(budget)].copy()
        view["sample_budget"] = int(budget)
        output = pd.concat(
            [_reference_frame(reference, sample_budget=budget), view],
            ignore_index=True,
            sort=False,
        )
        _write_csv(attempt_dir / "budgets" / f"points_n{budget:03d}.csv", output)


def _validate_sample_values(frame: pd.DataFrame, *, path: Path) -> None:
    samples = frame.loc[frame["run_id"].notna()]
    if set(samples["method"].astype(str)) != {"gflownet_tb"}:
        raise ValueError(f"{path} contains the wrong method")
    numeric = samples[["size", "depth"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric.to_numpy() < 0).any():
        raise ValueError(f"{path} contains invalid size/depth values")


def validate_attempt(
    protocol: GFNCheckpointEvaluationProtocol,
    task: EvaluationTask,
    *,
    attempt_dir: Path,
    training_artifact_root: Path,
) -> dict[str, int]:
    raw_frames: list[pd.DataFrame] = []
    for evaluation_seed in protocol.evaluation_seeds:
        path = attempt_dir / "raw" / f"seed_{evaluation_seed:02d}.csv"
        if not path.is_file():
            raise ValueError(f"raw evaluation file is missing: {path}")
        frame = pd.read_csv(path)
        expected_rows = len(protocol.training_seeds) * protocol.max_samples
        if len(frame) != expected_rows:
            raise ValueError(f"{path} has {len(frame)} rows; expected {expected_rows}")
        if set(frame["circuit_name"].astype(str)) != {task.circuit}:
            raise ValueError(f"{path} contains the wrong circuit")
        if set(frame["evaluation_seed"].astype(int)) != {evaluation_seed}:
            raise ValueError(f"{path} contains the wrong evaluation seed")
        _validate_sample_values(frame, path=path)
        for run_id, training_seed in enumerate(protocol.training_seeds):
            group = frame.loc[frame["training_seed"].astype(int) == training_seed]
            if set(group["run_id"].astype(int)) != {run_id}:
                raise ValueError(f"{path} contains the wrong run for training seed {training_seed}")
            if list(group["sample_id"].astype(int)) != list(range(protocol.max_samples)):
                raise ValueError(f"{path} training seed {training_seed} has unordered sample IDs")
            expected_checkpoint = str(
                (_circuit_artifact(training_artifact_root, task) / "saved_models" / f"run_{run_id}" / "last.pt").resolve()
            )
            if set(group["source_checkpoint"].astype(str)) != {expected_checkpoint}:
                raise ValueError(f"{path} contains the wrong source checkpoint for run {run_id}")
        raw_frames.append(frame)

    raw = pd.concat(raw_frames, ignore_index=True)
    expected_raw = len(protocol.training_seeds) * len(protocol.evaluation_seeds) * protocol.max_samples
    if len(raw) != expected_raw or raw.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("raw evaluation count or identity uniqueness mismatch")

    points_path = attempt_dir / "points.csv"
    if not points_path.is_file():
        raise ValueError(f"canonical points file is missing: {points_path}")
    canonical = pd.read_csv(points_path)
    references = canonical.loc[canonical["run_id"].isna()]
    samples = canonical.loc[canonical["run_id"].notna()]
    if len(references) != 1 or len(samples) != expected_raw:
        raise ValueError("canonical points count mismatch")
    if _sample_identity(canonical) != _sample_identity(raw) or _sample_payload(canonical) != _sample_payload(raw):
        raise ValueError("canonical points do not match raw evaluation")

    for budget in protocol.sample_budgets:
        path = attempt_dir / "budgets" / f"points_n{budget:03d}.csv"
        if not path.is_file():
            raise ValueError(f"budget view is missing: {path}")
        frame = pd.read_csv(path)
        references = frame.loc[frame["run_id"].isna()]
        samples = frame.loc[frame["run_id"].notna()]
        expected_rows = len(protocol.training_seeds) * len(protocol.evaluation_seeds) * budget
        if len(references) != 1 or len(samples) != expected_rows:
            raise ValueError(f"{path} count mismatch")
        if set(frame["sample_budget"].dropna().astype(int)) != {budget}:
            raise ValueError(f"{path} contains the wrong sample budget")
        prefix = raw.loc[raw["sample_id"] < budget]
        if _sample_identity(frame) != _sample_identity(prefix) or _sample_payload(frame) != _sample_payload(prefix):
            raise ValueError(f"{path} is not the exact nested prefix for budget {budget}")

    return {
        "checkpoint_count": len(protocol.training_seeds),
        "evaluation_seed_count": len(protocol.evaluation_seeds),
        "unique_sample_count": expected_raw,
    }


def _validate_existing_canonical(
    protocol: GFNCheckpointEvaluationProtocol,
    task: EvaluationTask,
    *,
    circuit_artifact: Path,
    training_artifact_root: Path,
) -> dict[str, int]:
    canonical_path = circuit_artifact / "points.csv"
    if not canonical_path.is_file():
        raise ValueError(f"canonical points file is missing: {canonical_path}")
    canonical_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    failures: list[str] = []
    for attempt_dir in sorted((circuit_artifact / "evaluation").glob("attempt_*"), reverse=True):
        try:
            counts = validate_attempt(
                protocol,
                task,
                attempt_dir=attempt_dir,
                training_artifact_root=training_artifact_root,
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{attempt_dir.name}: {exc}")
            continue
        if hashlib.sha256((attempt_dir / "points.csv").read_bytes()).hexdigest() != canonical_hash:
            failures.append(f"{attempt_dir.name}: canonical hash mismatch")
            continue
        return counts
    detail = "; ".join(failures) if failures else "no evaluation attempts exist"
    raise ValueError(f"existing canonical points.csv is not backed by a valid attempt: {detail}")


def validate_protocol(
    protocol: GFNCheckpointEvaluationProtocol,
    *,
    training_artifact_root: Path | None = None,
    circuit_index: int | None = None,
) -> dict[str, Any]:
    tasks = protocol.tasks() if circuit_index is None else [protocol.task(circuit_index)]
    if training_artifact_root is not None:
        for task in tasks:
            validate_training_artifacts(
                protocol,
                task,
                training_artifact_root=training_artifact_root.resolve(),
            )
    return {
        "campaign": protocol.data["campaign"],
        "protocol_hash": protocol.protocol_hash,
        "circuits": len(protocol.circuits),
        "validated_circuits": len(tasks),
        "checkpoints": len(protocol.circuits) * len(protocol.training_seeds),
        "unique_evaluation_rollouts": (
            len(protocol.circuits)
            * len(protocol.training_seeds)
            * len(protocol.evaluation_seeds)
            * protocol.max_samples
        ),
        "sample_budgets": list(protocol.sample_budgets),
    }


def run_task(
    protocol: GFNCheckpointEvaluationProtocol,
    *,
    circuit_index: int,
    training_artifact_root: Path,
    job_artifact_root: Path,
    evaluator_commit: str,
    device_name: str,
) -> dict[str, Any]:
    import torch

    if not COMMIT_HASH.fullmatch(evaluator_commit):
        raise ProtocolError("evaluator_commit must be a full lowercase commit hash")
    task = protocol.task(circuit_index)
    training_artifact_root = training_artifact_root.resolve()
    job_task_root = job_artifact_root.resolve() / "tasks" / task.task_id
    status_path = job_task_root / "task_status.json"
    circuit_artifact = _circuit_artifact(training_artifact_root, task)
    metadata: dict[str, Any] = {
        "task_id": task.task_id,
        "circuit": task.circuit,
        "circuit_index": task.circuit_index,
        "state": "running",
        "protocol_hash": protocol.protocol_hash,
        "evaluator_commit": evaluator_commit,
        "training_project_commit": protocol.data["training"]["project_commit"],
        "training_artifact_root": str(training_artifact_root),
        "device": device_name,
        "started_at": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    _write_json(status_path, metadata)
    started = time.monotonic()

    try:
        training_paths = validate_training_artifacts(
            protocol,
            task,
            training_artifact_root=training_artifact_root,
        )
        canonical_path = circuit_artifact / "points.csv"
        if canonical_path.exists():
            counts = _validate_existing_canonical(
                protocol,
                task,
                circuit_artifact=circuit_artifact,
                training_artifact_root=training_artifact_root,
            )
            metadata.update(counts)
            metadata["state"] = "already_complete"
            metadata["canonical_points"] = str(canonical_path)
        else:
            evaluation_root = circuit_artifact / "evaluation"
            evaluation_root.mkdir(parents=True, exist_ok=True)
            attempt_dir = _next_attempt_dir(evaluation_root)
            attempt_dir.mkdir(parents=True, exist_ok=False)
            metadata["attempt_dir"] = str(attempt_dir)
            _write_json(attempt_dir / "attempt.json", metadata)
            sample_started = time.monotonic()
            _sample_task(
                protocol,
                task,
                attempt_dir=attempt_dir,
                training_paths=training_paths,
                device=torch.device(device_name),
            )
            metadata["sample_wall_time_seconds"] = time.monotonic() - sample_started
            counts = validate_attempt(
                protocol,
                task,
                attempt_dir=attempt_dir,
                training_artifact_root=training_artifact_root,
            )
            metadata.update(counts)
            try:
                os.link(attempt_dir / "points.csv", canonical_path)
            except FileExistsError:
                _validate_existing_canonical(
                    protocol,
                    task,
                    circuit_artifact=circuit_artifact,
                    training_artifact_root=training_artifact_root,
                )
            metadata["state"] = "complete"
            metadata["canonical_points"] = str(canonical_path)
            metadata["finished_at"] = utc_now()
            metadata["wall_time_seconds"] = time.monotonic() - started
            _write_json(attempt_dir / "attempt.json", metadata)
    except Exception as exc:
        metadata["state"] = "failed"
        metadata["error"] = str(exc)
        metadata["finished_at"] = utc_now()
        metadata["wall_time_seconds"] = time.monotonic() - started
        if metadata.get("attempt_dir"):
            _write_json(Path(str(metadata["attempt_dir"])) / "attempt.json", metadata)
        _write_json(status_path, metadata)
        raise RuntimeError(f"{task.task_id}: {exc}") from exc

    metadata["finished_at"] = utc_now()
    metadata["wall_time_seconds"] = time.monotonic() - started
    _write_json(status_path, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint-only GFlowNet evaluation")
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate protocol and optional training artifacts")
    validate.add_argument("--training-artifact-root", default=None)
    validate.add_argument("--circuit-index", type=int, default=None)

    run = subparsers.add_parser("run-task", help="Evaluate one circuit's existing checkpoints")
    run.add_argument("--circuit-index", required=True, type=int)
    run.add_argument("--training-artifact-root", default=None)
    run.add_argument("--job-artifact-root", required=True)
    run.add_argument("--evaluator-commit", required=True)
    run.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        protocol = GFNCheckpointEvaluationProtocol.load(args.protocol)
        training_root = (
            Path(args.training_artifact_root)
            if args.training_artifact_root is not None
            else protocol.training_artifact_root
        )
        if args.command == "validate":
            result = validate_protocol(
                protocol,
                training_artifact_root=(training_root if args.training_artifact_root is not None else None),
                circuit_index=args.circuit_index,
            )
        elif args.command == "run-task":
            result = run_task(
                protocol,
                circuit_index=int(args.circuit_index),
                training_artifact_root=training_root,
                job_artifact_root=Path(args.job_artifact_root),
                evaluator_commit=str(args.evaluator_commit),
                device_name=str(args.device),
            )
        else:  # pragma: no cover
            parser.error(f"unknown command: {args.command}")
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ProtocolError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
