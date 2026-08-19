"""Stage-gated exploration tuning driver.

This module is the canonical entrypoint used by the Martin SLURM wrappers.
It never submits another stage.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping

import yaml

from src.exploration_analysis import analyze_stage, collect_task_records
from src.exploration_protocol import (
    ExperimentTask,
    ExplorationProtocol,
    PROTOCOL_PATH,
    ProtocolError,
    Setting,
    canonical_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _select_path(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(dotted_path)
        current = current[component]
    return current


def _values_equal(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return abs(float(lhs) - float(rhs)) <= 1e-12
    return lhs == rhs


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"required stage manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"invalid stage manifest: {path}")
    return value


def _load_selection(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"required stage selection is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"invalid stage selection: {path}")
    return value


def dependency_context(
    protocol: ExplorationProtocol,
    stage: str,
    *,
    artifact_base: Path,
    project_commit: str | None,
) -> tuple[dict[str, Any], list[Path]]:
    selections: dict[str, Any] = {}
    roots: list[Path] = []
    observed_commit: str | None = project_commit
    for dependency in protocol.dependency_stages(stage):
        root = protocol.artifact_root(dependency, artifact_base=artifact_base)
        manifest = _load_manifest(root / "stage_manifest.json")
        if manifest.get("state") != "complete":
            raise ProtocolError(f"dependency stage is not complete: {dependency}")
        if manifest.get("protocol_hash") != protocol.protocol_hash:
            raise ProtocolError(f"dependency protocol hash differs: {dependency}")
        manifest_commit = str(manifest.get("project_commit", ""))
        if not manifest_commit:
            raise ProtocolError(f"dependency project commit is missing: {dependency}")
        if observed_commit is None:
            observed_commit = manifest_commit
        if manifest_commit != observed_commit:
            raise ProtocolError(f"dependency project commit differs: {dependency}")
        selection = _load_selection(root / "selection.json")
        selections.update(selection.get("algorithms", {}))
        if dependency == "pcn_seeds":
            selections["pcn_seed_ranked"] = selection["algorithms"]["pcn"]["ranked"]
        if dependency == "pcn_noise":
            selections["pcn_noise_ranked"] = selection["algorithms"]["pcn"]["ranked"]
        roots.append(root)
    return selections, roots


def _existing_result_index(
    roots: Iterable[Path],
    *,
    protocol_hash: str,
    project_commit: str,
) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        manifest_path = root / "stage_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_manifest(manifest_path)
        if manifest.get("state") != "complete":
            continue
        if manifest.get("protocol_hash") != protocol_hash or manifest.get("project_commit") != project_commit:
            continue
        for task in manifest.get("tasks", []):
            status_path = root / "tasks" / str(task["task_id"]) / "task_status.json"
            if not status_path.is_file():
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") == "complete":
                attempt = Path(str(status["attempt_dir"]))
            elif status.get("state") == "reused":
                attempt = Path(str(status["source_attempt"]))
            else:
                continue
            if (attempt / "train" / "points.csv").is_file():
                index[str(task["reuse_key"])] = attempt
    return index


def _task_commands(
    task: ExperimentTask,
    *,
    attempt_dir: Path,
    python_executable: str,
    protocol: ExplorationProtocol,
) -> tuple[list[str], list[str]]:
    train_dir = attempt_dir / "train"
    common = protocol.data["common"]
    train = [
        python_executable,
        "-m",
        "src.run",
        "--config-name",
        task.base_config,
    ]
    if task.setting.fragment:
        train.append(f"+exp/exploration_tuning={task.setting.fragment}")
    overrides: dict[str, Any] = {
        "run_name": task.task_id,
        "dataset_cfg": task.dataset_cfg,
        "num_steps": int(common["num_steps"]),
        "available_actions": list(common["available_actions"]),
        "episodes": task.episodes,
        "eval_every": task.episodes,
        "seed": task.seed,
        "paper_mode.num_runs": 1,
        "paper_mode.infer_rollouts": 1,
        "discovery_metrics.enabled": True,
        "discovery_metrics.emit_every_trajectories": int(common["discovery_emit_every_trajectories"]),
        "hydra.run.dir": str(train_dir),
    }
    overrides.update(task.fixed_overrides)
    if not task.setting.fragment:
        overrides.update(task.setting.overrides)
    train.extend(f"{key}={_hydra_value(value)}" for key, value in overrides.items())
    sample = [
        python_executable,
        "-m",
        "src.sample_exp",
        "--experiment",
        str(train_dir),
        "--circuit",
        task.circuit_path,
        "--num-samples",
        str(task.evaluation_samples),
        "--seed",
        str(common["evaluation_seed"]),
        "--device",
        "cuda",
        "--pcn-sampling-mode",
        "target",
    ]
    return train, sample


def _validate_attempt(task: ExperimentTask, attempt_dir: Path, protocol: ExplorationProtocol) -> None:
    import csv

    train_dir = attempt_dir / "train"
    config_path = train_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"resolved Hydra config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected: dict[str, Any] = {
        "seed": task.seed,
        "episodes": task.episodes,
        "num_steps": int(protocol.data["common"]["num_steps"]),
        "available_actions": list(protocol.data["common"]["available_actions"]),
        "paper_mode.num_runs": 1,
        "paper_mode.infer_rollouts": 1,
        **task.setting.overrides,
        **task.fixed_overrides,
    }
    for path, value in expected.items():
        try:
            actual = _select_path(config, path)
        except KeyError as exc:
            raise ValueError(f"resolved config is missing {path}") from exc
        if not _values_equal(actual, value):
            raise ValueError(f"resolved config mismatch for {path}: expected {value!r}, got {actual!r}")
    checkpoints = list((train_dir / "saved_models").glob("run_*/last.pt"))
    if len(checkpoints) != 1:
        raise ValueError(f"expected one checkpoint, found {len(checkpoints)}")
    reports = list(train_dir.glob("*_report.json"))
    if len(reports) != 1:
        raise ValueError(f"expected one report JSON, found {len(reports)}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    if report.get("algorithm") != task.report_algorithm:
        raise ValueError(
            f"report algorithm mismatch: expected {task.report_algorithm}, got {report.get('algorithm')}"
        )
    discovery_path = train_dir / "discovery_metrics.csv"
    if not discovery_path.is_file():
        raise ValueError(f"discovery_metrics.csv is missing: {discovery_path}")
    with discovery_path.open("r", encoding="utf-8", newline="") as handle:
        discovery_rows = list(csv.DictReader(handle))
    final_circuit_rows = [
        row
        for row in discovery_rows
        if row.get("row_type") == "circuit"
        and str(row.get("is_final", "")).lower() == "true"
    ]
    if len(final_circuit_rows) != 1:
        raise ValueError(
            f"expected one final circuit discovery row, found {len(final_circuit_rows)}"
        )
    observed_trajectories = int(float(final_circuit_rows[0]["local_trajectory"]))
    if observed_trajectories != int(task.training_trajectories):
        raise ValueError(
            f"training trajectory mismatch: expected {task.training_trajectories}, "
            f"got {observed_trajectories}"
        )
    points_path = train_dir / "points.csv"
    if not points_path.is_file():
        raise ValueError(f"points.csv is missing: {points_path}")
    with points_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    references = [row for row in rows if not str(row.get("run_id", "")).strip()]
    samples = [row for row in rows if str(row.get("run_id", "")).strip()]
    if len(references) != 1 or len(samples) != task.evaluation_samples:
        raise ValueError(
            f"sample count mismatch: references={len(references)}, samples={len(samples)}, "
            f"expected={task.evaluation_samples}"
        )


def _next_attempt_dir(task_root: Path) -> Path:
    numbers = []
    for path in task_root.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return task_root / f"attempt_{max(numbers, default=0) + 1:03d}"


def _run_command(command: list[str], *, cwd: Path, env: Mapping[str, str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=cwd, env=dict(env), stdout=stdout, stderr=stderr, check=False)
    return int(completed.returncode)


def run_task(
    task: ExperimentTask,
    *,
    artifact_root: Path,
    protocol: ExplorationProtocol,
    project_commit: str,
    device: str,
    python_executable: str,
) -> dict[str, Any]:
    task_root = artifact_root / "tasks" / task.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    existing_status_path = task_root / "task_status.json"
    if existing_status_path.is_file():
        existing = json.loads(existing_status_path.read_text(encoding="utf-8"))
        if existing.get("state") == "complete":
            attempt = Path(str(existing["attempt_dir"]))
            try:
                _validate_attempt(task, attempt, protocol)
                return existing
            except ValueError:
                pass
        if existing.get("state") == "reused":
            return existing
    attempt_dir = _next_attempt_dir(task_root)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    train_command, sample_command = _task_commands(
        task,
        attempt_dir=attempt_dir,
        python_executable=python_executable,
        protocol=protocol,
    )
    started = time.monotonic()
    metadata: dict[str, Any] = {
        "task_id": task.task_id,
        "state": "running",
        "attempt_dir": str(attempt_dir),
        "protocol_hash": protocol.protocol_hash,
        "project_commit": project_commit,
        "device": device,
        "started_at": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "train_command": train_command,
        "sample_command": sample_command,
    }
    _write_json(attempt_dir / "attempt.json", metadata)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    env["HYDRA_FULL_ERROR"] = "1"
    env["GFC_TIMING"] = "1"
    train_started = time.monotonic()
    train_code = _run_command(
        train_command,
        cwd=protocol.repo_root,
        env=env,
        stdout_path=attempt_dir / "train.stdout.log",
        stderr_path=attempt_dir / "train.stderr.log",
    )
    metadata["train_exit_code"] = train_code
    metadata["train_wall_time_seconds"] = time.monotonic() - train_started
    sample_code: int | None = None
    error: str | None = None
    if train_code == 0:
        sample_started = time.monotonic()
        sample_code = _run_command(
            sample_command,
            cwd=protocol.repo_root,
            env=env,
            stdout_path=attempt_dir / "sample.stdout.log",
            stderr_path=attempt_dir / "sample.stderr.log",
        )
        metadata["sample_exit_code"] = sample_code
        metadata["sample_wall_time_seconds"] = time.monotonic() - sample_started
    try:
        if train_code != 0:
            raise ValueError(f"training exited with code {train_code}")
        if sample_code != 0:
            raise ValueError(f"sampling exited with code {sample_code}")
        _validate_attempt(task, attempt_dir, protocol)
        metadata["state"] = "complete"
        metadata["checkpoint_count"] = 1
        metadata["sample_count"] = task.evaluation_samples
        metadata["training_trajectories"] = task.training_trajectories
        metadata["resolved_config"] = str(attempt_dir / "train" / ".hydra" / "config.yaml")
    except ValueError as exc:
        metadata["state"] = "failed"
        error = str(exc)
        metadata["error"] = error
    metadata["finished_at"] = utc_now()
    metadata["wall_time_seconds"] = time.monotonic() - started
    _write_json(attempt_dir / "attempt.json", metadata)
    _write_json(existing_status_path, metadata)
    if error:
        raise RuntimeError(f"{task.task_id}: {error}")
    return metadata


def _gpu_devices(workers: int) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if len(devices) < workers:
        raise ProtocolError(
            f"run-stage requires {workers} allocated CUDA devices; CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return devices[:workers]


def validate_hydra_compositions(
    *,
    protocol: ExplorationProtocol,
    tasks: Iterable[ExperimentTask],
    output_dir: Path,
    python_executable: str,
) -> None:
    """Compose one representative config per distinct algorithm/setting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    representatives: dict[tuple[str, str], ExperimentTask] = {}
    for task in tasks:
        representatives.setdefault((task.algorithm, task.setting.key), task)
    for task in representatives.values():
        attempt = output_dir / f"_{task.task_id}"
        command, _ = _task_commands(
            task,
            attempt_dir=attempt,
            python_executable=python_executable,
            protocol=protocol,
        )
        command.extend(["--cfg", "job"])
        completed = subprocess.run(
            command,
            cwd=protocol.repo_root,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        name = f"{task.algorithm}__{task.setting.setting_id}"
        (output_dir / f"{name}.yaml").write_text(completed.stdout, encoding="utf-8")
        (output_dir / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise ProtocolError(
                f"Hydra composition failed for {task.algorithm}/{task.setting.setting_id}; "
                f"see {output_dir / f'{name}.stderr.log'}"
            )
        resolved = yaml.safe_load(completed.stdout)
        if not isinstance(resolved, Mapping):
            raise ProtocolError(f"Hydra composition produced no mapping for {name}")


def _runtime_projection(protocol: ExplorationProtocol, artifact_root: Path, workers: int) -> float:
    records = collect_task_records(artifact_root)
    train_seconds_per_trajectory: dict[str, list[float]] = {}
    sample_seconds_per_rollout: dict[str, list[float]] = {}
    for record in records:
        status = record["execution_status"]
        training_count = int(record["training_trajectories"])
        sample_count = int(record["evaluation_samples"])
        train_seconds_per_trajectory.setdefault(str(record["algorithm"]), []).append(
            float(status.get("train_wall_time_seconds", status.get("wall_time_seconds", 0.0)))
            / max(1, training_count)
        )
        sample_seconds_per_rollout.setdefault(str(record["algorithm"]), []).append(
            float(status.get("sample_wall_time_seconds", 0.0)) / max(1, sample_count)
        )
    coarse_settings = {
        algorithm: len(protocol.coarse_settings(algorithm))
        for algorithm in protocol.data["algorithms"]
    }
    total_gpu_seconds = 0.0
    for algorithm, values in train_seconds_per_trajectory.items():
        per_trajectory = sum(values) / len(values)
        sample_values = sample_seconds_per_rollout[algorithm]
        per_sample = sum(sample_values) / len(sample_values)
        tasks = coarse_settings[algorithm] * 2 * 3
        total_gpu_seconds += (per_trajectory * 200 + per_sample * 20) * tasks
    return total_gpu_seconds / max(1, workers) / 3600.0


def build_stage_manifest(
    *,
    protocol: ExplorationProtocol,
    stage: str,
    artifact_root: Path,
    artifact_base: Path,
    project_commit: str,
    settings: list[Setting],
    dependency_roots: list[Path],
) -> tuple[dict[str, Any], list[ExperimentTask]]:
    tasks = protocol.build_tasks(stage, settings)
    existing_path = artifact_root / "stage_manifest.json"
    if existing_path.is_file():
        existing = _load_manifest(existing_path)
        if existing.get("protocol_hash") != protocol.protocol_hash:
            raise ProtocolError("existing stage manifest uses a different protocol hash")
        if existing.get("project_commit") != project_commit:
            raise ProtocolError("existing stage manifest uses a different project commit")
        if canonical_json(existing.get("tasks")) != canonical_json([task.to_dict() for task in tasks]):
            raise ProtocolError("existing stage manifest task matrix differs")
        return existing, tasks
    artifact_root.mkdir(parents=True, exist_ok=True)
    reuse_index = _existing_result_index(
        dependency_roots,
        protocol_hash=protocol.protocol_hash,
        project_commit=project_commit,
    )
    manifest: dict[str, Any] = {
        "campaign": protocol.data["campaign"],
        "stage": stage,
        "state": "prepared",
        "protocol_path": str(protocol.path),
        "protocol_hash": protocol.protocol_hash,
        "project_commit": project_commit,
        "created_at": utc_now(),
        "artifact_root": str(artifact_root),
        "artifact_base": str(artifact_base),
        "dependencies": [str(path) for path in dependency_roots],
        "tasks": [task.to_dict() for task in tasks],
    }
    _write_json(existing_path, manifest)
    for task in tasks:
        source = reuse_index.get(task.reuse_key)
        if source is None:
            continue
        task_root = artifact_root / "tasks" / task.task_id
        task_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            task_root / "task_status.json",
            {
                "task_id": task.task_id,
                "state": "reused",
                "source_attempt": str(source),
                "protocol_hash": protocol.protocol_hash,
                "project_commit": project_commit,
                "wall_time_seconds": 0.0,
            },
        )
    return manifest, tasks


def validate_stage(
    *,
    protocol: ExplorationProtocol,
    stage: str,
    artifact_base: Path | None = None,
    project_commit: str | None = None,
    compose: bool = True,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    dependency_roots: list[Path] = []
    if protocol.stages[stage].get("dependencies"):
        resolved_artifact_base = artifact_base or Path(protocol.data["martin"]["artifact_base"])
        selections, dependency_roots = dependency_context(
            protocol,
            stage,
            artifact_base=resolved_artifact_base,
            project_commit=project_commit,
        )
    settings = protocol.settings_for_stage(stage, selections=selections)
    tasks = protocol.build_tasks(stage, settings)
    if compose:
        with TemporaryDirectory(prefix="gfc-exploration-compose-") as directory:
            validate_hydra_compositions(
                protocol=protocol,
                tasks=tasks,
                output_dir=Path(directory),
                python_executable=python_executable,
            )
    return {
        "stage": stage,
        "protocol_hash": protocol.protocol_hash,
        "settings": len(settings),
        "tasks": len(tasks),
        "dependencies": [str(path) for path in dependency_roots],
    }


def run_stage(args: argparse.Namespace, protocol: ExplorationProtocol) -> None:
    stage = str(args.stage)
    project_commit = str(args.project_commit or os.environ.get("GFC_PROJECT_COMMIT", ""))
    if not project_commit or project_commit == "SET_PROJECT_COMMIT_BEFORE_SYNC":
        raise ProtocolError("run-stage requires --project-commit or GFC_PROJECT_COMMIT")
    artifact_root = Path(args.artifact_root).resolve()
    artifact_base = Path(args.artifact_base).resolve() if args.artifact_base else artifact_root.parent
    selections, dependency_roots = dependency_context(
        protocol,
        stage,
        artifact_base=artifact_base,
        project_commit=project_commit,
    )
    settings = protocol.settings_for_stage(stage, selections=selections)
    manifest, tasks = build_stage_manifest(
        protocol=protocol,
        stage=stage,
        artifact_root=artifact_root,
        artifact_base=artifact_base,
        project_commit=project_commit,
        settings=settings,
        dependency_roots=dependency_roots,
    )
    validate_hydra_compositions(
        protocol=protocol,
        tasks=tasks,
        output_dir=artifact_root / "config_preflight",
        python_executable=str(args.python),
    )
    devices = _gpu_devices(int(args.workers))
    device_queue: queue.Queue[str] = queue.Queue()
    for device in devices:
        device_queue.put(device)
    failures: list[str] = []

    def execute(task: ExperimentTask) -> None:
        task_status = artifact_root / "tasks" / task.task_id / "task_status.json"
        if task_status.is_file():
            status = json.loads(task_status.read_text(encoding="utf-8"))
            if status.get("state") == "reused":
                return
        device = device_queue.get()
        try:
            run_task(
                task,
                artifact_root=artifact_root,
                protocol=protocol,
                project_commit=project_commit,
                device=device,
                python_executable=str(args.python),
            )
        finally:
            device_queue.put(device)

    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        future_to_task = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()
            except Exception as exc:  # keep other independent tasks running
                failures.append(f"{task.task_id}: {exc}")
                print(f"FAILED {task.task_id}: {exc}", file=sys.stderr, flush=True)
    if failures:
        manifest["state"] = "failed"
        manifest["failures"] = failures
        manifest["finished_at"] = utc_now()
        _write_json(artifact_root / "stage_manifest.json", manifest)
        raise RuntimeError(f"stage {stage} has {len(failures)} failed task(s)")
    result = analyze_stage(
        protocol=protocol,
        stage=stage,
        artifact_root=artifact_root,
        settings=settings,
    )
    if stage == "smoke":
        projected_hours = _runtime_projection(protocol, artifact_root, int(args.workers))
        result["projected_coarse_wall_time_hours"] = projected_hours
        selection_path = artifact_root / "selection.json"
        _write_json(selection_path, result)
        with (artifact_root / "report.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\nProjected four-GPU coarse-stage wall time: {projected_hours:.2f} hours.\n")
        if projected_hours > 48.0:
            manifest["state"] = "runtime_projection_failed"
            manifest["finished_at"] = utc_now()
            _write_json(artifact_root / "stage_manifest.json", manifest)
            raise RuntimeError(f"projected coarse runtime {projected_hours:.2f}h exceeds 48h")
    manifest["state"] = "complete"
    manifest["finished_at"] = utc_now()
    _write_json(artifact_root / "stage_manifest.json", manifest)


def analyze_command(args: argparse.Namespace, protocol: ExplorationProtocol) -> None:
    stage = str(args.stage)
    artifact_root = Path(args.artifact_root).resolve()
    manifest = _load_manifest(artifact_root / "stage_manifest.json")
    if manifest.get("protocol_hash") != protocol.protocol_hash:
        raise ProtocolError("stage manifest protocol hash differs")
    project_commit = str(manifest["project_commit"])
    artifact_base = Path(args.artifact_base).resolve() if args.artifact_base else artifact_root.parent
    selections, _ = dependency_context(
        protocol,
        stage,
        artifact_base=artifact_base,
        project_commit=project_commit,
    )
    settings = protocol.settings_for_stage(stage, selections=selections)
    analyze_stage(protocol=protocol, stage=stage, artifact_root=artifact_root, settings=settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-gated exploration tuning driver")
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate a protocol stage without training")
    validate.add_argument("--stage", required=True)
    validate.add_argument("--artifact-base", default=None)
    validate.add_argument("--project-commit", default=None)
    validate.add_argument("--python", default=sys.executable)
    run = subparsers.add_parser("run-stage", help="Run/resume, sample, and analyze a stage")
    run.add_argument("--stage", required=True)
    run.add_argument("--artifact-root", required=True)
    run.add_argument("--artifact-base", default=None)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--project-commit", default=None)
    run.add_argument("--python", default=sys.executable)
    analyze = subparsers.add_parser("analyze", help="Rebuild reports for a completed stage")
    analyze.add_argument("--stage", required=True)
    analyze.add_argument("--artifact-root", required=True)
    analyze.add_argument("--artifact-base", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    protocol = ExplorationProtocol.load(args.protocol)
    try:
        if args.command == "validate":
            artifact_base = Path(args.artifact_base).resolve() if args.artifact_base else None
            result = validate_stage(
                protocol=protocol,
                stage=str(args.stage),
                artifact_base=artifact_base,
                project_commit=args.project_commit,
                compose=True,
                python_executable=str(args.python),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "run-stage":
            if int(args.workers) <= 0:
                raise ProtocolError("--workers must be positive")
            run_stage(args, protocol)
        elif args.command == "analyze":
            analyze_command(args, protocol)
        else:  # pragma: no cover
            parser.error(f"unknown command: {args.command}")
    except (ProtocolError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
