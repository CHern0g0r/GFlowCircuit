"""Stage-gated runner for baseline optimizer and trajectory-budget tuning.

The module prepares and executes one reviewed stage. It never submits another
SLURM job and never advances past a failed selection gate.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from src.baseline_tuning_analysis import analyze_stage
from src.baseline_tuning_protocol import (
    BaselineTuningProtocol,
    PROTOCOL_PATH,
    ProtocolError,
)
from src.exploration_analysis import collect_task_records
from src.exploration_tuning import (
    _gpu_devices,
    _write_json,
    build_stage_manifest,
    run_task,
    utc_now,
    validate_hydra_compositions,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"invalid JSON object: {path}")
    return value


def dependency_context(
    protocol: BaselineTuningProtocol,
    stage: str,
    *,
    artifact_base: Path,
    project_commit: str | None,
) -> tuple[dict[str, Any], list[Path]]:
    payloads: dict[str, Any] = {}
    roots: list[Path] = []
    observed_commit = project_commit
    for dependency in protocol.dependency_stages(stage):
        root = protocol.artifact_root(dependency, artifact_base=artifact_base)
        manifest = _load_json(root / "stage_manifest.json")
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
        payloads[dependency] = _load_json(root / "selection.json")
        roots.append(root)
    return payloads, roots


def validate_stage(
    *,
    protocol: BaselineTuningProtocol,
    stage: str,
    artifact_base: Path | None = None,
    project_commit: str | None = None,
    compose: bool = True,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    roots: list[Path] = []
    if protocol.stages[stage].get("dependencies"):
        base = artifact_base or Path(protocol.data["martin"]["artifact_base"])
        payloads, roots = dependency_context(
            protocol,
            stage,
            artifact_base=base,
            project_commit=project_commit,
        )
    settings = protocol.settings_for_stage(stage, payloads=payloads)
    tasks = protocol.build_tasks(stage, settings)
    if compose and tasks:
        with TemporaryDirectory(prefix="gfc-baseline-compose-") as directory:
            validate_hydra_compositions(
                protocol=protocol,
                tasks=tasks,
                output_dir=Path(directory),
                python_executable=python_executable,
            )
    return {
        "stage": stage,
        "kind": protocol.stages[stage]["kind"],
        "protocol_hash": protocol.protocol_hash,
        "settings": len(settings),
        "tasks": len(tasks),
        "training_trajectories": sum(task.training_trajectories for task in tasks),
        "evaluation_rollouts": sum(task.evaluation_samples for task in tasks),
        "dependencies": [str(root) for root in roots],
    }


def _smoke_runtime_projection(protocol: BaselineTuningProtocol, artifact_root: Path) -> dict[str, Any]:
    records = collect_task_records(artifact_root)
    seconds_per_trajectory: dict[str, float] = {}
    seconds_per_sample: dict[str, float] = {}
    for record in records:
        status = record["execution_status"]
        algorithm = str(record["algorithm"])
        seconds_per_trajectory[algorithm] = float(status.get("train_wall_time_seconds", 0.0)) / max(
            1, int(record["training_trajectories"])
        )
        seconds_per_sample[algorithm] = float(status.get("sample_wall_time_seconds", 0.0)) / max(
            1, int(record["evaluation_samples"])
        )
    common = protocol.data["common"]
    grid_sum = sum(int(value) for value in common["budget_grid"])
    extension = int(common["conditional_extension_trajectory"])
    estimates: dict[str, Any] = {}
    for algorithm, cfg in protocol.algorithms.items():
        train_rate = seconds_per_trajectory.get(algorithm, 0.0)
        sample_rate = seconds_per_sample.get(algorithm, 0.0)
        profiles = len(cfg["profiles"])
        matrices = {
            "screen": (profiles * 2 * 3 * int(common["screen_training_trajectories"]), profiles * 2 * 3 * 50),
            "budget_curve": (2 * 2 * 3 * grid_sum, 2 * 5 * 2 * 3 * 50),
            "conditional_extension": (2 * 2 * 3 * extension, 2 * 2 * 3 * 50),
            "confirmation_worst_case": (3 * 2 * 10 * extension, 3 * 2 * 10 * 50),
        }
        estimates[algorithm] = {}
        for name, (trajectories, samples) in matrices.items():
            gpu_hours = (train_rate * trajectories + sample_rate * samples) / 3600.0
            estimates[algorithm][name] = {
                "training_trajectories": trajectories,
                "evaluation_rollouts": samples,
                "projected_gpu_hours": gpu_hours,
                "projected_four_gpu_wall_hours": gpu_hours / 4.0,
                "exceeds_72h_four_gpu_job": gpu_hours / 4.0 > 72.0,
            }
    return {
        "warning": (
            "Linear projection from a 20-trajectory smoke is provisional; "
            "split production stages if the 72h flag is true."
        ),
        "seconds_per_training_trajectory": seconds_per_trajectory,
        "seconds_per_evaluation_sample": seconds_per_sample,
        "estimates": estimates,
    }


def _record_roots_for_analysis(
    protocol: BaselineTuningProtocol,
    stage: str,
    artifact_root: Path,
    dependency_roots: list[Path],
) -> list[Path]:
    if protocol.stages[stage]["kind"] != "extend":
        return [artifact_root]
    direct = str(protocol.stages[stage]["dependencies"][0])
    direct_root = next(root for root in dependency_roots if root.name == protocol.data["martin"]["job_names"][direct])
    return [direct_root, artifact_root]


def run_stage(args: argparse.Namespace, protocol: BaselineTuningProtocol) -> None:
    stage = str(args.stage)
    project_commit = str(args.project_commit or os.environ.get("GFC_PROJECT_COMMIT", ""))
    if not project_commit or project_commit == "SET_PROJECT_COMMIT_BEFORE_SYNC":
        raise ProtocolError("run-stage requires --project-commit or GFC_PROJECT_COMMIT")
    artifact_root = Path(args.artifact_root).resolve()
    artifact_base = Path(args.artifact_base).resolve() if args.artifact_base else artifact_root.parent
    payloads, dependency_roots = dependency_context(
        protocol,
        stage,
        artifact_base=artifact_base,
        project_commit=project_commit,
    )
    settings = protocol.settings_for_stage(stage, payloads=payloads)
    manifest, tasks = build_stage_manifest(
        protocol=protocol,
        stage=stage,
        artifact_root=artifact_root,
        artifact_base=artifact_base,
        project_commit=project_commit,
        settings=settings,
        dependency_roots=dependency_roots,
    )
    if tasks:
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

        def execute(task: Any) -> None:
            task_status = artifact_root / "tasks" / task.task_id / "task_status.json"
            if task_status.is_file() and _load_json(task_status).get("state") == "reused":
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
            futures = {executor.submit(execute, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as exc:
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
        record_roots=_record_roots_for_analysis(protocol, stage, artifact_root, dependency_roots),
        dependency_payloads=payloads,
    )
    if protocol.stages[stage]["kind"] == "smoke":
        projection = _smoke_runtime_projection(protocol, artifact_root)
        _write_json(artifact_root / "runtime_projection.json", projection)
        result["runtime_projection"] = projection
        _write_json(artifact_root / "selection.json", result)
    manifest["state"] = "complete"
    manifest["finished_at"] = utc_now()
    _write_json(artifact_root / "stage_manifest.json", manifest)


def analyze_command(args: argparse.Namespace, protocol: BaselineTuningProtocol) -> None:
    stage = str(args.stage)
    artifact_root = Path(args.artifact_root).resolve()
    manifest = _load_json(artifact_root / "stage_manifest.json")
    if manifest.get("protocol_hash") != protocol.protocol_hash:
        raise ProtocolError("stage manifest protocol hash differs")
    artifact_base = Path(args.artifact_base).resolve() if args.artifact_base else artifact_root.parent
    payloads, dependency_roots = dependency_context(
        protocol,
        stage,
        artifact_base=artifact_base,
        project_commit=str(manifest["project_commit"]),
    )
    settings = protocol.settings_for_stage(stage, payloads=payloads)
    analyze_stage(
        protocol=protocol,
        stage=stage,
        artifact_root=artifact_root,
        settings=settings,
        record_roots=_record_roots_for_analysis(protocol, stage, artifact_root, dependency_roots),
        dependency_payloads=payloads,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline training hyperparameter-tuning driver")
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate and compose a stage without training")
    validate.add_argument("--stage", required=True)
    validate.add_argument("--artifact-base", default=None)
    validate.add_argument("--project-commit", default=None)
    validate.add_argument("--python", default=sys.executable)
    validate.add_argument(
        "--skip-compose",
        action="store_true",
        help="Validate only the declarative matrix when the local training environment is unavailable",
    )
    run = subparsers.add_parser("run-stage", help="Run/resume, sample, and analyze one stage")
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
    protocol = BaselineTuningProtocol.load(args.protocol)
    try:
        if args.command == "validate":
            result = validate_stage(
                protocol=protocol,
                stage=str(args.stage),
                artifact_base=Path(args.artifact_base).resolve() if args.artifact_base else None,
                project_commit=args.project_commit,
                compose=not bool(args.skip_compose),
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
