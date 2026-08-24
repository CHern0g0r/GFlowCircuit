"""Run and validate the final multi-budget baseline evaluation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Mapping

import yaml

if TYPE_CHECKING:
    import pandas as pd


PROTOCOL_PATH = Path("cfg/exp/baseline_evaluation/protocol.yaml")
EXPECTED_METHODS = ("reinforce", "drills", "ppo")
EXPECTED_CIRCUITS = (
    "C1355",
    "C5315",
    "adder",
    "apex1",
    "bc0",
    "dalu",
    "k2",
    "max",
    "multiplier",
)
IDENTITY_COLUMNS = ["method", "circuit_name", "training_seed", "evaluation_seed", "sample_id"]
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT_HASH = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationTask:
    method: str
    circuit: str
    circuit_index: int
    config_name: str
    report_algorithm: str
    report_file: str
    dataset_cfg: str
    circuit_path: str
    expected: dict[str, Any]

    @property
    def task_id(self) -> str:
        return f"{self.method}__{self.circuit}"


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


class BaselineEvaluationProtocol:
    def __init__(self, *, path: Path, data: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.repo_root = self.path.parents[3]
        self.data = dict(data)
        self.protocol_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._validate_structure()

    @classmethod
    def load(cls, path: Path | str = PROTOCOL_PATH) -> "BaselineEvaluationProtocol":
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ProtocolError("protocol root must be a mapping")
        return cls(path=resolved, data=data)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.data["methods"])

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

    def _validate_structure(self) -> None:
        required = {"version", "campaign", "common", "methods", "circuits", "martin"}
        missing = required - set(self.data)
        if missing:
            raise ProtocolError(f"protocol missing keys: {sorted(missing)}")
        if int(self.data["version"]) != 1:
            raise ProtocolError("only baseline evaluation protocol version 1 is supported")
        if self.methods != EXPECTED_METHODS:
            raise ProtocolError(f"methods must be ordered as {EXPECTED_METHODS}")
        if self.circuits != EXPECTED_CIRCUITS:
            raise ProtocolError(f"circuits must be ordered as {EXPECTED_CIRCUITS}")
        expected_seeds = tuple(range(10))
        if self.training_seeds != expected_seeds or self.evaluation_seeds != expected_seeds:
            raise ProtocolError("training and evaluation seeds must both be 0 through 9")
        if self.sample_budgets != (10, 50, 100, 200):
            raise ProtocolError("sample budgets must be [10, 50, 100, 200]")
        if self.max_samples != self.sample_budgets[-1]:
            raise ProtocolError("max_samples_per_seed must equal the largest sample budget")
        if int(self.data["common"]["training_trajectories"]) != 800:
            raise ProtocolError("training_trajectories must be 800")

        for method, cfg in self.data["methods"].items():
            config_path = self.repo_root / "cfg" / f"{cfg['config_name']}.yaml"
            if not config_path.is_file():
                raise ProtocolError(f"missing {method} config: {config_path}")
            episodes = int(cfg["expected"]["episodes"])
            trajectories_per_episode = int(cfg["trajectories_per_episode"])
            if episodes * trajectories_per_episode != 800:
                raise ProtocolError(f"{method} does not resolve to 800 trajectories")
        for circuit, cfg in self.data["circuits"].items():
            for key in ("dataset_cfg", "circuit_path"):
                path = self.repo_root / str(cfg[key])
                if not path.is_file():
                    raise ProtocolError(f"missing {circuit} {key}: {path}")

        martin = self.data["martin"]
        if martin.get("project") != "gflowcircuit-baselines":
            raise ProtocolError("Martin project must be gflowcircuit-baselines")
        if tuple(martin.get("job_names", {})) != self.methods:
            raise ProtocolError("Martin job names must cover methods in protocol order")
        for job_name in martin["job_names"].values():
            if not SAFE_COMPONENT.fullmatch(str(job_name)):
                raise ProtocolError(f"unsafe Martin job name: {job_name}")

    def task(self, *, method: str, circuit_index: int) -> EvaluationTask:
        if method not in self.methods:
            raise ProtocolError(f"unknown method: {method}")
        if circuit_index < 0 or circuit_index >= len(self.circuits):
            raise ProtocolError(f"circuit index must be in [0, {len(self.circuits) - 1}]")
        circuit = self.circuits[circuit_index]
        method_cfg = self.data["methods"][method]
        circuit_cfg = self.data["circuits"][circuit]
        expected = {
            **dict(self.data["common"]["expected"]),
            **dict(method_cfg["expected"]),
        }
        return EvaluationTask(
            method=method,
            circuit=circuit,
            circuit_index=int(circuit_index),
            config_name=str(method_cfg["config_name"]),
            report_algorithm=str(method_cfg["report_algorithm"]),
            report_file=str(method_cfg["report_file"]),
            dataset_cfg=str(circuit_cfg["dataset_cfg"]),
            circuit_path=str(circuit_cfg["circuit_path"]),
            expected=expected,
        )

    def tasks(self) -> list[EvaluationTask]:
        return [
            self.task(method=method, circuit_index=index)
            for method in self.methods
            for index in range(len(self.circuits))
        ]


def _train_command(
    task: EvaluationTask,
    *,
    train_dir: Path,
    python_executable: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "src.run",
        "--config-name",
        task.config_name,
        f"run_name=baseline_{task.method}_{task.circuit}",
        f"dataset_cfg={task.dataset_cfg}",
        f"hydra.run.dir={train_dir}",
    ]


def _validate_resolved_config(
    task: EvaluationTask,
    *,
    config: Mapping[str, Any],
) -> None:
    if str(config.get("dataset_cfg")) != task.dataset_cfg:
        raise ValueError(
            f"dataset_cfg mismatch: expected {task.dataset_cfg}, got {config.get('dataset_cfg')}"
        )
    expected_run_name = f"baseline_{task.method}_{task.circuit}"
    if str(config.get("run_name")) != expected_run_name:
        raise ValueError(
            f"run_name mismatch: expected {expected_run_name}, got {config.get('run_name')}"
        )
    for path, expected in task.expected.items():
        try:
            actual = _select_path(config, str(path))
        except KeyError as exc:
            raise ValueError(f"resolved config is missing {path}") from exc
        if not _values_equal(actual, expected):
            raise ValueError(f"resolved config mismatch for {path}: expected {expected!r}, got {actual!r}")


def validate_compositions(
    protocol: BaselineEvaluationProtocol,
    *,
    python_executable: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def compose(task: EvaluationTask) -> None:
        train_dir = output_dir / task.task_id / "train"
        command = _train_command(task, train_dir=train_dir, python_executable=python_executable)
        command.extend(["--cfg", "job", "--resolve"])
        completed = subprocess.run(
            command,
            cwd=protocol.repo_root,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_path = output_dir / f"{task.task_id}.yaml"
        stderr_path = output_dir / f"{task.task_id}.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise ProtocolError(f"Hydra composition failed for {task.task_id}; see {stderr_path}")
        resolved = yaml.safe_load(completed.stdout)
        if not isinstance(resolved, Mapping):
            raise ProtocolError(f"Hydra composition produced no mapping for {task.task_id}")
        try:
            _validate_resolved_config(task, config=resolved)
        except ValueError as exc:
            raise ProtocolError(f"invalid resolved config for {task.task_id}: {exc}") from exc

    tasks = protocol.tasks()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as executor:
        futures = {executor.submit(compose, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except (OSError, ProtocolError, ValueError) as exc:
                failures.append(f"{task.task_id}: {exc}")
    if failures:
        raise ProtocolError("Hydra composition failures:\n" + "\n".join(sorted(failures)))


def validate_protocol(
    protocol: BaselineEvaluationProtocol,
    *,
    python_executable: str,
    compose: bool,
) -> dict[str, Any]:
    tasks = protocol.tasks()
    if len(tasks) != 27 or len({task.task_id for task in tasks}) != 27:
        raise ProtocolError("protocol must expand to 27 unique tasks")
    if compose:
        with TemporaryDirectory(prefix="gfc-baseline-eval-compose-") as directory:
            validate_compositions(
                protocol,
                python_executable=python_executable,
                output_dir=Path(directory),
            )
    return {
        "campaign": protocol.data["campaign"],
        "protocol_hash": protocol.protocol_hash,
        "methods": len(protocol.methods),
        "circuits": len(protocol.circuits),
        "tasks": len(tasks),
        "trained_models": len(tasks) * len(protocol.training_seeds),
        "unique_evaluation_rollouts": (
            len(tasks)
            * len(protocol.training_seeds)
            * len(protocol.evaluation_seeds)
            * protocol.max_samples
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


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return int(completed.returncode)


def _validate_training_artifacts(
    task: EvaluationTask,
    *,
    train_dir: Path,
    protocol: BaselineEvaluationProtocol,
) -> None:
    config_path = train_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"resolved Hydra config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError(f"resolved Hydra config is invalid: {config_path}")
    _validate_resolved_config(task, config=config)

    checkpoints = sorted((train_dir / "saved_models").glob("run_*/last.pt"))
    if len(checkpoints) != len(protocol.training_seeds):
        raise ValueError(f"expected 10 checkpoints, found {len(checkpoints)}")
    observed: list[tuple[int, int]] = []
    for checkpoint in checkpoints:
        payload = _torch_load(checkpoint)
        observed.append((int(payload.get("run_idx", -1)), int(payload.get("seed", -1))))
    expected = list(enumerate(protocol.training_seeds))
    if sorted(observed) != expected:
        raise ValueError(f"checkpoint run/seed pairs differ: expected {expected}, got {sorted(observed)}")

    report_path = train_dir / task.report_file
    if not report_path.is_file():
        raise ValueError(f"training report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("algorithm") != task.report_algorithm:
        raise ValueError(
            f"report algorithm mismatch: expected {task.report_algorithm}, got {report.get('algorithm')}"
        )
    report_runs = report.get("runs", [])
    report_pairs = sorted((int(row["run_idx"]), int(row["seed"])) for row in report_runs)
    if report_pairs != expected:
        raise ValueError(f"report run/seed pairs differ: expected {expected}, got {report_pairs}")


def _reference_frame(reference: Mapping[str, Any], *, sample_budget: int | None = None) -> pd.DataFrame:
    import pandas as pd

    row = dict(reference)
    if sample_budget is not None:
        row["sample_budget"] = int(sample_budget)
    return pd.DataFrame([row])


def _sample_task(
    task: EvaluationTask,
    *,
    attempt_dir: Path,
    protocol: BaselineEvaluationProtocol,
    device: Any,
) -> None:
    import pandas as pd

    from src.sample_exp import evaluation_reference_row, sample_paired_evaluation_seed

    train_dir = attempt_dir / "train"
    circuit_path = (protocol.repo_root / task.circuit_path).resolve()
    raw_dir = attempt_dir / "evaluation" / "raw"
    budget_dir = attempt_dir / "evaluation" / "budgets"
    frames: list[pd.DataFrame] = []
    for evaluation_seed in protocol.evaluation_seeds:
        frame = sample_paired_evaluation_seed(
            experiment_dir=train_dir,
            circuit_path=circuit_path,
            method=task.method,
            circuit_name=task.circuit,
            num_samples=protocol.max_samples,
            evaluation_seed=evaluation_seed,
            device=device,
            num_steps=int(protocol.data["common"]["num_steps"]),
        )
        _write_csv(raw_dir / f"seed_{evaluation_seed:02d}.csv", frame)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    reference = evaluation_reference_row(
        circuit_path=circuit_path,
        method=task.method,
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
        _write_csv(budget_dir / f"points_n{budget:03d}.csv", output)


def _sample_identity(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    samples = frame.loc[frame["run_id"].notna()]
    return set(samples[IDENTITY_COLUMNS].itertuples(index=False, name=None))


def _sample_payload(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    samples = frame.loc[frame["run_id"].notna()]
    columns = [*IDENTITY_COLUMNS, "run_id", "size", "depth", "source_checkpoint"]
    return sorted(samples[columns].itertuples(index=False, name=None))


def validate_attempt(
    task: EvaluationTask,
    *,
    attempt_dir: Path,
    protocol: BaselineEvaluationProtocol,
) -> dict[str, int]:
    import pandas as pd

    train_dir = attempt_dir / "train"
    _validate_training_artifacts(task, train_dir=train_dir, protocol=protocol)

    raw_frames: list[pd.DataFrame] = []
    for evaluation_seed in protocol.evaluation_seeds:
        path = attempt_dir / "evaluation" / "raw" / f"seed_{evaluation_seed:02d}.csv"
        if not path.is_file():
            raise ValueError(f"raw evaluation file is missing: {path}")
        frame = pd.read_csv(path)
        expected_rows = len(protocol.training_seeds) * protocol.max_samples
        if len(frame) != expected_rows:
            raise ValueError(f"{path} has {len(frame)} rows; expected {expected_rows}")
        if set(frame["evaluation_seed"].astype(int)) != {evaluation_seed}:
            raise ValueError(f"{path} contains the wrong evaluation seed")
        for training_seed in protocol.training_seeds:
            group = frame.loc[frame["training_seed"].astype(int) == training_seed]
            if list(group["sample_id"].astype(int)) != list(range(protocol.max_samples)):
                raise ValueError(
                    f"{path} training seed {training_seed} does not contain ordered samples 0..{protocol.max_samples - 1}"
                )
        raw_frames.append(frame)

    raw = pd.concat(raw_frames, ignore_index=True)
    expected_raw = len(protocol.training_seeds) * len(protocol.evaluation_seeds) * protocol.max_samples
    if len(raw) != expected_raw:
        raise ValueError(f"raw evaluation has {len(raw)} rows; expected {expected_raw}")
    if raw.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("raw evaluation contains duplicate sample identities")
    expected_training_seeds = set(protocol.training_seeds)
    if set(raw["training_seed"].astype(int)) != expected_training_seeds:
        raise ValueError("raw evaluation contains the wrong training seeds")

    points_path = attempt_dir / "points.csv"
    if not points_path.is_file():
        raise ValueError(f"canonical points file is missing: {points_path}")
    canonical = pd.read_csv(points_path)
    references = canonical.loc[canonical["run_id"].isna()]
    samples = canonical.loc[canonical["run_id"].notna()]
    if len(references) != 1 or len(samples) != expected_raw:
        raise ValueError(
            f"canonical points count mismatch: references={len(references)}, samples={len(samples)}"
        )
    if _sample_identity(canonical) != _sample_identity(raw) or _sample_payload(canonical) != _sample_payload(raw):
        raise ValueError("canonical points do not match raw evaluation samples")

    for budget in protocol.sample_budgets:
        path = attempt_dir / "evaluation" / "budgets" / f"points_n{budget:03d}.csv"
        if not path.is_file():
            raise ValueError(f"budget view is missing: {path}")
        frame = pd.read_csv(path)
        references = frame.loc[frame["run_id"].isna()]
        samples = frame.loc[frame["run_id"].notna()]
        expected_rows = len(protocol.training_seeds) * len(protocol.evaluation_seeds) * budget
        if len(references) != 1 or len(samples) != expected_rows:
            raise ValueError(
                f"{path} count mismatch: references={len(references)}, samples={len(samples)}, expected={expected_rows}"
            )
        if set(frame["sample_budget"].dropna().astype(int)) != {budget}:
            raise ValueError(f"{path} contains the wrong sample_budget")
        expected_identity = _sample_identity(raw.loc[raw["sample_id"] < budget])
        expected_payload = _sample_payload(raw.loc[raw["sample_id"] < budget])
        if _sample_identity(frame) != expected_identity or _sample_payload(frame) != expected_payload:
            raise ValueError(f"{path} is not the exact nested prefix for budget {budget}")

    return {
        "checkpoint_count": len(protocol.training_seeds),
        "evaluation_seed_count": len(protocol.evaluation_seeds),
        "unique_sample_count": expected_raw,
    }


def run_task(
    protocol: BaselineEvaluationProtocol,
    *,
    method: str,
    circuit_index: int,
    artifact_root: Path,
    project_commit: str,
    python_executable: str,
    device_name: str,
) -> dict[str, Any]:
    import torch

    if not COMMIT_HASH.fullmatch(project_commit):
        raise ProtocolError("run-task requires a full 40- or 64-character lowercase project commit hash")
    task = protocol.task(method=method, circuit_index=circuit_index)
    task_root = artifact_root.resolve() / "tasks" / task.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = _next_attempt_dir(task_root)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    train_dir = attempt_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=False)
    train_command = _train_command(task, train_dir=train_dir, python_executable=python_executable)
    status_path = task_root / "task_status.json"
    started = time.monotonic()
    metadata: dict[str, Any] = {
        "task_id": task.task_id,
        "method": task.method,
        "circuit": task.circuit,
        "circuit_index": task.circuit_index,
        "state": "running",
        "attempt_dir": str(attempt_dir),
        "protocol_hash": protocol.protocol_hash,
        "project_commit": project_commit,
        "device": device_name,
        "started_at": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "train_command": train_command,
    }
    _write_json(attempt_dir / "attempt.json", metadata)
    _write_json(status_path, metadata)

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["GFC_TIMING"] = "1"
    error: str | None = None
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
    try:
        if train_code != 0:
            raise ValueError(f"training exited with code {train_code}")
        _validate_training_artifacts(task, train_dir=train_dir, protocol=protocol)
        sample_started = time.monotonic()
        _sample_task(
            task,
            attempt_dir=attempt_dir,
            protocol=protocol,
            device=torch.device(device_name),
        )
        metadata["sample_wall_time_seconds"] = time.monotonic() - sample_started
        counts = validate_attempt(task, attempt_dir=attempt_dir, protocol=protocol)
        metadata.update(counts)
        metadata["state"] = "complete"
    except Exception as exc:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final multi-budget baseline evaluation runner")
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate the full matrix without training")
    validate.add_argument("--python", default=sys.executable)
    validate.add_argument("--skip-compose", action="store_true")

    run = subparsers.add_parser("run-task", help="Train and evaluate one method/circuit array task")
    run.add_argument("--method", required=True, choices=EXPECTED_METHODS)
    run.add_argument("--circuit-index", required=True, type=int)
    run.add_argument("--artifact-root", required=True)
    run.add_argument("--project-commit", required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        protocol = BaselineEvaluationProtocol.load(args.protocol)
        if args.command == "validate":
            result = validate_protocol(
                protocol,
                python_executable=str(args.python),
                compose=not bool(args.skip_compose),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "run-task":
            result = run_task(
                protocol,
                method=str(args.method),
                circuit_index=int(args.circuit_index),
                artifact_root=Path(args.artifact_root),
                project_commit=str(args.project_commit),
                python_executable=str(args.python),
                device_name=str(args.device),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        else:  # pragma: no cover
            parser.error(f"unknown command: {args.command}")
    except (ProtocolError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
