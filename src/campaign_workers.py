"""Process-isolated method scheduling; imports neither PyTorch nor pyspiel."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from src.circuit_artifacts import sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def attempt_identity(protocol: dict, method: str, circuit: str, abc_path: Path, device: str) -> dict:
    files = sorted([*REPO.glob("src/**/*.py"), *REPO.glob("cfg/**/*.yaml"), *REPO.glob("scr/**/*.py")])
    source = {str(p.relative_to(REPO)): sha256(p) for p in files}
    spec = importlib.util.find_spec("pyspiel")
    if spec is None or spec.origin is None:
        raise RuntimeError("pyspiel extension is unavailable")
    abc_path = abc_path.resolve()
    if not abc_path.is_file() or not os.access(abc_path, os.X_OK):
        raise ValueError(f"ABC is not executable: {abc_path}")
    return {"version": 1, "method": method, "circuit": circuit, "device": device,
            "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
            "protocol_sha256": protocol["protocol_sha256"], "smoke": protocol.get("smoke", False),
            "source_protocol_sha256": protocol["source_protocol_sha256"],
            "circuit_sha256": sha256(REPO / protocol["baseline"]["circuits"][circuit]["circuit_path"]),
            "abc_sha256": sha256(abc_path), "pyspiel_sha256": sha256(Path(spec.origin))}


def inventory(root: Path, paths: list[Path]) -> dict:
    files = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file())
        else:
            files.add(path)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its directory: {relative}")
    return path


def verify_inventory(root: Path, files: dict) -> None:
    if not files:
        raise ValueError("Missing artifact inventory")
    for name, digest in files.items():
        if sha256(checked_path(root, name)) != digest:
            raise ValueError(f"Artifact changed: {root / name}")


def validate_inputs(attempt: Path, status: dict, *, verify_hashes: bool = True) -> None:
    """Validate complete populations and their saved artifacts without loading policies."""
    if not status.get("training_complete") or not status.get("sampling_complete"):
        raise ValueError("Training and sampling must be complete")
    if verify_hashes:
        verify_inventory(attempt, status.get("input_files", {}))
    protocol, identity = status["protocol"], status["identity"]
    common = protocol["baseline"]["common"]
    seeds, eval_seeds = common["training_seeds"], common["evaluation_seeds"]
    train = attempt / "train"
    config = train / ".hydra/config.yaml"
    from src.diversity_campaign import task_config
    _, _, report_file = task_config(protocol, identity["method"], identity["circuit"])
    if not (train / report_file).is_file() or not config.is_file():
        raise ValueError("Missing training report/config")
    checkpoints = sorted((train / "saved_models").glob("run_*/last.pt"))
    archives = sorted((train / "pareto_archives").glob("run_*/*/manifest.json"))
    samples = sorted((attempt / "final_sampling").glob("run_*/seed_*/manifest.json"))
    if len(checkpoints) != len(seeds) or len(archives) != len(seeds) or len(samples) != len(seeds) * len(eval_seeds):
        raise ValueError("Incomplete checkpoint/archive/sample population")
    expected_method = "drills_a2c" if identity["method"] == "drills" else identity["method"]
    seen_archives, seen_samples = set(), set()
    for path in archives + samples:
        data = json.loads(path.read_text())
        meta = data["metadata"]
        run_id = meta["run_id"]
        if (not data["complete"] or meta["method"] != expected_method or
                meta["source_sha256"] != identity["circuit_sha256"] or
                run_id not in range(len(seeds)) or meta["training_seed"] != seeds[run_id]):
            raise ValueError(f"Manifest identity mismatch: {path}")
        if path in archives:
            if data["kind"] != "training_archive" or data["snapshot"]["local_trajectory"] != common["training_trajectories"]:
                raise ValueError(f"Incomplete training archive: {path}")
            if run_id in seen_archives:
                raise ValueError("Duplicate training archive")
            seen_archives.add(run_id)
        else:
            key = (run_id, meta["evaluation_seed"])
            if key in seen_samples or key[1] not in eval_seeds or data["kind"] != "final_sampling":
                raise ValueError(f"Duplicate/invalid sample population: {path}")
            seen_samples.add(key)
            count = common["max_samples_per_seed"]
            if [r["sample_index"] for r in data["records"]] != list(range(count)):
                raise ValueError(f"Incomplete ordered samples: {path}")
            checkpoint = train / f"saved_models/run_{run_id}/last.pt"
            if sha256(checkpoint) != meta["checkpoint_sha256"] or sha256(config) != meta["config_sha256"]:
                raise ValueError(f"Checkpoint/config changed: {path}")
        for row in [data["reference"], *data["records"]]:
            if sha256(checked_path(path.parent, row["aig_path"])) != row["aig_sha256"]:
                raise ValueError(f"AIG changed: {path}")
        for name, digest in data.get("record_files", {}).items():
            if sha256(checked_path(path.parent, name)) != digest:
                raise ValueError(f"Archive events changed: {path}")
    for seed in eval_seeds:
        for suffix in ["", *(f"_n{b}" for b in common["sample_budgets"])]:
            if not (attempt / f"samples_seed_{seed}{suffix}.csv").is_file():
                raise ValueError("Missing sampling CSV")


def seal_inputs(attempt: Path, status: dict) -> None:
    validate_inputs(attempt, status, verify_hashes=False)
    status["input_files"] = inventory(attempt, [attempt / "train", attempt / "final_sampling",
                                                *attempt.glob("samples_seed_*.csv")])


def select_attempt(root: Path, identity: dict) -> tuple[Path, dict] | None:
    for attempt in sorted(root.glob("attempt_*"), reverse=True):
        path = attempt / "status.json"
        if not path.exists():
            continue  # killed before startup status; retain and start fresh
        status = json.loads(path.read_text())
        if status.get("identity") != identity:
            continue
        if status.get("training_complete") and status.get("sampling_complete"):
            validate_inputs(attempt, status)
            return attempt, status
    return None


def mapping_complete(attempt: Path, status: dict) -> bool:
    if not status.get("mapping_complete") or not status.get("complete"):
        return False
    verify_inventory(attempt, status.get("mapping_files", {}))
    result = json.loads((attempt / "diversity/status.json").read_text())
    metrics = json.loads((attempt / "diversity/metrics.json").read_text())
    if not result["complete"] or result["failed_count"] or not metrics["complete"] or metrics["failures"]:
        raise ValueError("Inconsistent completed mapping status")
    return True


@contextmanager
def method_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Method already running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stop_processes(active: dict) -> None:
    for process, _ in active.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(p.poll() is None for p, _ in active.values()):
        time.sleep(.05)
    for process, stream in active.values():
        # Also kill descendants when their parent has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        stream.close()


def run_method(protocol: dict, *, protocol_path: Path, method: str, stage: str, workers: int,
               circuits: list[str] | None, artifact_root: Path, abc_path: Path,
               device: str = "cuda") -> dict:
    if method not in protocol["methods"] or stage not in {"train-sample", "mapping"} or not 1 <= workers <= 2:
        raise ValueError("Choose a protocol method/stage and one or two workers")
    available = list(protocol["baseline"]["circuits"])
    if circuits is not None and (not circuits or len(set(circuits)) != len(circuits) or set(circuits) - set(available)):
        raise ValueError("Invalid circuit subset")
    selected = [c for c in available if circuits is None or c in circuits]
    root, abc_path = artifact_root.resolve(), abc_path.resolve()
    directory = root / "methods" / method
    with method_lock(directory / "method.lock"):
        invocation = directory / f"{stage}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        invocation.mkdir()
        status = {"method": method, "stage": stage, "complete": False,
                  "job_id": os.environ.get("SLURM_JOB_ID"), "circuits": {
                      c: {"state": "pending", "attempt": None, "exit_code": None,
                          "log": str(invocation / f"{c}.log")} for c in selected}}
        def persist():
            write_json(invocation / "status.json", status)
            write_json(directory / "status.json", status)
        pending, active, identities = list(selected), {}, {}
        stopped = False
        def terminate(signum, frame):
            nonlocal stopped
            stopped = True
        handlers = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
        persist()
        try:
            while pending or active:
                if stopped:
                    break
                while pending and len(active) < workers and not stopped:
                    circuit = pending.pop(0)
                    row = status["circuits"][circuit]
                    try:
                        identity = attempt_identity(protocol, method, circuit, abc_path, device)
                        identities[circuit] = identity
                        existing = select_attempt(root / f"{method}_{circuit}", identity)
                        if existing:
                            attempt, saved = existing
                            row["attempt"] = str(attempt)
                            if stage == "train-sample" or mapping_complete(attempt, saved):
                                row.update(state="skipped", exit_code=0)
                                persist()
                                continue
                        elif stage == "mapping":
                            raise ValueError("No matching completed training/sampling attempt")
                        if stage == "mapping":
                            command = [sys.executable, "-m", "src.diversity_campaign", "resume-mapping",
                                       "--attempt", str(attempt), "--abc-path", str(abc_path)]
                        else:
                            command = [sys.executable, "-m", "src.diversity_campaign", "--protocol", str(protocol_path),
                                       "run-task", "--method", method, "--circuit", circuit,
                                       "--artifact-root", str(root), "--abc-path", str(abc_path),
                                       "--device", device, "--stage", "train-sample"]
                        if stage == "train-sample" and protocol.get("smoke"):
                            command.insert(command.index("run-task"), "--smoke")
                        stream = Path(row["log"]).open("w")
                        try:
                            process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                                       start_new_session=True, env={**os.environ,
                                                           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                                           "MKL_NUM_THREADS": "1"})
                        except BaseException:
                            stream.close()
                            raise
                        active[circuit] = process, stream
                        row.update(state="running", pid=process.pid, command=command)
                    except Exception as exc:
                        row.update(state="failed", exit_code=1, error=str(exc))
                    persist()
                for circuit, (process, stream) in list(active.items()):
                    row = status["circuits"][circuit]
                    # Publish the attempt path while training is in progress.
                    if row["attempt"] is None:
                        attempts = sorted((root / f"{method}_{circuit}").glob("attempt_*"))
                        if attempts:
                            row["attempt"] = str(attempts[-1])
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    del active[circuit]
                    row.update(exit_code=code, state="failed" if code else "complete")
                    if code == 0:
                        try:
                            found = select_attempt(root / f"{method}_{circuit}", identities[circuit])
                            if found is None:
                                raise ValueError("Worker exited without complete outputs")
                            row["attempt"] = str(found[0])
                            if stage == "mapping" and not mapping_complete(*found):
                                raise ValueError("Worker exited without complete mapping")
                        except Exception as exc:
                            row.update(state="failed", exit_code=1, error=str(exc))
                    persist()
                if active:
                    persist()
                    time.sleep(.2)
        finally:
            stop_processes(active)
            for c in [*pending, *active]:
                status["circuits"][c].update(state="interrupted", exit_code=143)
            status["complete"] = all(r["state"] in {"complete", "skipped"} for r in status["circuits"].values())
            persist()
            for s, handler in handlers.items():
                signal.signal(s, handler)
        return status
