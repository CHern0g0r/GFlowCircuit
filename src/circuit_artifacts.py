"""Verified circuit serialization shared by training archives and evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def implementation_provenance() -> dict:
    repo = Path(__file__).resolve().parents[1]
    files = sorted([*repo.glob("src/**/*.py"), *repo.glob("cfg/**/*.yaml")])
    hashes = {str(p.relative_to(repo)): sha256(p) for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    module = sys.modules.get("pyspiel")
    extension = getattr(module, "__file__", None)
    return {"git_commit": commit, "source_config_sha256": digest,
            "pyspiel_sha256": sha256(Path(extension)) if extension else None}


def circuit_metrics(path: Path, num_steps: int = 20) -> tuple[int, int]:
    import pyspiel

    state = pyspiel.load_game("circuit", {
        "num_steps": int(num_steps), "file_path": str(path.resolve()),
    }).new_initial_state()
    obs = state.observation_tensor(0)
    return int(obs[2]), int(obs[3])


def export_aig(*, circuit: str, actions: Sequence[int], path: Path,
               expected: tuple[int, int], num_steps: int = 20) -> str:
    """Replay separately; never serialize or strash the live training state."""
    import pyspiel

    state = pyspiel.load_game("circuit", {
        "num_steps": int(num_steps), "file_path": str(Path(circuit).resolve()),
    }).new_initial_state()
    for action in actions:
        if int(action) not in state.legal_actions():
            raise ValueError(f"Illegal replay action {action} for {circuit}")
        state.apply_action(int(action))
    obs = state.observation_tensor(0)
    actual = int(obs[2]), int(obs[3])
    if actual != tuple(expected):
        raise ValueError(f"Replay metrics mismatch: expected {expected}, got {actual}")
    if actions and not state.is_terminal():
        raise ValueError("Only terminal recipes may be exported")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.aig")
    try:
        if int(pyspiel.save_circuit(state, str(temporary))) != 1:
            raise RuntimeError(f"AIG export failed: {path}")
        if circuit_metrics(temporary, num_steps) != tuple(expected):
            raise ValueError(f"Reloaded AIG metrics mismatch: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(path)


def save_final_samples(*, rows: list[dict], root: Path, circuit: Path,
                       num_steps: int, metadata: dict, checkpoint: Path,
                       config_path: Path) -> Path:
    """Capture the existing ordered samples without consuming additional RNG."""
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Final sampling requires a fresh directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    initial = circuit_metrics(circuit, num_steps)
    reference_hash = export_aig(circuit=str(circuit), actions=[], path=root / "reference.aig",
                                expected=initial, num_steps=num_steps)
    metadata = {**metadata, "method": "drills_a2c" if metadata.get("method") == "drills" else metadata.get("method")}
    manifest = {
        "schema_version": 1, "kind": "final_sampling", "complete": False,
        "metadata": {**metadata, "source_sha256": sha256(circuit), "num_steps": num_steps,
                     "implementation": implementation_provenance(),
                     "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
                     "config_sha256": sha256(config_path)},
        "reference": {"aig_path": "reference.aig", "aig_sha256": reference_hash,
                      "aig_size": initial[0], "aig_depth": initial[1]},
        "records": [],
    }
    path = root / "manifest.json"
    write_json(path, manifest)
    for index, row in enumerate(rows):
        target = root / "aig" / f"{index:06d}.aig"
        digest = export_aig(circuit=str(circuit), actions=row["actions"], path=target,
                            expected=(int(row["size"]), int(row["depth"])), num_steps=num_steps)
        manifest["records"].append({
            "sample_index": index, "sample_id": index, "artifact_id": digest,
            "aig_sha256": digest, "aig_path": str(target.relative_to(root)),
            "aig_size": int(row["size"]), "aig_depth": int(row["depth"]),
            "actions": list(row["actions"]), "origin": "final_sampling",
        })
        write_json(path, manifest)
    manifest["complete"] = True
    write_json(path, manifest)
    return path
