"""Append-only historical circuit artifacts with a mutable Pareto front."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from src.circuit_artifacts import append_json, export_aig, sha256, write_json
from src.discovery_metrics import CircuitDiscoveryArchive, DiscoveryPoint, _weakly_dominates


class PersistentCircuitArchive(CircuitDiscoveryArchive):
    def __init__(self, *, circuit: str, initial_size: int, initial_depth: int,
                 root: Path, metadata: dict[str, Any], num_steps: int,
                 exporter: Callable = export_aig) -> None:
        super().__init__(circuit=circuit, initial_size=initial_size, initial_depth=initial_depth)
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Training requires a fresh archive: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.exporter = exporter
        self.num_steps = int(num_steps)
        if self.num_steps <= 0:
            raise ValueError("Archive recipe horizon must be positive")
        self.metadata = {**metadata, "circuit": str(Path(circuit).resolve()),
                         "source_sha256": sha256(Path(circuit)), "num_steps": self.num_steps}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.export_seconds = 0.0
        self.admissions = 0
        self.repeats = 0
        self.complete = False
        self.reference_hash = self._export([], self.root / "reference.aig", (initial_size, initial_depth))
        self.persist()

    def _export(self, actions: Sequence[int], path: Path, expected: tuple[int, int]) -> str:
        start = time.monotonic()
        try:
            return self.exporter(circuit=self.circuit, actions=actions, path=path,
                                 expected=expected, num_steps=self.num_steps)
        finally:
            self.export_seconds += time.monotonic() - start

    def record_terminal(self, *, initial_size: int, initial_depth: int,
                        final_size: int, final_depth: int, actions: Sequence[int],
                        origin: str = "training") -> bool:
        if (initial_size, initial_depth) != (self.initial_size, self.initial_depth):
            raise ValueError("Initial circuit metrics changed")
        if origin not in ("training", "calibration"):
            raise ValueError(f"Not a training discovery: {origin}")
        if actions is None or len(actions) != self.num_steps:
            raise ValueError("Archive requires a complete terminal action sequence")
        if min(final_size, final_depth) < 0:
            self.record_failure()
            return False
        index = self.attempted_trajectories + 1
        key = int(final_size), int(final_depth)
        row = {"trajectory_index": index, "sample_index": index, "origin": origin,
               "actions": list(actions), "aig_size": key[0], "aig_depth": key[1]}
        admitted = not any(_weakly_dominates(p, key) for p in self._points)
        if admitted:
            candidate = self.root / "candidate.aig"
            try:
                digest = self._export(actions, candidate, key)
            except Exception as exc:
                write_json(self.root / "status.json", {"complete": False, "error": str(exc),
                                                       "trajectory_index": index})
                raise
            row["aig_sha256"] = digest
            row["artifact_id"] = digest
            if digest in self.artifacts:
                candidate.unlink()
                self.repeats += 1
                row["admission"] = "repeat"
            else:
                target = self.root / "aig" / f"{digest}.aig"
                target.parent.mkdir(exist_ok=True)
                candidate.replace(target)
                row["admission"] = "tied_coordinate" if key in self._points else "new_coordinate"
                self.artifacts[digest] = {**row, "aig_path": str(target.relative_to(self.root))}
                self.admissions += 1
            retired = [p for p in self._points if _weakly_dominates(key, p)]
            event = {**row, "retired_coordinates": retired}
            append_json(self.root / "events.jsonl", event)
            self.events.append(event)
            for p in retired:
                del self._points[p]
            self._points.setdefault(key, DiscoveryPoint(*key, index))
        else:
            row["admission"] = "dominated"
        self.attempted_trajectories = index
        append_json(self.root / "training_terminals.jsonl", row)
        self.persist()
        return admitted

    def record_failure(self) -> int:
        index = super().record_failure()
        append_json(self.root / "training_terminals.jsonl", {
            "trajectory_index": index, "status": "failed", "origin": "training"})
        self.persist()
        return index

    def snapshot(self) -> dict[str, Any]:
        active = [a for a in self.artifacts.values() if (a["aig_size"], a["aig_depth"]) in self._points]
        return {**super().snapshot(), "historical_artifact_count": len(self.artifacts),
                "historical_coordinate_count": len({(a["aig_size"], a["aig_depth"]) for a in self.artifacts.values()}),
                "active_generated_artifact_count": len(active),
                "admission_count": self.admissions, "repeat_count": self.repeats,
                "artifact_bytes": sum(p.stat().st_size for p in self.root.glob("aig/*.aig"))
                                  + (self.root / "reference.aig").stat().st_size,
                "export_seconds": self.export_seconds}

    def persist(self) -> None:
        front = [{"size": p.size, "depth": p.depth,
                  "artifact_ids": [h for h, a in self.artifacts.items()
                                   if (a["aig_size"], a["aig_depth"]) == (p.size, p.depth)],
                  "includes_reference": (p.size, p.depth) == (self.initial_size, self.initial_depth)}
                 for p in self.points]
        write_json(self.root / "front.json", front)
        write_json(self.root / "manifest.json", {
            "schema_version": 1, "kind": "training_archive", "metadata": self.metadata,
            "serialization": "pyspiel.save_circuit AIGER SHA-256; not graph-isomorphism canonical",
            "reference": {"aig_size": self.initial_size, "aig_depth": self.initial_depth,
                          "aig_path": "reference.aig", "aig_sha256": self.reference_hash},
            "records": list(self.artifacts.values()), "front": front,
            "snapshot": self.snapshot(), "complete": self.complete,
            "record_files": {name: sha256(self.root / name) for name in (
                "training_terminals.jsonl", "events.jsonl") if (self.root / name).exists()},
        })
        write_json(self.root / "status.json", {"complete": self.complete})

    def finalize(self) -> None:
        self.complete = True
        self.persist()
