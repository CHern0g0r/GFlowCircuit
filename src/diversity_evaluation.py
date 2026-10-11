"""Resumable, policy-free LUT evaluation of archive and sampling manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.circuit_artifacts import sha256, write_json
from src.sampling_diversity import (
    _best_value, _resolve_abc, map_to_lut, pareto_front,
    recipe_diversity_metrics, spearman_rank_correlation,
)


def hypervolume(points: Sequence[tuple[int, int]], reference: tuple[int, int]) -> float:
    if min(reference) <= 0:
        raise ValueError("Original size/depth must be positive for normalization")
    current_y, volume = 1.0, 0.0
    for size, depth in sorted(set(points)):
        x, y = size / reference[0], depth / reference[1]
        if x < 1 and y < current_y:
            volume += (1 - x) * (current_y - y)
            current_y = y
    return volume


def population_metrics(rows: list[dict], reference: tuple[int, int], *,
                       permutations: int = 32, include_reference: bool = False,
                       artifacts_available: bool = True) -> dict:
    rows = [dict(row) for row in rows]
    for row in rows:
        row["aig_product"] = row["aig_size"] * row["aig_depth"]
        if "lut_size" in row:
            row["lut_product"] = row["lut_size"] * row["lut_depth"]
    coordinates = {(r["aig_size"], r["aig_depth"]) for r in rows}
    front_rows = rows + ([{"aig_size": reference[0], "aig_depth": reference[1],
                           "sample_index": -1}] if include_reference else [])
    aig_front = pareto_front(front_rows, size_key="aig_size", depth_key="aig_depth")
    active_indices = {i for p in aig_front for i in p["sample_indices"] if i != -1}
    mapped = [r for r in rows if "lut_size" in r]
    unique_aigs = len({r["aig_sha256"] for r in rows}) if artifacts_available else None
    unique_luts = len({r["lut_sha256"] for r in mapped})
    lut_pairs = {(r["lut_size"], r["lut_depth"]) for r in mapped}
    lut_front = pareto_front(mapped, size_key="lut_size", depth_key="lut_depth")
    result: dict[str, Any] = {
        "occurrence_count": len(rows), "coordinate_count": len(coordinates),
        "unique_aig_artifacts": unique_aigs,
        "unique_aig_fraction": unique_aigs / len(rows) if rows and unique_aigs is not None else None,
        "coordinate_fraction": len(coordinates) / len(rows) if rows else None,
        "hypervolume": hypervolume(list(coordinates), reference),
        "front_coordinate_count": len(aig_front),
        "front_generated_artifact_count": len({r["aig_sha256"] for r in rows
                                                if r["sample_index"] in active_indices}) if artifacts_available else None,
        "front_includes_reference": any(-1 in p["sample_indices"] for p in aig_front),
        "aig_front": aig_front,
        "mapping": {"requested": len(rows) if artifacts_available else 0,
                    "successful": len(mapped), "complete": artifacts_available and len(mapped) == len(rows)},
        "lut_coordinate_count": len(lut_pairs) if mapped or not rows else None,
        "unique_lut_artifacts": unique_luts if mapped or not rows else None,
        "unique_lut_fraction": unique_luts / len(mapped) if mapped else None,
        "lut_coordinate_fraction": len(lut_pairs) / len(mapped) if mapped else None,
        "lut_front_coordinate_count": len(lut_front) if mapped or not rows else None,
        "lut_front": lut_front,
        "recipe_diversity": recipe_diversity_metrics([r["actions"] for r in rows],
                                                     curve_permutations=permutations, seed=0) if rows else None,
        "quality": {},
    }
    # Metric indices refer to the stable manifest rows; hashes identify shared files.
    for prefix, population in (("aig", rows), ("lut", mapped)):
        quality = {}
        for name, suffix, ties in (("best_size", "size", ("depth",)),
                                   ("best_depth", "depth", ("size",)),
                                   ("best_size_depth_product", "product", ("size", "depth"))):
            value = _best_value(population, value_key=f"{prefix}_{suffix}",
                                tie_keys=tuple(f"{prefix}_{t}" for t in ties)) if population else None
            if value is not None:
                winner = next(r for r in population if r["sample_index"] == value["selected_sample_index"])
                value["selected_artifact"] = winner.get(f"{prefix}_path")
                value["selected_sha256"] = winner.get(f"{prefix}_sha256")
            quality[name] = value
        result["quality"][prefix] = quality
    result["rank_correlation"] = {
        metric: spearman_rank_correlation([r[f"aig_{metric}"] for r in mapped],
                                         [r[f"lut_{metric}"] for r in mapped]) if len(mapped) > 1 else None
        for metric in ("size", "depth")
    }
    return result


def _safe_input(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes manifest directory: {relative}")
    return path


def evaluate(manifests: Sequence[Path], *, output_dir: Path, abc_path: Path,
             k: int = 6, resume: bool = False, timeout_seconds: float = 120,
             budgets: Sequence[int] = (10, 50, 100, 200),
             milestone_interval: int = 50, permutations: int = 32) -> dict:
    if not 2 <= k <= 32 or timeout_seconds <= 0 or milestone_interval <= 0 or permutations <= 0:
        raise ValueError("Invalid mapping or metric settings")
    if not manifests or not budgets or any(b <= 0 for b in budgets):
        raise ValueError("Provide input manifests and positive sampling budgets")
    output_dir = output_dir.resolve()
    abc_path = _resolve_abc(str(abc_path))
    manifest_paths = [p.resolve() for p in manifests]
    if len(set(manifest_paths)) != len(manifest_paths):
        raise ValueError("Duplicate input manifest")
    identity = {"schema_version": 1, "inputs": {str(p): sha256(p) for p in manifest_paths},
                "abc_sha256": sha256(abc_path), "k": k,
                "command": f"read; strash; if -K {k}; print_stats; write_blif",
                "budgets": list(budgets), "milestone_interval": milestone_interval,
                "curve_permutations": permutations}
    identity_path = output_dir / "identity.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError("Evaluation output exists; use --resume")
        if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("Evaluation identity changed; use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(identity_path, identity)
    write_json(output_dir / "status.json", {"complete": False, "stage": "mapping"})
    start = time.monotonic()
    cache_root = output_dir / "mapping_cache"
    cache_root.mkdir(exist_ok=True)
    failures, groups, stats = [], [], []
    identities = set()
    mapping_outcomes: dict[str, dict | str] = {}
    model_sources: dict[tuple, tuple] = {}
    model_checkpoints: dict[tuple, tuple] = {}

    def mapped_record(record: dict, manifest_path: Path) -> dict:
        row = dict(record)
        try:
            aig = _safe_input(manifest_path.parent, row["aig_path"])
            if sha256(aig) != row["aig_sha256"]:
                raise ValueError(f"AIG checksum mismatch: {aig}")
            row["aig_path"] = str(aig)
            key = hashlib.sha256(json.dumps([row["aig_sha256"], identity["abc_sha256"],
                                             k, identity["command"]]).encode()).hexdigest()
            cached_path = cache_root / f"{key}.json"
            lut_path = cache_root / f"{key}.blif"
            log_path = cache_root / f"{key}.log"
            outcome = mapping_outcomes.get(key)
            if isinstance(outcome, str):
                raise RuntimeError(outcome)
            cached = outcome or (json.loads(cached_path.read_text()) if cached_path.exists() else None)
            if cached is None or not lut_path.exists() or sha256(lut_path) != cached["lut_sha256"]:
                temporary = cache_root / f"{key}.tmp.blif"
                temporary.unlink(missing_ok=True)
                try:
                    size, depth = map_to_lut(abc_path=abc_path, aig_path=aig, lut_path=temporary,
                                             timeout_seconds=timeout_seconds, k=k, log_path=log_path)
                    temporary.replace(lut_path)
                except Exception as exc:
                    mapping_outcomes[key] = str(exc)
                    if not log_path.exists():
                        log_path.write_text(str(exc))
                    raise
                finally:
                    temporary.unlink(missing_ok=True)
                cached = {"lut_size": size, "lut_depth": depth, "lut_sha256": sha256(lut_path),
                          "lut_path": str(lut_path), "mapping_key": key,
                          "mapping_log": str(log_path), "abc_sha256": identity["abc_sha256"], "k": k}
                write_json(cached_path, cached)
            mapping_outcomes[key] = cached
            row.update(cached)
            row["mapping_status"] = "complete"
        except Exception as exc:
            row["mapping_status"] = "failed"
            row["mapping_error"] = str(exc)
            failures.append({"manifest": str(manifest_path), "sample_index": row.get("sample_index"),
                             "error": str(exc)})
        return row

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("kind") not in {"training_archive", "final_sampling"}:
            raise ValueError(f"Unsupported manifest: {manifest_path}")
        metadata = manifest["metadata"]
        model_key = tuple(str(metadata.get(x)) for x in ("method", "circuit", "training_seed", "run_id"))
        source = (metadata.get("source_sha256"), metadata.get("num_steps"),
                  manifest["reference"]["aig_size"], manifest["reference"]["aig_depth"])
        if model_sources.setdefault(model_key, source) != source:
            raise ValueError("Cannot mix source circuits or environment settings within a model")
        checkpoint = metadata.get("checkpoint_sha256")
        sampling_identity = (checkpoint, metadata.get("config_sha256"), metadata.get("gflownet_batch_size"))
        if checkpoint and model_checkpoints.setdefault(model_key, sampling_identity) != sampling_identity:
            raise ValueError("Cannot mix checkpoints across evaluation seeds")
        for name, digest in manifest.get("record_files", {}).items():
            if sha256(_safe_input(manifest_path.parent, name)) != digest:
                raise ValueError(f"Archive record checksum mismatch: {name}")
        group_key = tuple(str(metadata.get(x)) for x in
                          ("method", "circuit", "training_seed", "run_id", "evaluation_seed")) + (manifest["kind"],)
        if group_key in identities:
            raise ValueError("Multiple attempts for one population; evaluate separately")
        identities.add(group_key)
        if not manifest.get("complete"):
            failures.append({"manifest": str(manifest_path), "error": "Input stage incomplete"})
        reference = manifest["reference"]
        original = int(reference["aig_size"]), int(reference["aig_depth"])
        hypervolume([], original)  # Validate before any expensive mapping.
        records = manifest["records"]
        indices = [r["sample_index"] for r in records]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate occurrence indices")
        if manifest["kind"] == "final_sampling" and indices != list(range(len(indices))):
            raise ValueError("Final samples must be an ordered contiguous sequence")
        mapped = [mapped_record(r, manifest_path) for r in records]
        mapped_reference = mapped_record({**reference, "sample_index": -1}, manifest_path)
        stats.extend({**metadata, "population": manifest["kind"], **r} for r in mapped)
        base = {"metadata": metadata, "manifest": str(manifest_path),
                "reference": mapped_reference, "training_archive_available": manifest["kind"] == "training_archive"}

        def add(population: str, rows: list[dict], **extra: Any) -> None:
            groups.append({**base, "population": population, **extra,
                           "metrics": population_metrics(rows, original, permutations=permutations,
                                                         include_reference=population.startswith("training"))})

        if manifest["kind"] == "final_sampling":
            for budget in budgets:
                if budget > len(mapped):
                    failures.append({"manifest": str(manifest_path), "error": f"Missing budget {budget}"})
                add("final_sampling", mapped[:budget], sample_budget=budget)
        else:
            add("training_archive", mapped, archive_snapshot=manifest["snapshot"])
            active = {a for point in manifest["front"] for a in point["artifact_ids"]}
            add("training_final_front", [r for r in mapped if r["artifact_id"] in active])
            terminal_path = manifest_path.parent / "training_terminals.jsonl"
            terminals = [json.loads(line) for line in terminal_path.read_text().splitlines()] if terminal_path.exists() else []
            valid = [r for r in terminals if r.get("status") != "failed"]
            groups.append({**base, "population": "training_all_terminals",
                           "failed_count": len(terminals) - len(valid),
                           "metrics": population_metrics(valid, original, permutations=permutations,
                                                         include_reference=True, artifacts_available=False)})
            count = manifest["snapshot"]["local_trajectory"]
            for milestone in sorted(set(range(milestone_interval, count + 1, milestone_interval)) | {count}):
                add("training_archive_milestone", [r for r in mapped if r["trajectory_index"] <= milestone],
                    local_trajectory=milestone)
    training_models = {tuple(g["metadata"].get(k) for k in ("method", "circuit", "training_seed", "run_id"))
                       for g in groups if g["population"] == "training_archive"}
    for group in groups:
        group["training_archive_available"] = tuple(group["metadata"].get(k) for k in (
            "method", "circuit", "training_seed", "run_id")) in training_models
    result = {"schema_version": 1, "groups": groups, "failures": failures,
              "mapping_configuration": {key: identity[key] for key in ("abc_sha256", "k", "command")},
              "complete": not failures, "mapping_seconds": time.monotonic() - start,
              "artifact_bytes": sum(p.stat().st_size for p in cache_root.iterdir() if p.is_file())}
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "mapping_records.json", stats)
    write_json(output_dir / "fronts.json", [{"metadata": g["metadata"], "population": g["population"],
                                              "sample_budget": g.get("sample_budget"),
                                              "local_trajectory": g.get("local_trajectory"),
                                              "aig": g["metrics"]["aig_front"], "lut": g["metrics"]["lut_front"]} for g in groups])
    fields = sorted({key for row in stats for key in row})
    with (output_dir / "stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in stats)
    write_json(output_dir / "status.json", {"complete": not failures, "failed_count": len(failures)})
    return result


def aggregate(reports: Sequence[Path]) -> dict:
    """Average evaluation seeds within models before comparing trained models."""
    model_values = defaultdict(lambda: defaultdict(list))
    seen = set()
    mapping_configurations: set[str] = set()
    def scalars(value: dict, prefix: str = "") -> dict:
        out = {}
        for key, item in value.items():
            if key in {"selected_sample_index", "sample_indices", "selected_artifact", "selected_sha256"}:
                continue
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                out.update(scalars(item, name))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                out[name] = item
        return out
    for path in reports:
        report = json.loads(path.read_text())
        if not report["complete"]:
            raise ValueError(f"Cannot aggregate incomplete evaluation: {path}")
        config = report.get("mapping_configuration")
        mapping_configurations.add(json.dumps(config, sort_keys=True))
        if len(mapping_configurations) > 1:
            raise ValueError("Cannot compare reports with different LUT mapping settings")
        for group in report["groups"]:
            meta = group["metadata"]
            key = (meta["method"], meta["circuit"], group["population"],
                   group.get("sample_budget"), group.get("local_trajectory"), meta["training_seed"])
            occurrence_key = (*key, meta.get("evaluation_seed"))
            if occurrence_key in seen:
                raise ValueError("Duplicate model/evaluation-seed population in aggregation")
            seen.add(occurrence_key)
            for metric, value in scalars(group["metrics"]).items():
                model_values[key][metric].append(value)
    populations = defaultdict(lambda: defaultdict(list))
    for key, metrics in model_values.items():
        for metric, values in metrics.items():
            populations[key[:-1]][metric].append(statistics.fmean(values))
    return {"schema_version": 1, "groups": [
        {"method": key[0], "circuit": key[1], "population": key[2], "sample_budget": key[3],
         "local_trajectory": key[4], "metrics": {
             metric: {"mean": statistics.fmean(values), "std": statistics.pstdev(values),
                      "training_seed_count": len(values)} for metric, values in metrics.items()}}
        for key, metrics in populations.items()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("evaluate")
    run.add_argument("--manifest", type=Path, action="append", default=[])
    run.add_argument("--input-root", type=Path, action="append", default=[])
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--abc-path", type=Path, required=True)
    run.add_argument("--k", type=int, default=6)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--timeout-seconds", type=float, default=120)
    run.add_argument("--milestone-interval", type=int, default=50)
    run.add_argument("--curve-permutations", type=int, default=32)
    run.add_argument("--budgets", type=int, nargs="+", default=[10, 50, 100, 200])
    report = subs.add_parser("aggregate")
    report.add_argument("--report", type=Path, action="append", required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "aggregate":
        write_json(args.output, aggregate(args.report))
    else:
        manifests = args.manifest + [p for root in args.input_root for p in sorted(root.rglob("manifest.json"))]
        result = evaluate(manifests, output_dir=args.output_dir, abc_path=args.abc_path,
                          k=args.k, resume=args.resume, timeout_seconds=args.timeout_seconds, budgets=args.budgets,
                          milestone_interval=args.milestone_interval, permutations=args.curve_permutations)
        if not result["complete"]:
            raise SystemExit("Evaluation incomplete; inspect status.json and metrics.json")


if __name__ == "__main__":
    main()
