"""Artifact validation and decision report for TB Experiment 5.5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.gflownet_tb.diagnostics import HealthThresholds, health_gates
from src.experiments.tb_batch_size_influence import (
    BATCH_SIZES,
    EXPECTED_CIRCUITS,
    EXPECTED_SEEDS,
    EXPERIMENT_NAME,
    INITIALIZATION,
    LOG_Z_LEARNING_RATE,
    MAX_TRAJECTORIES,
    MILESTONES,
    RUN_SCHEMA_VERSION,
    SCHEDULE_TRAJECTORIES,
)
from src.experiments.tb_logz_calibration import canonical_trajectory_epsilon_values
from src.experiments.tb_logz_calibration_report import (
    ArtifactValidationError,
    IncompleteRunSetError,
    _read_json,
    _read_jsonl,
    _sha256,
    _write_csv,
    _write_json,
)


BASELINE_BATCH_SIZE = 4
EXPECTED_STRATA = ("fixed_uniform", "fresh_on_policy")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95


def _run_dir(root: Path, batch_size: int, circuit: str, seed: int) -> Path:
    return root / f"batch_{batch_size}" / circuit / f"seed_{seed}"


def _validate_table(path: Path, expected_rows: int) -> None:
    if not path.is_file():
        raise IncompleteRunSetError(f"missing table: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        raise ArtifactValidationError(f"invalid table {path}: {exc}") from exc
    if len(rows) != expected_rows:
        raise ArtifactValidationError(
            f"wrong row count in {path}: expected {expected_rows}, got {len(rows)}"
        )


def _expected_counters(batch_size: int) -> dict[str, int]:
    return {
        "training_trajectories": MAX_TRAJECTORIES,
        "new_training_trajectories": MAX_TRAJECTORIES - 64,
        "calibration_trajectories": 64,
        "calibration_training_presentations": 64,
        "training_presentations": MAX_TRAJECTORIES,
        "optimizer_updates": MAX_TRAJECTORIES // batch_size,
        "validation_rollouts": 256 + len(MILESTONES) * (128 + 50),
    }


def validate_run(run_dir: Path, *, batch_size: int, circuit: str, seed: int) -> dict[str, Any]:
    """Validate one completed batch/circuit/seed cell before aggregation."""
    summary = _read_json(run_dir / "run_summary.json")
    resolved = _read_json(run_dir / "resolved_config.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    if not summary.get("complete") or summary.get("numerical_failure") is not None:
        raise IncompleteRunSetError(f"run is incomplete or failed: {run_dir}")
    if (summary.get("circuit"), int(summary.get("seed", -1))) != (circuit, seed):
        raise ArtifactValidationError(f"run identity mismatch: {run_dir}")
    required = {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "variant": INITIALIZATION,
        "max_trajectories": MAX_TRAJECTORIES,
        "schedule_trajectories": SCHEDULE_TRAJECTORIES,
        "configured_optimizer_updates": MAX_TRAJECTORIES // batch_size,
        "milestones": list(MILESTONES),
        "trajectories_per_update": batch_size,
        "policy_learning_rate": 0.001,
        "log_z_learning_rate_resolved": LOG_Z_LEARNING_RATE,
        "calibration_trajectories": 64,
        "fixed_validation_trajectories": 256,
        "fresh_validation_trajectories": 128,
        "search_trajectories": 50,
        "search_budgets": [1, 2, 5, 10, 20, 50],
        "epsilon_schedule_indexing": "canonical_trajectory_groups",
        "epsilon_schedule_unit_trajectories": 4,
    }
    mismatches = {
        key: {"expected": expected, "actual": resolved.get(key)}
        for key, expected in required.items() if resolved.get(key) != expected
    }
    if mismatches:
        raise ArtifactValidationError(f"scientific configuration mismatch in {run_dir}: {mismatches}")
    for key in (
        "scientific_configuration_fingerprint",
        "paired_configuration_fingerprint",
        "batch_pairing_configuration_fingerprint",
    ):
        if summary.get(key) != resolved.get(key):
            raise ArtifactValidationError(f"{key} mismatch: {run_dir}")
    if summary.get("source_tree_sha256") != metadata.get("source_tree_sha256"):
        raise ArtifactValidationError(f"source-tree metadata mismatch: {run_dir}")
    for key in (
        "pre_calibration_parameter_checksum",
        "post_initialization_parameter_checksum",
        "fixed_sequence_checksum",
        "calibration_sequence_checksum",
    ):
        if summary.get(key) != metadata.get(key):
            raise ArtifactValidationError(f"{key} mismatch: {run_dir}")

    for filename, checksum_key, expected_count in (
        ("fixed_validation.pt", "fixed_validation_checksum", 256),
        ("calibration.pt", "calibration_cache_checksum", 64),
    ):
        path = run_dir / filename
        if not path.is_file():
            raise IncompleteRunSetError(f"missing cache: {path}")
        if _sha256(path) != summary.get(checksum_key):
            raise ArtifactValidationError(f"cache checksum mismatch: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            trajectories = payload["trajectories"]
        except Exception as exc:
            raise ArtifactValidationError(f"unreadable cache {path}: {exc}") from exc
        if len(trajectories) != expected_count:
            raise ArtifactValidationError(f"wrong cache trajectory count: {path}")
        for trajectory in trajectories:
            if len(trajectory.steps) != 20:
                raise ArtifactValidationError(f"wrong-horizon cached trajectory: {path}")
            if any(step.action not in step.legal_actions for step in trajectory.steps):
                raise ArtifactValidationError(f"illegal cached trajectory action: {path}")
            if float(trajectory.log_pb_sum) != 0.0:
                raise ArtifactValidationError(f"cached trajectory has nonzero log P_B: {path}")
        if filename == "calibration.pt":
            actions = [[int(step.action) for step in trajectory.steps] for trajectory in trajectories]
            if payload.get("actions") != actions or payload.get("batch_order") != "collection_order":
                raise ArtifactValidationError(f"calibration manifest mismatch: {path}")

    milestones = {int(row["trajectory_budget"]): row for row in summary.get("milestones", [])}
    if set(milestones) != set(MILESTONES):
        raise IncompleteRunSetError(
            f"milestones are {sorted(milestones)}, expected {list(MILESTONES)}: {run_dir}"
        )
    file_milestones = {
        int(row["trajectory_budget"]): row for row in _read_jsonl(run_dir / "milestones.jsonl")
    }
    if file_milestones != milestones:
        raise ArtifactValidationError(f"summary/milestones disagreement: {run_dir}")
    for budget in MILESTONES:
        milestone = milestones[budget]
        if int(milestone.get("optimizer_update", -1)) != budget // batch_size:
            raise ArtifactValidationError(f"milestone update mismatch at {budget}: {run_dir}")
        if [row["n"] for row in milestone["search"]["budgets"]] != [1, 2, 5, 10, 20, 50]:
            raise ArtifactValidationError(f"wrong search budgets at {budget}: {run_dir}")
        _validate_table(run_dir / "tables" / f"trajectory_{budget}_validation.csv", 2)
        _validate_table(run_dir / "tables" / f"trajectory_{budget}_best_of_n.csv", 6)
        checkpoint_path = run_dir / "checkpoints" / f"trajectory_{budget}.pt"
        if not checkpoint_path.is_file():
            raise IncompleteRunSetError(f"missing checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ArtifactValidationError(f"unreadable checkpoint {checkpoint_path}: {exc}") from exc
        required_checkpoint = {
            "policy", "optimizer", "scheduler", "replay", "archive", "counters",
            "global_rng", "train_action_generator_state", "circuit_rng_state", "replay_rng_state",
            "fixed_cache_checksum", "calibration_cache_checksum", "resolved_config", "run_metadata",
        }
        if missing := required_checkpoint.difference(checkpoint):
            raise ArtifactValidationError(
                f"checkpoint missing keys {sorted(missing)}: {checkpoint_path}"
            )
        if int(checkpoint["counters"]["training_trajectories"]) != budget:
            raise ArtifactValidationError(f"checkpoint budget mismatch: {checkpoint_path}")
        if int(checkpoint["counters"]["optimizer_updates"]) != budget // batch_size:
            raise ArtifactValidationError(f"checkpoint update mismatch: {checkpoint_path}")
        if checkpoint["resolved_config"].get(
            "scientific_configuration_fingerprint"
        ) != summary.get("scientific_configuration_fingerprint"):
            raise ArtifactValidationError(f"checkpoint fingerprint mismatch: {checkpoint_path}")

    updates = [
        row for row in _read_jsonl(run_dir / "metrics.jsonl")
        if row.get("row_type") == "training_update"
    ]
    expected_updates = MAX_TRAJECTORIES // batch_size
    if len(updates) != expected_updates:
        raise IncompleteRunSetError(
            f"expected {expected_updates} updates, found {len(updates)}: {run_dir}"
        )
    calibration_updates = 64 // batch_size
    if any(row.get("training_source") != "calibration" for row in updates[:calibration_updates]):
        raise ArtifactValidationError(f"calibration update prefix mismatch: {run_dir}")
    if [row.get("calibration_batch_start") for row in updates[:calibration_updates]] != list(
        range(0, 64, batch_size)
    ):
        raise ArtifactValidationError(f"calibration minibatch order mismatch: {run_dir}")
    if any(row.get("training_source") != "new_on_policy" for row in updates[calibration_updates:]):
        raise ArtifactValidationError(f"on-policy update suffix mismatch: {run_dir}")
    expected_budgets = [64] * calibration_updates + list(
        range(64 + batch_size, MAX_TRAJECTORIES + 1, batch_size)
    )
    if [int(row.get("trajectory_budget", -1)) for row in updates] != expected_budgets:
        raise ArtifactValidationError(f"optimizer trajectory-budget progression mismatch: {run_dir}")
    expected_epsilons = canonical_trajectory_epsilon_values(
        first_trajectory=1,
        count=MAX_TRAJECTORIES,
    )
    observed_epsilons: list[float] = []
    for index, row in enumerate(updates):
        values = [float(value) for value in row.get("epsilon_uniform_values", [])]
        if len(values) != batch_size:
            raise ArtifactValidationError(f"epsilon vector width mismatch at update {index + 1}: {run_dir}")
        observed_epsilons.extend(values)
    if not np.allclose(observed_epsilons, expected_epsilons, rtol=0.0, atol=1e-12):
        raise ArtifactValidationError(f"trajectory-indexed epsilon schedule mismatch: {run_dir}")

    for key, expected in _expected_counters(batch_size).items():
        if int(summary["counters"].get(key, -1)) != expected:
            raise ArtifactValidationError(f"counter {key} mismatch: {run_dir}")
    trajectory_rows = _read_jsonl(run_dir / "trajectories.jsonl")
    expected_sources = {
        "fixed_uniform": 256,
        "calibration": 64,
        "training": MAX_TRAJECTORIES - 64,
        "fresh_on_policy": 128 * len(MILESTONES),
        "search": 50 * len(MILESTONES),
    }
    actual_sources = {
        source: sum(row.get("source") == source for row in trajectory_rows)
        for source in expected_sources
    }
    if actual_sources != expected_sources:
        raise ArtifactValidationError(f"trajectory source counts mismatch in {run_dir}: {actual_sources}")
    resources = summary.get("resource_usage", {})
    if int(resources.get("peak_host_rss_kib", 0)) <= 0:
        raise ArtifactValidationError(f"missing peak host RSS: {run_dir}")
    if str(resolved.get("device", "")).startswith("cuda"):
        for key in (
            "peak_cuda_memory_allocated_bytes", "peak_cuda_memory_reserved_bytes",
        ):
            if int(resources.get(key, 0)) <= 0:
                raise ArtifactValidationError(f"missing {key}: {run_dir}")
    return {
        "dir": run_dir,
        "summary": summary,
        "resolved": resolved,
        "metadata": metadata,
        "milestones": milestones,
        "updates": updates,
    }


def load_run_matrix(roots: Sequence[Path]) -> dict[tuple[int, str, int], dict[str, Any]]:
    resolved_roots = [root.resolve() for root in roots]
    runs: dict[tuple[int, str, int], dict[str, Any]] = {}
    missing: list[str] = []
    for batch_size in BATCH_SIZES:
        for circuit in EXPECTED_CIRCUITS:
            for seed in EXPECTED_SEEDS:
                key = (batch_size, circuit, seed)
                matches = [
                    _run_dir(root, batch_size, circuit, seed)
                    for root in resolved_roots
                    if (_run_dir(root, batch_size, circuit, seed) / "run_summary.json").is_file()
                ]
                if not matches:
                    missing.append(f"batch={batch_size}, circuit={circuit}, seed={seed}")
                    continue
                if len(matches) > 1:
                    raise ArtifactValidationError(f"duplicate run cell {key}: {matches}")
                runs[key] = validate_run(
                    matches[0], batch_size=batch_size, circuit=circuit, seed=seed
                )
    if missing:
        raise IncompleteRunSetError("missing run cells: " + "; ".join(missing))
    if len(runs) != len(BATCH_SIZES) * len(EXPECTED_CIRCUITS) * len(EXPECTED_SEEDS):
        raise IncompleteRunSetError(f"wrong run-matrix size: {len(runs)}")
    sources = {run["summary"]["source_tree_sha256"] for run in runs.values()}
    pairings = {run["resolved"]["batch_pairing_configuration_fingerprint"] for run in runs.values()}
    if len(sources) != 1 or len(pairings) != 1:
        raise ArtifactValidationError("run matrix mixes source trees or batch-pairing configurations")
    batch_fingerprints: set[str] = set()
    for batch_size in BATCH_SIZES:
        fingerprints = {
            runs[(batch_size, circuit, seed)]["resolved"]["scientific_configuration_fingerprint"]
            for circuit in EXPECTED_CIRCUITS for seed in EXPECTED_SEEDS
        }
        if len(fingerprints) != 1:
            raise ArtifactValidationError(f"batch {batch_size} mixes scientific configurations")
        batch_fingerprints.update(fingerprints)
    if len(batch_fingerprints) != len(BATCH_SIZES):
        raise ArtifactValidationError("batch variants do not have distinct scientific fingerprints")
    for seed in EXPECTED_SEEDS:
        checksums = {
            runs[(batch_size, circuit, seed)]["summary"]["pre_calibration_parameter_checksum"]
            for batch_size in BATCH_SIZES for circuit in EXPECTED_CIRCUITS
        }
        if len(checksums) != 1:
            raise ArtifactValidationError(f"initial-policy pairing mismatch for seed {seed}")
    for circuit in EXPECTED_CIRCUITS:
        for seed in EXPECTED_SEEDS:
            for key in (
                "post_initialization_parameter_checksum",
                "fixed_sequence_checksum",
                "calibration_sequence_checksum",
            ):
                checksums = {
                    runs[(batch_size, circuit, seed)]["summary"][key]
                    for batch_size in BATCH_SIZES
                }
                if len(checksums) != 1:
                    raise ArtifactValidationError(
                        f"{key} pairing mismatch for {circuit}/seed_{seed}"
                    )
    return runs


def paired_mean_bootstrap(
    values: Sequence[float], *, seed: int, samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires a nonempty finite one-dimensional sample")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    estimates = np.mean(array[indices], axis=1)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "standard_error": float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "samples": samples,
        "seed": seed,
    }


def _health_rows(runs: Mapping[tuple[int, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = HealthThresholds()
    for (batch_size, circuit, seed), run in sorted(runs.items()):
        for budget in MILESTONES:
            milestone = run["milestones"][budget]
            optimizer_health = milestone["optimizer_health"]
            for stratum in EXPECTED_STRATA:
                gate = health_gates(
                    validation=milestone[stratum],
                    gradient_p99_median_ratio=float(
                        optimizer_health["policy_gradient_p99_median_ratio"]
                    ),
                    clipping_enabled=bool(optimizer_health["gradient_clipping_enabled"]),
                    clipping_rate=float(optimizer_health["gradient_clipping_rate"]),
                    thresholds=thresholds,
                )
                rows.append({
                    "batch_size": batch_size,
                    "circuit": circuit,
                    "seed": seed,
                    "trajectory_budget": budget,
                    "stratum": stratum,
                    "pass": bool(gate["pass"]),
                    "failed": gate["failed"],
                })
    return rows


def classify_batch_sizes(
    runs: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    endpoint = MAX_TRAJECTORIES
    health_rows = _health_rows(runs)
    health_lookup = {
        (row["batch_size"], row["circuit"], row["seed"], row["trajectory_budget"], row["stratum"]): row
        for row in health_rows
    }
    comparisons: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    candidates: dict[int, dict[str, Any]] = {}
    for candidate_index, batch_size in enumerate(BATCH_SIZES):
        if batch_size == BASELINE_BATCH_SIZE:
            continue
        circuit_results: dict[str, Any] = {}
        health_pass = all(
            health_lookup[(batch_size, circuit, seed, endpoint, stratum)]["pass"]
            for circuit in EXPECTED_CIRCUITS
            for seed in EXPECTED_SEEDS
            for stratum in EXPECTED_STRATA
        )
        any_improvement = False
        all_noninferior = True
        for circuit_index, circuit in enumerate(EXPECTED_CIRCUITS):
            rms_improvements = []
            hv_improvements = []
            for seed in EXPECTED_SEEDS:
                baseline = runs[(BASELINE_BATCH_SIZE, circuit, seed)]["milestones"][endpoint]
                candidate = runs[(batch_size, circuit, seed)]["milestones"][endpoint]
                baseline_rms = float(baseline["fixed_uniform"]["residual"]["centered_rms"])
                candidate_rms = float(candidate["fixed_uniform"]["residual"]["centered_rms"])
                rms_improvements.append(
                    (baseline_rms - candidate_rms) / max(abs(baseline_rms), 1e-12)
                )
                hv_improvements.append(
                    float(candidate["training_archive"]["hypervolume"])
                    - float(baseline["training_archive"]["hypervolume"])
                )
            rms = paired_mean_bootstrap(
                rms_improvements,
                seed=55_000 + candidate_index * 100 + circuit_index * 10,
            )
            hv = paired_mean_bootstrap(
                hv_improvements,
                seed=56_000 + candidate_index * 100 + circuit_index * 10,
            )
            rms_improvement_pass = rms["ci_low"] >= 0.05
            hv_improvement_pass = hv["ci_low"] >= 0.005
            rms_noninferior = rms["ci_low"] >= -0.05
            hv_noninferior = hv["ci_low"] >= -0.005
            any_improvement = any_improvement or rms_improvement_pass or hv_improvement_pass
            all_noninferior = all_noninferior and rms_noninferior and hv_noninferior
            circuit_results[circuit] = {
                "rms": rms,
                "hypervolume": hv,
                "rms_improvement_pass": rms_improvement_pass,
                "hypervolume_improvement_pass": hv_improvement_pass,
                "rms_noninferior": rms_noninferior,
                "hypervolume_noninferior": hv_noninferior,
            }
            bootstrap_rows.extend([
                {"batch_size": batch_size, "circuit": circuit, "metric": "fixed_uniform_centered_rms_relative_improvement", **rms},
                {"batch_size": batch_size, "circuit": circuit, "metric": "archive_hypervolume_absolute_improvement", **hv},
            ])
        qualifies = health_pass and any_improvement and all_noninferior
        endpoint_hv = [
            float(runs[(batch_size, circuit, seed)]["milestones"][endpoint]["training_archive"]["hypervolume"])
            for circuit in EXPECTED_CIRCUITS for seed in EXPECTED_SEEDS
        ]
        endpoint_rms = [
            float(runs[(batch_size, circuit, seed)]["milestones"][endpoint]["fixed_uniform"]["residual"]["centered_rms"])
            for circuit in EXPECTED_CIRCUITS for seed in EXPECTED_SEEDS
        ]
        wall_times = [
            float(runs[(batch_size, circuit, seed)]["summary"]["wall_time_seconds"])
            for circuit in EXPECTED_CIRCUITS for seed in EXPECTED_SEEDS
        ]
        candidate = {
            "batch_size": batch_size,
            "health_pass": health_pass,
            "any_improvement": any_improvement,
            "all_noninferior": all_noninferior,
            "qualifies": qualifies,
            "circuits": circuit_results,
            "mean_archive_hypervolume": float(np.mean(endpoint_hv)),
            "mean_fixed_uniform_centered_rms": float(np.mean(endpoint_rms)),
            "mean_wall_time_seconds": float(np.mean(wall_times)),
        }
        comparisons.append(candidate)
        candidates[batch_size] = candidate

    qualifying = [row for row in comparisons if row["qualifies"]]
    selected = BASELINE_BATCH_SIZE
    within_one_se: list[int] = []
    if qualifying:
        best = max(qualifying, key=lambda row: row["mean_archive_hypervolume"])
        best_batch = int(best["batch_size"])
        for candidate in qualifying:
            batch_size = int(candidate["batch_size"])
            paired_differences = []
            for seed in EXPECTED_SEEDS:
                best_seed = np.mean([
                    runs[(best_batch, circuit, seed)]["milestones"][endpoint]["training_archive"]["hypervolume"]
                    for circuit in EXPECTED_CIRCUITS
                ])
                candidate_seed = np.mean([
                    runs[(batch_size, circuit, seed)]["milestones"][endpoint]["training_archive"]["hypervolume"]
                    for circuit in EXPECTED_CIRCUITS
                ])
                paired_differences.append(float(best_seed - candidate_seed))
            difference = float(np.mean(paired_differences))
            standard_error = (
                float(np.std(paired_differences, ddof=1) / math.sqrt(len(paired_differences)))
                if len(paired_differences) > 1 else 0.0
            )
            candidate["hypervolume_gap_from_best"] = difference
            candidate["paired_standard_error_from_best"] = standard_error
            candidate["within_one_standard_error"] = difference <= standard_error + 1e-12
            if candidate["within_one_standard_error"]:
                within_one_se.append(batch_size)
        selected = min(
            (candidates[batch_size] for batch_size in within_one_se),
            key=lambda row: (
                row["batch_size"], row["mean_fixed_uniform_centered_rms"], row["mean_wall_time_seconds"],
            ),
        )["batch_size"]
    decision = "select_batch_size" if qualifying else "reject_batch_size_influence_retain_four"
    return {
        "complete": True,
        "decision": decision,
        "baseline_batch_size": BASELINE_BATCH_SIZE,
        "selected_batch_size": int(selected),
        "qualifying_batch_sizes": sorted(int(row["batch_size"]) for row in qualifying),
        "within_one_standard_error": sorted(within_one_se),
        "health_gates": health_rows,
        "comparisons": comparisons,
        "bootstrap": bootstrap_rows,
        "requires_new_experiment5_curve": bool(qualifying),
    }


def _seed_metric_rows(runs: Mapping[tuple[int, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (batch_size, circuit, seed), run in sorted(runs.items()):
        for budget in MILESTONES:
            milestone = run["milestones"][budget]
            for stratum in EXPECTED_STRATA:
                residual = milestone[stratum]["residual"]
                rows.append({
                    "batch_size": batch_size, "circuit": circuit, "seed": seed,
                    "trajectory_budget": budget, "stratum": stratum,
                    "centered_rms": residual["centered_rms"],
                    "log_z_target_gap": residual["log_z_target_gap"],
                    "bias_fraction": residual["bias_fraction"],
                    "standardized_bias": residual["standardized_bias"],
                    "training_archive_hypervolume": milestone["training_archive"]["hypervolume"],
                    "best_of_n_auc": milestone["search"]["log2_n_hypervolume_auc"],
                })
    return rows


def _best_of_n_rows(runs: Mapping[tuple[int, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"batch_size": batch_size, "circuit": circuit, "seed": seed, "trajectory_budget": budget, **row}
        for (batch_size, circuit, seed), run in sorted(runs.items())
        for budget in MILESTONES
        for row in run["milestones"][budget]["search"]["budgets"]
    ]


def _resource_rows(runs: Mapping[tuple[int, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "batch_size": batch_size,
            "circuit": circuit,
            "seed": seed,
            "wall_time_seconds": run["summary"]["wall_time_seconds"],
            **run["summary"]["resource_usage"],
        }
        for (batch_size, circuit, seed), run in sorted(runs.items())
    ]


def _write_plots(
    output_dir: Path,
    runs: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    endpoint = MAX_TRAJECTORIES
    for circuit in EXPECTED_CIRCUITS:
        passing = []
        for batch_size in BATCH_SIZES:
            count = 0
            for seed in EXPECTED_SEEDS:
                milestone = runs[(batch_size, circuit, seed)]["milestones"][endpoint]
                optimizer_health = milestone["optimizer_health"]
                for stratum in EXPECTED_STRATA:
                    gate = health_gates(
                        validation=milestone[stratum],
                        gradient_p99_median_ratio=float(
                            optimizer_health["policy_gradient_p99_median_ratio"]
                        ),
                        clipping_enabled=bool(optimizer_health["gradient_clipping_enabled"]),
                        clipping_rate=float(optimizer_health["gradient_clipping_rate"]),
                        thresholds=HealthThresholds(),
                    )
                    count += int(gate["pass"])
            passing.append(count)
        plt.plot(BATCH_SIZES, passing, marker="o", label=circuit)
    plt.axhline(
        len(EXPECTED_SEEDS) * len(EXPECTED_STRATA), color="black", linestyle="--"
    )
    plt.xscale("log", base=2)
    plt.xlabel("batch size")
    plt.ylabel("passing endpoint seed/stratum gates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "00_health_gates.png", dpi=160)
    plt.close()
    for metric, getter, filename, ylabel in (
        (
            "rms",
            lambda milestone: milestone["fixed_uniform"]["residual"]["centered_rms"],
            "01_fixed_uniform_centered_rms.png",
            "mean fixed-uniform centered RMS",
        ),
        (
            "hv",
            lambda milestone: milestone["training_archive"]["hypervolume"],
            "02_archive_hypervolume.png",
            "mean archive hypervolume",
        ),
        (
            "auc",
            lambda milestone: milestone["search"]["log2_n_hypervolume_auc"],
            "03_best_of_n_auc.png",
            "mean best-of-N AUC",
        ),
    ):
        del metric
        for circuit in EXPECTED_CIRCUITS:
            values = [
                float(np.mean([
                    getter(runs[(batch_size, circuit, seed)]["milestones"][endpoint])
                    for seed in EXPECTED_SEEDS
                ]))
                for batch_size in BATCH_SIZES
            ]
            plt.plot(BATCH_SIZES, values, marker="o", label=circuit)
        plt.xscale("log", base=2)
        plt.xlabel("batch size")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()
    for circuit in EXPECTED_CIRCUITS:
        wall = [
            float(np.mean([
                runs[(batch_size, circuit, seed)]["summary"]["wall_time_seconds"]
                for seed in EXPECTED_SEEDS
            ]))
            for batch_size in BATCH_SIZES
        ]
        plt.plot(BATCH_SIZES, wall, marker="o", label=circuit)
    plt.xscale("log", base=2)
    plt.xlabel("batch size")
    plt.ylabel("mean wall time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "04_wall_time.png", dpi=160)
    plt.close()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for circuit in EXPECTED_CIRCUITS:
        host = [
            float(np.mean([
                runs[(batch_size, circuit, seed)]["summary"]["resource_usage"]["peak_host_rss_kib"]
                for seed in EXPECTED_SEEDS
            ])) / (1024 ** 2)
            for batch_size in BATCH_SIZES
        ]
        cuda = [
            float(np.mean([
                runs[(batch_size, circuit, seed)]["summary"]["resource_usage"]["peak_cuda_memory_allocated_bytes"]
                for seed in EXPECTED_SEEDS
            ])) / (1024 ** 3)
            for batch_size in BATCH_SIZES
        ]
        axes[0].plot(BATCH_SIZES, host, marker="o", label=circuit)
        axes[1].plot(BATCH_SIZES, cuda, marker="o", label=circuit)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("batch size")
        axis.legend()
    axes[0].set_ylabel("mean peak host RSS (GiB)")
    axes[1].set_ylabel("mean peak CUDA allocated (GiB)")
    figure.tight_layout()
    figure.savefig(plot_dir / "05_peak_memory.png", dpi=160)
    plt.close(figure)


def _decision_markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 5.5 batch-size decision",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        f"- selected batch size: `{decision['selected_batch_size']}`",
        f"- qualifying alternatives: `{decision['qualifying_batch_sizes']}`",
        f"- within one paired standard error of best hypervolume: `{decision['within_one_standard_error']}`",
        f"- rerun Experiment 5: `{decision['requires_new_experiment5_curve']}`",
        "",
        "The control is the unresolved Experiment 5 descriptive configuration (`zcal`, logZ LR 0.01), not a selected healthy optimizer.",
        "",
        "| batch | health | improvement | noninferior | qualifies | mean HV | mean fixed RMS | mean wall s |",
        "|--:|:--:|:--:|:--:|:--:|--:|--:|--:|",
    ]
    for row in decision["comparisons"]:
        lines.append(
            f"| {row['batch_size']} | {row['health_pass']} | {row['any_improvement']} | "
            f"{row['all_noninferior']} | {row['qualifies']} | "
            f"{row['mean_archive_hypervolume']:.6f} | "
            f"{row['mean_fixed_uniform_centered_rms']:.6f} | "
            f"{row['mean_wall_time_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        runs = load_run_matrix(args.runs_root)
        decision = classify_batch_sizes(runs)
        _write_json(output_dir / "decision_summary.json", decision)
        (output_dir / "decision_report.md").write_text(
            _decision_markdown(decision), encoding="utf-8"
        )
        _write_csv(output_dir / "seed_milestone_metrics.csv", _seed_metric_rows(runs))
        _write_csv(output_dir / "health_gates.csv", decision["health_gates"])
        _write_csv(output_dir / "paired_batch_comparisons.csv", decision["comparisons"])
        _write_csv(output_dir / "bootstrap.csv", decision["bootstrap"])
        _write_csv(output_dir / "best_of_n.csv", _best_of_n_rows(runs))
        _write_csv(output_dir / "resource_usage.csv", _resource_rows(runs))
        _write_plots(output_dir, runs)
        return 0 if decision["qualifying_batch_sizes"] else 3
    except IncompleteRunSetError as exc:
        _write_json(output_dir / "decision_summary.json", {
            "complete": False, "failure_type": "incomplete_run_matrix", "message": str(exc),
        })
        return 2
    except Exception as exc:
        _write_json(output_dir / "decision_summary.json", {
            "complete": False,
            "failure_type": "artifact_or_execution_failure",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        return 1


def add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="validate and compare Experiment 5.5 batches")
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=report)


__all__ = [
    "classify_batch_sizes", "load_run_matrix", "paired_mean_bootstrap", "report",
    "validate_run",
]
