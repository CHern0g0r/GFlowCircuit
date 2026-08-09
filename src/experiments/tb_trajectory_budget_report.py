"""Artifact validation and budget-selection report for TB Experiment 5."""

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
from src.experiments.tb_logz_calibration_report import (
    ArtifactValidationError,
    IncompleteRunSetError,
    _read_json,
    _read_jsonl,
    _sha256,
    _write_csv,
    _write_json,
)
from src.experiments.tb_trajectory_budget import (
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


EXPECTED_STRATA = ("fixed_uniform", "fresh_on_policy")
CANDIDATE_BUDGETS = MILESTONES[:-1]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95


def _strictly_below(value: float, threshold: float) -> bool:
    """Apply a strict protocol boundary without binary-float boundary leakage."""
    return value < threshold and not math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)


def _run_dir(root: Path, circuit: str, seed: int) -> Path:
    return root / circuit / f"seed_{seed}"


def _validate_table(path: Path, expected_rows: int) -> None:
    if not path.is_file():
        raise IncompleteRunSetError(f"missing milestone table: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        raise ArtifactValidationError(f"invalid milestone table {path}: {exc}") from exc
    if len(rows) != expected_rows:
        raise ArtifactValidationError(
            f"wrong row count in {path}: expected {expected_rows}, got {len(rows)}"
        )


def validate_run(run_dir: Path, *, circuit: str, seed: int) -> dict[str, Any]:
    """Validate one complete scientific run before it enters aggregation."""
    summary = _read_json(run_dir / "run_summary.json")
    resolved = _read_json(run_dir / "resolved_config.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    if not summary.get("complete") or summary.get("numerical_failure") is not None:
        raise IncompleteRunSetError(f"run is incomplete or failed: {run_dir}")
    if (summary.get("circuit"), int(summary.get("seed", -1))) != (circuit, seed):
        raise ArtifactValidationError(f"run identity mismatch: {run_dir}")
    required_config = {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "variant": INITIALIZATION,
        "max_trajectories": MAX_TRAJECTORIES,
        "schedule_trajectories": SCHEDULE_TRAJECTORIES,
        "configured_optimizer_updates": MAX_TRAJECTORIES // 4,
        "milestones": list(MILESTONES),
        "trajectories_per_update": 4,
        "policy_learning_rate": 0.001,
        "log_z_learning_rate_resolved": LOG_Z_LEARNING_RATE,
        "calibration_trajectories": 64,
        "fixed_validation_trajectories": 256,
        "fresh_validation_trajectories": 128,
        "search_trajectories": 50,
        "search_budgets": [1, 2, 5, 10, 20, 50],
    }
    mismatches = {
        key: {"expected": expected, "actual": resolved.get(key)}
        for key, expected in required_config.items()
        if resolved.get(key) != expected
    }
    if mismatches:
        raise ArtifactValidationError(f"scientific configuration mismatch in {run_dir}: {mismatches}")
    for key in ("scientific_configuration_fingerprint", "paired_configuration_fingerprint"):
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
            if float(payload.get("epsilon_uniform", -1.0)) != 0.5:
                raise ArtifactValidationError(f"calibration epsilon mismatch: {path}")

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
        if [row["n"] for row in milestones[budget]["search"]["budgets"]] != [1, 2, 5, 10, 20, 50]:
            raise ArtifactValidationError(f"wrong nested search budgets at {budget}: {run_dir}")
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
            raise ArtifactValidationError(f"checkpoint missing keys {sorted(missing)}: {checkpoint_path}")
        if int(checkpoint["counters"]["training_trajectories"]) != budget:
            raise ArtifactValidationError(f"checkpoint budget mismatch: {checkpoint_path}")
        if checkpoint["resolved_config"].get("scientific_configuration_fingerprint") != summary.get(
            "scientific_configuration_fingerprint"
        ):
            raise ArtifactValidationError(f"checkpoint fingerprint mismatch: {checkpoint_path}")

    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    updates = [row for row in metrics if row.get("row_type") == "training_update"]
    if len(updates) != MAX_TRAJECTORIES // 4:
        raise IncompleteRunSetError(
            f"expected {MAX_TRAJECTORIES // 4} updates, found {len(updates)}: {run_dir}"
        )
    if [row.get("training_source") for row in updates[:16]] != ["calibration"] * 16:
        raise ArtifactValidationError(f"first 16 updates are not calibration minibatches: {run_dir}")
    if [row.get("calibration_batch_start") for row in updates[:16]] != list(range(0, 64, 4)):
        raise ArtifactValidationError(f"calibration minibatch order mismatch: {run_dir}")
    if any(row.get("training_source") != "new_on_policy" for row in updates[16:]):
        raise ArtifactValidationError(f"post-calibration source mismatch: {run_dir}")
    if not math.isclose(float(updates[-1]["epsilon_uniform"]), 0.01, rel_tol=0.0, abs_tol=1e-12):
        raise ArtifactValidationError(f"epsilon did not clamp at 0.01: {run_dir}")
    expected_counters = {
        "training_trajectories": 6_400,
        "new_training_trajectories": 6_336,
        "calibration_trajectories": 64,
        "calibration_training_presentations": 64,
        "training_presentations": 6_400,
        "optimizer_updates": 1_600,
        "validation_rollouts": 256 + len(MILESTONES) * (128 + 50),
    }
    for key, expected in expected_counters.items():
        if int(summary["counters"].get(key, -1)) != expected:
            raise ArtifactValidationError(f"counter {key} mismatch: {run_dir}")
    trajectory_rows = _read_jsonl(run_dir / "trajectories.jsonl")
    expected_sources = {
        "fixed_uniform": 256,
        "calibration": 64,
        "training": 6_336,
        "fresh_on_policy": 128 * len(MILESTONES),
        "search": 50 * len(MILESTONES),
    }
    actual_sources = {
        source: sum(row.get("source") == source for row in trajectory_rows)
        for source in expected_sources
    }
    if actual_sources != expected_sources:
        raise ArtifactValidationError(
            f"trajectory source counts mismatch in {run_dir}: {actual_sources}"
        )
    return {
        "dir": run_dir,
        "summary": summary,
        "resolved": resolved,
        "metadata": metadata,
        "milestones": milestones,
        "updates": updates,
    }


def load_run_matrix(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    runs = {
        (circuit, seed): validate_run(_run_dir(root, circuit, seed), circuit=circuit, seed=seed)
        for circuit in EXPECTED_CIRCUITS
        for seed in EXPECTED_SEEDS
    }
    fingerprints = {run["resolved"]["scientific_configuration_fingerprint"] for run in runs.values()}
    sources = {run["summary"]["source_tree_sha256"] for run in runs.values()}
    if len(fingerprints) != 1 or len(sources) != 1:
        raise ArtifactValidationError("run matrix mixes scientific configurations or source trees")
    for seed in EXPECTED_SEEDS:
        checksums = {
            runs[(circuit, seed)]["summary"]["pre_calibration_parameter_checksum"]
            for circuit in EXPECTED_CIRCUITS
        }
        if len(checksums) != 1:
            raise ArtifactValidationError(f"initial-policy pairing mismatch for seed {seed}")
    return runs


def paired_median_bootstrap(
    values: Sequence[float], *, seed: int, samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires a nonempty finite one-dimensional sample")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    estimates = np.median(array[indices], axis=1)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "median": float(np.median(array)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": BOOTSTRAP_CONFIDENCE,
        "samples": samples,
        "seed": seed,
    }


def _health_rows(runs: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = HealthThresholds()
    for (circuit, seed), run in sorted(runs.items()):
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
                    "circuit": circuit,
                    "seed": seed,
                    "trajectory_budget": budget,
                    "stratum": stratum,
                    "pass": bool(gate["pass"]),
                    "failed": gate["failed"],
                })
    return rows


def classify_budget_selection(
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    health_rows = _health_rows(runs)
    health_lookup = {
        (row["circuit"], row["seed"], row["trajectory_budget"], row["stratum"]): row
        for row in health_rows
    }
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    circuit_candidates: dict[str, int | None] = {}
    for circuit_index, circuit in enumerate(EXPECTED_CIRCUITS):
        selected: int | None = None
        for budget_index, budget in enumerate(CANDIDATE_BUDGETS):
            successor = budget * 2
            health_pass = all(
                health_lookup[(circuit, seed, point, stratum)]["pass"]
                for seed in EXPECTED_SEEDS
                for point in (budget, successor)
                for stratum in EXPECTED_STRATA
            )
            rms_pass = True
            stratum_results: dict[str, dict[str, Any]] = {}
            for stratum_index, stratum in enumerate(EXPECTED_STRATA):
                reductions = []
                for seed in EXPECTED_SEEDS:
                    before = float(runs[(circuit, seed)]["milestones"][budget][stratum]["residual"]["centered_rms"])
                    after = float(runs[(circuit, seed)]["milestones"][successor][stratum]["residual"]["centered_rms"])
                    reductions.append((before - after) / max(abs(before), 1e-12))
                bootstrap = paired_median_bootstrap(
                    reductions,
                    seed=50_000 + circuit_index * 1_000 + budget_index * 10 + stratum_index,
                )
                passes = _strictly_below(bootstrap["median"], 0.05) and _strictly_below(
                    bootstrap["ci_high"], 0.10
                )
                rms_pass = rms_pass and passes
                result = {"reductions": reductions, "pass": passes, **bootstrap}
                stratum_results[stratum] = result
                bootstrap_rows.append({
                    "circuit": circuit,
                    "trajectory_budget": budget,
                    "successor_budget": successor,
                    "stratum": stratum,
                    **result,
                })
            hv_before = [
                float(runs[(circuit, seed)]["milestones"][budget]["training_archive"]["hypervolume"])
                for seed in EXPECTED_SEEDS
            ]
            hv_after = [
                float(runs[(circuit, seed)]["milestones"][successor]["training_archive"]["hypervolume"])
                for seed in EXPECTED_SEEDS
            ]
            hv_gain = float(np.mean(hv_after) - np.mean(hv_before))
            hv_pass = _strictly_below(abs(hv_gain), 0.005)
            auc_before = [
                float(runs[(circuit, seed)]["milestones"][budget]["search"]["log2_n_hypervolume_auc"])
                for seed in EXPECTED_SEEDS
            ]
            auc_after = [
                float(runs[(circuit, seed)]["milestones"][successor]["search"]["log2_n_hypervolume_auc"])
                for seed in EXPECTED_SEEDS
            ]
            mean_auc_before = float(np.mean(auc_before))
            mean_auc_after = float(np.mean(auc_after))
            auc_gain = (mean_auc_after - mean_auc_before) / max(abs(mean_auc_before), 1e-12)
            auc_pass = _strictly_below(auc_gain, 0.05)
            eligible = health_pass and rms_pass and hv_pass and auc_pass
            comparison_rows.append({
                "circuit": circuit,
                "trajectory_budget": budget,
                "successor_budget": successor,
                "health_pass": health_pass,
                "rms_pass": rms_pass,
                "fixed_uniform_rms": stratum_results["fixed_uniform"],
                "fresh_on_policy_rms": stratum_results["fresh_on_policy"],
                "mean_hypervolume_before": float(np.mean(hv_before)),
                "mean_hypervolume_after": float(np.mean(hv_after)),
                "absolute_mean_hypervolume_gain": abs(hv_gain),
                "hypervolume_pass": hv_pass,
                "mean_best_of_n_auc_before": mean_auc_before,
                "mean_best_of_n_auc_after": mean_auc_after,
                "relative_best_of_n_auc_gain": auc_gain,
                "best_of_n_auc_pass": auc_pass,
                "eligible": eligible,
            })
            if selected is None and eligible:
                selected = budget
        circuit_candidates[circuit] = selected
    complete_selection = all(value is not None for value in circuit_candidates.values())
    selected_budget = max(value for value in circuit_candidates.values() if value is not None) if complete_selection else None
    return {
        "complete": True,
        "decision": "select_common_trajectory_budget" if complete_selection else "reject_no_finite_budget_within_cap",
        "selected_budget": selected_budget,
        "circuit_candidates": circuit_candidates,
        "health_gates": health_rows,
        "comparisons": comparison_rows,
        "bootstrap": bootstrap_rows,
        "recommendation": (
            f"Use {selected_budget} complete training trajectories per circuit/seed."
            if complete_selection else
            "Extend the trajectory cap or run Experiment 6, then repeat the entire budget curve."
        ),
    }


def _seed_metric_rows(runs: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (circuit, seed), run in sorted(runs.items()):
        for budget in MILESTONES:
            milestone = run["milestones"][budget]
            for stratum in EXPECTED_STRATA:
                residual = milestone[stratum]["residual"]
                rows.append({
                    "circuit": circuit, "seed": seed, "trajectory_budget": budget,
                    "stratum": stratum, "centered_rms": residual["centered_rms"],
                    "log_z_target_gap": residual["log_z_target_gap"],
                    "bias_fraction": residual["bias_fraction"],
                    "standardized_bias": residual["standardized_bias"],
                    "training_archive_hypervolume": milestone["training_archive"]["hypervolume"],
                    "best_of_n_auc": milestone["search"]["log2_n_hypervolume_auc"],
                })
    return rows


def _best_of_n_rows(runs: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"circuit": circuit, "seed": seed, "trajectory_budget": budget, **row}
        for (circuit, seed), run in sorted(runs.items())
        for budget in MILESTONES
        for row in run["milestones"][budget]["search"]["budgets"]
    ]


def _write_plots(
    output_dir: Path,
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    health = decision["health_gates"]
    for circuit in EXPECTED_CIRCUITS:
        values = [sum(
            row["pass"] for row in health
            if row["circuit"] == circuit and row["trajectory_budget"] == budget
        ) for budget in MILESTONES]
        plt.plot(MILESTONES, values, marker="o", label=circuit)
    plt.axhline(len(EXPECTED_SEEDS) * len(EXPECTED_STRATA), color="black", linestyle="--")
    plt.xscale("log", base=2); plt.xlabel("training trajectories"); plt.ylabel("passing seed/stratum gates")
    plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "01_health_gates.png", dpi=160); plt.close()

    for stratum in EXPECTED_STRATA:
        for circuit in EXPECTED_CIRCUITS:
            values = [float(np.mean([
                runs[(circuit, seed)]["milestones"][budget][stratum]["residual"]["centered_rms"]
                for seed in EXPECTED_SEEDS
            ])) for budget in MILESTONES]
            plt.plot(MILESTONES, values, marker="o", label=f"{circuit}/{stratum}")
    plt.xscale("log", base=2); plt.xlabel("training trajectories"); plt.ylabel("mean centered RMS")
    plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "02_centered_rms.png", dpi=160); plt.close()

    for metric, key, filename, ylabel in (
        ("training_archive", "hypervolume", "03_archive_hypervolume.png", "mean archive hypervolume"),
        ("search", "log2_n_hypervolume_auc", "04_best_of_n_auc.png", "mean best-of-N AUC"),
    ):
        for circuit in EXPECTED_CIRCUITS:
            values = [float(np.mean([
                runs[(circuit, seed)]["milestones"][budget][metric][key]
                for seed in EXPECTED_SEEDS
            ])) for budget in MILESTONES]
            plt.plot(MILESTONES, values, marker="o", label=circuit)
        plt.xscale("log", base=2); plt.xlabel("training trajectories"); plt.ylabel(ylabel)
        plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / filename, dpi=160); plt.close()


def _decision_markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 5 trajectory-budget decision",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        f"- `bc0` candidate: `{decision['circuit_candidates']['bc0']}`",
        f"- `dalu` candidate: `{decision['circuit_candidates']['dalu']}`",
        f"- common `B*`: `{decision['selected_budget']}`",
        f"- recommendation: {decision['recommendation']}",
        "",
        "The run uses the Experiment 4 descriptive control (`zcal`, logZ LR 0.01), not a validated Experiment 4 winner.",
        "The 6,400 checkpoint is confirmation-only because it has no completed 12,800-trajectory successor.",
        "",
        "| circuit | B | 2B | health | RMS | |HV gain| | AUC gain | eligible |",
        "|:--|--:|--:|:--:|:--:|--:|--:|:--:|",
    ]
    for row in decision["comparisons"]:
        lines.append(
            f"| {row['circuit']} | {row['trajectory_budget']} | {row['successor_budget']} | "
            f"{row['health_pass']} | {row['rms_pass']} | {row['absolute_mean_hypervolume_gain']:.6f} | "
            f"{row['relative_best_of_n_auc_gain']:.6f} | {row['eligible']} |"
        )
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        runs = load_run_matrix(args.runs_root.resolve())
        decision = classify_budget_selection(runs)
        _write_json(output_dir / "decision_summary.json", decision)
        (output_dir / "decision_report.md").write_text(_decision_markdown(decision), encoding="utf-8")
        _write_csv(output_dir / "seed_milestone_metrics.csv", _seed_metric_rows(runs))
        _write_csv(output_dir / "health_gates.csv", decision["health_gates"])
        _write_csv(output_dir / "paired_budget_comparisons.csv", decision["comparisons"])
        _write_csv(output_dir / "bootstrap.csv", decision["bootstrap"])
        _write_csv(output_dir / "best_of_n.csv", _best_of_n_rows(runs))
        _write_plots(output_dir, runs, decision)
        return 0 if decision["selected_budget"] is not None else 3
    except IncompleteRunSetError as exc:
        _write_json(output_dir / "decision_summary.json", {
            "complete": False, "failure_type": "incomplete_run_matrix", "message": str(exc),
        })
        return 2
    except Exception as exc:
        _write_json(output_dir / "decision_summary.json", {
            "complete": False, "failure_type": "artifact_or_execution_failure",
            "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        })
        return 1


def add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="validate and select the Experiment 5 budget")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=report)


__all__ = [
    "ArtifactValidationError", "IncompleteRunSetError", "classify_budget_selection",
    "load_run_matrix", "paired_median_bootstrap", "report", "validate_run",
]
