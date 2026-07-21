"""Authoritative aggregation for the active-baseline diagnosis experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.gflownet_tb.diagnostics import HealthThresholds, health_gates


EXPECTED_MILESTONES = (200, 400, 800)
EXPECTED_STRATA = ("fixed_uniform", "fresh_on_policy")


class IncompleteRunSetError(RuntimeError):
    pass


class ArtifactValidationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IncompleteRunSetError(f"missing artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactValidationError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise IncompleteRunSetError(f"missing artifact: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except Exception as exc:
        raise ArtifactValidationError(f"invalid JSONL artifact {path}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _milestone_map(summary: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result = {int(row["trajectory_budget"]): dict(row) for row in summary["milestones"]}
    if set(result) != set(EXPECTED_MILESTONES):
        raise IncompleteRunSetError(
            f"milestones are {sorted(result)}, expected {list(EXPECTED_MILESTONES)}"
        )
    return result


def _validate_run(run_dir: Path, *, circuit: str, seed: int) -> dict[str, Any]:
    summary = _read_json(run_dir / "run_summary.json")
    resolved = _read_json(run_dir / "resolved_config.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    if not summary.get("complete"):
        raise IncompleteRunSetError(f"run is incomplete: {run_dir}")
    if summary.get("numerical_failure") is not None:
        raise ArtifactValidationError(f"run reports numerical failure: {run_dir}")
    if summary.get("circuit") != circuit or int(summary.get("seed", -1)) != seed:
        raise ArtifactValidationError(f"run identity mismatch: {run_dir}")
    fingerprint = summary.get("scientific_configuration_fingerprint")
    if fingerprint != resolved.get("scientific_configuration_fingerprint"):
        raise ArtifactValidationError(f"configuration fingerprint mismatch: {run_dir}")
    if summary.get("initial_parameter_checksum") != metadata.get("initial_parameter_checksum"):
        raise ArtifactValidationError(f"initialization checksum mismatch: {run_dir}")
    fixed_path = run_dir / "fixed_validation.pt"
    if not fixed_path.is_file():
        raise IncompleteRunSetError(f"missing fixed validation cache: {fixed_path}")
    if _sha256(fixed_path) != summary.get("fixed_validation_checksum"):
        raise ArtifactValidationError(f"fixed validation cache is corrupt: {run_dir}")
    try:
        fixed_payload = torch.load(fixed_path, map_location="cpu", weights_only=False)
        fixed_trajectories = fixed_payload["trajectories"]
    except Exception as exc:
        raise ArtifactValidationError(f"fixed validation cache cannot be loaded: {run_dir}: {exc}") from exc
    if len(fixed_trajectories) != 256:
        raise ArtifactValidationError(f"fixed validation cache has wrong size: {run_dir}")
    for trajectory in fixed_trajectories:
        if len(trajectory.steps) != 20:
            raise ArtifactValidationError(f"fixed validation trajectory has wrong horizon: {run_dir}")
        if any(step.action not in step.legal_actions for step in trajectory.steps):
            raise ArtifactValidationError(f"fixed validation trajectory contains an illegal action: {run_dir}")
        if float(trajectory.log_pb_sum) != 0.0:
            raise ArtifactValidationError(f"fixed validation log P_B is non-zero: {run_dir}")
    milestone_rows = _read_jsonl(run_dir / "milestones.jsonl")
    summary_milestones = _milestone_map(summary)
    file_milestones = {int(row["trajectory_budget"]): row for row in milestone_rows}
    if set(file_milestones) != set(EXPECTED_MILESTONES):
        raise IncompleteRunSetError(f"milestones.jsonl incomplete: {run_dir}")
    for budget in EXPECTED_MILESTONES:
        if file_milestones[budget] != summary_milestones[budget]:
            raise ArtifactValidationError(f"summary/milestone disagreement at {budget}: {run_dir}")
        for suffix in ("validation.csv", "best_of_n.csv"):
            table_path = run_dir / "tables" / f"trajectory_{budget}_{suffix}"
            if not table_path.is_file():
                raise IncompleteRunSetError(f"missing milestone CSV for {budget}: {run_dir}")
            try:
                with table_path.open(newline="", encoding="utf-8") as handle:
                    table_rows = list(csv.DictReader(handle))
            except Exception as exc:
                raise ArtifactValidationError(f"corrupt milestone CSV {table_path}: {exc}") from exc
            expected_rows = 2 if suffix == "validation.csv" else 6
            if len(table_rows) != expected_rows:
                raise ArtifactValidationError(
                    f"milestone CSV {table_path} has {len(table_rows)} rows, expected {expected_rows}"
                )
        checkpoint = run_dir / "checkpoints" / f"trajectory_{budget}.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise IncompleteRunSetError(f"missing checkpoint: {checkpoint}")
        try:
            checkpoint_value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ArtifactValidationError(f"corrupt checkpoint {checkpoint}: {exc}") from exc
        required_checkpoint_keys = {
            "policy", "optimizer", "scheduler", "replay", "archive", "counters",
            "global_rng", "train_action_generator_state", "fixed_cache_checksum",
            "resolved_config", "run_metadata", "source_tree_sha256",
        }
        missing_keys = required_checkpoint_keys.difference(checkpoint_value)
        if missing_keys:
            raise ArtifactValidationError(
                f"checkpoint {checkpoint} is missing keys: {sorted(missing_keys)}"
            )
        if checkpoint_value["fixed_cache_checksum"] != summary.get("fixed_validation_checksum"):
            raise ArtifactValidationError(f"checkpoint fixed-cache checksum mismatch: {checkpoint}")
        if int(checkpoint_value["counters"]["training_trajectories"]) != budget:
            raise ArtifactValidationError(f"checkpoint counter mismatch: {checkpoint}")
        if (
            checkpoint_value["resolved_config"]["scientific_configuration_fingerprint"]
            != fingerprint
        ):
            raise ArtifactValidationError(f"checkpoint configuration mismatch: {checkpoint}")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    initial_rows = [row for row in metrics if row.get("row_type") == "initial_fixed_validation"]
    if len(initial_rows) != 1:
        raise ArtifactValidationError(
            f"expected exactly one initial fixed-validation row in {run_dir}"
        )
    update_rows = [row for row in metrics if row.get("row_type") == "training_update"]
    if len(update_rows) != 200:
        raise IncompleteRunSetError(f"expected 200 training updates in {run_dir}, found {len(update_rows)}")
    counters = summary["counters"]
    expected_counters = {
        "training_trajectories": 800,
        "optimizer_updates": 200,
        "training_presentations": 800,
    }
    for key, expected in expected_counters.items():
        if int(counters.get(key, -1)) != expected:
            raise ArtifactValidationError(f"counter {key} mismatch in {run_dir}")
    trajectory_rows = _read_jsonl(run_dir / "trajectories.jsonl")
    expected_sources = {
        "fixed_uniform": 256,
        "training": 800,
        "fresh_on_policy": 128 * 3,
        "search": 50 * 3,
    }
    actual_sources = {
        source: sum(row.get("source") == source for row in trajectory_rows)
        for source in expected_sources
    }
    if actual_sources != expected_sources:
        raise ArtifactValidationError(
            f"trajectory archive source counts mismatch in {run_dir}: {actual_sources}"
        )
    if any(int(row.get("trajectory_length", -1)) != 20 for row in trajectory_rows):
        raise ArtifactValidationError(f"trajectory archive contains a wrong horizon: {run_dir}")
    return {
        "dir": run_dir,
        "summary": summary,
        "resolved": resolved,
        "metadata": metadata,
        "milestones": summary_milestones,
        "initial": initial_rows[0],
        "updates": update_rows,
    }


def classify_diagnosis(runs: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    circuits = sorted({key[0] for key in runs})
    seeds = sorted({key[1] for key in runs})
    offset_rows: list[dict[str, Any]] = []
    pooled_bias: list[float] = []
    for circuit in circuits:
        for stratum in EXPECTED_STRATA:
            reductions = [
                float(runs[(circuit, seed)]["milestones"][800][stratum]["residual"]["recentered_mse_reduction"])
                for seed in seeds
            ]
            biases = [
                float(runs[(circuit, seed)]["milestones"][800][stratum]["residual"]["bias_fraction"])
                for seed in seeds
            ]
            pooled_bias.extend(biases)
            median_reduction = float(np.median(reductions))
            offset_rows.append(
                {
                    "circuit": circuit,
                    "stratum": stratum,
                    "median_recentered_mse_reduction": median_reduction,
                    "threshold": 0.5,
                    "margin": median_reduction - 0.5,
                    "passes": median_reduction >= 0.5,
                    "per_seed": reductions,
                }
            )
    pooled_bias_median = float(np.median(pooled_bias))
    offset_supported = all(row["passes"] for row in offset_rows) and pooled_bias_median > 0.5

    circuit_rows: list[dict[str, Any]] = []
    for circuit in circuits:
        rms_decreases: list[float] = []
        hv_gains: list[float] = []
        auc_400: list[float] = []
        auc_800: list[float] = []
        for seed in seeds:
            milestones = runs[(circuit, seed)]["milestones"]
            rms_before = float(milestones[400]["fixed_uniform"]["residual"]["centered_rms"])
            rms_after = float(milestones[800]["fixed_uniform"]["residual"]["centered_rms"])
            rms_decreases.append((rms_before - rms_after) / max(rms_before, 1e-12))
            hv_gains.append(
                float(milestones[800]["training_archive"]["hypervolume"])
                - float(milestones[400]["training_archive"]["hypervolume"])
            )
            auc_400.append(float(milestones[400]["search"]["log2_n_hypervolume_auc"]))
            auc_800.append(float(milestones[800]["search"]["log2_n_hypervolume_auc"]))
        median_rms = float(np.median(rms_decreases))
        mean_hv_gain = float(np.mean(hv_gains))
        mean_auc_400 = float(np.mean(auc_400))
        mean_auc_800 = float(np.mean(auc_800))
        auc_relative = (mean_auc_800 - mean_auc_400) / max(abs(mean_auc_400), 1e-12)
        checks = {
            "centered_rms_decrease": median_rms >= 0.05,
            "archive_hypervolume_gain": mean_hv_gain >= 0.005,
            "best_of_n_auc_increase": auc_relative >= 0.05,
        }
        circuit_rows.append(
            {
                "circuit": circuit,
                "median_fixed_centered_rms_decrease": median_rms,
                "fixed_centered_rms_margin": median_rms - 0.05,
                "mean_archive_hypervolume_gain": mean_hv_gain,
                "archive_hypervolume_margin": mean_hv_gain - 0.005,
                "mean_best_of_n_auc_400": mean_auc_400,
                "mean_best_of_n_auc_800": mean_auc_800,
                "relative_best_of_n_auc_increase": auc_relative,
                "best_of_n_auc_margin": auc_relative - 0.05,
                "checks": checks,
                "supports_undertraining": any(checks.values()),
            }
        )
    undertraining_supported = any(row["supports_undertraining"] for row in circuit_rows)

    gate_rows: list[dict[str, Any]] = []
    all_endpoint_healthy = True
    thresholds = HealthThresholds()
    for (circuit, seed), run in sorted(runs.items()):
        for budget in EXPECTED_MILESTONES:
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
                row = {
                    "circuit": circuit,
                    "seed": seed,
                    "trajectory_budget": budget,
                    "stratum": stratum,
                    "pass": gate["pass"],
                    "failed": gate["failed"],
                    "absolute_log_z_target_gap": abs(
                        float(milestone[stratum]["residual"]["log_z_target_gap"])
                    ),
                    "log_z_target_gap_margin": thresholds.log_z_target_gap
                    - abs(float(milestone[stratum]["residual"]["log_z_target_gap"])),
                    "bias_fraction": float(milestone[stratum]["residual"]["bias_fraction"]),
                    "bias_fraction_margin": thresholds.bias_fraction
                    - float(milestone[stratum]["residual"]["bias_fraction"]),
                    "standardized_bias": float(
                        milestone[stratum]["residual"]["standardized_bias"]
                    ),
                    "standardized_bias_margin": thresholds.standardized_bias
                    - float(milestone[stratum]["residual"]["standardized_bias"]),
                    "gradient_p99_median_ratio": float(
                        optimizer_health["policy_gradient_p99_median_ratio"]
                    ),
                    "gradient_ratio_margin": thresholds.gradient_p99_median_ratio
                    - float(optimizer_health["policy_gradient_p99_median_ratio"]),
                    "clipping_rate": float(optimizer_health["gradient_clipping_rate"]),
                    "collapse_fraction": float(
                        milestone[stratum]["policy"]["collapse_fraction"]
                    ),
                    "collapse_fraction_margin": thresholds.collapse_fraction
                    - float(milestone[stratum]["policy"]["collapse_fraction"]),
                    **{f"gate_{name}": value for name, value in gate["checks"].items()},
                }
                gate_rows.append(row)
                if budget == 800 and not gate["pass"]:
                    all_endpoint_healthy = False

    if undertraining_supported:
        undertraining_decision = "undertraining_supported"
        conclusive = True
        exit_code = 0
    elif all_endpoint_healthy:
        undertraining_decision = "800_not_demonstrably_insufficient"
        conclusive = True
        exit_code = 0
    else:
        undertraining_decision = "inconclusive_unhealthy_active_baseline"
        conclusive = False
        exit_code = 3
    if not conclusive:
        recommendation = "Repair the failed active-baseline health gates before selecting a trajectory budget."
    elif undertraining_supported:
        recommendation = "Proceed to the budget-extension experiment while retaining the active optimizer as the control."
    elif offset_supported:
        recommendation = "Proceed to calibrated logZ initialization; 800 remains only a candidate budget."
    else:
        recommendation = "Investigate centered policy error or representation before prioritizing logZ initialization."
    return {
        "global_offset": {
            "decision": "supported" if offset_supported else "rejected",
            "supported": offset_supported,
            "per_circuit_stratum": offset_rows,
            "pooled_median_bias_fraction": pooled_bias_median,
            "pooled_bias_threshold": 0.5,
            "pooled_bias_margin": pooled_bias_median - 0.5,
        },
        "undertraining": {
            "decision": undertraining_decision,
            "supported": undertraining_supported,
            "per_circuit": circuit_rows,
            "all_endpoint_health_gates_pass": all_endpoint_healthy,
        },
        "health_gates": gate_rows,
        "complete_and_conclusive": conclusive,
        "recommended_next_experiment": recommendation,
        "exit_code": exit_code,
    }


def _seed_rows(runs: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (circuit, seed), run in sorted(runs.items()):
        initial_value = run["initial"]["fixed_uniform"]
        initial_residual = initial_value["residual"]
        rows.append(
            {
                "circuit": circuit,
                "seed": seed,
                "trajectory_budget": 0,
                "stratum": "fixed_uniform",
                "learned_log_z_mse": initial_residual["learned_log_z_mse"],
                "centered_mse": initial_residual["centered_mse"],
                "centered_rms": initial_residual["centered_rms"],
                "residual_mean": initial_residual["residual_mean"],
                "bias_fraction": initial_residual["bias_fraction"],
                "standardized_bias": initial_residual["standardized_bias"],
                "recentered_mse_reduction": initial_residual["recentered_mse_reduction"],
                "log_z_target_gap": initial_residual["log_z_target_gap"],
                "regression_slope": initial_value["regression"]["slope"],
                "regression_correlation": initial_value["regression"]["correlation"],
                "collapse_fraction": initial_value["policy"]["collapse_fraction"],
                "training_archive_hypervolume": 0.0,
                "best_of_n_auc": None,
            }
        )
        for budget, milestone in sorted(run["milestones"].items()):
            for stratum in EXPECTED_STRATA:
                value = milestone[stratum]
                residual = value["residual"]
                rows.append(
                    {
                        "circuit": circuit,
                        "seed": seed,
                        "trajectory_budget": budget,
                        "stratum": stratum,
                        "learned_log_z_mse": residual["learned_log_z_mse"],
                        "centered_mse": residual["centered_mse"],
                        "centered_rms": residual["centered_rms"],
                        "residual_mean": residual["residual_mean"],
                        "bias_fraction": residual["bias_fraction"],
                        "standardized_bias": residual["standardized_bias"],
                        "recentered_mse_reduction": residual["recentered_mse_reduction"],
                        "log_z_target_gap": residual["log_z_target_gap"],
                        "regression_slope": value["regression"]["slope"],
                        "regression_correlation": value["regression"]["correlation"],
                        "collapse_fraction": value["policy"]["collapse_fraction"],
                        "training_archive_hypervolume": milestone["training_archive"]["hypervolume"],
                        "best_of_n_auc": milestone["search"]["log2_n_hypervolume_auc"],
                    }
                )
    return rows


def _best_of_n_rows(runs: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (circuit, seed), run in sorted(runs.items()):
        for budget, milestone in sorted(run["milestones"].items()):
            for value in milestone["search"]["budgets"]:
                rows.append(
                    {
                        "circuit": circuit,
                        "seed": seed,
                        "trajectory_budget": budget,
                        **value,
                    }
                )
    return rows


def _milestone_aggregate_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    circuits = sorted({str(row["circuit"]) for row in seed_rows})
    for circuit in circuits:
        for budget in EXPECTED_MILESTONES:
            for stratum in EXPECTED_STRATA:
                selected = [
                    row for row in seed_rows
                    if row["circuit"] == circuit
                    and int(row["trajectory_budget"]) == budget
                    and row["stratum"] == stratum
                ]
                rows.append(
                    {
                        "circuit": circuit,
                        "trajectory_budget": budget,
                        "stratum": stratum,
                        "seed_count": len(selected),
                        "median_learned_log_z_mse": float(np.median([
                            float(row["learned_log_z_mse"]) for row in selected
                        ])),
                        "median_centered_mse": float(np.median([
                            float(row["centered_mse"]) for row in selected
                        ])),
                        "median_centered_rms": float(np.median([
                            float(row["centered_rms"]) for row in selected
                        ])),
                        "median_bias_fraction": float(np.median([
                            float(row["bias_fraction"]) for row in selected
                        ])),
                        "median_log_z_target_gap": float(np.median([
                            float(row["log_z_target_gap"]) for row in selected
                        ])),
                        "mean_training_archive_hypervolume": float(np.mean([
                            float(row["training_archive_hypervolume"]) for row in selected
                        ])),
                        "mean_best_of_n_auc": float(np.mean([
                            float(row["best_of_n_auc"]) for row in selected
                        ])),
                    }
                )
    return rows


def _plot_reports(
    output_dir: Path,
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = plot_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    def lines(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(8, 5))
        for circuit in sorted({row["circuit"] for row in seed_rows}):
            for stratum in EXPECTED_STRATA:
                points = []
                for budget in EXPECTED_MILESTONES:
                    values = [
                        float(row[metric]) for row in seed_rows
                        if row["circuit"] == circuit
                        and row["stratum"] == stratum
                        and row["trajectory_budget"] == budget
                    ]
                    points.append(float(np.median(values)))
                plt.plot(EXPECTED_MILESTONES, points, marker="o", label=f"{circuit}/{stratum}")
        plt.xlabel("training trajectories")
        plt.ylabel(ylabel)
        plt.legend(fontsize=7)
        plt.grid(alpha=0.25)
        save(filename)

    lines("learned_log_z_mse", "learned-logZ residual MSE", "01_learned_mse.png")
    lines("centered_mse", "analytically centered MSE", "02_centered_mse.png")
    lines("log_z_target_gap", "logZ - analytic target", "03_logz_target_gap.png")
    lines("bias_fraction", "TB MSE bias fraction", "04_bias_fraction.png")

    plt.figure(figsize=(8, 5))
    for (circuit, seed), run in sorted(runs.items()):
        x = [int(row["trajectory_budget"]) for row in run["updates"]]
        y = [float(row["policy_gradient_norm"]) for row in run["updates"]]
        plt.plot(x, y, alpha=0.7, label=f"{circuit}/s{seed}")
    plt.yscale("log")
    plt.xlabel("training trajectories")
    plt.ylabel("policy gradient norm")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    save("05_policy_gradient_norm.png")

    plt.figure(figsize=(8, 5))
    for circuit in sorted({row["circuit"] for row in seed_rows}):
        values = []
        for budget in EXPECTED_MILESTONES:
            per_seed = [
                float(run["milestones"][budget]["training_archive"]["hypervolume"])
                for (name, _), run in runs.items() if name == circuit
            ]
            values.append(float(np.mean(per_seed)))
        plt.plot(EXPECTED_MILESTONES, values, marker="o", label=circuit)
    plt.xlabel("training trajectories")
    plt.ylabel("mean training archive hypervolume")
    plt.legend()
    plt.grid(alpha=0.25)
    save("06_training_archive_hypervolume.png")

    plt.figure(figsize=(8, 5))
    for circuit in sorted({row["circuit"] for row in best_rows}):
        for budget in (400, 800):
            ns = sorted({int(row["n"]) for row in best_rows})
            values = [
                float(np.mean([
                    float(row["hypervolume"]) for row in best_rows
                    if row["circuit"] == circuit
                    and row["trajectory_budget"] == budget
                    and int(row["n"]) == n
                ]))
                for n in ns
            ]
            plt.plot(ns, values, marker="o", label=f"{circuit}/{budget}")
    plt.xscale("log", base=2)
    plt.xlabel("best-of-N rollouts")
    plt.ylabel("mean nested archive hypervolume")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    save("07_best_of_n_hypervolume.png")
    return paths


def _markdown_report(summary: Mapping[str, Any]) -> str:
    offset = summary["global_offset"]
    under = summary["undertraining"]
    failed_endpoint = [
        row for row in summary["health_gates"]
        if row["trajectory_budget"] == 800 and not row["pass"]
    ]
    lines = [
        "# Active-baseline diagnosis",
        "",
        f"- Global-offset hypothesis: **{offset['decision']}**.",
        f"- Undertraining diagnosis: **{under['decision']}**.",
        f"- Complete and scientifically conclusive: **{summary['complete_and_conclusive']}**.",
        f"- Recommended next experiment: {summary['recommended_next_experiment']}",
        "",
        "## Global-offset endpoint test",
        "",
        (
            f"The pooled median bias fraction at 800 trajectories is "
            f"{offset['pooled_median_bias_fraction']:.6f} (strict threshold > 0.5)."
        ),
        "",
        "| Circuit | Validation stratum | Median MSE removed | Margin | Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in offset["per_circuit_stratum"]:
        lines.append(
            f"| {row['circuit']} | {row['stratum']} | "
            f"{row['median_recentered_mse_reduction']:.6f} | {row['margin']:.6f} | {row['passes']} |"
        )
    lines.extend(
        [
            "",
            "## Undertraining test (400 to 800 trajectories)",
            "",
            "| Circuit | Centered-RMS decrease | HV gain | Best-of-N AUC increase | Supported |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in under["per_circuit"]:
        lines.append(
            f"| {row['circuit']} | {row['median_fixed_centered_rms_decrease']:.6f} | "
            f"{row['mean_archive_hypervolume_gain']:.6f} | "
            f"{row['relative_best_of_n_auc_increase']:.6f} | {row['supports_undertraining']} |"
        )
    lines.extend(["", "## Health gates at 800 trajectories", ""])
    if not failed_endpoint:
        lines.append("All six seeds on both validation strata pass every endpoint health gate.")
    else:
        lines.append("The following endpoint evaluations fail at least one gate:")
        lines.append("")
        for row in failed_endpoint:
            lines.append(
                f"- {row['circuit']} seed {row['seed']} / {row['stratum']}: "
                + ", ".join(row["failed"])
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Earlier milestones are diagnostic only for the global-offset decision; the formal offset decision uses the 800-trajectory endpoint. Evaluation trajectories are excluded from the training archive, and best-of-N curves use nested prefixes of the same 50 search rollouts.",
            "",
        ]
    )
    return "\n".join(lines)


def report_experiment(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    circuits = list(dict.fromkeys(args.expected_circuits))
    seeds = list(dict.fromkeys(int(seed) for seed in args.expected_seeds))
    try:
        runs: dict[tuple[str, int], dict[str, Any]] = {}
        for circuit in circuits:
            for seed in seeds:
                run_dir = args.runs_root.resolve() / circuit / f"seed_{seed}"
                runs[(circuit, seed)] = _validate_run(run_dir, circuit=circuit, seed=seed)
        fingerprints = {
            run["resolved"]["scientific_configuration_fingerprint"] for run in runs.values()
        }
        if len(fingerprints) != 1:
            raise ArtifactValidationError("scientific configuration fingerprints differ across runs")
        source_checksums = {run["metadata"]["source_tree_sha256"] for run in runs.values()}
        if len(source_checksums) != 1:
            raise ArtifactValidationError("source-tree checksums differ across runs")
        for seed in seeds:
            checksums = {
                runs[(circuit, seed)]["metadata"]["initial_parameter_checksum"]
                for circuit in circuits
            }
            if len(checksums) != 1:
                raise ArtifactValidationError(
                    f"paired circuit runs for seed {seed} have unmatched initializations"
                )
        diagnosis = classify_diagnosis(runs)
        seed_rows = _seed_rows(runs)
        milestone_rows = _milestone_aggregate_rows(seed_rows)
        best_rows = _best_of_n_rows(runs)
        update_rows = [
            {"circuit": circuit, "seed": seed, **row}
            for (circuit, seed), run in sorted(runs.items())
            for row in run["updates"]
        ]
        diagnosis.update(
            {
                "schema_version": 1,
                "run_count": len(runs),
                "circuits": circuits,
                "seeds": seeds,
                "milestones": list(EXPECTED_MILESTONES),
                "scientific_configuration_fingerprint": next(iter(fingerprints)),
                "source_tree_sha256": next(iter(source_checksums)),
                "initial_diagnostics": [
                    {
                        "circuit": circuit,
                        "median_fixed_recentered_mse_reduction": float(np.median([
                            runs[(circuit, seed)]["initial"]["fixed_uniform"]["residual"][
                                "recentered_mse_reduction"
                            ]
                            for seed in seeds
                        ])),
                        "median_fixed_bias_fraction": float(np.median([
                            runs[(circuit, seed)]["initial"]["fixed_uniform"]["residual"][
                                "bias_fraction"
                            ]
                            for seed in seeds
                        ])),
                    }
                    for circuit in circuits
                ],
            }
        )
        plots = _plot_reports(output_dir, runs, seed_rows, best_rows)
        diagnosis["plots"] = plots
        _write_json(output_dir / "diagnosis_summary.json", diagnosis)
        (output_dir / "diagnosis_report.md").write_text(
            _markdown_report(diagnosis), encoding="utf-8"
        )
        _write_csv(output_dir / "seed_metrics.csv", seed_rows)
        _write_csv(output_dir / "milestone_metrics.csv", milestone_rows)
        _write_csv(output_dir / "gate_results.csv", diagnosis["health_gates"])
        _write_csv(output_dir / "offset_results.csv", diagnosis["global_offset"]["per_circuit_stratum"])
        _write_csv(output_dir / "circuit_summary.csv", diagnosis["undertraining"]["per_circuit"])
        _write_csv(output_dir / "best_of_n.csv", best_rows)
        _write_csv(output_dir / "optimizer_updates.csv", update_rows)
        print(json.dumps(diagnosis, indent=2, sort_keys=True))
        return int(diagnosis["exit_code"])
    except IncompleteRunSetError as exc:
        failure = {"complete": False, "failure_type": "incomplete_run_set", "message": str(exc)}
        _write_json(output_dir / "diagnosis_summary.json", failure)
        print(str(exc))
        return 2
    except Exception as exc:
        failure = {"complete": False, "failure_type": "artifact_or_execution_failure", "message": str(exc)}
        _write_json(output_dir / "diagnosis_summary.json", failure)
        print(str(exc))
        return 1


def add_report_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("report", help="validate and diagnose all active-baseline runs")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-circuits", nargs="+", default=["bc0", "dalu"])
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.set_defaults(handler=report_experiment)


__all__ = [
    "ArtifactValidationError",
    "IncompleteRunSetError",
    "add_report_parser",
    "classify_diagnosis",
    "report_experiment",
]
