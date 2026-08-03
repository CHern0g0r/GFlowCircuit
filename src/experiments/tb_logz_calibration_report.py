"""Validation, paired analysis, and decision report for TB Experiment 3."""

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


EXPECTED_VARIANTS = ("z0", "zcal")
EXPECTED_CIRCUITS = ("bc0", "dalu")
EXPECTED_SEEDS = (0, 1, 2)
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
        for line in path.read_text(encoding="utf-8").splitlines():
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
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def oscillation_diagnostics(target_gaps: Sequence[float]) -> dict[str, Any]:
    """Apply the protocol's persistent target-gap oscillation rule."""
    values = [float(value) for value in target_gaps]
    if len(values) != 200 or not all(math.isfinite(value) for value in values):
        return {"persistent": True, "sign_changes": None, "reason": "requires_200_finite_updates"}
    signs = [1 if value > 0 else -1 for value in values if value != 0.0]
    changes = sum(left != right for left, right in zip(signs, signs[1:]))
    preceding = float(np.mean(np.abs(values[100:150])))
    final = float(np.mean(np.abs(values[150:200])))
    return {
        "persistent": bool(changes > 4 and final > preceding),
        "sign_changes": int(changes),
        "sign_change_threshold": 4,
        "preceding_50_mean_absolute_gap": preceding,
        "final_50_mean_absolute_gap": final,
        "final_gap_worsened": final > preceding,
    }


def _validate_run(run_dir: Path, *, variant: str, circuit: str, seed: int) -> dict[str, Any]:
    summary = _read_json(run_dir / "run_summary.json")
    resolved = _read_json(run_dir / "resolved_config.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    if not summary.get("complete") or summary.get("numerical_failure") is not None:
        raise IncompleteRunSetError(f"run is incomplete or failed: {run_dir}")
    identity = (summary.get("variant"), summary.get("circuit"), int(summary.get("seed", -1)))
    if identity != (variant, circuit, seed):
        raise ArtifactValidationError(f"run identity mismatch: {run_dir}: {identity}")
    if resolved.get("schema_version") != 2 or resolved.get("experiment") != "calibrated_log_z_initialization":
        raise ArtifactValidationError(f"wrong Experiment 3 schema: {run_dir}")
    if summary.get("scientific_configuration_fingerprint") != resolved.get("scientific_configuration_fingerprint"):
        raise ArtifactValidationError(f"configuration fingerprint mismatch: {run_dir}")
    if summary.get("paired_configuration_fingerprint") != resolved.get("paired_configuration_fingerprint"):
        raise ArtifactValidationError(f"paired configuration fingerprint mismatch: {run_dir}")
    if summary.get("pre_calibration_parameter_checksum") != metadata.get("pre_calibration_parameter_checksum"):
        raise ArtifactValidationError(f"pre-calibration checksum mismatch: {run_dir}")
    if summary.get("post_initialization_parameter_checksum") != metadata.get("post_initialization_parameter_checksum"):
        raise ArtifactValidationError(f"post-initialization checksum mismatch: {run_dir}")
    for key in ("fixed_sequence_checksum", "calibration_sequence_checksum"):
        if summary.get(key) != metadata.get(key):
            raise ArtifactValidationError(f"{key} mismatch: {run_dir}")
    loaded_payloads: dict[str, Any] = {}
    for filename, key, expected_count in (
        ("fixed_validation.pt", "fixed_validation_checksum", 256),
        ("calibration.pt", "calibration_cache_checksum", 64),
    ):
        path = run_dir / filename
        if not path.is_file() or _sha256(path) != summary.get(key):
            raise ArtifactValidationError(f"missing or corrupt {filename}: {run_dir}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        loaded_payloads[filename] = payload
        trajectories = payload["trajectories"]
        if len(trajectories) != expected_count:
            raise ArtifactValidationError(f"wrong trajectory count in {filename}: {run_dir}")
        for trajectory in trajectories:
            if len(trajectory.steps) != 20 or any(step.action not in step.legal_actions for step in trajectory.steps):
                raise ArtifactValidationError(f"illegal or wrong-horizon trajectory in {filename}: {run_dir}")
            if float(trajectory.log_pb_sum) != 0.0:
                raise ArtifactValidationError(f"non-zero log P_B in {filename}: {run_dir}")
    calibration_payload = loaded_payloads["calibration.pt"]
    calibration_trajectories = calibration_payload["trajectories"]
    expected_actions = [[int(step.action) for step in trajectory.steps] for trajectory in calibration_trajectories]
    if calibration_payload.get("actions") != expected_actions:
        raise ArtifactValidationError(f"calibration action manifest mismatch: {run_dir}")
    if float(calibration_payload.get("epsilon_uniform", -1.0)) != 0.5:
        raise ArtifactValidationError(f"calibration epsilon mismatch: {run_dir}")
    if calibration_payload.get("batch_order") != "collection_order":
        raise ArtifactValidationError(f"calibration batch order mismatch: {run_dir}")
    target = float(calibration_payload["analytic_log_z_target"])
    reconstructed_target = float(np.mean(
        np.asarray(calibration_payload["log_r"], dtype=np.float64)
        + np.asarray(calibration_payload["log_pb"], dtype=np.float64)
        - np.asarray(calibration_payload["initial_log_pf"], dtype=np.float64)
    ))
    if not math.isclose(target, reconstructed_target, rel_tol=0.0, abs_tol=1e-5):
        raise ArtifactValidationError(f"calibration target is inconsistent with cached scores: {run_dir}")
    if target != float(metadata.get("calibration_analytic_target")):
        raise ArtifactValidationError(f"calibration target metadata mismatch: {run_dir}")
    expected_initial_log_z = 0.0 if variant == "z0" else target
    if not math.isclose(float(metadata.get("assigned_initial_log_z")), expected_initial_log_z,
                        rel_tol=0.0, abs_tol=1e-7):
        raise ArtifactValidationError(f"assigned initial logZ mismatch: {run_dir}")
    milestones = {int(row["trajectory_budget"]): row for row in summary["milestones"]}
    if set(milestones) != set(EXPECTED_MILESTONES):
        raise IncompleteRunSetError(f"milestones are {sorted(milestones)}, expected {list(EXPECTED_MILESTONES)}")
    file_milestones = {int(row["trajectory_budget"]): row for row in _read_jsonl(run_dir / "milestones.jsonl")}
    if file_milestones != milestones:
        raise ArtifactValidationError(f"summary/milestones disagreement: {run_dir}")
    for budget in EXPECTED_MILESTONES:
        for suffix, expected_rows in (("validation.csv", 2), ("best_of_n.csv", 6)):
            table = run_dir / "tables" / f"trajectory_{budget}_{suffix}"
            if not table.is_file():
                raise IncompleteRunSetError(f"missing milestone table: {table}")
            with table.open(newline="", encoding="utf-8") as handle:
                if len(list(csv.DictReader(handle))) != expected_rows:
                    raise ArtifactValidationError(f"wrong row count in milestone table: {table}")
        checkpoint = run_dir / "checkpoints" / f"trajectory_{budget}.pt"
        if not checkpoint.is_file():
            raise IncompleteRunSetError(f"missing checkpoint: {checkpoint}")
        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {"policy", "optimizer", "scheduler", "replay", "archive", "counters", "global_rng",
                    "train_action_generator_state", "circuit_rng_state", "replay_rng_state",
                    "fixed_cache_checksum", "calibration_cache_checksum", "resolved_config", "run_metadata"}
        if missing := required.difference(value):
            raise ArtifactValidationError(f"checkpoint missing keys {sorted(missing)}: {checkpoint}")
        if int(value["counters"]["training_trajectories"]) != budget:
            raise ArtifactValidationError(f"checkpoint budget mismatch: {checkpoint}")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    updates = [row for row in metrics if row.get("row_type") == "training_update"]
    if len(updates) != 200:
        raise IncompleteRunSetError(f"expected 200 training updates in {run_dir}, found {len(updates)}")
    if [row.get("training_source") for row in updates[:16]] != ["calibration"] * 16:
        raise ArtifactValidationError(f"first 16 updates are not calibration minibatches: {run_dir}")
    if [row.get("calibration_batch_start") for row in updates[:16]] != list(range(0, 64, 4)):
        raise ArtifactValidationError(f"calibration minibatch order mismatch: {run_dir}")
    if any(row.get("training_source") != "new_on_policy" for row in updates[16:]):
        raise ArtifactValidationError(f"post-calibration update source mismatch: {run_dir}")
    expected_counters = {
        "training_trajectories": 800, "new_training_trajectories": 736,
        "calibration_trajectories": 64, "calibration_training_presentations": 64,
        "training_presentations": 800, "optimizer_updates": 200,
    }
    for key, expected in expected_counters.items():
        if int(summary["counters"].get(key, -1)) != expected:
            raise ArtifactValidationError(f"counter {key} mismatch: {run_dir}")
    trajectory_rows = _read_jsonl(run_dir / "trajectories.jsonl")
    expected_sources = {"fixed_uniform": 256, "calibration": 64, "training": 736,
                        "fresh_on_policy": 128 * 3, "search": 50 * 3}
    actual_sources = {source: sum(row.get("source") == source for row in trajectory_rows) for source in expected_sources}
    if actual_sources != expected_sources:
        raise ArtifactValidationError(f"trajectory source counts mismatch: {run_dir}: {actual_sources}")
    return {
        "summary": summary, "resolved": resolved, "metadata": metadata,
        "milestones": milestones, "updates": updates,
        "oscillation": oscillation_diagnostics([float(row["target_gap"]) for row in updates]),
    }


def _paired_bootstrap(values: Sequence[float], *, seed: int, samples: int = 10_000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    estimates = array[indices].mean(axis=1)
    return {
        "mean_difference": float(array.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "samples": samples,
    }


def _wins(values: Sequence[float], *, lower_is_better: bool) -> dict[str, int]:
    signs = [-value if lower_is_better else value for value in values]
    return {"zcal_wins": sum(value > 0 for value in signs),
            "ties": sum(value == 0 for value in signs),
            "z0_wins": sum(value < 0 for value in signs)}


def classify_calibration(runs: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> dict[str, Any]:
    circuits = sorted({key[1] for key in runs})
    seeds = sorted({key[2] for key in runs})
    thresholds = HealthThresholds()
    improvement_rows: list[dict[str, Any]] = []
    nonharm_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    all_health_at_200_800 = {variant: True for variant in EXPECTED_VARIANTS}
    for circuit in circuits:
        for stratum in EXPECTED_STRATA:
            z0_gap = [abs(float(runs[("z0", circuit, seed)]["milestones"][200][stratum]["residual"]["log_z_target_gap"])) for seed in seeds]
            zcal_gap = [abs(float(runs[("zcal", circuit, seed)]["milestones"][200][stratum]["residual"]["log_z_target_gap"])) for seed in seeds]
            z0_bias = [float(runs[("z0", circuit, seed)]["milestones"][200][stratum]["residual"]["bias_fraction"]) for seed in seeds]
            zcal_bias = [float(runs[("zcal", circuit, seed)]["milestones"][200][stratum]["residual"]["bias_fraction"]) for seed in seeds]
            gap_ratio = float(np.median(zcal_gap) / max(np.median(z0_gap), 1e-12))
            bias_ratio = float(np.median(zcal_bias) / max(np.median(z0_bias), 1e-12))
            improvement_rows.append({"circuit": circuit, "stratum": stratum,
                                     "median_absolute_gap_z0": float(np.median(z0_gap)),
                                     "median_absolute_gap_zcal": float(np.median(zcal_gap)),
                                     "gap_ratio": gap_ratio, "gap_pass": gap_ratio <= 0.5,
                                     "median_bias_z0": float(np.median(z0_bias)),
                                     "median_bias_zcal": float(np.median(zcal_bias)),
                                     "bias_ratio": bias_ratio, "bias_pass": bias_ratio <= 0.5,
                                     "pass": gap_ratio <= 0.5 and bias_ratio <= 0.5})
            z0_rms = [float(runs[("z0", circuit, seed)]["milestones"][800][stratum]["residual"]["centered_rms"]) for seed in seeds]
            zcal_rms = [float(runs[("zcal", circuit, seed)]["milestones"][800][stratum]["residual"]["centered_rms"]) for seed in seeds]
            rms_ratio = float(np.median(zcal_rms) / max(np.median(z0_rms), 1e-12))
            nonharm_rows.append({"metric": "centered_rms", "circuit": circuit, "stratum": stratum,
                                 "z0": float(np.median(z0_rms)), "zcal": float(np.median(zcal_rms)),
                                 "ratio": rms_ratio, "threshold": 1.1, "pass": rms_ratio <= 1.1})
            differences = [right - left for left, right in zip(z0_rms, zcal_rms)]
            bootstrap_rows.append({"metric": "centered_rms", "circuit": circuit, "stratum": stratum,
                                   **_paired_bootstrap(differences, seed=17 + len(bootstrap_rows)),
                                   **_wins(differences, lower_is_better=True)})
    for circuit in circuits:
        z0_hv = [float(runs[("z0", circuit, seed)]["milestones"][800]["training_archive"]["hypervolume"]) for seed in seeds]
        zcal_hv = [float(runs[("zcal", circuit, seed)]["milestones"][800]["training_archive"]["hypervolume"]) for seed in seeds]
        mean_z0, mean_zcal = float(np.mean(z0_hv)), float(np.mean(zcal_hv))
        ratio = math.inf if mean_z0 == 0.0 and mean_zcal < 0.0 else (1.0 if mean_z0 == 0.0 else mean_zcal / mean_z0)
        nonharm_rows.append({"metric": "archive_hypervolume", "circuit": circuit, "stratum": None,
                             "z0": mean_z0, "zcal": mean_zcal, "ratio": ratio,
                             "threshold": 0.9, "pass": ratio >= 0.9})
        differences = [right - left for left, right in zip(z0_hv, zcal_hv)]
        bootstrap_rows.append({"metric": "archive_hypervolume", "circuit": circuit, "stratum": None,
                               **_paired_bootstrap(differences, seed=91 + len(bootstrap_rows)),
                               **_wins(differences, lower_is_better=False)})
    for (variant, circuit, seed), run in sorted(runs.items()):
        for budget in EXPECTED_MILESTONES:
            milestone = run["milestones"][budget]
            optimizer_health = milestone["optimizer_health"]
            for stratum in EXPECTED_STRATA:
                gate = health_gates(
                    validation=milestone[stratum],
                    gradient_p99_median_ratio=float(optimizer_health["policy_gradient_p99_median_ratio"]),
                    clipping_enabled=False, clipping_rate=0.0, thresholds=thresholds,
                )
                health_rows.append({"variant": variant, "circuit": circuit, "seed": seed,
                                    "trajectory_budget": budget, "stratum": stratum,
                                    "pass": gate["pass"], "failed": gate["failed"]})
                if budget in (200, 800) and not gate["pass"]:
                    all_health_at_200_800[variant] = False
    oscillation_rows = [{"variant": variant, "circuit": circuit, "seed": seed, **run["oscillation"]}
                        for (variant, circuit, seed), run in sorted(runs.items())]
    improvement_pass = all(row["pass"] for row in improvement_rows)
    nonharm_pass = all(row["pass"] for row in nonharm_rows)
    oscillation_pass = not any(row["persistent"] for row in oscillation_rows)
    numerical_pass = all(run["summary"].get("numerical_failure") is None for run in runs.values())
    all_cis_include_zero = all(row["ci_low"] <= 0.0 <= row["ci_high"] for row in bootstrap_rows)
    simplicity_tie = all(all_health_at_200_800.values()) and all_cis_include_zero
    if not numerical_pass or not oscillation_pass or not improvement_pass or not nonharm_pass:
        decision, selected = "reject_calibrated_initialization", "z0"
    elif simplicity_tie:
        decision, selected = "prefer_simpler_z0", "z0"
    else:
        decision, selected = "support_calibrated_initialization", "zcal"
    return {
        "decision": decision, "selected_variant": selected,
        "calibration_hypothesis_supported": selected == "zcal",
        "checks": {"numerical_pass": numerical_pass, "oscillation_pass": oscillation_pass,
                   "improvement_200_pass": improvement_pass, "nonharm_800_pass": nonharm_pass,
                   "both_variants_healthy_200_800": all(all_health_at_200_800.values()),
                   "all_bootstrap_intervals_include_zero": all_cis_include_zero,
                   "simplicity_tie_break": simplicity_tie},
        "improvement_200": improvement_rows, "nonharm_800": nonharm_rows,
        "bootstrap": bootstrap_rows, "oscillation": oscillation_rows, "health_gates": health_rows,
    }


def _seed_rows(runs: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (variant, circuit, seed), run in sorted(runs.items()):
        for budget, milestone in sorted(run["milestones"].items()):
            for stratum in EXPECTED_STRATA:
                residual = milestone[stratum]["residual"]
                rows.append({"variant": variant, "circuit": circuit, "seed": seed,
                             "trajectory_budget": budget, "stratum": stratum,
                             "log_z_target_gap": residual["log_z_target_gap"],
                             "absolute_log_z_target_gap": abs(float(residual["log_z_target_gap"])),
                             "bias_fraction": residual["bias_fraction"],
                             "centered_rms": residual["centered_rms"],
                             "learned_log_z": residual["learned_log_z"],
                             "archive_hypervolume": milestone["training_archive"]["hypervolume"],
                             "best_of_n_auc": milestone["search"]["log2_n_hypervolume_auc"]})
    return rows


def _best_of_n_rows(runs: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"variant": variant, "circuit": circuit, "seed": seed,
         "trajectory_budget": budget, **value}
        for (variant, circuit, seed), run in sorted(runs.items())
        for budget, milestone in sorted(run["milestones"].items())
        for value in milestone["search"]["budgets"]
    ]


def _plot_reports(output_dir: Path, seed_rows: Sequence[Mapping[str, Any]], runs: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for metric, ylabel, filename in (
        ("absolute_log_z_target_gap", "absolute logZ target gap", "target_gap.png"),
        ("bias_fraction", "bias fraction", "bias_fraction.png"),
        ("centered_rms", "centered residual RMS", "centered_rms.png"),
        ("archive_hypervolume", "archive hypervolume", "hypervolume.png"),
    ):
        for variant in EXPECTED_VARIANTS:
            for circuit in EXPECTED_CIRCUITS:
                selected = [row for row in seed_rows if row["variant"] == variant and row["circuit"] == circuit
                            and row["stratum"] == "fixed_uniform"]
                budgets = sorted({int(row["trajectory_budget"]) for row in selected})
                means = [float(np.mean([float(row[metric]) for row in selected if int(row["trajectory_budget"]) == budget]))
                         for budget in budgets]
                plt.plot(budgets, means, marker="o", label=f"{variant}:{circuit}")
        plt.xlabel("training trajectories")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = plot_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))
    for variant in EXPECTED_VARIANTS:
        for circuit in EXPECTED_CIRCUITS:
            for seed in EXPECTED_SEEDS:
                updates = runs[(variant, circuit, seed)]["updates"]
                plt.plot([row["optimizer_update"] for row in updates], [row["target_gap"] for row in updates],
                         alpha=0.65, label=f"{variant}:{circuit}:{seed}")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("optimizer update")
    plt.ylabel("training-batch target gap")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    path = plot_dir / "target_gap_updates.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path))
    for metric, ylabel, filename in (
        ("policy_gradient_norm", "policy gradient norm", "policy_gradient_norm.png"),
        ("log_z_gradient_norm", "logZ gradient norm", "logz_gradient_norm.png"),
        ("policy_parameter_update_norm", "policy update norm", "policy_update_norm.png"),
        ("absolute_log_z_update", "absolute logZ update", "logz_update_norm.png"),
    ):
        for variant in EXPECTED_VARIANTS:
            selected = [row for (run_variant, _, _), run in runs.items() if run_variant == variant for row in run["updates"]]
            means = [float(np.mean([float(row[metric]) for row in selected if int(row["optimizer_update"]) == update]))
                     for update in range(1, 201)]
            plt.plot(range(1, 201), means, label=variant)
        plt.xlabel("optimizer update")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = plot_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))
    return paths


def _markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Experiment 3: calibrated logZ initialization",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        f"Selected variant: **{summary['selected_variant']}**",
        "",
        "## Deterministic checks",
        "",
        *[f"- {key}: `{value}`" for key, value in summary["checks"].items()],
        "",
        "See the CSV tables and plots for circuit-, stratum-, seed-, and milestone-level evidence.",
    ])


def report_experiment(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(args.expected_variants))
    circuits = list(dict.fromkeys(args.expected_circuits))
    seeds = list(dict.fromkeys(int(seed) for seed in args.expected_seeds))
    try:
        runs = {(variant, circuit, seed): _validate_run(
                    args.runs_root.resolve() / variant / circuit / f"seed_{seed}",
                    variant=variant, circuit=circuit, seed=seed)
                for variant in variants for circuit in circuits for seed in seeds}
        paired_fingerprints = {run["resolved"]["paired_configuration_fingerprint"] for run in runs.values()}
        sources = {run["metadata"]["source_tree_sha256"] for run in runs.values()}
        if len(paired_fingerprints) != 1 or len(sources) != 1:
            raise ArtifactValidationError("paired configuration or source-tree checksums differ")
        for circuit in circuits:
            for seed in seeds:
                left, right = runs[("z0", circuit, seed)], runs[("zcal", circuit, seed)]
                for key in ("pre_calibration_parameter_checksum", "fixed_sequence_checksum", "calibration_sequence_checksum"):
                    if left["summary"][key] != right["summary"][key]:
                        raise ArtifactValidationError(f"paired {key} mismatch for {circuit} seed {seed}")
        decision = classify_calibration(runs)
        decision.update({"schema_version": 2, "complete": True, "run_count": len(runs),
                         "variants": variants, "circuits": circuits, "seeds": seeds,
                         "milestones": list(EXPECTED_MILESTONES),
                         "paired_configuration_fingerprint": next(iter(paired_fingerprints)),
                         "source_tree_sha256": next(iter(sources))})
        seed_rows = _seed_rows(runs)
        best_rows = _best_of_n_rows(runs)
        update_rows = [
            {"variant": variant, "circuit": circuit, "seed": seed,
             "optimizer_update": row["optimizer_update"], "trajectory_budget": row["trajectory_budget"],
             "target_gap": row["target_gap"], "loss": row["loss"],
             "policy_gradient_norm": row["policy_gradient_norm"],
             "log_z_gradient_norm": row["log_z_gradient_norm"],
             "policy_parameter_update_norm": row["policy_parameter_update_norm"],
             "absolute_log_z_update": row["absolute_log_z_update"]}
            for (variant, circuit, seed), run in sorted(runs.items()) for row in run["updates"]
        ]
        decision["plots"] = _plot_reports(output_dir, seed_rows, runs)
        ledger = [{"experiment": 3, "variant": variant,
                   "hypothesis": "calibrated logZ reaches health earlier without 800-trajectory harm",
                   "circuits": ",".join(circuits), "seeds": ",".join(map(str, seeds)), "maximum_B": 800,
                   "status": "selected" if decision["selected_variant"] == variant else "rejected",
                   "rejection_reason": "" if decision["selected_variant"] == variant else decision["decision"],
                   "artifact_path": str(args.runs_root.resolve() / variant)} for variant in variants]
        _write_json(output_dir / "decision_summary.json", decision)
        (output_dir / "decision_report.md").write_text(_markdown(decision), encoding="utf-8")
        _write_csv(output_dir / "seed_metrics.csv", seed_rows)
        _write_csv(output_dir / "best_of_n.csv", best_rows)
        _write_csv(output_dir / "optimizer_updates.csv", update_rows)
        _write_csv(output_dir / "improvement_200.csv", decision["improvement_200"])
        _write_csv(output_dir / "nonharm_800.csv", decision["nonharm_800"])
        _write_csv(output_dir / "bootstrap.csv", decision["bootstrap"])
        _write_csv(output_dir / "oscillation.csv", decision["oscillation"])
        _write_csv(output_dir / "health_gates.csv", decision["health_gates"])
        _write_csv(output_dir / "phase_ledger.csv", ledger)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["decision"] != "reject_calibrated_initialization" else 3
    except IncompleteRunSetError as exc:
        _write_json(output_dir / "decision_summary.json", {"complete": False, "failure_type": "incomplete_run_set", "message": str(exc)})
        print(str(exc))
        return 2
    except Exception as exc:
        _write_json(output_dir / "decision_summary.json", {"complete": False, "failure_type": "artifact_or_execution_failure", "message": str(exc)})
        print(str(exc))
        return 1


def add_report_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("report", help="validate and compare all Experiment 3 runs")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-variants", nargs="+", default=list(EXPECTED_VARIANTS))
    parser.add_argument("--expected-circuits", nargs="+", default=list(EXPECTED_CIRCUITS))
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=list(EXPECTED_SEEDS))
    parser.set_defaults(handler=report_experiment)


__all__ = ["ArtifactValidationError", "IncompleteRunSetError", "add_report_parser",
           "classify_calibration", "oscillation_diagnostics", "report_experiment"]
