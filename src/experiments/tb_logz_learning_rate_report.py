"""Validation, staged selection, and reports for TB Experiment 4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.algorithms.gflownet_tb.diagnostics import HealthThresholds, health_gates
from src.experiments.tb_logz_calibration_report import (
    ArtifactValidationError,
    IncompleteRunSetError,
    _validate_run,
    _write_csv,
    _write_json,
)


EXPECTED_INITIALIZATIONS = ("z0", "zcal")
EXPECTED_RATES = (0.003, 0.01, 0.03, 0.1)
SCREEN_SEEDS = (0, 1)
CONFIRMATION_SEEDS = (0, 1, 2)
EXPECTED_STRATA = ("fixed_uniform", "fresh_on_policy")
EXPECTED_MILESTONES = (200, 400, 800)
FINAL_BUDGET = 800
SCHEMA_VERSION = 1


def rate_slug(rate: float) -> str:
    values = {0.003: "rate_0p003", 0.01: "rate_0p01", 0.03: "rate_0p03", 0.1: "rate_0p1"}
    try:
        return values[float(rate)]
    except KeyError as exc:
        raise ValueError(f"unsupported Experiment 4 rate: {rate}") from exc


def _run_dir(root: Path, initialization: str, rate: float, circuit: str, seed: int) -> Path:
    return root / initialization / rate_slug(rate) / circuit / f"seed_{seed}"


def _load_experiment4_run(
    root: Path, *, initialization: str, rate: float, circuit: str, seed: int,
) -> dict[str, Any]:
    return _validate_run(
        _run_dir(root, initialization, rate, circuit, seed),
        variant=initialization,
        circuit=circuit,
        seed=seed,
        expected_experiment="log_z_learning_rate",
        schema_version=3,
        log_z_learning_rate=rate,
    )


def _load_experiment3_control(
    root: Path, *, initialization: str, circuit: str, seed: int,
) -> dict[str, Any]:
    return _validate_run(
        root / initialization / circuit / f"seed_{seed}",
        variant=initialization,
        circuit=circuit,
        seed=seed,
    )


def _gate_rows(
    runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = HealthThresholds()
    for (initialization, rate, circuit, seed), run in sorted(runs.items()):
        for budget in EXPECTED_MILESTONES:
            milestone = run["milestones"][budget]
            gradient_ratio = float(milestone["optimizer_health"]["policy_gradient_p99_median_ratio"])
            for stratum in EXPECTED_STRATA:
                gate = health_gates(
                    validation=milestone[stratum],
                    gradient_p99_median_ratio=gradient_ratio,
                    clipping_enabled=False,
                    clipping_rate=0.0,
                    thresholds=thresholds,
                )
                rows.append({
                    "initialization": initialization,
                    "log_z_learning_rate": rate,
                    "circuit": circuit,
                    "seed": seed,
                    "trajectory_budget": budget,
                    "stratum": stratum,
                    "decisive": budget == FINAL_BUDGET,
                    "pass": gate["pass"],
                    "failed": gate["failed"],
                })
    return rows


def _cell_values(
    runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]],
    initialization: str,
    rate: float,
    circuit: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    gaps: list[float] = []
    biases: list[float] = []
    per_seed_gap: list[float] = []
    per_seed_bias: list[float] = []
    hypervolumes: list[float] = []
    fixed_biases: list[float] = []
    for seed in seeds:
        milestone = runs[(initialization, rate, circuit, seed)]["milestones"][FINAL_BUDGET]
        seed_gaps = [abs(float(milestone[stratum]["residual"]["log_z_target_gap"])) for stratum in EXPECTED_STRATA]
        seed_biases = [float(milestone[stratum]["residual"]["bias_fraction"]) for stratum in EXPECTED_STRATA]
        gaps.extend(seed_gaps)
        biases.extend(seed_biases)
        per_seed_gap.append(float(np.mean(seed_gaps)))
        per_seed_bias.append(float(np.mean(seed_biases)))
        fixed_biases.append(float(milestone["fixed_uniform"]["residual"]["bias_fraction"]))
        hypervolumes.append(float(milestone["training_archive"]["hypervolume"]))
    return {
        "mean_absolute_target_gap": float(np.mean(gaps)),
        "mean_bias_fraction": float(np.mean(biases)),
        "median_fixed_bias_fraction": float(np.median(fixed_biases)),
        "mean_archive_hypervolume": float(np.mean(hypervolumes)),
        "per_seed_gap": per_seed_gap,
        "per_seed_bias": per_seed_bias,
    }


def _paired_standard_error(left: Sequence[float], right: Sequence[float]) -> float:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if len(differences) < 2:
        return 0.0
    return float(np.std(differences, ddof=1) / math.sqrt(len(differences)))


def _validate_screen_pairing(
    runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    sources = {str(run["metadata"]["source_tree_sha256"]) for run in runs.values()}
    pairing = {str(run["resolved"].get("pairing_configuration_fingerprint")) for run in runs.values()}
    if len(sources) != 1 or len(pairing) != 1 or "None" in pairing:
        raise ArtifactValidationError("screen source tree or pairing configuration differs")
    for seed in SCREEN_SEEDS:
        seed_runs = [run for key, run in runs.items() if key[2:] == ("bc0", seed)]
        for key in ("pre_calibration_parameter_checksum", "fixed_sequence_checksum", "calibration_sequence_checksum"):
            if len({str(run["summary"][key]) for run in seed_runs}) != 1:
                raise ArtifactValidationError(f"screen pairing mismatch for {key}, seed {seed}")
        for initialization in EXPECTED_INITIALIZATIONS:
            initialized = [run for (variant, _, circuit, run_seed), run in runs.items()
                           if variant == initialization and circuit == "bc0" and run_seed == seed]
            if len({str(run["summary"]["post_initialization_parameter_checksum"]) for run in initialized}) != 1:
                raise ArtifactValidationError(
                    f"post-initialization pairing mismatch for {initialization}, seed {seed}"
                )
    return {"source_tree_sha256": next(iter(sources)), "pairing_configuration_fingerprint": next(iter(pairing))}


def classify_screen(
    runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    health_rows = _gate_rows(runs)
    cells: list[dict[str, Any]] = []
    values: dict[tuple[str, float], dict[str, Any]] = {}
    for initialization in EXPECTED_INITIALIZATIONS:
        control = _cell_values(runs, initialization, 0.01, "bc0", SCREEN_SEEDS)
        for rate in EXPECTED_RATES:
            current = _cell_values(runs, initialization, rate, "bc0", SCREEN_SEEDS)
            values[(initialization, rate)] = current
            final_gates = [row for row in health_rows if row["initialization"] == initialization
                           and row["log_z_learning_rate"] == rate and row["decisive"]]
            health_pass = bool(final_gates) and all(bool(row["pass"]) for row in final_gates)
            oscillations = [runs[(initialization, rate, "bc0", seed)]["oscillation"] for seed in SCREEN_SEEDS]
            oscillation_pass = not any(bool(row["persistent"]) for row in oscillations)
            denominator = max(float(control["median_fixed_bias_fraction"]), 1e-12)
            bias_ratio = float(current["median_fixed_bias_fraction"]) / denominator
            bias_control_pass = bias_ratio <= 1.1
            cells.append({
                "initialization": initialization,
                "log_z_learning_rate": rate,
                **{key: value for key, value in current.items() if not key.startswith("per_seed_")},
                "fixed_bias_ratio_to_0p01": bias_ratio,
                "health_pass": health_pass,
                "oscillation_pass": oscillation_pass,
                "bias_control_pass": bias_control_pass,
                "eligible": health_pass and oscillation_pass and bias_control_pass,
                "rejection_reasons": [name for name, passed in (
                    ("final_health", health_pass), ("persistent_oscillation", oscillation_pass),
                    ("fixed_bias_ratio", bias_control_pass),
                ) if not passed],
                "oscillation": oscillations,
                "simplicity_preferred": False,
            })

    retained: list[dict[str, Any]] = []
    simplicity: list[dict[str, Any]] = []
    for initialization in EXPECTED_INITIALIZATIONS:
        eligible = [row for row in cells if row["initialization"] == initialization and row["eligible"]]
        control_row = next(row for row in cells if row["initialization"] == initialization
                           and math.isclose(float(row["log_z_learning_rate"]), 0.01))
        preferred = False
        if eligible and control_row["eligible"]:
            best = min(eligible, key=lambda row: (
                row["mean_absolute_target_gap"], row["mean_bias_fraction"],
                -row["mean_archive_hypervolume"], row["log_z_learning_rate"],
            ))
            control_values = values[(initialization, 0.01)]
            best_values = values[(initialization, float(best["log_z_learning_rate"]))]
            gap_se = _paired_standard_error(control_values["per_seed_gap"], best_values["per_seed_gap"])
            bias_se = _paired_standard_error(control_values["per_seed_bias"], best_values["per_seed_bias"])
            gap_within = control_values["mean_absolute_target_gap"] <= best_values["mean_absolute_target_gap"] + gap_se
            bias_within = control_values["mean_bias_fraction"] <= best_values["mean_bias_fraction"] + bias_se
            preferred = gap_within and bias_within
            simplicity.append({
                "initialization": initialization,
                "best_rate": best["log_z_learning_rate"],
                "control_rate": 0.01,
                "gap_standard_error": gap_se,
                "bias_standard_error": bias_se,
                "gap_within_one_se": gap_within,
                "bias_within_one_se": bias_within,
                "prefer_0p01": preferred,
            })
        if preferred:
            control_row["simplicity_preferred"] = True
            retained.append(control_row)
        else:
            retained.extend(eligible)

    ranked = sorted(retained, key=lambda row: (
        row["mean_absolute_target_gap"], row["mean_bias_fraction"],
        -row["mean_archive_hypervolume"], row["initialization"], row["log_z_learning_rate"],
    ))
    candidates = []
    for rank, row in enumerate(ranked[:2], 1):
        candidates.append({
            "rank": rank,
            "initialization": row["initialization"],
            "log_z_learning_rate": row["log_z_learning_rate"],
            "rate_slug": rate_slug(float(row["log_z_learning_rate"])),
            "dalu_source": "experiment3" if math.isclose(float(row["log_z_learning_rate"]), 0.01) else "experiment4",
        })
    return {
        "decision": "continue_to_dalu" if candidates else "reject_no_healthy_screen_candidate",
        "candidates": candidates,
        "cells": cells,
        "simplicity": simplicity,
        "health_gates": health_rows,
    }


def _seed_rows(runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (initialization, rate, circuit, seed), run in sorted(runs.items()):
        for budget, milestone in sorted(run["milestones"].items()):
            for stratum in EXPECTED_STRATA:
                residual = milestone[stratum]["residual"]
                rows.append({
                    "initialization": initialization, "log_z_learning_rate": rate,
                    "circuit": circuit, "seed": seed, "trajectory_budget": budget,
                    "stratum": stratum, "log_z_target_gap": residual["log_z_target_gap"],
                    "absolute_log_z_target_gap": abs(float(residual["log_z_target_gap"])),
                    "bias_fraction": residual["bias_fraction"], "centered_rms": residual["centered_rms"],
                    "learned_log_z": residual["learned_log_z"],
                    "archive_hypervolume": milestone["training_archive"]["hypervolume"],
                    "best_of_n_auc": milestone["search"]["log2_n_hypervolume_auc"],
                })
    return rows


def _update_rows(runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "initialization": initialization, "log_z_learning_rate": rate,
        "circuit": circuit, "seed": seed, "optimizer_update": row["optimizer_update"],
        "trajectory_budget": row["trajectory_budget"], "target_gap": row["target_gap"],
        "loss": row["loss"], "policy_gradient_norm": row["policy_gradient_norm"],
        "log_z_gradient_norm": row["log_z_gradient_norm"],
        "policy_parameter_update_norm": row["policy_parameter_update_norm"],
        "absolute_log_z_update": row["absolute_log_z_update"],
    } for (initialization, rate, circuit, seed), run in sorted(runs.items()) for row in run["updates"]]


def _best_of_n_rows(runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "initialization": initialization, "log_z_learning_rate": rate,
        "circuit": circuit, "seed": seed, "trajectory_budget": budget, **row,
    } for (initialization, rate, circuit, seed), run in sorted(runs.items())
      for budget, milestone in sorted(run["milestones"].items())
      for row in milestone["search"]["budgets"]]


def _oscillation_rows(runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"initialization": initialization, "log_z_learning_rate": rate,
             "circuit": circuit, "seed": seed, **run["oscillation"]}
            for (initialization, rate, circuit, seed), run in sorted(runs.items())]


def _plot_metrics(output_dir: Path, seed_rows: Sequence[Mapping[str, Any]], *, prefix: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for metric, label in (("absolute_log_z_target_gap", "absolute logZ target gap"),
                          ("bias_fraction", "bias fraction"),
                          ("archive_hypervolume", "archive hypervolume")):
        for initialization in EXPECTED_INITIALIZATIONS:
            for rate in EXPECTED_RATES:
                selected = [row for row in seed_rows if row["initialization"] == initialization
                            and math.isclose(float(row["log_z_learning_rate"]), rate)
                            and row["stratum"] == "fixed_uniform"]
                if not selected:
                    continue
                budgets = sorted({int(row["trajectory_budget"]) for row in selected})
                means = [float(np.mean([float(row[metric]) for row in selected
                                       if int(row["trajectory_budget"]) == budget])) for budget in budgets]
                plt.plot(budgets, means, marker="o", label=f"{initialization}:{rate:g}")
        plt.xlabel("training trajectories")
        plt.ylabel(label)
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = plot_dir / f"{prefix}_{metric}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        outputs.append(str(path))
    return outputs


def _write_common_tables(
    output_dir: Path,
    runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]],
    decision: Mapping[str, Any],
    *,
    prefix: str,
) -> list[str]:
    seed_rows = _seed_rows(runs)
    _write_csv(output_dir / "seed_metrics.csv", seed_rows)
    _write_csv(output_dir / "best_of_n.csv", _best_of_n_rows(runs))
    _write_csv(output_dir / "optimizer_updates.csv", _update_rows(runs))
    _write_csv(output_dir / "health_gates.csv", decision["health_gates"])
    _write_csv(output_dir / "oscillation.csv", _oscillation_rows(runs))
    _write_csv(output_dir / "candidate_ranking.csv", decision["cells"])
    return _plot_metrics(output_dir, seed_rows, prefix=prefix)


def screen_report(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    try:
        runs = {(initialization, rate, "bc0", seed): _load_experiment4_run(
                    args.runs_root.resolve(), initialization=initialization, rate=rate,
                    circuit="bc0", seed=seed)
                for initialization in EXPECTED_INITIALIZATIONS for rate in EXPECTED_RATES
                for seed in SCREEN_SEEDS}
        provenance = _validate_screen_pairing(runs)
        decision = classify_screen(runs)
        decision.update({"schema_version": SCHEMA_VERSION, "complete": True,
                         "phase": "bc0_screen", "run_count": len(runs), **provenance})
        decision["plots"] = _write_common_tables(output_dir, runs, decision, prefix="screen")
        manifest = {"schema_version": SCHEMA_VERSION, "complete": True,
                    "screen_decision": decision["decision"], "candidates": decision["candidates"],
                    "screen_runs_root": str(args.runs_root.resolve()), **provenance}
        _write_json(output_dir / "decision_summary.json", decision)
        _write_json(output_dir / "confirmation_candidates.json", manifest)
        _write_csv(output_dir / "simplicity.csv", decision["simplicity"])
        _write_csv(output_dir / "phase_ledger.csv", [{
            "experiment": 4, "phase": "bc0_screen", "status": decision["decision"],
            "run_count": len(runs), "candidate_count": len(decision["candidates"]),
            "artifact_path": str(output_dir),
        }])
        (output_dir / "decision_report.md").write_text(
            "# Experiment 4 bc0 screen\n\n"
            f"Decision: **{decision['decision']}**\n\n"
            f"Confirmation candidates: `{json.dumps(decision['candidates'], sort_keys=True)}`\n",
            encoding="utf-8",
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["candidates"] else 3
    except IncompleteRunSetError as exc:
        _write_json(output_dir / "decision_summary.json", {"complete": False, "failure_type": "incomplete_run_set", "message": str(exc)})
        return 2
    except Exception as exc:
        _write_json(output_dir / "decision_summary.json", {"complete": False, "failure_type": "artifact_or_execution_failure", "message": str(exc)})
        return 1


_COMPATIBILITY_KEYS = (
    "config_name", "num_steps", "available_actions", "trajectories_per_update",
    "policy_learning_rate", "reward_alpha", "reward_eps", "reward_improvement_clip",
    "exploration_epsilon_enabled", "exploration_epsilon_start", "exploration_epsilon_end",
    "exploration_warmup_updates", "exploration_decay_updates", "schedule_trajectories",
    "configured_optimizer_updates", "calibration_trajectories", "calibration_epsilon",
    "calibration_batch_order", "gradient_clipping", "fixed_validation_trajectories",
    "fresh_validation_trajectories", "search_trajectories", "search_budgets",
)


def _validate_scientific_compatibility(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    left, right = candidate["resolved"], control["resolved"]
    mismatches = [key for key in _COMPATIBILITY_KEYS if left.get(key) != right.get(key)]
    if mismatches:
        raise ArtifactValidationError(f"Experiment 3 control is scientifically incompatible: {mismatches}")


def _validate_external_control(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    _validate_scientific_compatibility(candidate, control)
    for key in ("pre_calibration_parameter_checksum", "fixed_sequence_checksum", "calibration_sequence_checksum"):
        if candidate["summary"].get(key) != control["summary"].get(key):
            raise ArtifactValidationError(f"Experiment 3 control pairing mismatch: {key}")


def classify_confirmation(
    screen_runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]],
    dalu_runs: Mapping[tuple[str, float, str, int], Mapping[str, Any]],
    controls: Mapping[tuple[str, str, int], Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    health_rows = _gate_rows(dalu_runs)
    cells: list[dict[str, Any]] = []
    for candidate in candidates:
        initialization = str(candidate["initialization"])
        rate = float(candidate["log_z_learning_rate"])
        current = _cell_values(dalu_runs, initialization, rate, "dalu", CONFIRMATION_SEEDS)
        control_biases = [float(controls[(initialization, "dalu", seed)]["milestones"][800]
                                ["fixed_uniform"]["residual"]["bias_fraction"])
                          for seed in CONFIRMATION_SEEDS]
        bias_ratio = current["median_fixed_bias_fraction"] / max(float(np.median(control_biases)), 1e-12)
        gates = [row for row in health_rows if row["initialization"] == initialization
                 and math.isclose(float(row["log_z_learning_rate"]), rate) and row["decisive"]]
        health_pass = bool(gates) and all(bool(row["pass"]) for row in gates)
        oscillation_pass = not any(dalu_runs[(initialization, rate, "dalu", seed)]["oscillation"]["persistent"]
                                   for seed in CONFIRMATION_SEEDS)
        bias_pass = bias_ratio <= 1.1
        bc0 = _cell_values(screen_runs, initialization, rate, "bc0", SCREEN_SEEDS)
        cells.append({
            "initialization": initialization, "log_z_learning_rate": rate,
            "mean_absolute_target_gap": float(np.mean([bc0["mean_absolute_target_gap"], current["mean_absolute_target_gap"]])),
            "mean_bias_fraction": float(np.mean([bc0["mean_bias_fraction"], current["mean_bias_fraction"]])),
            "mean_archive_hypervolume": float(np.mean([bc0["mean_archive_hypervolume"], current["mean_archive_hypervolume"]])),
            "dalu_fixed_bias_ratio_to_experiment3_0p01": bias_ratio,
            "health_pass": health_pass, "oscillation_pass": oscillation_pass,
            "bias_control_pass": bias_pass,
            "eligible": health_pass and oscillation_pass and bias_pass,
            "rejection_reasons": [name for name, passed in (
                ("final_health", health_pass), ("persistent_oscillation", oscillation_pass),
                ("fixed_bias_ratio", bias_pass),
            ) if not passed],
        })
    eligible = sorted((row for row in cells if row["eligible"]), key=lambda row: (
        row["mean_absolute_target_gap"], row["mean_bias_fraction"],
        -row["mean_archive_hypervolume"], row["initialization"], row["log_z_learning_rate"],
    ))
    selected = eligible[0] if eligible else None
    return {
        "decision": "select_log_z_setting" if selected else "reject_no_healthy_confirmed_candidate",
        "selected": None if selected is None else {
            "initialization": selected["initialization"],
            "log_z_learning_rate": selected["log_z_learning_rate"],
        },
        "cells": cells,
        "health_gates": health_rows,
    }


def final_report(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    try:
        manifest = json.loads(args.candidates_manifest.read_text(encoding="utf-8"))
        candidates = manifest.get("candidates", [])
        if not manifest.get("complete") or not (1 <= len(candidates) <= 2):
            raise IncompleteRunSetError("confirmation manifest has no runnable candidate set")
        screen_root = args.screen_runs_root.resolve()
        exp3_root = args.experiment3_runs_root.resolve()
        confirmation_root = args.confirmation_runs_root.resolve()
        screen_runs: dict[tuple[str, float, str, int], dict[str, Any]] = {}
        dalu_runs: dict[tuple[str, float, str, int], dict[str, Any]] = {}
        controls: dict[tuple[str, str, int], dict[str, Any]] = {}
        for candidate in candidates:
            initialization, rate = str(candidate["initialization"]), float(candidate["log_z_learning_rate"])
            for seed in SCREEN_SEEDS:
                screen_runs[(initialization, rate, "bc0", seed)] = _load_experiment4_run(
                    screen_root, initialization=initialization, rate=rate, circuit="bc0", seed=seed)
            for seed in CONFIRMATION_SEEDS:
                control = _load_experiment3_control(exp3_root, initialization=initialization, circuit="dalu", seed=seed)
                controls[(initialization, "dalu", seed)] = control
                screen_reference = screen_runs[(initialization, rate, "bc0", seed % len(SCREEN_SEEDS))]
                _validate_scientific_compatibility(screen_reference, control)
                if math.isclose(rate, 0.01):
                    run = control
                else:
                    run = _load_experiment4_run(
                        confirmation_root, initialization=initialization, rate=rate, circuit="dalu", seed=seed)
                    _validate_external_control(run, control)
                dalu_runs[(initialization, rate, "dalu", seed)] = run
        decision = classify_confirmation(screen_runs, dalu_runs, controls, candidates)
        source_pairs = sorted({(run["metadata"]["source_tree_sha256"], controls[(initialization, "dalu", seed)]
                               ["metadata"]["source_tree_sha256"])
                              for (initialization, _, _, seed), run in dalu_runs.items()})
        decision.update({
            "schema_version": SCHEMA_VERSION, "complete": True, "phase": "dalu_confirmation",
            "candidates": candidates, "experiment3_control_provenance_exception": True,
            "source_tree_pairs": source_pairs,
        })
        combined = {**screen_runs, **dalu_runs}
        decision["plots"] = _write_common_tables(output_dir, combined, decision, prefix="final")
        _write_json(output_dir / "decision_summary.json", decision)
        _write_csv(output_dir / "phase_ledger.csv", [{
            "experiment": 4, "phase": "dalu_confirmation", "status": decision["decision"],
            "selected": decision["selected"], "artifact_path": str(output_dir),
        }])
        selected_text = "none" if decision["selected"] is None else json.dumps(decision["selected"], sort_keys=True)
        (output_dir / "decision_report.md").write_text(
            "# Experiment 4 final report\n\n"
            f"Decision: **{decision['decision']}**\n\nSelected setting: `{selected_text}`\n\n"
            "Experiment 3 `dalu` rate-0.01 artifacts were accepted only through the documented "
            "scientific-compatibility and pairing-checksum adapter.\n",
            encoding="utf-8",
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["selected"] is not None else 3
    except FileNotFoundError as exc:
        _write_json(output_dir / "decision_summary.json", {"complete": False, "failure_type": "incomplete_run_set", "message": str(exc)})
        return 2
    except IncompleteRunSetError as exc:
        _write_json(output_dir / "decision_summary.json", {"complete": False, "failure_type": "incomplete_run_set", "message": str(exc)})
        return 2
    except Exception as exc:
        _write_json(output_dir / "decision_summary.json", {"complete": False, "failure_type": "artifact_or_execution_failure", "message": str(exc)})
        return 1


def add_report_parsers(subparsers: argparse._SubParsersAction) -> None:
    screen = subparsers.add_parser("screen-report", help="validate and rank the bc0 screen")
    screen.add_argument("--runs-root", type=Path, required=True)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.set_defaults(handler=screen_report)

    final = subparsers.add_parser("final-report", help="validate dalu confirmations and select a setting")
    final.add_argument("--screen-runs-root", type=Path, required=True)
    final.add_argument("--confirmation-runs-root", type=Path, required=True)
    final.add_argument("--experiment3-runs-root", type=Path, required=True)
    final.add_argument("--candidates-manifest", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.set_defaults(handler=final_report)


__all__ = [
    "add_report_parsers", "classify_confirmation", "classify_screen", "final_report",
    "rate_slug", "screen_report",
]
