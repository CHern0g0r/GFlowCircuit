"""Selection rules and reports for baseline training hyperparameter tuning."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from src.baseline_tuning_protocol import BaselineSetting, BaselineTuningProtocol
from src.exploration_analysis import (
    bootstrap_mean_ci,
    collect_task_records,
    discovery_curve_rows,
    per_seed_metrics,
    summarize_settings,
)


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(fmean(data)) if data else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def enriched_seed_metrics(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows, grouped_points = per_seed_metrics(records)
    metadata = {
        (
            str(record["algorithm"]),
            str(record["setting"]["setting_id"]),
            str(record["circuit"]),
            int(record["seed"]),
        ): record
        for record in records
    }
    for row in seed_rows:
        record = metadata[(row["algorithm"], row["setting_id"], row["circuit"], int(row["seed"]))]
        row["profile_id"] = str(record["profile_id"])
        row["role"] = str(record["role"])
        row["training_trajectories"] = int(record["training_trajectories"])
        row["training_wall_time_seconds"] = float(
            record["execution_status"].get("train_wall_time_seconds", 0.0)
        )
        row["sampling_wall_time_seconds"] = float(
            record["execution_status"].get("sample_wall_time_seconds", 0.0)
        )
        row["gpu_hours"] = float(row["wall_time_seconds"]) / 3600.0
    summary, pooled = summarize_settings(seed_rows, grouped_points)
    summary_metadata = {
        (str(row["algorithm"]), str(row["setting_id"])): row
        for row in seed_rows
    }
    for row in summary:
        seed = summary_metadata[(str(row["algorithm"]), str(row["setting_id"]))]
        row["profile_id"] = seed["profile_id"]
        row["role"] = seed["role"]
        row["training_trajectories"] = seed["training_trajectories"]
    return seed_rows, summary, pooled


def _rank_candidates(
    summary_rows: list[dict[str, Any]],
    *,
    algorithm: str,
    circuits: list[str],
    tie_threshold: float,
) -> list[dict[str, Any]]:
    rows = [row for row in summary_rows if row["algorithm"] == algorithm]
    maxima = {
        circuit: max(float(row["mean_hypervolume"]) for row in rows if row["circuit"] == circuit)
        for circuit in circuits
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["profile_id"]), int(row["training_trajectories"]))].append(row)
    ranked: list[dict[str, Any]] = []
    for (profile, budget), candidates in grouped.items():
        by_circuit = {str(row["circuit"]): row for row in candidates}
        relative = []
        products = []
        for circuit in circuits:
            row = by_circuit[circuit]
            maximum = maxima[circuit]
            relative.append(1.0 if maximum == 0.0 else float(row["mean_hypervolume"]) / maximum)
            products.append(float(row["mean_product_improvement"]))
        ranked.append(
            {
                "profile_id": profile,
                "training_trajectories": budget,
                "selection_score": _mean(relative),
                "minimum_relative_score": min(relative),
                "mean_product_improvement": _mean(products),
            }
        )
    return _rank_scored_candidates(ranked, tie_threshold=tie_threshold)


def _rank_scored_candidates(
    rows: list[dict[str, Any]],
    *,
    tie_threshold: float,
) -> list[dict[str, Any]]:
    pending = list(rows)
    output: list[dict[str, Any]] = []
    while pending:
        best = max(float(row["selection_score"]) for row in pending)
        tied = [row for row in pending if best - float(row["selection_score"]) <= tie_threshold + 1e-12]
        tied.sort(
            key=lambda row: (
                -float(row["minimum_relative_score"]),
                -float(row["mean_product_improvement"]),
                int(row["training_trajectories"]),
                str(row["profile_id"]),
            )
        )
        winner = tied[0]
        output.append(winner)
        pending.remove(winner)
    return output


def select_screen(
    protocol: BaselineTuningProtocol,
    stage: str,
    settings: list[BaselineSetting],
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = protocol.stages[stage]
    algorithm = str(cfg["algorithm"])
    ranked = _rank_candidates(
        summary_rows,
        algorithm=algorithm,
        circuits=[str(value) for value in cfg["circuits"]],
        tie_threshold=float(protocol.data["common"]["practical_tie_threshold"]),
    )
    if len(ranked) < 2:
        raise ValueError(f"{algorithm} screen produced fewer than two profiles")
    selection = {
        "status": "screened",
        "selected_profile": ranked[0]["profile_id"],
        "runner_up_profile": ranked[1]["profile_id"],
        "top_profiles": [ranked[0]["profile_id"], ranked[1]["profile_id"]],
        "rankings": ranked,
        "settings": [setting.to_dict() for setting in settings],
    }
    factorial = cfg.get("factorial")
    interaction_profiles = [str(value) for value in cfg.get("interaction_profiles", [])]
    if factorial is not None:
        interaction_profiles = [str(value) for value in factorial["interaction_profiles"]]
    if interaction_profiles:
        interaction_in_top_two = [
            str(value) for value in selection["top_profiles"] if value in interaction_profiles
        ]
        selection.update(
            {
                "interaction_profiles": interaction_profiles,
                "interaction_in_top_two": interaction_in_top_two,
                "next_action": (
                    "run_new_budget_curve"
                    if interaction_in_top_two
                    else "continue_existing_confirmation_path"
                ),
            }
        )
    return {
        algorithm: {
            **selection,
        }
    }


def factorial_contrasts(
    protocol: BaselineTuningProtocol,
    stage: str,
    summary_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute paired factorial effects on circuit-normalized sampled HV."""
    cfg = protocol.stages[stage]
    factorial = cfg.get("factorial")
    if factorial is None:
        return []
    algorithm = str(cfg["algorithm"])
    factors = [str(value) for value in factorial["factors"]]
    cells = {
        str(profile): tuple(int(value) for value in raw_cell)
        for profile, raw_cell in factorial["cells"].items()
    }
    circuits = [str(value) for value in cfg["circuits"]]
    relevant_summaries = [
        row for row in summary_rows
        if str(row["algorithm"]) == algorithm and str(row["circuit"]) in circuits
    ]
    maxima = {
        circuit: max(
            float(row["mean_hypervolume"])
            for row in relevant_summaries
            if str(row["circuit"]) == circuit
        )
        for circuit in circuits
    }
    blocks: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in seed_rows:
        if str(row["algorithm"]) != algorithm or str(row["circuit"]) not in circuits:
            continue
        profile = str(row["profile_id"])
        if profile not in cells:
            continue
        circuit = str(row["circuit"])
        scale = maxima[circuit]
        normalized = 1.0 if scale == 0.0 else float(row["hypervolume"]) / scale
        blocks[(circuit, int(row["seed"]))][profile] = normalized
    expected_profiles = set(cells)
    for block, values in blocks.items():
        if set(values) != expected_profiles:
            raise ValueError(f"incomplete factorial block for {algorithm}/{block}")
    expected_blocks = len(circuits) * len(protocol.data["common"]["screen_seeds"])
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"factorial screen requires {expected_blocks} paired blocks, found {len(blocks)}"
        )

    rows: list[dict[str, Any]] = []
    contrast_index = 0
    for order in range(1, len(factors) + 1):
        for factor_indices in combinations(range(len(factors)), order):
            effects = []
            denominator = float(2 ** (len(factors) - order))
            for values in blocks.values():
                contrast = 0.0
                for profile, cell in cells.items():
                    sign = 1.0
                    for index in factor_indices:
                        sign *= 1.0 if cell[index] else -1.0
                    contrast += sign * values[profile]
                effects.append(contrast / denominator)
            low, high = bootstrap_mean_ci(
                effects,
                repetitions=int(protocol.data["common"]["bootstrap_repetitions"]),
                seed=int(protocol.data["common"]["bootstrap_seed"]) + contrast_index,
            )
            selected_factors = [factors[index] for index in factor_indices]
            rows.append(
                {
                    "contrast_id": "__x__".join(selected_factors),
                    "contrast_type": "factorial_effect",
                    "factors": "|".join(selected_factors),
                    "order": order,
                    "mean_effect": _mean(effects),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_blocks": len(effects),
                    "metric": "circuit_normalized_sampled_hypervolume",
                }
            )
            contrast_index += 1
    return rows


def paired_profile_contrasts(
    protocol: BaselineTuningProtocol,
    stage: str,
    summary_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate declared paired profile contrasts on circuit-normalized HV."""
    cfg = protocol.stages[stage]
    declared = cfg.get("paired_contrasts", [])
    if not declared:
        return []
    algorithm = str(cfg["algorithm"])
    circuits = [str(value) for value in cfg["circuits"]]
    relevant_summaries = [
        row
        for row in summary_rows
        if str(row["algorithm"]) == algorithm and str(row["circuit"]) in circuits
    ]
    maxima = {
        circuit: max(
            float(row["mean_hypervolume"])
            for row in relevant_summaries
            if str(row["circuit"]) == circuit
        )
        for circuit in circuits
    }
    profiles = {
        str(profile)
        for contrast in declared
        for profile in contrast["terms"]
    }
    blocks: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in seed_rows:
        if str(row["algorithm"]) != algorithm or str(row["circuit"]) not in circuits:
            continue
        profile = str(row["profile_id"])
        if profile not in profiles:
            continue
        circuit = str(row["circuit"])
        scale = maxima[circuit]
        normalized = 1.0 if scale == 0.0 else float(row["hypervolume"]) / scale
        blocks[(circuit, int(row["seed"]))][profile] = normalized
    expected_blocks = len(circuits) * len(protocol.data["common"]["screen_seeds"])
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"paired contrasts require {expected_blocks} paired blocks, found {len(blocks)}"
        )

    rows: list[dict[str, Any]] = []
    for contrast_index, declared_contrast in enumerate(declared):
        terms = {
            str(profile): float(weight)
            for profile, weight in declared_contrast["terms"].items()
        }
        effects = []
        for block, values in blocks.items():
            if not set(terms) <= set(values):
                raise ValueError(
                    f"incomplete paired contrast {declared_contrast['id']} for {algorithm}/{block}"
                )
            effects.append(sum(weight * values[profile] for profile, weight in terms.items()))
        low, high = bootstrap_mean_ci(
            effects,
            repetitions=int(protocol.data["common"]["bootstrap_repetitions"]),
            seed=int(protocol.data["common"]["bootstrap_seed"]) + 100 + contrast_index,
        )
        rows.append(
            {
                "contrast_id": str(declared_contrast["id"]),
                "contrast_type": str(declared_contrast.get("kind", "declared")),
                "terms": "|".join(
                    f"{profile}:{weight:g}" for profile, weight in terms.items()
                ),
                "mean_effect": _mean(effects),
                "ci95_low": low,
                "ci95_high": high,
                "paired_blocks": len(effects),
                "metric": "circuit_normalized_sampled_hypervolume",
            }
        )
    return rows


def _normalized_pair_differences(
    seed_rows: list[dict[str, Any]],
    *,
    algorithm: str,
    profile_id: str,
    current_budget: int,
    successor_budget: int,
    repetitions: int,
    bootstrap_seed: int,
) -> tuple[float, float, float]:
    relevant = [
        row for row in seed_rows
        if row["algorithm"] == algorithm
        and row["profile_id"] == profile_id
        and int(row["training_trajectories"]) in {current_budget, successor_budget}
    ]
    maxima: dict[str, float] = defaultdict(float)
    for row in relevant:
        maxima[str(row["circuit"])] = max(maxima[str(row["circuit"])], float(row["hypervolume"]))
    grouped: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in relevant:
        scale = maxima[str(row["circuit"])]
        normalized = 1.0 if scale == 0.0 else float(row["hypervolume"]) / scale
        grouped[(str(row["circuit"]), int(row["seed"]))][int(row["training_trajectories"])] = normalized
    differences = []
    for key, values in grouped.items():
        if set(values) != {current_budget, successor_budget}:
            raise ValueError(f"unpaired budget comparison for {algorithm}/{profile_id}/{key}")
        differences.append(values[successor_budget] - values[current_budget])
    low, high = bootstrap_mean_ci(differences, repetitions=repetitions, seed=bootstrap_seed)
    return _mean(differences), low, high


def select_budget(
    protocol: BaselineTuningProtocol,
    stage: str,
    summary_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    *,
    top_profiles: list[str],
) -> dict[str, Any]:
    cfg = protocol.stages[stage]
    algorithm = str(cfg["algorithm"])
    circuits = [str(value) for value in cfg["circuits"]]
    common = protocol.data["common"]
    ranked = _rank_candidates(
        summary_rows,
        algorithm=algorithm,
        circuits=circuits,
        tie_threshold=float(common["practical_tie_threshold"]),
    )
    budgets = sorted({int(row["training_trajectories"]) for row in ranked})
    by_key = {(str(row["profile_id"]), int(row["training_trajectories"])): row for row in ranked}
    global_best = max(float(row["selection_score"]) for row in ranked)
    plateau_rows: list[dict[str, Any]] = []
    for profile in top_profiles:
        for current, successor in zip(budgets, budgets[1:]):
            row = dict(by_key[(profile, current)])
            next_row = by_key[(profile, successor)]
            difference, ci_low, ci_high = _normalized_pair_differences(
                seed_rows,
                algorithm=algorithm,
                profile_id=profile,
                current_budget=current,
                successor_budget=successor,
                repetitions=int(common["bootstrap_repetitions"]),
                bootstrap_seed=int(common["bootstrap_seed"]),
            )
            relative_gain = (
                float(next_row["selection_score"]) - float(row["selection_score"])
            ) / max(abs(float(row["selection_score"])), 1e-12)
            row.update(
                {
                    "successor_budget": successor,
                    "successor_selection_score": next_row["selection_score"],
                    "successor_relative_gain": relative_gain,
                    "paired_mean_normalized_gain": difference,
                    "paired_ci95_low": ci_low,
                    "paired_ci95_high": ci_high,
                    "within_two_percent_of_global_best": float(row["selection_score"])
                    >= (1.0 - float(common["plateau_relative_tolerance"])) * global_best,
                    "successor_not_significantly_better": ci_low <= 0.0,
                }
            )
            row["eligible"] = bool(
                row["within_two_percent_of_global_best"]
                and row["successor_not_significantly_better"]
            )
            plateau_rows.append(row)
    eligible = [row for row in plateau_rows if row["eligible"]]
    if not eligible:
        status = (
            "extend_required"
            if max(budgets) < int(common["conditional_extension_trajectory"])
            else "budget_unresolved_at_6400"
        )
        return {
            algorithm: {
                "status": status,
                "top_profiles": top_profiles,
                "rankings": ranked,
                "plateau_checks": plateau_rows,
                "largest_tested_budget": max(budgets),
            }
        }
    eligible.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            int(row["training_trajectories"]),
            -float(row["minimum_relative_score"]),
            str(row["profile_id"]),
        )
    )
    winner = eligible[0]
    runner_up_candidates = [
        row
        for row in ranked
        if int(row["training_trajectories"]) == int(winner["training_trajectories"])
        and str(row["profile_id"]) != str(winner["profile_id"])
    ]
    if not runner_up_candidates:
        raise ValueError(f"{algorithm} budget curve produced no runner-up profile")
    runner_up = _rank_scored_candidates(
        runner_up_candidates,
        tie_threshold=float(common["practical_tie_threshold"]),
    )[0]["profile_id"]
    return {
        algorithm: {
            "status": "selected",
            "selected_profile": winner["profile_id"],
            "runner_up_profile": runner_up,
            "top_profiles": top_profiles,
            "selected_budget": int(winner["training_trajectories"]),
            "successor_budget": int(winner["successor_budget"]),
            "selection_score": winner["selection_score"],
            "rankings": ranked,
            "plateau_checks": plateau_rows,
        }
    }


def _role_pair_ci(
    seed_rows: list[dict[str, Any]],
    *,
    lhs_role: str,
    rhs_role: str,
    repetitions: int,
    bootstrap_seed: int,
) -> tuple[float, float, float]:
    relevant = [row for row in seed_rows if row["role"] in {lhs_role, rhs_role}]
    maxima: dict[str, float] = defaultdict(float)
    for row in relevant:
        maxima[str(row["circuit"])] = max(maxima[str(row["circuit"])], float(row["hypervolume"]))
    pairs: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in relevant:
        scale = maxima[str(row["circuit"])]
        value = 1.0 if scale == 0.0 else float(row["hypervolume"]) / scale
        pairs[(str(row["circuit"]), int(row["seed"]))][str(row["role"])] = value
    differences = []
    for key, values in pairs.items():
        if set(values) != {lhs_role, rhs_role}:
            raise ValueError(f"unpaired confirmation comparison: {key}")
        differences.append(values[lhs_role] - values[rhs_role])
    low, high = bootstrap_mean_ci(differences, repetitions=repetitions, seed=bootstrap_seed)
    return _mean(differences), low, high


def confirm_selection(
    protocol: BaselineTuningProtocol,
    stage: str,
    settings: list[BaselineSetting],
    summary_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    algorithm = str(protocol.stages[stage]["algorithm"])
    common = protocol.data["common"]
    by_role = {setting.role: setting for setting in settings}
    scores = _rank_candidates(
        summary_rows,
        algorithm=algorithm,
        circuits=[str(value) for value in protocol.stages[stage]["circuits"]],
        tie_threshold=float(common["practical_tie_threshold"]),
    )
    score_by_role = {}
    for row in scores:
        match = next(
            setting for setting in settings
            if setting.profile_id == row["profile_id"]
            and setting.training_trajectories == row["training_trajectories"]
        )
        score_by_role[match.role] = float(row["selection_score"])
    profile_diff, profile_low, profile_high = _role_pair_ci(
        seed_rows,
        lhs_role="selected",
        rhs_role="runner_up",
        repetitions=int(common["bootstrap_repetitions"]),
        bootstrap_seed=int(common["bootstrap_seed"]),
    )
    successor_minus_selected, budget_low, budget_high = _role_pair_ci(
        seed_rows,
        lhs_role="successor",
        rhs_role="selected",
        repetitions=int(common["bootstrap_repetitions"]),
        bootstrap_seed=int(common["bootstrap_seed"]) + 1,
    )
    selected_score = score_by_role["selected"]
    runner_score = score_by_role["runner_up"]
    successor_score = score_by_role["successor"]
    profile_gate = (
        selected_score + float(common["practical_tie_threshold"]) >= runner_score
        and profile_high >= 0.0
    )
    budget_gate = (
        selected_score >= (1.0 - float(common["plateau_relative_tolerance"]))
        * max(selected_score, successor_score)
        and budget_low <= 0.0
    )
    return {
        algorithm: {
            "status": "confirmed" if profile_gate and budget_gate else "confirmation_failed",
            "selected_profile": by_role["selected"].profile_id,
            "runner_up_profile": by_role["runner_up"].profile_id,
            "selected_budget": by_role["selected"].training_trajectories,
            "successor_budget": by_role["successor"].training_trajectories,
            "profile_gate": profile_gate,
            "budget_gate": budget_gate,
            "selected_score": selected_score,
            "runner_up_score": runner_score,
            "successor_score": successor_score,
            "selected_minus_runner_up": {
                "mean": profile_diff,
                "ci95_low": profile_low,
                "ci95_high": profile_high,
            },
            "successor_minus_selected": {
                "mean": successor_minus_selected,
                "ci95_low": budget_low,
                "ci95_high": budget_high,
            },
        }
    }


def analyze_stage(
    *,
    protocol: BaselineTuningProtocol,
    stage: str,
    artifact_root: Path,
    settings: list[BaselineSetting],
    record_roots: list[Path] | None = None,
    dependency_payloads: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record_roots = record_roots or [artifact_root]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in record_roots:
        manifest_path = root / "stage_manifest.json"
        if not manifest_path.is_file():
            continue
        for record in collect_task_records(root):
            key = str(record["reuse_key"])
            if key not in seen:
                seen.add(key)
                records.append(record)
    seed_rows, summary_rows, pooled_rows = enriched_seed_metrics(records) if records else ([], [], [])
    kind = str(protocol.stages[stage]["kind"])
    selections: dict[str, Any] = {}
    if kind == "screen":
        selections = select_screen(protocol, stage, settings, summary_rows)
    elif kind in {"budget", "extend"}:
        algorithm = str(protocol.stages[stage]["algorithm"])
        if kind == "budget":
            top_profiles = protocol.budget_profiles(
                stage,
                payloads=dependency_payloads or {},
            )
        else:
            budget_stage = str(protocol.stages[stage]["dependencies"][0])
            top_profiles = list(dependency_payloads[budget_stage]["algorithms"][algorithm]["top_profiles"])
        selections = select_budget(
            protocol,
            stage,
            summary_rows,
            seed_rows,
            top_profiles=[str(value) for value in top_profiles],
        )
    elif kind == "confirm":
        selections = confirm_selection(protocol, stage, settings, summary_rows, seed_rows)
    elif kind == "common":
        for setting in settings:
            selections[setting.algorithm] = {
                "status": "common_evaluated",
                "selected_profile": setting.profile_id,
                "common_budget": setting.training_trajectories,
            }
    payload: dict[str, Any] = {
        "stage": stage,
        "kind": kind,
        "protocol_hash": protocol.protocol_hash,
        "algorithms": selections,
    }
    factorial_rows = factorial_contrasts(protocol, stage, summary_rows, seed_rows)
    paired_rows = paired_profile_contrasts(protocol, stage, summary_rows, seed_rows)
    interaction_rows = [*factorial_rows, *paired_rows]
    if interaction_rows:
        algorithm = str(protocol.stages[stage]["algorithm"])
        selections[algorithm]["interaction_contrasts"] = interaction_rows
        payload["interaction_contrasts"] = interaction_rows
    _write_csv(artifact_root / "per_seed_metrics.csv", seed_rows)
    _write_csv(artifact_root / "setting_summary.csv", summary_rows)
    _write_csv(artifact_root / "pooled_front.csv", pooled_rows)
    _write_csv(artifact_root / "discovery_curves.csv", discovery_curve_rows(records))
    if interaction_rows:
        _write_csv(artifact_root / "interaction_contrasts.csv", interaction_rows)
    (artifact_root / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Baseline training tuning: {stage}",
        "",
        f"Protocol hash: `{protocol.protocol_hash}`",
        "",
        "Primary metric: mean per-training-seed hypervolume from exactly 50 paired post-training samples.",
        "Pooled hypervolume is descriptive only. Equal trajectories measure sample efficiency, not wall-clock speed.",
        "",
        "| Algorithm | Status | Profile | Budget |",
        "| --- | --- | --- | ---: |",
    ]
    for algorithm, result in sorted(selections.items()):
        lines.append(
            f"| {algorithm} | {result.get('status', '')} | "
            f"`{result.get('selected_profile', '')}` | "
            f"{result.get('selected_budget', result.get('common_budget', ''))} |"
        )
    lines.extend(
        [
            "",
            (
                "Review `setting_summary.csv`, `per_seed_metrics.csv`, "
                "`pooled_front.csv`, `interaction_contrasts.csv` when present, "
                "and `selection.json` before advancing."
            ),
            (
                "Reports include optimizer-update counts, wall time, and GPU-hours "
                "to separate sample efficiency from compute efficiency."
            ),
        ]
    )
    (artifact_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "analyze_stage",
    "confirm_selection",
    "enriched_seed_metrics",
    "factorial_contrasts",
    "paired_profile_contrasts",
    "select_budget",
    "select_screen",
]
