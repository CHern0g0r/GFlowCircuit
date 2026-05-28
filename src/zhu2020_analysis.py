from __future__ import annotations

import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


PAPER_TABLE2: dict[str, dict[str, float]] = {
    "i10": {
        "initial_size": 2675,
        "initial_depth": 50,
        "resyn2_1_size": 1829,
        "resyn2_1_depth": 32,
        "resyn2_2_size": 1804,
        "resyn2_2_depth": 32,
        "resyn2_inf_size": 1789,
        "resyn2_inf_depth": 32,
        "rl1_size": 1730.2,
        "rl1_depth": 40.3,
        "rl2_size": 1839.4,
        "rl2_depth": 31.9,
    },
    "c1355": {
        "initial_size": 504,
        "initial_depth": 25,
        "resyn2_1_size": 390,
        "resyn2_1_depth": 16,
        "resyn2_2_size": 390,
        "resyn2_2_depth": 16,
        "resyn2_inf_size": 390,
        "resyn2_inf_depth": 16,
        "rl1_size": 386.2,
        "rl1_depth": 17.6,
        "rl2_size": 390.0,
        "rl2_depth": 16.0,
    },
    "c7552": {
        "initial_size": 2093,
        "initial_depth": 29,
        "resyn2_1_size": 1469,
        "resyn2_1_depth": 26,
        "resyn2_2_size": 1416,
        "resyn2_2_depth": 26,
        "resyn2_inf_size": 1398,
        "resyn2_inf_depth": 26,
        "rl1_size": 1395.4,
        "rl1_depth": 27.4,
        "rl2_size": 1460.8,
        "rl2_depth": 22.1,
    },
    "c6288": {
        "initial_size": 2337,
        "initial_depth": 120,
        "resyn2_1_size": 1870,
        "resyn2_1_depth": 89,
        "resyn2_2_size": 1870,
        "resyn2_2_depth": 89,
        "resyn2_inf_size": 1870,
        "resyn2_inf_depth": 89,
        "rl1_size": 1870.0,
        "rl1_depth": 88.0,
        "rl2_size": 1882.0,
        "rl2_depth": 88.0,
    },
    "c5315": {
        "initial_size": 1780,
        "initial_depth": 37,
        "resyn2_1_size": 1306,
        "resyn2_1_depth": 28,
        "resyn2_2_size": 1295,
        "resyn2_2_depth": 26,
        "resyn2_inf_size": 1294,
        "resyn2_inf_depth": 26,
        "rl1_size": 1337.4,
        "rl1_depth": 27.2,
        "rl2_size": 1364.7,
        "rl2_depth": 25.4,
    },
    "dalu": {
        "initial_size": 1371,
        "initial_depth": 35,
        "resyn2_1_size": 1106,
        "resyn2_1_depth": 31,
        "resyn2_2_size": 1103,
        "resyn2_2_depth": 31,
        "resyn2_inf_size": 1103,
        "resyn2_inf_depth": 31,
        "rl1_size": 1039.8,
        "rl1_depth": 33.2,
        "rl2_size": 1095.6,
        "rl2_depth": 30.0,
    },
    "k2": {
        "initial_size": 1998,
        "initial_depth": 23,
        "resyn2_1_size": 1234,
        "resyn2_1_depth": 13,
        "resyn2_2_size": 1186,
        "resyn2_2_depth": 13,
        "resyn2_inf_size": 1145,
        "resyn2_inf_depth": 13,
        "rl1_size": 1128.4,
        "rl1_depth": 19.8,
        "rl2_size": 1187.5,
        "rl2_depth": 13.0,
    },
    "mainpla": {
        "initial_size": 5346,
        "initial_depth": 38,
        "resyn2_1_size": 3678,
        "resyn2_1_depth": 26,
        "resyn2_2_size": 3583,
        "resyn2_2_depth": 26,
        "resyn2_inf_size": 3504,
        "resyn2_inf_depth": 25,
        "rl1_size": 3438.4,
        "rl1_depth": 25.0,
        "rl2_size": 3504.0,
        "rl2_depth": 25.5,
    },
    "apex1": {
        "initial_size": 2665,
        "initial_depth": 27,
        "resyn2_1_size": 1999,
        "resyn2_1_depth": 17,
        "resyn2_2_size": 1966,
        "resyn2_2_depth": 17,
        "resyn2_inf_size": 1941,
        "resyn2_inf_depth": 17,
        "rl1_size": 1921.6,
        "rl1_depth": 19.2,
        "rl2_size": 2004.7,
        "rl2_depth": 17.0,
    },
    "bc0": {
        "initial_size": 1592,
        "initial_depth": 31,
        "resyn2_1_size": 933,
        "resyn2_1_depth": 17,
        "resyn2_2_size": 899,
        "resyn2_2_depth": 17,
        "resyn2_inf_size": 875,
        "resyn2_inf_depth": 17,
        "rl1_size": 819.4,
        "rl1_depth": 18.6,
        "rl2_size": 851.7,
        "rl2_depth": 17.5,
    },
}

EXPECTED_RUN_SETTINGS = {
    "algorithm": "reinforce",
    "reward.type": "zhu_size",
    "num_steps": 20,
    "episodes": 200,
    "gamma": 0.9,
    "policy_learning_rate": 8e-4,
    "value_learning_rate": 3e-3,
    "baseline": "zhu_resyn2",
    "paper_mode.per_circuit_mode": True,
    "paper_mode.num_runs": 10,
    "paper_mode.infer_rollouts": 10,
    "entropy_beta": 0.0,
    "clip_grad_norm_policy": None,
    "clip_grad_norm_value": None,
    "normalize_returns": False,
}

EXPECTED_ACTIONS = [0, 1, 2, 3, 4]
EXPECTED_RESYN2_SEQUENCE = [0, 1, 2, 0, 1, 3, 0, 4, 3, 0]


@dataclass(frozen=True)
class ReportSelection:
    circuit: str
    selected_path: Path
    selected_run_id: int
    selected_hydra_config: Path | None
    duplicate_paths: tuple[Path, ...]
    report: dict[str, Any]


def nested_get(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def discover_reports(output_dir: Path, outputs_dir: Path | None = None) -> list[ReportSelection]:
    grouped: dict[str, list[tuple[int, Path, dict[str, Any], Path | None]]] = {}
    for path in sorted(output_dir.glob("zhu2020_*_reproduce_*.json")):
        match = re.match(r"zhu2020_(?P<circuit>.+)_reproduce_(?P<run_id>\d+)\.json$", path.name)
        if not match:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        circuit = match.group("circuit")
        run_id = int(match.group("run_id"))
        hydra_path = find_hydra_config(report, outputs_dir) if outputs_dir is not None else None
        grouped.setdefault(circuit, []).append((run_id, path, report, hydra_path))

    selections: list[ReportSelection] = []
    for circuit, candidates in sorted(grouped.items()):
        candidates = sorted(candidates, key=lambda item: (item[3] is not None, item[0], item[1].name))
        run_id, path, report, hydra_path = candidates[-1]
        duplicates = tuple(item[1] for item in candidates if item[1] != path)
        selections.append(
            ReportSelection(
                circuit=circuit,
                selected_path=path,
                selected_run_id=run_id,
                selected_hydra_config=hydra_path,
                duplicate_paths=duplicates,
                report=report,
            )
        )
    return selections


def find_hydra_config(report: dict[str, Any], outputs_dir: Path | None) -> Path | None:
    if outputs_dir is None:
        return None
    run_name = str(nested_get(report, "hydra_config.run_name") or "")
    if not run_name:
        return None
    candidates = sorted(outputs_dir.glob(f"**/*{run_name}/.hydra/config.yaml"))
    return candidates[-1] if candidates else None


def final_per_circuit_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in report.get("runs", []):
        final_eval = run.get("final_eval", {})
        per_circuit = final_eval.get("per_circuit", [])
        if len(per_circuit) != 1:
            continue
        row = dict(per_circuit[0])
        row["run_idx"] = run.get("run_idx")
        row["seed"] = run.get("seed")
        row["mean_normalized_improvement_vs_resyn2_2"] = final_eval.get(
            "mean_normalized_improvement_vs_resyn2_2"
        )
        rows.append(row)
    return rows


def aggregate_report(selection: ReportSelection) -> dict[str, Any]:
    rows = final_per_circuit_rows(selection.report)
    if not rows:
        raise ValueError(f"No final per-circuit rows found in {selection.selected_path}")

    final_sizes = [float(row["final_size"]) for row in rows]
    final_depths = [float(row["final_depth"]) for row in rows]
    resyn2_2_sizes = [float(row["resyn2_2_size"]) for row in rows]
    resyn2_inf_sizes = [float(row["resyn2_inf_size"]) for row in rows]
    normalized = [float(row["normalized_improvement_vs_resyn2_2"]) for row in rows]
    initial_size = float(rows[0]["initial_size"])
    initial_depth = float(rows[0]["initial_depth"])

    paper = PAPER_TABLE2.get(selection.circuit)
    mean_size = fmean(final_sizes)
    mean_depth = fmean(final_depths)
    out = {
        "circuit": selection.circuit,
        "json_path": str(selection.selected_path),
        "hydra_config_path": str(selection.selected_hydra_config) if selection.selected_hydra_config else "",
        "duplicate_jsons": ";".join(str(path) for path in selection.duplicate_paths),
        "n_runs": len(rows),
        "initial_size": initial_size,
        "initial_depth": initial_depth,
        "resyn2_1_size": float(rows[0]["resyn2_1_size"]),
        "resyn2_2_size": float(rows[0]["resyn2_2_size"]),
        "resyn2_inf_size": float(rows[0]["resyn2_inf_size"]),
        "mean_final_size": mean_size,
        "std_final_size": pstdev(final_sizes) if len(final_sizes) > 1 else 0.0,
        "min_final_size": min(final_sizes),
        "max_final_size": max(final_sizes),
        "mean_final_depth": mean_depth,
        "std_final_depth": pstdev(final_depths) if len(final_depths) > 1 else 0.0,
        "min_final_depth": min(final_depths),
        "max_final_depth": max(final_depths),
        "win_rate_vs_resyn2_2": fmean(1.0 if s < b else 0.0 for s, b in zip(final_sizes, resyn2_2_sizes)),
        "win_rate_vs_resyn2_inf": fmean(1.0 if s < b else 0.0 for s, b in zip(final_sizes, resyn2_inf_sizes)),
        "mean_normalized_improvement_vs_resyn2_2": fmean(normalized),
    }
    if paper:
        out.update(
            {
                "paper_initial_size": paper["initial_size"],
                "paper_initial_depth": paper["initial_depth"],
                "paper_resyn2_1_size": paper["resyn2_1_size"],
                "paper_resyn2_2_size": paper["resyn2_2_size"],
                "paper_resyn2_inf_size": paper["resyn2_inf_size"],
                "paper_rl1_size": paper["rl1_size"],
                "paper_rl1_depth": paper["rl1_depth"],
                "delta_size_vs_paper_rl1": mean_size - paper["rl1_size"],
                "delta_depth_vs_paper_rl1": mean_depth - paper["rl1_depth"],
                "normalized_delta_size_vs_paper_rl1": (mean_size - paper["rl1_size"])
                / max(1.0, paper["initial_size"]),
                "baseline_size_status": baseline_status(out, paper),
            }
        )
    return out


def baseline_status(row: dict[str, Any], paper: dict[str, float]) -> str:
    keys = ("initial_size", "resyn2_1_size", "resyn2_2_size", "resyn2_inf_size")
    mismatches = []
    for key in keys:
        paper_key = f"paper_{key}" if key != "initial_size" else "paper_initial_size"
        if int(round(float(row[key]))) != int(round(float(paper[paper_key.replace("paper_", "")]))):
            mismatches.append(key)
    return "match" if not mismatches else "toolchain_or_benchmark_drift:" + ",".join(mismatches)


def audit_config(report: dict[str, Any]) -> list[dict[str, str]]:
    cfg = report.get("hydra_config", {})
    rows = []
    for key, expected in EXPECTED_RUN_SETTINGS.items():
        actual = nested_get(report, key) if key == "algorithm" else nested_get(cfg, key)
        ok = values_equal(actual, expected)
        rows.append(
            {
                "check": key,
                "expected": repr(expected),
                "actual": repr(actual),
                "status": "pass" if ok else "fail",
            }
        )
    actual_actions = nested_get(cfg, "available_actions")
    rows.append(
        {
            "check": "available_actions",
            "expected": repr(EXPECTED_ACTIONS),
            "actual": repr(actual_actions),
            "status": "pass" if actual_actions == EXPECTED_ACTIONS else "fail",
        }
    )
    encoder = nested_get(cfg, "encoder")
    rows.extend(audit_encoder_config(encoder if isinstance(encoder, dict) else {}))
    return rows


def audit_configs(selections: list[ReportSelection]) -> list[dict[str, str]]:
    checks: dict[str, Any] = dict(EXPECTED_RUN_SETTINGS)
    checks["available_actions"] = EXPECTED_ACTIONS
    checks["encoder.type"] = "hybrid"
    checks["encoder.graph.type"] = "zhu_gcn"
    checks["encoder.vector.source"] = "zhu10"
    checks["value.input"] = "zhu10"

    rows: list[dict[str, str]] = []
    for key, expected in checks.items():
        actuals = {}
        for selection in selections:
            report = selection.report
            cfg = report.get("hydra_config", {})
            actuals[selection.circuit] = nested_get(report, key) if key == "algorithm" else nested_get(cfg, key)
        rows.append(
            {
                "check": key,
                "expected": repr(expected),
                "actual": summarize_actuals(actuals),
                "status": "pass" if all(values_equal(actual, expected) for actual in actuals.values()) else "fail",
            }
        )
    return rows


def summarize_actuals(actuals: dict[str, Any]) -> str:
    values = {repr(value) for value in actuals.values()}
    if len(values) == 1:
        return f"all={next(iter(values))}"
    return "; ".join(f"{circuit}={actual!r}" for circuit, actual in sorted(actuals.items()))


def audit_encoder_config(encoder: dict[str, Any]) -> list[dict[str, str]]:
    checks = {
        "encoder.type": "hybrid",
        "encoder.graph.type": "zhu_gcn",
        "encoder.vector.source": "zhu10",
        "value.input": "zhu10",
    }
    # value.input is checked against the report config elsewhere if this helper
    # is used directly; keep the row in the protocol table label set.
    rows = []
    for key, expected in checks.items():
        if key == "value.input":
            continue
        actual = nested_get({"encoder": encoder}, key)
        rows.append(
            {
                "check": key,
                "expected": repr(expected),
                "actual": repr(actual),
                "status": "pass" if actual == expected else "fail",
            }
        )
    return rows


def source_audit(repo_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    resyn2_path = repo_root / "src" / "baselines" / "resyn2.py"
    trainer_path = repo_root / "src" / "algorithms" / "reinforce" / "trainer.py"
    episode_path = repo_root / "src" / "algorithms" / "reinforce" / "episode.py"
    rewards_path = repo_root / "src" / "models" / "rewards.py"

    sequence = extract_resyn2_sequence(resyn2_path)
    checks.append(
        {
            "check": "RESYN2_ACTION_SEQUENCE",
            "expected": repr(EXPECTED_RESYN2_SEQUENCE),
            "actual": repr(sequence),
            "status": "pass" if sequence == EXPECTED_RESYN2_SEQUENCE else "fail",
        }
    )

    source_expectations = [
        (
            "resyn2_inf_stops_after_5_unchanged",
            resyn2_path,
            "unchanged_runs >= 5",
        ),
        (
            "evaluation_selects_best_final_return",
            trainer_path,
            'max(candidates, key=lambda r: float(r["final_return"]))',
        ),
        (
            "zhu_baseline_applied_to_reward",
            episode_path,
            'baseline == "zhu_resyn2"',
        ),
        (
            "zhu_reward_subtracts_baseline",
            rewards_path,
            "return gain - self.baseline_scale * self.baseline_per_step",
        ),
    ]
    for label, path, needle in source_expectations:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        checks.append(
            {
                "check": label,
                "expected": needle,
                "actual": "found" if needle in text else "missing",
                "status": "pass" if needle in text else "fail",
            }
        )
    return checks


def extract_resyn2_sequence(path: Path) -> list[int] | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"RESYN2_ACTION_SEQUENCE\s*=\s*(\[[^\]]+\])", text)
    if not match:
        return None
    value = ast.literal_eval(match.group(1))
    return [int(item) for item in value]


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def tensorboard_crosscheck(selections: list[ReportSelection]) -> list[dict[str, Any]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return [
            {
                "circuit": "all",
                "status": "skipped",
                "checked_points": 0,
                "max_abs_error": "",
                "note": "tensorboard is not installed",
            }
        ]

    tag_to_history_key = {
        "eval/mean_final_return": "test_mean_final_return",
        "eval/mean_size_reduction": "test_mean_size_reduction",
        "eval/mean_depth_reduction": "test_mean_depth_reduction",
        "eval/win_rate_vs_resyn2_2": "test_win_rate_vs_resyn2_2",
        "eval/mean_normalized_improvement_vs_resyn2_2": "test_mean_normalized_improvement_vs_resyn2_2",
    }
    rows: list[dict[str, Any]] = []
    for selection in selections:
        if selection.selected_hydra_config is None:
            rows.append(
                {
                    "circuit": selection.circuit,
                    "status": "missing_hydra",
                    "checked_points": 0,
                    "max_abs_error": "",
                    "note": "No local Hydra config found for selected JSON",
                }
            )
            continue
        experiment_dir = selection.selected_hydra_config.parents[1]
        tb_dir = experiment_dir / "tensorboard"
        if not tb_dir.is_dir():
            rows.append(
                {
                    "circuit": selection.circuit,
                    "status": "missing_tensorboard",
                    "checked_points": 0,
                    "max_abs_error": "",
                    "note": str(tb_dir),
                }
            )
            continue

        max_abs_error = 0.0
        checked = 0
        missing: list[str] = []
        for run in selection.report.get("runs", []):
            run_idx = int(run.get("run_idx", -1))
            history = run.get("history", [])
            if not history:
                missing.append(f"run_{run_idx}:history")
                continue
            expected_row = history[-1]
            run_dir = tb_dir / f"run_{run_idx}"
            if not run_dir.is_dir():
                missing.append(f"run_{run_idx}:dir")
                continue
            accumulator = event_accumulator.EventAccumulator(
                str(run_dir),
                size_guidance={event_accumulator.SCALARS: 0},
            )
            accumulator.Reload()
            tags = set(accumulator.Tags().get("scalars") or [])
            for tag, history_key in tag_to_history_key.items():
                if tag not in tags:
                    missing.append(f"run_{run_idx}:{tag}")
                    continue
                scalars = accumulator.Scalars(tag)
                if not scalars:
                    missing.append(f"run_{run_idx}:{tag}:empty")
                    continue
                actual = float(sorted(scalars, key=lambda event: event.step)[-1].value)
                expected = float(expected_row[history_key])
                max_abs_error = max(max_abs_error, abs(actual - expected))
                checked += 1
        status = "pass" if checked > 0 and not missing and max_abs_error < 1e-5 else "fail"
        rows.append(
            {
                "circuit": selection.circuit,
                "status": status,
                "checked_points": checked,
                "max_abs_error": max_abs_error if checked else "",
                "note": "; ".join(missing[:8]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(format_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text.replace("|", "\\|")


def build_markdown_report(
    *,
    selections: list[ReportSelection],
    aggregate_rows: list[dict[str, Any]],
    config_audit_rows: list[dict[str, str]],
    source_audit_rows: list[dict[str, str]],
    tensorboard_rows: list[dict[str, Any]],
) -> str:
    provenance_rows = [
        {
            "circuit": selection.circuit,
            "json": selection.selected_path.name,
            "hydra": str(selection.selected_hydra_config or ""),
            "run_count": len(selection.report.get("runs", [])),
            "duplicates": "; ".join(path.name for path in selection.duplicate_paths),
        }
        for selection in selections
    ]
    baseline_rows = [
        {
            "circuit": row["circuit"],
            "ours_init": row["initial_size"],
            "paper_init": row.get("paper_initial_size", ""),
            "ours_r2_1": row["resyn2_1_size"],
            "paper_r2_1": row.get("paper_resyn2_1_size", ""),
            "ours_r2_2": row["resyn2_2_size"],
            "paper_r2_2": row.get("paper_resyn2_2_size", ""),
            "ours_r2_inf": row["resyn2_inf_size"],
            "paper_r2_inf": row.get("paper_resyn2_inf_size", ""),
            "status": row.get("baseline_size_status", "missing_paper_reference"),
        }
        for row in aggregate_rows
    ]
    result_rows = [
        {
            "circuit": row["circuit"],
            "n": row["n_runs"],
            "ours_size_mean": row["mean_final_size"],
            "ours_size_std": row["std_final_size"],
            "paper_rl1_size": row.get("paper_rl1_size", ""),
            "delta_size": row.get("delta_size_vs_paper_rl1", ""),
            "norm_delta": row.get("normalized_delta_size_vs_paper_rl1", ""),
            "ours_depth_mean": row["mean_final_depth"],
            "paper_rl1_depth": row.get("paper_rl1_depth", ""),
            "win_vs_r2_2": row["win_rate_vs_resyn2_2"],
            "win_vs_r2_inf": row["win_rate_vs_resyn2_inf"],
        }
        for row in aggregate_rows
    ]
    failed_config = [row for row in config_audit_rows if row["status"] != "pass"]
    failed_source = [row for row in source_audit_rows if row["status"] != "pass"]
    drift_count = sum(
        1
        for row in aggregate_rows
        if str(row.get("baseline_size_status", "")).startswith("toolchain_or_benchmark_drift")
    )
    return "\n\n".join(
        [
            "# Zhu 2020 Reproduction Analysis",
            (
                "This report analyzes the completed size-objective reproduction against Zhu et al. "
                "Table 2 RL-1. Saved JSON reports and saved Hydra configs are treated as the source "
                "of truth."
            ),
            "## Diagnosis\n\n"
            + "\n".join(
                [
                    f"- Selected {len(selections)} benchmark reports.",
                    f"- Protocol config audit failures: {len(failed_config)}.",
                    f"- Source audit failures: {len(failed_source)}.",
                    f"- Baseline/toolchain drift detected for {drift_count} benchmark(s).",
                    "- RL-vs-paper differences should be interpreted after checking baseline drift.",
                ]
            ),
            "## Provenance\n\n"
            + markdown_table(provenance_rows, ["circuit", "json", "hydra", "run_count", "duplicates"]),
            "## Protocol Config Audit\n\n"
            + markdown_table(config_audit_rows, ["check", "expected", "actual", "status"]),
            "## Source Audit\n\n" + markdown_table(source_audit_rows, ["check", "expected", "actual", "status"]),
            "## TensorBoard Cross-Check\n\n"
            + markdown_table(
                tensorboard_rows,
                ["circuit", "status", "checked_points", "max_abs_error", "note"],
            ),
            "## Baseline Comparison\n\n"
            + markdown_table(
                baseline_rows,
                [
                    "circuit",
                    "ours_init",
                    "paper_init",
                    "ours_r2_1",
                    "paper_r2_1",
                    "ours_r2_2",
                    "paper_r2_2",
                    "ours_r2_inf",
                    "paper_r2_inf",
                    "status",
                ],
            ),
            "## RL-1 Result Comparison\n\n"
            + markdown_table(
                result_rows,
                [
                    "circuit",
                    "n",
                    "ours_size_mean",
                    "ours_size_std",
                    "paper_rl1_size",
                    "delta_size",
                    "norm_delta",
                    "ours_depth_mean",
                    "paper_rl1_depth",
                    "win_vs_r2_2",
                    "win_vs_r2_inf",
                ],
            ),
            "## Notes\n\n"
            + "\n".join(
                [
                    "- The actual run settings are `entropy_beta=0.0`, no gradient clipping, and no return normalization.",
                    "- `i10` has duplicate JSON reports; the selected report is the one with a matching saved Hydra directory and highest reproduce id.",
                    "- TensorBoard is supporting evidence only; final JSON evaluation may be a fresh stochastic best-of-10 evaluation after the last logged training evaluation.",
                ]
            ),
        ]
    )


def write_analysis_outputs(repo_root: Path, output_root: Path) -> dict[str, Path]:
    selections = discover_reports(repo_root / "output", repo_root / "outputs")
    if not selections:
        raise FileNotFoundError(f"No Zhu 2020 reports found under {repo_root / 'output'}")

    aggregate_rows = [aggregate_report(selection) for selection in selections]
    config_audit_rows = audit_configs(selections)
    source_audit_rows = source_audit(repo_root)
    tensorboard_rows = tensorboard_crosscheck(selections)

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_root / "zhu2020_reproduction_report.md",
        "results_csv": output_root / "zhu2020_results.csv",
        "protocol_csv": output_root / "zhu2020_protocol_audit.csv",
        "source_csv": output_root / "zhu2020_source_audit.csv",
        "tensorboard_csv": output_root / "zhu2020_tensorboard_crosscheck.csv",
    }
    paths["report"].write_text(
        build_markdown_report(
            selections=selections,
            aggregate_rows=aggregate_rows,
            config_audit_rows=config_audit_rows,
            source_audit_rows=source_audit_rows,
            tensorboard_rows=tensorboard_rows,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(paths["results_csv"], aggregate_rows)
    write_csv(paths["protocol_csv"], config_audit_rows)
    write_csv(paths["source_csv"], source_audit_rows)
    write_csv(paths["tensorboard_csv"], tensorboard_rows)
    return paths
