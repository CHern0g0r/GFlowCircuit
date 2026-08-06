from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.experiments.tb_logz_calibration import _validate_configuration
from src.experiments.tb_logz_calibration_report import oscillation_diagnostics
from src.experiments.tb_logz_learning_rate_report import (
    ArtifactValidationError,
    CONFIRMATION_SEEDS,
    DALU_REPLICATION_SEEDS,
    EXPECTED_INITIALIZATIONS,
    EXPECTED_RATES,
    SCREEN_SEEDS,
    _COMPATIBILITY_KEYS,
    _cross_circuit_rows,
    _validate_external_control,
    classify_confirmation,
    classify_screen,
    dalu_replication_report,
    rate_slug,
    screen_report,
)


def _validation(*, gap: float, bias: float, healthy: bool = True) -> dict:
    return {
        "finite": healthy,
        "residual": {
            "log_z_target_gap": gap,
            "bias_fraction": bias,
            "standardized_bias": 0.1 if healthy else 1.0,
            "centered_rms": 0.2,
            "learned_log_z": 40.0,
        },
        "policy": {
            "max_normalization_error": 0.0,
            "max_illegal_probability": 0.0,
            "collapse_fraction": 0.0,
        },
    }


def _run(*, gap: float = 0.05, bias: float = 0.01, unhealthy_early: bool = False,
         persistent: bool = False, hypervolume: float = 0.2) -> dict:
    milestones = {}
    for budget in (200, 400, 800):
        healthy = not (unhealthy_early and budget < 800)
        validation = _validation(gap=gap, bias=bias, healthy=healthy)
        milestones[budget] = {
            "fixed_uniform": validation,
            "fresh_on_policy": validation,
            "optimizer_health": {"policy_gradient_p99_median_ratio": 1.0},
            "training_archive": {"hypervolume": hypervolume},
            "search": {"log2_n_hypervolume_auc": hypervolume},
        }
    return {"milestones": milestones, "oscillation": {"persistent": persistent}}


def _screen_runs() -> dict:
    result = {}
    for initialization in EXPECTED_INITIALIZATIONS:
        for rate in EXPECTED_RATES:
            for seed in SCREEN_SEEDS:
                gap = 0.02 + abs(math.log10(rate) - math.log10(0.01)) * 0.01
                result[(initialization, rate, "bc0", seed)] = _run(
                    gap=gap, bias=0.01, unhealthy_early=True,
                    hypervolume=0.2 + (0.001 if initialization == "zcal" else 0.0),
                )
    return result


def _dalu_runs() -> dict:
    return {
        (initialization, rate, "dalu", seed): _run(
            gap=0.02 + abs(math.log10(rate) - math.log10(0.01)) * 0.01,
            bias=0.01,
        )
        for initialization in EXPECTED_INITIALIZATIONS
        for rate in EXPECTED_RATES
        for seed in DALU_REPLICATION_SEEDS
    }


class ConfigurationTest(unittest.TestCase):
    def _resolved(self, rate: float) -> dict:
        return {
            "experiment": "log_z_learning_rate", "variant": "zcal", "num_steps": 20,
            "available_actions": list(range(7)), "trajectories_per_update": 4,
            "policy_learning_rate": 0.001, "log_z_learning_rate_resolved": rate,
            "reward_alpha": 4.0, "reward_eps": 1e-8, "reward_improvement_clip": 2.0,
            "exploration_epsilon_enabled": True, "exploration_epsilon_start": 0.5,
            "exploration_epsilon_end": 0.01, "exploration_warmup_updates": 20,
            "configured_optimizer_updates": 200, "calibration_trajectories": 64,
            "calibration_epsilon": 0.5, "schedule_trajectories": 800,
        }

    def test_accepts_only_protocol_rates(self) -> None:
        for rate in EXPECTED_RATES:
            _validate_configuration(self._resolved(rate))
        with self.assertRaises(ValueError):
            _validate_configuration(self._resolved(0.02))

    def test_rate_slugs_are_stable(self) -> None:
        self.assertEqual(rate_slug(0.003), "rate_0p003")
        self.assertEqual(rate_slug(0.1), "rate_0p1")
        with self.assertRaises(ValueError):
            rate_slug(0.02)


class ScreenDecisionTest(unittest.TestCase):
    def test_report_distinguishes_incomplete_and_corrupt_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(runs_root=root / "runs", output_dir=root / "report")
            self.assertEqual(screen_report(args), 2)
            first = args.runs_root / "z0/rate_0p003/bc0/seed_0"
            first.mkdir(parents=True)
            (first / "run_summary.json").write_text("not-json", encoding="utf-8")
            self.assertEqual(screen_report(args), 1)

    def test_early_health_is_diagnostic_and_candidate_cap_is_two(self) -> None:
        decision = classify_screen(_screen_runs())
        self.assertEqual(decision["decision"], "continue_to_dalu")
        self.assertEqual(len(decision["candidates"]), 2)
        self.assertEqual({row["log_z_learning_rate"] for row in decision["candidates"]}, {0.01})
        early = [row for row in decision["health_gates"] if row["trajectory_budget"] == 200]
        self.assertTrue(early)
        self.assertTrue(any(not row["pass"] for row in early))
        self.assertTrue(all(not row["decisive"] for row in early))

    def test_bias_ratio_boundary_and_persistent_oscillation(self) -> None:
        runs = _screen_runs()
        for seed in SCREEN_SEEDS:
            runs[("z0", 0.1, "bc0", seed)] = _run(bias=0.011)
            runs[("zcal", 0.1, "bc0", seed)] = _run(bias=0.01101)
            runs[("zcal", 0.03, "bc0", seed)] = _run(persistent=True)
        decision = classify_screen(runs)
        rows = {(row["initialization"], row["log_z_learning_rate"]): row for row in decision["cells"]}
        self.assertTrue(rows[("z0", 0.1)]["bias_control_pass"])
        self.assertFalse(rows[("zcal", 0.1)]["bias_control_pass"])
        self.assertFalse(rows[("zcal", 0.03)]["oscillation_pass"])

    def test_no_healthy_candidate_is_explicit_rejection(self) -> None:
        runs = _screen_runs()
        for run in runs.values():
            run["milestones"][800]["fixed_uniform"] = _validation(gap=1.0, bias=0.9, healthy=False)
        decision = classify_screen(runs)
        self.assertEqual(decision["decision"], "reject_no_healthy_screen_candidate")
        self.assertEqual(decision["candidates"], [])

    def test_oscillation_exact_boundaries(self) -> None:
        four_changes = [1.0] * 40 + [-1.0] * 40 + [1.0] * 40 + [-1.0] * 40 + [2.0] * 40
        self.assertEqual(oscillation_diagnostics(four_changes)["sign_changes"], 4)
        self.assertFalse(oscillation_diagnostics(four_changes)["persistent"])
        five_changes = [1.0] * 30 + [-1.0] * 30 + [1.0] * 30 + [-1.0] * 30 + [1.0] * 30 + [-2.0] * 50
        self.assertEqual(oscillation_diagnostics(five_changes)["sign_changes"], 5)
        self.assertTrue(oscillation_diagnostics(five_changes)["persistent"])

    def test_dalu_replication_classification_uses_three_seeds(self) -> None:
        decision = classify_screen(
            _dalu_runs(),
            circuit="dalu",
            seeds=DALU_REPLICATION_SEEDS,
            success_decision="dalu_replication_has_eligible_cells",
            failure_decision="dalu_replication_no_eligible_cells",
        )
        self.assertEqual(decision["decision"], "dalu_replication_has_eligible_cells")
        self.assertEqual(len(decision["candidates"]), 2)
        decisive = [row for row in decision["health_gates"] if row["decisive"]]
        self.assertEqual(len(decisive), 2 * 4 * 3 * 2)

    def test_cross_circuit_rows_are_paired_only_on_screen_seeds(self) -> None:
        rows = _cross_circuit_rows(_screen_runs(), _dalu_runs())
        self.assertEqual(len(rows), 2 * 4 * 2 * 3 * 2)
        self.assertEqual({row["seed"] for row in rows}, set(SCREEN_SEEDS))
        self.assertTrue(all(row["dalu_minus_bc0_absolute_gap"] == 0.0 for row in rows))

    def test_dalu_replication_report_rejects_an_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                bc0_runs_root=root / "bc0",
                dalu_runs_root=root / "dalu",
                output_dir=root / "report",
            )
            self.assertEqual(dalu_replication_report(args), 2)


class ConfirmationDecisionTest(unittest.TestCase):
    def test_experiment3_adapter_allows_source_difference_but_requires_pairing(self) -> None:
        resolved = {key: 1 for key in _COMPATIBILITY_KEYS}
        checksums = {
            "pre_calibration_parameter_checksum": "policy",
            "fixed_sequence_checksum": "fixed",
            "calibration_sequence_checksum": "calibration",
        }
        candidate = {"resolved": dict(resolved), "summary": dict(checksums),
                     "metadata": {"source_tree_sha256": "experiment4"}}
        control = {"resolved": dict(resolved), "summary": dict(checksums),
                   "metadata": {"source_tree_sha256": "experiment3"}}
        _validate_external_control(candidate, control)
        control["summary"]["fixed_sequence_checksum"] = "different"
        with self.assertRaises(ArtifactValidationError):
            _validate_external_control(candidate, control)
        control["summary"]["fixed_sequence_checksum"] = "fixed"
        control["resolved"]["policy_learning_rate"] = 0.5
        with self.assertRaises(ArtifactValidationError):
            _validate_external_control(candidate, control)

    def test_selects_only_healthy_confirmed_cell(self) -> None:
        screen = _screen_runs()
        candidates = [
            {"initialization": "z0", "log_z_learning_rate": 0.003},
            {"initialization": "zcal", "log_z_learning_rate": 0.03},
        ]
        dalu, controls = {}, {}
        for initialization in EXPECTED_INITIALIZATIONS:
            for seed in CONFIRMATION_SEEDS:
                controls[(initialization, "dalu", seed)] = _run(bias=0.01)
        for seed in CONFIRMATION_SEEDS:
            dalu[("z0", 0.003, "dalu", seed)] = _run(gap=0.03, bias=0.01)
            dalu[("zcal", 0.03, "dalu", seed)] = _run(gap=0.01, bias=0.01, persistent=True)
        decision = classify_confirmation(screen, dalu, controls, candidates)
        self.assertEqual(decision["selected"], {"initialization": "z0", "log_z_learning_rate": 0.003})


if __name__ == "__main__":
    unittest.main()
    _COMPATIBILITY_KEYS,
    _validate_external_control,
