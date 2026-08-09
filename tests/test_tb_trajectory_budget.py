from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.experiments.tb_trajectory_budget import (
    MAX_TRAJECTORIES,
    MILESTONES,
    SCHEDULE_TRAJECTORIES,
    _validate_experiment5_args,
)
from src.experiments.tb_logz_calibration import _resolved_configuration, _validate_configuration
from src.experiments.tb_trajectory_budget_report import (
    EXPECTED_CIRCUITS,
    EXPECTED_SEEDS,
    classify_budget_selection,
    paired_median_bootstrap,
    report,
)


def _validation(centered_rms: float = 1.0, *, healthy: bool = True) -> dict:
    return {
        "finite": healthy,
        "residual": {
            "centered_rms": centered_rms,
            "log_z_target_gap": 0.01 if healthy else 1.0,
            "bias_fraction": 0.01 if healthy else 0.9,
            "standardized_bias": 0.1 if healthy else 1.0,
        },
        "policy": {
            "max_normalization_error": 0.0,
            "max_illegal_probability": 0.0,
            "collapse_fraction": 0.0,
        },
    }


def _synthetic_runs(
    *, eligible_from: dict[str, int] | None = None,
) -> dict[tuple[str, int], dict]:
    eligible_from = eligible_from or {"bc0": 200, "dalu": 200}
    runs = {}
    for circuit in EXPECTED_CIRCUITS:
        for seed in EXPECTED_SEEDS:
            milestones = {}
            for budget in MILESTONES:
                healthy = budget >= eligible_from[circuit]
                # A 2% reduction per doubling is below both RMS thresholds.
                rms = 1.0 * (0.98 ** int(round(math.log2(budget / 200))))
                milestones[budget] = {
                    "fixed_uniform": _validation(rms, healthy=healthy),
                    "fresh_on_policy": _validation(rms, healthy=healthy),
                    "optimizer_health": {
                        "policy_gradient_p99_median_ratio": 1.0,
                        "gradient_clipping_enabled": False,
                        "gradient_clipping_rate": 0.0,
                    },
                    "training_archive": {"hypervolume": 0.2 + 0.0001 * math.log2(budget / 200)},
                    "search": {"log2_n_hypervolume_auc": 0.2},
                }
            runs[(circuit, seed)] = {"milestones": milestones}
    return runs


class ConfigurationTest(unittest.TestCase):
    def _resolved_contract(self) -> dict:
        return {
            "experiment": "trajectory_budget_selection", "variant": "zcal",
            "max_trajectories": 6_400, "schedule_trajectories": 800,
            "num_steps": 20, "available_actions": list(range(7)),
            "trajectories_per_update": 4, "policy_learning_rate": 0.001,
            "log_z_learning_rate_resolved": 0.01, "reward_alpha": 4.0,
            "reward_eps": 1e-8, "reward_improvement_clip": 2.0,
            "exploration_epsilon_enabled": True, "exploration_epsilon_start": 0.5,
            "exploration_epsilon_end": 0.01, "exploration_warmup_updates": 20,
            "configured_optimizer_updates": 1_600, "calibration_trajectories": 64,
            "calibration_epsilon": 0.5,
        }

    def test_shared_engine_accepts_only_experiment5_logz_control(self) -> None:
        resolved = self._resolved_contract()
        _validate_configuration(resolved)
        resolved["variant"] = "z0"
        with self.assertRaises(ValueError):
            _validate_configuration(resolved)
        resolved = self._resolved_contract()
        resolved["log_z_learning_rate_resolved"] = 0.003
        with self.assertRaises(ValueError):
            _validate_configuration(resolved)

    def test_scientific_budget_is_fixed(self) -> None:
        valid = SimpleNamespace(
            max_trajectories=MAX_TRAJECTORIES,
            schedule_trajectories=SCHEDULE_TRAJECTORIES,
            milestones=list(MILESTONES),
        )
        _validate_experiment5_args(valid)
        valid.max_trajectories = 3_200
        with self.assertRaises(ValueError):
            _validate_experiment5_args(valid)

    def test_internal_smoke_budget_escape_is_not_a_cli_setting(self) -> None:
        args = SimpleNamespace(
            max_trajectories=72, schedule_trajectories=800,
            milestones=[68, 72], _allow_test_budget=True,
        )
        _validate_experiment5_args(args)

    def test_resume_fingerprint_includes_maximum_and_milestones(self) -> None:
        tb = SimpleNamespace(
            batch_size=4, trajectories_per_episode=4, log_z_learning_rate=None,
            reward_alpha=4.0, reward_eps=1e-8, reward_improvement_clip=2.0,
            exploration_epsilon_enabled=True, exploration_epsilon_start=0.5,
            exploration_epsilon_end=0.01, exploration_warmup_episodes=20,
            exploration_decay_episodes=None,
        )
        cfg = SimpleNamespace(
            learning_rate=0.001, episodes=200, num_steps=20,
            available_actions=list(range(7)), tb=tb,
        )

        class FakeOmegaConf:
            @staticmethod
            def select(value, path):
                current = value
                for component in path.split("."):
                    current = getattr(current, component, None)
                    if current is None:
                        break
                return current

            @staticmethod
            def to_container(value, resolve=True):
                return {"test_config": True}

        def resolved(maximum: int, milestones: list[int]) -> dict:
            args = SimpleNamespace(
                variant="zcal", experiment_name="trajectory_budget_selection",
                run_schema_version=4, log_z_learning_rate=0.01,
                config_name="tb_zhuDOP", circuit="bc0", seed=0,
                output_dir=Path("/tmp/experiment5-test"), device="cpu",
                max_trajectories=maximum, schedule_trajectories=800,
                milestones=milestones,
            )
            fake_module = SimpleNamespace(OmegaConf=FakeOmegaConf)
            with patch.dict(sys.modules, {"omegaconf": fake_module}):
                return _resolved_configuration(args, cfg, Path("/tmp/bc0.blif"))

        full = resolved(6_400, list(MILESTONES))
        shorter = resolved(3_200, [200, 400, 800, 1_600, 3_200])
        self.assertNotEqual(
            full["scientific_configuration_fingerprint"],
            shorter["scientific_configuration_fingerprint"],
        )
        self.assertEqual(full["scientific_configuration"]["max_trajectories"], 6_400)
        self.assertEqual(full["scientific_configuration"]["milestones"], list(MILESTONES))


class BootstrapTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_uses_median(self) -> None:
        first = paired_median_bootstrap([0.01, 0.02, 0.03, 0.04, 0.5], seed=7)
        second = paired_median_bootstrap([0.01, 0.02, 0.03, 0.04, 0.5], seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["median"], 0.03)
        self.assertEqual(first["samples"], 10_000)


class SelectionTest(unittest.TestCase):
    def test_selects_smallest_candidate_and_cross_circuit_maximum(self) -> None:
        decision = classify_budget_selection(
            _synthetic_runs(eligible_from={"bc0": 200, "dalu": 800})
        )
        self.assertEqual(decision["circuit_candidates"], {"bc0": 200, "dalu": 800})
        self.assertEqual(decision["selected_budget"], 800)
        self.assertEqual(decision["decision"], "select_common_trajectory_budget")

    def test_both_strata_must_pass_rms_gate(self) -> None:
        runs = _synthetic_runs()
        for seed in EXPECTED_SEEDS:
            runs[("bc0", seed)]["milestones"][400]["fresh_on_policy"]["residual"]["centered_rms"] = 0.5
        decision = classify_budget_selection(runs)
        first = next(row for row in decision["comparisons"]
                     if row["circuit"] == "bc0" and row["trajectory_budget"] == 200)
        self.assertFalse(first["rms_pass"])
        self.assertFalse(first["eligible"])

    def test_thresholds_are_strict(self) -> None:
        runs = _synthetic_runs()
        for seed in EXPECTED_SEEDS:
            runs[("bc0", seed)]["milestones"][200]["training_archive"]["hypervolume"] = 0.2
            runs[("bc0", seed)]["milestones"][400]["training_archive"]["hypervolume"] = 0.205
        decision = classify_budget_selection(runs)
        first = next(row for row in decision["comparisons"]
                     if row["circuit"] == "bc0" and row["trajectory_budget"] == 200)
        self.assertFalse(first["hypervolume_pass"])

    def test_no_fallback_to_largest_checkpoint(self) -> None:
        runs = _synthetic_runs(eligible_from={"bc0": 6_400, "dalu": 6_400})
        decision = classify_budget_selection(runs)
        self.assertIsNone(decision["selected_budget"])
        self.assertEqual(decision["circuit_candidates"], {"bc0": None, "dalu": None})
        self.assertEqual(decision["decision"], "reject_no_finite_budget_within_cap")

    def test_report_exit_codes_distinguish_incomplete_and_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(runs_root=root / "runs", output_dir=root / "report")
            self.assertEqual(report(args), 2)
            first = args.runs_root / "bc0" / "seed_0"
            first.mkdir(parents=True)
            (first / "run_summary.json").write_text("not-json", encoding="utf-8")
            self.assertEqual(report(args), 1)
            payload = json.loads((args.output_dir / "decision_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_type"], "artifact_or_execution_failure")


try:
    import pyspiel
except ImportError:
    pyspiel = None


@unittest.skipIf(pyspiel is None, "OpenSpiel Python bindings are unavailable")
class TrajectoryBudgetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        circuit = Path(__file__).resolve().parents[1] / "data/hdl-benchmarks/mcnc/Combinational/blif/bc0.blif"
        try:
            pyspiel.load_game("circuit", {"num_steps": 20, "file_path": str(circuit)})
        except Exception as exc:
            raise unittest.SkipTest(f"OpenSpiel circuit game is unavailable: {exc}") from exc

    @unittest.skipUnless(
        os.environ.get("GFLOWCIRCUIT_RUN_SLOW_INTEGRATION") == "1",
        "set GFLOWCIRCUIT_RUN_SLOW_INTEGRATION=1 for Experiment 5 integration",
    )
    def test_continuous_reduced_run_checkpoints_and_archive_isolation(self) -> None:
        from src.experiments.tb_trajectory_budget import run_experiment

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            args = SimpleNamespace(
                config_name="tb_zhuDOP", circuit="bc0", seed=0, output_dir=output,
                max_trajectories=72, schedule_trajectories=800, milestones=[68, 72],
                device="cpu", resume_checkpoint=None, _allow_test_budget=True,
            )
            self.assertEqual(run_experiment(args), 0)
            first = torch.load(output / "checkpoints/trajectory_68.pt", map_location="cpu", weights_only=False)
            second = torch.load(output / "checkpoints/trajectory_72.pt", map_location="cpu", weights_only=False)
            self.assertEqual(first["counters"]["training_trajectories"], 68)
            self.assertEqual(second["counters"]["training_trajectories"], 72)
            self.assertEqual(second["counters"]["training_presentations"], 72)
            self.assertEqual(second["counters"]["validation_rollouts"], 256 + 2 * (128 + 50))
            rows = [json.loads(line) for line in (output / "trajectories.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row["source"] == "training" for row in rows), 8)
            self.assertEqual(sum(row["source"] == "search" for row in rows), 100)
            self.assertEqual(len(second["archive"]["records"]), 72)


if __name__ == "__main__":
    unittest.main()
