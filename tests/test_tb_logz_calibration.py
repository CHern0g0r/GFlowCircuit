from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

try:
    import pyspiel  # noqa: F401
except ImportError:
    pyspiel_stub = types.ModuleType("pyspiel")
    pyspiel_stub.State = object
    sys.modules.setdefault("pyspiel", pyspiel_stub)

from src.experiments.tb_experiment_common import (
    calibration_target,
    differentiable_trajectory_scores,
    initialize_log_z,
)
from src.experiments.tb_logz_calibration_report import (
    classify_calibration,
    oscillation_diagnostics,
    report_experiment,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.log_z = nn.Parameter(torch.tensor(0.0))
        self.num_actions = 2

    def forward(self, observations):
        return self.logits.expand(len(observations), -1)

    def log_prob_legal_batch(self, logits, legal_rows, actions):
        return torch.log_softmax(logits, dim=-1).gather(
            1, torch.as_tensor(actions, dtype=torch.long, device=logits.device)[:, None]
        ).squeeze(1)


def trajectory(*actions: int, log_reward: float = 0.0):
    return SimpleNamespace(
        steps=[SimpleNamespace(observation=object(), legal_actions=[0, 1], action=action) for action in actions],
        log_pb_sum=torch.tensor(0.0),
        log_reward=log_reward,
    )


class CalibrationMechanicsTest(unittest.TestCase):
    def test_target_and_zcal_assignment(self) -> None:
        policy = TinyPolicy()
        trajectories = [trajectory(0, log_reward=1.0), trajectory(1, log_reward=3.0)]
        target = calibration_target(policy, trajectories)
        self.assertAlmostEqual(target, 2.0 + float(torch.log(torch.tensor(2.0))), places=6)
        self.assertEqual(initialize_log_z(policy, "z0", target), 0.0)
        self.assertEqual(float(policy.log_z.detach()), 0.0)
        self.assertAlmostEqual(initialize_log_z(policy, "zcal", target), target)
        self.assertAlmostEqual(float(policy.log_z.detach()), target, places=6)

    def test_cached_rescoring_tracks_policy_and_retains_gradients(self) -> None:
        policy = TinyPolicy()
        cached = [trajectory(0, 0), trajectory(1, 1)]
        before, _, _ = differentiable_trajectory_scores(policy, cached)
        before.sum().backward()
        self.assertIsNotNone(policy.logits.grad)
        policy.logits.grad = None
        with torch.no_grad():
            policy.logits.copy_(torch.tensor([4.0, -4.0]))
        after, _, _ = differentiable_trajectory_scores(policy, cached)
        self.assertFalse(torch.equal(before.detach(), after.detach()))
        after.sum().backward()
        self.assertIsNotNone(policy.logits.grad)

    def test_initialization_rejects_unknown_variant(self) -> None:
        with self.assertRaises(ValueError):
            initialize_log_z(TinyPolicy(), "other", 1.0)


def _validation(*, gap: float, bias: float, rms: float, healthy: bool = True) -> dict:
    return {
        "finite": healthy,
        "residual": {
            "log_z_target_gap": gap,
            "bias_fraction": bias,
            "standardized_bias": 0.1 if healthy else 1.0,
            "centered_rms": rms,
        },
        "policy": {
            "max_normalization_error": 0.0,
            "max_illegal_probability": 0.0,
            "collapse_fraction": 0.0,
        },
    }


def _fake_runs(
    *, improvement: bool = True, harm: bool = False, oscillation: bool = False,
    tie: bool = False, numerical: bool = False,
) -> dict[tuple[str, str, int], dict]:
    result = {}
    for variant in ("z0", "zcal"):
        for circuit in ("bc0", "dalu"):
            for seed in range(3):
                milestones = {}
                for budget in (200, 400, 800):
                    if budget == 200:
                        gap = 0.4 if variant == "z0" else (0.1 if improvement else 0.3)
                        bias = 0.04 if variant == "z0" else (0.01 if improvement else 0.03)
                    else:
                        gap, bias = 0.1, 0.01
                    rms = 1.0
                    if variant == "zcal" and harm and budget == 800:
                        rms = 1.2
                    hv = 0.1
                    if variant == "zcal" and not tie:
                        hv = 0.11
                    validation = _validation(gap=gap, bias=bias, rms=rms)
                    milestones[budget] = {
                        "fixed_uniform": validation,
                        "fresh_on_policy": validation,
                        "training_archive": {"hypervolume": hv},
                        "optimizer_health": {"policy_gradient_p99_median_ratio": 1.0},
                    }
                osc = {"persistent": oscillation and variant == "zcal", "sign_changes": 5}
                result[(variant, circuit, seed)] = {
                    "milestones": milestones,
                    "oscillation": osc,
                    "summary": {"numerical_failure": "failure" if numerical and variant == "zcal" else None},
                }
    return result


class CalibrationDecisionTest(unittest.TestCase):
    def test_report_distinguishes_missing_and_corrupt_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                runs_root=root / "runs",
                output_dir=root / "report",
                expected_variants=["z0", "zcal"],
                expected_circuits=["bc0", "dalu"],
                expected_seeds=[0, 1, 2],
            )
            self.assertEqual(report_experiment(args), 2)
            run_dir = root / "runs/z0/bc0/seed_0"
            run_dir.mkdir(parents=True)
            (run_dir / "run_summary.json").write_text("not-json", encoding="utf-8")
            self.assertEqual(report_experiment(args), 1)

    def test_oscillation_rule_ignores_zeros(self) -> None:
        stable = oscillation_diagnostics([1.0] * 100 + [0.0] * 10 + [1.0] * 90)
        self.assertFalse(stable["persistent"])
        worsening = [(-1.0 if index % 2 else 1.0) * (1.0 if index < 150 else 2.0) for index in range(200)]
        self.assertTrue(oscillation_diagnostics(worsening)["persistent"])

    def test_zcal_is_selected_when_beneficial_and_nonharmful(self) -> None:
        result = classify_calibration(_fake_runs())
        self.assertEqual(result["decision"], "support_calibrated_initialization")
        self.assertEqual(result["selected_variant"], "zcal")

    def test_insufficient_improvement_rejects_zcal(self) -> None:
        result = classify_calibration(_fake_runs(improvement=False))
        self.assertFalse(result["checks"]["improvement_200_pass"])
        self.assertEqual(result["selected_variant"], "z0")

    def test_800_harm_rejects_zcal(self) -> None:
        result = classify_calibration(_fake_runs(harm=True))
        self.assertFalse(result["checks"]["nonharm_800_pass"])

    def test_oscillation_and_numerical_failure_reject(self) -> None:
        self.assertFalse(classify_calibration(_fake_runs(oscillation=True))["checks"]["oscillation_pass"])
        self.assertFalse(classify_calibration(_fake_runs(numerical=True))["checks"]["numerical_pass"])

    def test_healthy_statistical_tie_prefers_z0(self) -> None:
        result = classify_calibration(_fake_runs(tie=True))
        self.assertEqual(result["decision"], "prefer_simpler_z0")
        self.assertEqual(result["selected_variant"], "z0")


if __name__ == "__main__":
    unittest.main()
