from __future__ import annotations

import copy
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

try:
    import pyspiel  # noqa: F401
except ImportError:
    pyspiel_stub = types.ModuleType("pyspiel")
    pyspiel_stub.State = object
    sys.modules.setdefault("pyspiel", pyspiel_stub)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.gflownet_tb.diagnostics import (
    SequenceArchive,
    SequenceRecord,
    derive_seed,
    health_gates,
    nested_search_metrics,
    parameter_gradient_norm,
    parameter_snapshot,
    parameter_update_norm,
    residual_diagnostics,
)
from src.algorithms.gflownet_tb.sampler import _sample_behavior_actions
from src.experiments.tb_active_baseline import (
    _epsilon_for_update,
    _git_commit,
    _resolved_configuration,
    _score_trajectory_set,
)
from src.experiments.tb_active_baseline_report import classify_diagnosis, report_experiment


class ActiveBaselineDiagnosticsTest(unittest.TestCase):
    def test_myhpc_source_commit_sidecar_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = "a" * 40
            (root / ".myhpc-source-commit").write_text(expected + "\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(_git_commit(root), expected)

    def test_config_resolution_records_null_and_resolved_log_z_rate(self) -> None:
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            self.skipTest(f"OmegaConf is unavailable: {exc}")

        cfg = OmegaConf.create(
            {
                "learning_rate": 0.001,
                "episodes": 200,
                "num_steps": 20,
                "available_actions": list(range(7)),
                "tb": {
                    "trajectories_per_episode": 4,
                    "reward_alpha": 4.0,
                    "reward_eps": 1e-8,
                    "reward_improvement_clip": 2.0,
                    "log_z_learning_rate": None,
                    "exploration_epsilon_enabled": True,
                    "exploration_epsilon_start": 0.5,
                    "exploration_epsilon_end": 0.01,
                    "exploration_warmup_episodes": 20,
                    "exploration_decay_episodes": None,
                },
            }
        )
        args = SimpleNamespace(
            config_name="tb_zhuDOP",
            seed=0,
            output_dir=Path("out"),
            device="cpu",
            max_trajectories=800,
            schedule_trajectories=800,
            milestones=[200, 400, 800],
        )
        resolved = _resolved_configuration(args, cfg, Path("bc0.blif"))
        self.assertIsNone(resolved["log_z_learning_rate_configured"])
        self.assertEqual(resolved["log_z_learning_rate_resolved"], 0.01)

    def test_seed_derivation_is_stable_and_stream_specific(self) -> None:
        self.assertEqual(derive_seed(7, "training_actions"), derive_seed(7, "training_actions"))
        self.assertNotEqual(derive_seed(7, "training_actions"), derive_seed(7, "evaluation_search"))
        self.assertNotEqual(derive_seed(7, "training_actions"), derive_seed(8, "training_actions"))

    def test_residual_decomposition_and_recentering_invariance(self) -> None:
        log_pf = torch.tensor([-2.0, -1.0, -3.0], dtype=torch.float64)
        log_r = torch.tensor([-1.5, -0.5, -2.0], dtype=torch.float64)
        log_pb = torch.zeros(3, dtype=torch.float64)
        first = residual_diagnostics(log_z=0.0, log_pf=log_pf, log_r=log_r, log_pb=log_pb)
        shifted = residual_diagnostics(log_z=9.0, log_pf=log_pf, log_r=log_r, log_pb=log_pb)
        self.assertAlmostEqual(first["centered_mse"], shifted["centered_mse"], places=12)
        self.assertAlmostEqual(
            first["learned_log_z_mse"],
            first["centered_mse"] + first["squared_bias"],
            places=12,
        )
        self.assertAlmostEqual(
            first["bias_fraction"],
            first["squared_bias"] / (first["learned_log_z_mse"] + 1e-12),
            places=12,
        )
        expected_target = float((log_r + log_pb - log_pf).mean())
        self.assertAlmostEqual(first["analytic_log_z_target"], expected_target, places=12)

    def test_optional_generator_is_reproducible(self) -> None:
        probs = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32)
        left = torch.Generator().manual_seed(123)
        right = torch.Generator().manual_seed(123)
        first = [_sample_behavior_actions(probs, left) for _ in range(20)]
        second = [_sample_behavior_actions(probs, right) for _ in range(20)]
        self.assertTrue(torch.equal(torch.stack(first), torch.stack(second)))

    def test_default_generator_path_remains_categorical(self) -> None:
        probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        sentinel = torch.tensor([1])
        with patch("src.algorithms.gflownet_tb.sampler.Categorical.sample", return_value=sentinel) as sample:
            result = _sample_behavior_actions(probs, None)
        sample.assert_called_once_with()
        self.assertIs(result, sentinel)

    def test_active_epsilon_schedule_retains_800_trajectory_horizon(self) -> None:
        kwargs = {
            "schedule_updates": 200,
            "enabled": True,
            "start": 0.5,
            "end": 0.01,
            "warmup_updates": 20,
            "decay_updates": None,
        }
        self.assertEqual(_epsilon_for_update(20, **kwargs), 0.5)
        self.assertAlmostEqual(_epsilon_for_update(200, **kwargs), 0.01)
        self.assertGreater(_epsilon_for_update(25, **kwargs), 0.01)

    def test_gradient_and_update_norms_match_hand_calculation(self) -> None:
        parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
        parameter.grad = torch.tensor([6.0, 8.0])
        self.assertEqual(parameter_gradient_norm([parameter]), 10.0)
        before = parameter_snapshot([parameter])
        with torch.no_grad():
            parameter.add_(torch.tensor([0.0, 5.0]))
        self.assertEqual(parameter_update_norm(before, [parameter]), 5.0)

    def test_cached_sequences_are_rescored_after_policy_change(self) -> None:
        class TinyPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logits = nn.Parameter(torch.tensor([0.0, 0.0]))
                self.log_z = nn.Parameter(torch.tensor(0.0))
                self.num_actions = 2

            def forward(self, observations):
                return self.logits.expand(len(observations), -1)

            def masked_probs(self, logits, legal_rows):
                return torch.softmax(logits, dim=-1)

            def log_prob_legal_batch(self, logits, legal_rows, actions):
                action_tensor = torch.as_tensor(actions, dtype=torch.long)
                return torch.log_softmax(logits, dim=-1).gather(1, action_tensor[:, None]).squeeze(1)

        policy = TinyPolicy()
        step = SimpleNamespace(observation=object(), legal_actions=[0, 1], action=0)
        trajectory = SimpleNamespace(
            steps=[step],
            log_reward=0.0,
            log_pb_sum=torch.tensor(0.0),
            comparable_return=0.0,
            terminal_reward=1.0,
        )
        before = _score_trajectory_set(
            policy, [trajectory], reward_eps=1e-8, reward_improvement_clip=2.0
        )
        with torch.no_grad():
            policy.logits.copy_(torch.tensor([4.0, -4.0]))
        after = _score_trajectory_set(
            policy, [trajectory], reward_eps=1e-8, reward_improvement_clip=2.0
        )
        self.assertNotEqual(before["log_pf_values"], after["log_pf_values"])
        self.assertGreater(after["log_pf_values"][0], before["log_pf_values"][0])

    def test_sequence_archive_keeps_sequences_but_deduplicates_endpoints(self) -> None:
        archive = SequenceArchive(circuit="tiny", initial_size=10, initial_depth=10)
        for actions in ((0, 1), (1, 0)):
            archive.record(
                actions=actions,
                initial_size=10,
                initial_depth=10,
                final_size=8,
                final_depth=8,
                comparable_return=0.36,
            )
        snapshot = archive.snapshot()
        self.assertEqual(snapshot["trajectory_count"], 2)
        self.assertEqual(snapshot["distinct_sequences"], 2)
        self.assertEqual(snapshot["distinct_terminal_points"], 1)
        self.assertEqual(snapshot["nondominated_count"], 1)
        restored = SequenceArchive.from_state_dict(archive.state_dict())
        self.assertEqual(restored.state_dict(), archive.state_dict())

    def test_nested_hypervolume_and_log_auc_are_deterministic(self) -> None:
        records = [
            SequenceRecord(i, (i,), 10, 10, 10 - min(i, 5), 10 - min(i, 5), float(i))
            for i in range(1, 51)
        ]
        first = nested_search_metrics(records, initial_size=10, initial_depth=10)
        second = nested_search_metrics(records, initial_size=10, initial_depth=10)
        self.assertEqual(first, second)
        self.assertGreater(first["log2_n_hypervolume_auc"], 0.0)
        self.assertEqual([row["n"] for row in first["budgets"]], [1, 2, 5, 10, 20, 50])

    def test_health_gate_boundaries_are_inclusive_except_clipping(self) -> None:
        validation = {
            "finite": True,
            "residual": {
                "log_z_target_gap": 0.5,
                "bias_fraction": 0.05,
                "standardized_bias": 0.25,
            },
            "policy": {
                "max_normalization_error": 1e-6,
                "max_illegal_probability": 1e-6,
                "collapse_fraction": 0.95,
            },
        }
        gate = health_gates(
            validation=validation,
            gradient_p99_median_ratio=20.0,
            clipping_enabled=False,
            clipping_rate=1.0,
        )
        self.assertTrue(gate["pass"])
        clipped = health_gates(
            validation=validation,
            gradient_p99_median_ratio=20.0,
            clipping_enabled=True,
            clipping_rate=0.05,
        )
        self.assertFalse(clipped["checks"]["clipping_rate"])


def _validation(
    *, reduction: float, bias: float = 0.6, rms: float = 1.0, healthy: bool = True
) -> dict:
    return {
        "finite": healthy,
        "residual": {
            "recentered_mse_reduction": reduction,
            "bias_fraction": bias if healthy else 0.2,
            "centered_rms": rms,
            "log_z_target_gap": 0.0 if healthy else 2.0,
            "standardized_bias": 0.0 if healthy else 2.0,
            "learned_log_z_mse": 1.0,
            "centered_mse": 1.0 - reduction,
            "residual_mean": 0.0,
        },
        "policy": {
            "max_normalization_error": 0.0,
            "max_illegal_probability": 0.0,
            "collapse_fraction": 0.0,
        },
        "regression": {"slope": 1.0, "correlation": 1.0},
    }


def _fake_runs(
    *, undertrained: bool, healthy: bool = True, offset: bool = True
) -> dict[tuple[str, int], dict]:
    result = {}
    for circuit in ("bc0", "dalu"):
        for seed in range(3):
            milestones = {}
            for budget in (200, 400, 800):
                rms = 1.0
                if budget == 800 and undertrained:
                    rms = 0.9
                hv = 0.01 if budget < 800 else (0.02 if undertrained else 0.01)
                auc = 0.1 if budget < 800 else (0.12 if undertrained else 0.1)
                milestones[budget] = {
                    "fixed_uniform": _validation(
                        reduction=0.6 if offset else 0.4,
                        bias=0.6 if offset else 0.04,
                        rms=rms,
                        healthy=healthy,
                    ),
                    "fresh_on_policy": _validation(
                        reduction=0.6 if offset else 0.4,
                        bias=0.6 if offset else 0.04,
                        rms=rms,
                        healthy=healthy,
                    ),
                    "training_archive": {"hypervolume": hv},
                    "search": {"log2_n_hypervolume_auc": auc},
                    "optimizer_health": {
                        "policy_gradient_p99_median_ratio": 1.0,
                        "gradient_clipping_enabled": False,
                        "gradient_clipping_rate": 0.0,
                    },
                }
            result[(circuit, seed)] = {"milestones": milestones}
    return result


class ActiveBaselineClassificationTest(unittest.TestCase):
    def test_report_exit_codes_distinguish_missing_and_corrupt_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "report"
            args = SimpleNamespace(
                runs_root=root / "runs",
                output_dir=output,
                expected_circuits=["bc0", "dalu"],
                expected_seeds=[0, 1, 2],
            )
            self.assertEqual(report_experiment(args), 2)
            run_dir = root / "runs/bc0/seed_0"
            run_dir.mkdir(parents=True)
            (run_dir / "run_summary.json").write_text("not-json", encoding="utf-8")
            self.assertEqual(report_experiment(args), 1)

    def test_global_offset_bias_threshold_is_strict(self) -> None:
        runs = _fake_runs(undertrained=True, offset=True)
        for run in runs.values():
            for stratum in ("fixed_uniform", "fresh_on_policy"):
                run["milestones"][800][stratum]["residual"]["recentered_mse_reduction"] = 0.5
                run["milestones"][800][stratum]["residual"]["bias_fraction"] = 0.5
        result = classify_diagnosis(runs)
        self.assertEqual(result["global_offset"]["decision"], "rejected")

    def test_offset_and_undertraining_supported(self) -> None:
        result = classify_diagnosis(_fake_runs(undertrained=True, offset=True))
        self.assertEqual(result["global_offset"]["decision"], "supported")
        self.assertEqual(result["undertraining"]["decision"], "undertraining_supported")
        self.assertEqual(result["exit_code"], 0)

    def test_healthy_no_change_retains_800_as_candidate(self) -> None:
        result = classify_diagnosis(_fake_runs(undertrained=False, healthy=True, offset=False))
        self.assertEqual(result["global_offset"]["decision"], "rejected")
        self.assertEqual(result["undertraining"]["decision"], "800_not_demonstrably_insufficient")
        self.assertEqual(result["exit_code"], 0)

    def test_unhealthy_no_change_is_inconclusive(self) -> None:
        result = classify_diagnosis(_fake_runs(undertrained=False, healthy=False))
        self.assertEqual(result["undertraining"]["decision"], "inconclusive_unhealthy_active_baseline")
        self.assertEqual(result["exit_code"], 3)


if __name__ == "__main__":
    unittest.main()
