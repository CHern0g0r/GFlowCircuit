from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.algorithms.gflownet_tb.behavior import epsilon_mixed_probs
from src.experiments.tb_batch_size_influence import (
    BATCH_SIZES,
    MAX_TRAJECTORIES,
    MILESTONES,
    _configure,
)
from src.experiments.tb_batch_size_influence_report import (
    EXPECTED_CIRCUITS,
    EXPECTED_SEEDS,
    classify_batch_sizes,
    paired_mean_bootstrap,
    report,
)
from src.experiments.tb_logz_calibration import (
    _resolved_configuration,
    _set_trajectories_per_update,
    canonical_trajectory_epsilon_values,
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


def _synthetic_runs() -> dict[tuple[int, str, int], dict]:
    runs = {}
    for batch_size in BATCH_SIZES:
        for circuit in EXPECTED_CIRCUITS:
            for seed in EXPECTED_SEEDS:
                milestones = {
                    budget: {
                        "fixed_uniform": _validation(),
                        "fresh_on_policy": _validation(),
                        "optimizer_health": {
                            "policy_gradient_p99_median_ratio": 1.0,
                            "gradient_clipping_enabled": False,
                            "gradient_clipping_rate": 0.0,
                        },
                        "training_archive": {"hypervolume": 0.2},
                        "search": {"log2_n_hypervolume_auc": 0.1},
                    }
                    for budget in MILESTONES
                }
                runs[(batch_size, circuit, seed)] = {
                    "milestones": milestones,
                    "summary": {"wall_time_seconds": float(100 + batch_size)},
                }
    return runs


class EpsilonVectorTest(unittest.TestCase):
    def test_scalar_behavior_is_preserved(self) -> None:
        probs = torch.tensor([[0.8, 0.2], [0.25, 0.75]])
        expected = 0.75 * probs + 0.25 * torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        actual = epsilon_mixed_probs(probs, [[0, 1], [0, 1]], 0.25)
        self.assertTrue(torch.allclose(actual, expected))

    def test_vector_epsilon_is_applied_per_row(self) -> None:
        probs = torch.tensor([[0.8, 0.2], [0.25, 0.75]])
        actual = epsilon_mixed_probs(probs, [[0, 1], [0, 1]], [0.0, 1.0])
        self.assertTrue(torch.allclose(actual[0], probs[0]))
        self.assertTrue(torch.allclose(actual[1], torch.tensor([0.5, 0.5])))

    def test_vector_shape_range_and_finiteness_are_validated(self) -> None:
        probs = torch.tensor([[0.8, 0.2], [0.25, 0.75]])
        for invalid in ([0.1], [0.1, 1.1], [0.1, float("nan")]):
            with self.assertRaises(ValueError):
                epsilon_mixed_probs(probs, [[0, 1], [0, 1]], invalid)

    def test_canonical_schedule_matches_original_batch_four_groups(self) -> None:
        values = canonical_trajectory_epsilon_values(first_trajectory=1, count=1_600)
        self.assertEqual(len(values), 1_600)
        self.assertEqual(values[:80], [0.5] * 80)
        self.assertEqual(values[80:84], [values[80]] * 4)
        self.assertLess(values[80], 0.5)
        self.assertTrue(np.allclose(values[796:800], [0.01] * 4, rtol=0.0, atol=1e-12))
        self.assertTrue(np.allclose(values[800:], [0.01] * 800, rtol=0.0, atol=1e-12))


class ConfigurationTest(unittest.TestCase):
    def test_batch_size_override_uses_existing_tb_trajectory_field(self) -> None:
        class StructuredTB:
            __slots__ = ("trajectories_per_episode",)

            def __init__(self) -> None:
                self.trajectories_per_episode = 4

        cfg = SimpleNamespace(tb=StructuredTB())
        _set_trajectories_per_update(cfg, 32)
        self.assertEqual(cfg.tb.trajectories_per_episode, 32)
        self.assertFalse(hasattr(cfg.tb, "batch_size"))

    def test_scientific_cli_values_are_fixed_for_every_batch(self) -> None:
        for batch_size in BATCH_SIZES:
            args = SimpleNamespace(
                batch_size=batch_size, circuit="bc0", seed=2,
                max_trajectories=1, milestones=[1],
            )
            _configure(args, preflight=False)
            self.assertEqual(args.max_trajectories, MAX_TRAJECTORIES)
            self.assertEqual(args.milestones, list(MILESTONES))
            self.assertEqual(MAX_TRAJECTORIES // batch_size, args.max_trajectories // batch_size)
            self.assertEqual(64 % batch_size, 0)

    def test_preflight_is_fixed_to_seed_zero_and_128(self) -> None:
        args = SimpleNamespace(batch_size=32, circuit="dalu")
        _configure(args, preflight=True)
        self.assertEqual((args.seed, args.max_trajectories, args.milestones), (0, 128, [128]))

    def test_batch_pairing_fingerprint_excludes_batch_size(self) -> None:
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

        def resolve(batch_size: int) -> dict:
            tb = SimpleNamespace(
                trajectories_per_episode=batch_size,
                log_z_learning_rate=0.01, reward_alpha=4.0, reward_eps=1e-8,
                reward_improvement_clip=2.0, exploration_epsilon_enabled=True,
                exploration_epsilon_start=0.5, exploration_epsilon_end=0.01,
                exploration_warmup_episodes=20, exploration_decay_episodes=None,
            )
            cfg = SimpleNamespace(
                learning_rate=0.001, episodes=200, num_steps=20,
                available_actions=list(range(7)), tb=tb,
            )
            args = SimpleNamespace(
                variant="zcal", experiment_name="batch_size_influence",
                run_schema_version=5, log_z_learning_rate=0.01,
                config_name="tb_zhuDOP", seed=0, output_dir=Path("/tmp/test"),
                device="cpu", max_trajectories=1_600, schedule_trajectories=800,
                milestones=list(MILESTONES),
            )
            with patch.dict(sys.modules, {"omegaconf": SimpleNamespace(OmegaConf=FakeOmegaConf)}):
                return _resolved_configuration(args, cfg, Path("/tmp/bc0.blif"))

        one = resolve(1)
        thirty_two = resolve(32)
        self.assertNotEqual(
            one["scientific_configuration_fingerprint"],
            thirty_two["scientific_configuration_fingerprint"],
        )
        self.assertEqual(
            one["batch_pairing_configuration_fingerprint"],
            thirty_two["batch_pairing_configuration_fingerprint"],
        )


class DecisionTest(unittest.TestCase):
    def test_rms_improvement_on_one_circuit_selects_healthy_noninferior_batch(self) -> None:
        runs = _synthetic_runs()
        for seed in EXPECTED_SEEDS:
            runs[(1, "bc0", seed)]["milestones"][MAX_TRAJECTORIES]["fixed_uniform"]["residual"]["centered_rms"] = 0.9
        decision = classify_batch_sizes(runs)
        self.assertEqual(decision["selected_batch_size"], 1)
        self.assertEqual(decision["qualifying_batch_sizes"], [1])

    def test_hypervolume_improvement_can_qualify(self) -> None:
        runs = _synthetic_runs()
        for seed in EXPECTED_SEEDS:
            runs[(8, "dalu", seed)]["milestones"][MAX_TRAJECTORIES]["training_archive"]["hypervolume"] = 0.21
        decision = classify_batch_sizes(runs)
        self.assertIn(8, decision["qualifying_batch_sizes"])

    def test_smallest_qualifier_within_one_standard_error_is_selected(self) -> None:
        runs = _synthetic_runs()
        for batch_size in (8, 16):
            for seed in EXPECTED_SEEDS:
                runs[(batch_size, "bc0", seed)]["milestones"][MAX_TRAJECTORIES]["fixed_uniform"]["residual"]["centered_rms"] = 0.9
        decision = classify_batch_sizes(runs)
        self.assertEqual(decision["qualifying_batch_sizes"], [8, 16])
        self.assertEqual(decision["within_one_standard_error"], [8, 16])
        self.assertEqual(decision["selected_batch_size"], 8)

    def test_health_failure_rejects_candidate(self) -> None:
        runs = _synthetic_runs()
        for seed in EXPECTED_SEEDS:
            runs[(1, "bc0", seed)]["milestones"][MAX_TRAJECTORIES]["fixed_uniform"]["residual"]["centered_rms"] = 0.9
        runs[(1, "dalu", 0)]["milestones"][MAX_TRAJECTORIES]["fresh_on_policy"] = _validation(1.0, healthy=False)
        decision = classify_batch_sizes(runs)
        self.assertNotIn(1, decision["qualifying_batch_sizes"])

    def test_cross_circuit_hypervolume_harm_rejects_candidate(self) -> None:
        runs = _synthetic_runs()
        for seed in EXPECTED_SEEDS:
            runs[(1, "bc0", seed)]["milestones"][MAX_TRAJECTORIES]["fixed_uniform"]["residual"]["centered_rms"] = 0.9
            runs[(1, "dalu", seed)]["milestones"][MAX_TRAJECTORIES]["training_archive"]["hypervolume"] = 0.19
        decision = classify_batch_sizes(runs)
        self.assertNotIn(1, decision["qualifying_batch_sizes"])

    def test_no_improvement_retains_four(self) -> None:
        decision = classify_batch_sizes(_synthetic_runs())
        self.assertEqual(decision["selected_batch_size"], 4)
        self.assertEqual(decision["decision"], "reject_batch_size_influence_retain_four")

    def test_bootstrap_is_deterministic_and_uses_mean(self) -> None:
        first = paired_mean_bootstrap([0.0, 0.1, 0.2], seed=7)
        second = paired_mean_bootstrap([0.0, 0.1, 0.2], seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 0.1)

    def test_report_distinguishes_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(runs_root=[root / "runs"], output_dir=root / "report")
            self.assertEqual(report(args), 2)
            payload = json.loads((args.output_dir / "decision_summary.json").read_text())
            self.assertEqual(payload["failure_type"], "incomplete_run_matrix")


try:
    import pyspiel
except ImportError:
    pyspiel = None


@unittest.skipIf(pyspiel is None, "OpenSpiel Python bindings are unavailable")
class BatchSizeIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("GFLOWCIRCUIT_RUN_SLOW_INTEGRATION") == "1",
        "set GFLOWCIRCUIT_RUN_SLOW_INTEGRATION=1 for Experiment 5.5 integration",
    )
    def test_reduced_small_and_large_batches_and_resume(self) -> None:
        from src.experiments.tb_batch_size_influence import run_experiment

        with tempfile.TemporaryDirectory() as temporary:
            for batch_size in (1, 32):
                output = Path(temporary) / f"batch_{batch_size}"
                args = SimpleNamespace(
                    batch_size=batch_size, config_name="tb_zhuDOP", circuit="bc0", seed=0,
                    output_dir=output, max_trajectories=128, milestones=[128],
                    device="cpu", resume_checkpoint=None, _allow_test_budget=True,
                )
                self.assertEqual(run_experiment(args), 0)
                checkpoint = output / "checkpoints/trajectory_128.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(payload["counters"]["optimizer_updates"], 128 // batch_size)
                self.assertEqual(payload["counters"]["training_trajectories"], 128)
                args.resume_checkpoint = checkpoint
                self.assertEqual(run_experiment(args), 0)


if __name__ == "__main__":
    unittest.main()
