from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    import pyspiel
except ImportError:
    pyspiel = None


@unittest.skipIf(pyspiel is None, "OpenSpiel Python bindings are unavailable")
class ActiveBaselineCircuitIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.circuits = [
            cls.repo_root / "data/hdl-benchmarks/mcnc/Combinational/blif/bc0.blif",
            cls.repo_root / "data/hdl-benchmarks/mcnc/Combinational/blif/dalu.blif",
        ]
        try:
            pyspiel.load_game(
                "circuit", {"num_steps": 20, "file_path": str(cls.circuits[0])}
            )
        except Exception as exc:
            raise unittest.SkipTest(f"OpenSpiel circuit game is unavailable: {exc}") from exc

    def test_bc0_and_dalu_short_rollouts_are_legal_and_finite(self) -> None:
        from src.algorithms.gflownet_tb.sampler import sample_tb_trajectories
        from src.models.rewards import DiffOfProductReward

        class UniformPolicy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.log_z = torch.nn.Parameter(torch.tensor(0.0))
                self.num_actions = 7

            def forward(self, observations):
                return torch.zeros(len(observations), self.num_actions)

            def masked_probs(self, logits, legal_rows):
                probabilities = torch.zeros_like(logits)
                for row, legal in enumerate(legal_rows):
                    probabilities[row, legal] = 1.0 / len(legal)
                return probabilities

            def log_prob_legal_batch(self, logits, legal_rows, actions):
                return torch.tensor(
                    [-float(torch.log(torch.tensor(len(legal)))) for legal in legal_rows]
                )

        policy = UniformPolicy()
        available_actions = list(range(7))
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            trajectories = sample_tb_trajectories(
                file_paths=[str(path) for path in self.circuits],
                num_steps=20,
                policy=policy,
                reward_class=DiffOfProductReward,
                reward_alpha=4.0,
                reward_eps=1e-8,
                reward_improvement_clip=2.0,
                sample_actions=True,
                available_actions=available_actions,
                epsilon_uniform=0.5,
                action_generator=generator,
            )
        self.assertEqual(len(trajectories), 2)
        for trajectory in trajectories:
            self.assertEqual(len(trajectory.steps), 20)
            self.assertTrue(torch.isfinite(trajectory.log_pf_sum))
            self.assertTrue(all(step.action in step.legal_actions for step in trajectory.steps))
            self.assertEqual(float(trajectory.log_pb_sum), 0.0)

    @unittest.skipUnless(
        os.environ.get("GFLOWCIRCUIT_RUN_SLOW_INTEGRATION") == "1",
        "set GFLOWCIRCUIT_RUN_SLOW_INTEGRATION=1 for runner/checkpoint integration",
    )
    def test_reduced_resume_matches_uninterrupted_run_and_writes_artifacts(self) -> None:
        from src.experiments.tb_active_baseline import run_experiment

        def args(output_dir: Path, *, maximum: int, milestones: list[int], resume=None):
            return SimpleNamespace(
                config_name="tb_zhuDOP",
                circuit="bc0",
                seed=0,
                output_dir=output_dir,
                max_trajectories=maximum,
                schedule_trajectories=800,
                milestones=milestones,
                device="cpu",
                resume_checkpoint=resume,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            self.assertEqual(run_experiment(args(uninterrupted, maximum=8, milestones=[4, 8])), 0)
            self.assertEqual(run_experiment(args(resumed, maximum=4, milestones=[4])), 0)
            checkpoint_4 = resumed / "checkpoints/trajectory_4.pt"
            self.assertEqual(
                run_experiment(
                    args(resumed, maximum=8, milestones=[4, 8], resume=checkpoint_4)
                ),
                0,
            )
            full = torch.load(
                uninterrupted / "checkpoints/trajectory_8.pt",
                map_location="cpu",
                weights_only=False,
            )
            split = torch.load(
                resumed / "checkpoints/trajectory_8.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(full["counters"], split["counters"])
            self.assertEqual(full["archive"], split["archive"])
            self.assertTrue(
                torch.equal(
                    full["train_action_generator_state"],
                    split["train_action_generator_state"],
                )
            )
            for name in full["policy"]:
                self.assertTrue(torch.equal(full["policy"][name], split["policy"][name]), name)

            def assert_nested_equal(left, right):
                if isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right))
                elif isinstance(left, dict):
                    self.assertEqual(set(left), set(right))
                    for key in left:
                        assert_nested_equal(left[key], right[key])
                elif isinstance(left, (list, tuple)):
                    self.assertEqual(len(left), len(right))
                    for l_item, r_item in zip(left, right, strict=True):
                        assert_nested_equal(l_item, r_item)
                else:
                    self.assertEqual(left, right)

            assert_nested_equal(full["optimizer"], split["optimizer"])
            for artifact in (
                "resolved_config.json",
                "run_metadata.json",
                "fixed_validation.pt",
                "metrics.jsonl",
                "milestones.jsonl",
                "trajectories.jsonl",
                "run_summary.json",
            ):
                self.assertTrue((resumed / artifact).is_file(), artifact)


if __name__ == "__main__":
    unittest.main()
