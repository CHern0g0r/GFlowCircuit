from __future__ import annotations

import json
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
class LogZCalibrationCircuitIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.circuit = cls.repo_root / "data/hdl-benchmarks/mcnc/Combinational/blif/bc0.blif"
        try:
            pyspiel.load_game("circuit", {"num_steps": 20, "file_path": str(cls.circuit)})
        except Exception as exc:
            raise unittest.SkipTest(f"OpenSpiel circuit game is unavailable: {exc}") from exc

    @unittest.skipUnless(
        os.environ.get("GFLOWCIRCUIT_RUN_SLOW_INTEGRATION") == "1",
        "set GFLOWCIRCUIT_RUN_SLOW_INTEGRATION=1 for calibrated runner integration",
    )
    def test_variants_share_calibration_and_resume_is_exact(self) -> None:
        from src.experiments.tb_logz_calibration import run_experiment

        def args(output: Path, variant: str, maximum: int, milestones: list[int], resume=None):
            return SimpleNamespace(
                variant=variant,
                config_name="tb_zhuDOP",
                circuit="bc0",
                seed=0,
                output_dir=output,
                max_trajectories=maximum,
                schedule_trajectories=800,
                milestones=milestones,
                device="cpu",
                resume_checkpoint=resume,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            z0 = root / "z0"
            zcal = root / "zcal"
            self.assertEqual(run_experiment(args(z0, "z0", 68, [68])), 0)
            self.assertEqual(run_experiment(args(zcal, "zcal", 68, [68])), 0)
            z0_summary = json.loads((z0 / "run_summary.json").read_text(encoding="utf-8"))
            zcal_summary = json.loads((zcal / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(z0_summary["pre_calibration_parameter_checksum"],
                             zcal_summary["pre_calibration_parameter_checksum"])
            self.assertEqual(z0_summary["calibration_sequence_checksum"],
                             zcal_summary["calibration_sequence_checksum"])
            self.assertEqual(z0_summary["fixed_sequence_checksum"],
                             zcal_summary["fixed_sequence_checksum"])
            calibration = torch.load(zcal / "calibration.pt", map_location="cpu", weights_only=False)
            self.assertEqual(len(calibration["trajectories"]), 64)
            for trajectory in calibration["trajectories"]:
                self.assertEqual(len(trajectory.steps), 20)
                self.assertTrue(all(step.action in step.legal_actions for step in trajectory.steps))
                self.assertTrue(torch.isfinite(trajectory.log_pf_sum))

            uninterrupted = root / "uninterrupted"
            split = root / "split"
            self.assertEqual(run_experiment(args(uninterrupted, "zcal", 72, [68, 72])), 0)
            self.assertEqual(run_experiment(args(split, "zcal", 68, [68])), 0)
            checkpoint = split / "checkpoints/trajectory_68.pt"
            self.assertEqual(run_experiment(args(split, "zcal", 72, [68, 72], checkpoint)), 0)
            full = torch.load(uninterrupted / "checkpoints/trajectory_72.pt", map_location="cpu", weights_only=False)
            resumed = torch.load(split / "checkpoints/trajectory_72.pt", map_location="cpu", weights_only=False)
            self.assertEqual(full["counters"], resumed["counters"])
            self.assertEqual(full["archive"], resumed["archive"])
            self.assertTrue(torch.equal(full["train_action_generator_state"], resumed["train_action_generator_state"]))
            for name, value in full["policy"].items():
                self.assertTrue(torch.equal(value, resumed["policy"][name]), name)


if __name__ == "__main__":
    unittest.main()
