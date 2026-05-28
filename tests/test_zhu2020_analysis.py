from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.zhu2020_analysis import (
    EXPECTED_RESYN2_SEQUENCE,
    aggregate_report,
    audit_config,
    discover_reports,
    extract_resyn2_sequence,
)


def _minimal_report(*, run_name: str, final_size: int, run_id: int = 1) -> dict:
    return {
        "algorithm": "reinforce",
        "hydra_config": {
            "run_name": run_name,
            "algorithm": {"name": "reinforce"},
            "reward": {"type": "zhu_size"},
            "num_steps": 20,
            "episodes": 200,
            "gamma": 0.9,
            "policy_learning_rate": 8e-4,
            "value_learning_rate": 3e-3,
            "baseline": "zhu_resyn2",
            "paper_mode": {"per_circuit_mode": True, "num_runs": 10, "infer_rollouts": 10},
            "entropy_beta": 0.0,
            "clip_grad_norm_policy": None,
            "clip_grad_norm_value": None,
            "normalize_returns": False,
            "available_actions": [0, 1, 2, 3, 4],
            "encoder": {
                "type": "hybrid",
                "graph": {"type": "zhu_gcn"},
                "vector": {"source": "zhu10"},
            },
            "value": {"input": "zhu10"},
        },
        "runs": [
            {
                "run_idx": 0,
                "seed": run_id,
                "final_eval": {
                    "mean_normalized_improvement_vs_resyn2_2": 0.1,
                    "per_circuit": [
                        {
                            "file_path": "example.blif",
                            "initial_size": 100,
                            "initial_depth": 10,
                            "final_size": final_size,
                            "final_depth": 8,
                            "final_qor": final_size * 8,
                            "final_return": 0.1,
                            "comparable_return": 0.2,
                            "num_steps_taken": 20,
                            "size_reduction_pct": 100.0 * (100 - final_size) / 100,
                            "resyn2_1_size": 90,
                            "resyn2_2_size": 85,
                            "resyn2_inf_size": 80,
                            "normalized_improvement_vs_resyn2_2": (85 - final_size) / 100,
                        }
                    ],
                },
            }
        ],
    }


class Zhu2020AnalysisTest(unittest.TestCase):
    def test_zhu_size_reward_formula(self) -> None:
        rewards_path = REPO_ROOT / "src" / "models" / "rewards.py"
        spec = importlib.util.spec_from_file_location("rewards_for_test", rewards_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        rewards = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rewards)

        reward = rewards.ZhuSizeReward(100, 10, baseline_per_step=0.01)
        self.assertAlmostEqual(reward(90, 10, 100, 10), 0.09)
        self.assertAlmostEqual(reward.last_gain, 0.1)

    def test_extract_resyn2_sequence_from_source(self) -> None:
        path = REPO_ROOT / "src" / "baselines" / "resyn2.py"
        self.assertEqual(extract_resyn2_sequence(path), EXPECTED_RESYN2_SEQUENCE)
        self.assertEqual(
            (EXPECTED_RESYN2_SEQUENCE * 2)[:20],
            [
                0,
                1,
                2,
                0,
                1,
                3,
                0,
                4,
                3,
                0,
                0,
                1,
                2,
                0,
                1,
                3,
                0,
                4,
                3,
                0,
            ],
        )

    def test_discover_reports_prefers_matching_hydra_and_highest_id(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "output"
            outputs = tmp_path / "outputs"
            output.mkdir()
            low = _minimal_report(run_name="zhu2020_i10_reproduce_1", final_size=95, run_id=1)
            high = _minimal_report(run_name="zhu2020_i10_reproduce_2", final_size=75, run_id=2)
            (output / "zhu2020_i10_reproduce_1.json").write_text(json.dumps(low), encoding="utf-8")
            (output / "zhu2020_i10_reproduce_2.json").write_text(json.dumps(high), encoding="utf-8")
            hydra_dir = outputs / "12:00_zhu2020_i10_reproduce_2" / ".hydra"
            hydra_dir.mkdir(parents=True)
            (hydra_dir / "config.yaml").write_text("run_name: zhu2020_i10_reproduce_2\n", encoding="utf-8")

            selections = discover_reports(output, outputs)
            self.assertEqual(len(selections), 1)
            self.assertEqual(selections[0].selected_run_id, 2)
            self.assertEqual(selections[0].selected_hydra_config, hydra_dir / "config.yaml")
            self.assertEqual(selections[0].duplicate_paths, (output / "zhu2020_i10_reproduce_1.json",))

    def test_aggregate_report_and_config_audit(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            report = _minimal_report(run_name="zhu2020_apex1_reproduce_1", final_size=75)
            path = output / "zhu2020_apex1_reproduce_1.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            selection = discover_reports(output)[0]
            row = aggregate_report(selection)
            self.assertEqual(row["mean_final_size"], 75)
            self.assertEqual(row["win_rate_vs_resyn2_2"], 1.0)
            self.assertEqual(row["win_rate_vs_resyn2_inf"], 1.0)

            audit = audit_config(report)
            self.assertTrue(all(item["status"] == "pass" for item in audit))


if __name__ == "__main__":
    unittest.main()
