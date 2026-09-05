from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.sampling_diversity import (
    action_sequence_string,
    compute_group_metrics,
    discover_checkpoints,
    map_to_lut6,
    parse_abc_lut_stats,
    pareto_front,
    recipe_diversity_metrics,
    spearman_rank_correlation,
)


class SamplingDiversityMetricTest(unittest.TestCase):
    def test_action_sequence_uses_abc_names(self) -> None:
        self.assertEqual(
            action_sequence_string([1, 3, 0]),
            "rewrite;rewrite -z;balance",
        )

    def test_recipe_entropy_and_action_entropy(self) -> None:
        metrics = recipe_diversity_metrics(
            [[0, 1], [0, 1], [1, 1], [1, 0]],
            curve_permutations=10,
            seed=7,
        )
        self.assertTrue(math.isclose(metrics["recipe_entropy_nats"], 1.0397207708))
        per_position = metrics["action_entropy_nats"]["per_position"]
        self.assertTrue(math.isclose(per_position[0]["entropy_nats"], math.log(2.0)))
        self.assertGreater(per_position[1]["entropy_nats"], 0.0)
        curve = metrics["unique_recipe_fraction_curve"]
        self.assertEqual(len(curve), 4)
        self.assertEqual(curve[0]["mean"], 1.0)
        self.assertEqual(curve[-1]["mean"], 0.75)

    def test_spearman_uses_average_tie_ranks(self) -> None:
        self.assertTrue(
            math.isclose(
                spearman_rank_correlation([1, 1, 3], [2, 2, 8]) or 0.0,
                1.0,
            )
        )
        self.assertIsNone(spearman_rank_correlation([1, 1], [2, 3]))

    def test_pareto_front_preserves_all_samples_at_coordinate(self) -> None:
        rows = [
            {"sample_index": 0, "lut_size": 10, "lut_depth": 4},
            {"sample_index": 1, "lut_size": 8, "lut_depth": 5},
            {"sample_index": 2, "lut_size": 10, "lut_depth": 4},
            {"sample_index": 3, "lut_size": 12, "lut_depth": 6},
        ]
        self.assertEqual(
            pareto_front(rows, size_key="lut_size", depth_key="lut_depth"),
            [
                {"size": 8, "depth": 5, "sample_indices": [1]},
                {"size": 10, "depth": 4, "sample_indices": [0, 2]},
            ],
        )

    def test_group_metrics_include_requested_aig_and_lut_quality(self) -> None:
        rows = [
            {
                "sample_index": 0,
                "aig_size": 10,
                "aig_depth": 4,
                "lut_size": 8,
                "lut_depth": 3,
            },
            {
                "sample_index": 1,
                "aig_size": 8,
                "aig_depth": 5,
                "lut_size": 7,
                "lut_depth": 4,
            },
            {
                "sample_index": 2,
                "aig_size": 10,
                "aig_depth": 4,
                "lut_size": 8,
                "lut_depth": 3,
            },
        ]
        metrics = compute_group_metrics(
            rows,
            [[0, 1], [1, 0], [0, 1]],
            curve_permutations=5,
            seed=3,
        )
        self.assertEqual(metrics["aig_diversity"]["unique_size_depth_pairs"], 2)
        self.assertEqual(metrics["lut_diversity"]["unique_size_depth_pairs"], 2)
        self.assertEqual(metrics["quality"]["aig"]["best_size"]["value"], 8)
        self.assertEqual(
            metrics["quality"]["aig"]["best_size_depth_product"]["value"],
            40,
        )
        self.assertEqual(metrics["quality"]["lut"]["best_depth"]["value"], 3)
        self.assertEqual(
            metrics["quality"]["lut"]["best_size_depth_product"]["value"],
            24,
        )

    def test_parse_abc_lut_stats(self) -> None:
        output = "network : i/o = 4/2  lat = 0  nd = 17  edge = 31  lev = 5\n"
        self.assertEqual(parse_abc_lut_stats(output), (17, 5))
        with self.assertRaises(ValueError):
            parse_abc_lut_stats("ABC did not print network statistics")

    def test_lut_mapping_canonicalizes_blif_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aig_path = root / "sample.aig"
            lut_path = root / "sample.blif"
            aig_path.write_bytes(b"aig fixture")
            lut_path.write_text(
                "# generated header\n.model original_name\n.inputs a\n.outputs y\n.names a y\n1 1\n.end\n",
                encoding="utf-8",
            )
            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "network : nd = 3 edge = 4 lev = 2\n",
                    "stderr": "",
                },
            )()
            with patch(
                "src.sampling_diversity.subprocess.run", return_value=completed
            ) as run:
                self.assertEqual(
                    map_to_lut6(
                        abc_path=Path("/test/abc"),
                        aig_path=aig_path,
                        lut_path=lut_path,
                        timeout_seconds=5.0,
                    ),
                    (3, 2),
                )
            self.assertNotIn("# generated", lut_path.read_text(encoding="utf-8"))
            self.assertTrue(
                lut_path.read_text(encoding="utf-8").startswith(".model canonical\n")
            )
            command = run.call_args.args[0]
            self.assertEqual(command[0], "/test/abc")
            self.assertIn("if -K 6", command[2])


class SamplingDiversityInputTest(unittest.TestCase):
    def test_discovers_run_checkpoints_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "saved_models"
            for run_id in (10, 2, 0):
                checkpoint = root / f"run_{run_id}" / "last.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"fixture")
            discovered = discover_checkpoints(root)
            self.assertEqual(
                [path.parent.name for path in discovered], ["run_0", "run_2", "run_10"]
            )


if __name__ == "__main__":
    unittest.main()
