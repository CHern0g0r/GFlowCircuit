from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

from src.circuit_artifacts import sha256, write_json
from src.discovery_metrics import TrainingDiscoveryTracker, record_training_trajectory
from src.diversity_evaluation import aggregate, evaluate, hypervolume, population_metrics
from src.pareto_archive import PersistentCircuitArchive
from src.sampling_diversity import map_to_lut


def fake_export(*, circuit, actions, path, expected, num_steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([list(actions), expected]))
    return sha256(path)


class ArchiveFixture:
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.blif"
        self.source.write_text("fixture")

    def archive(self, name="archive", exporter=fake_export):
        return PersistentCircuitArchive(circuit=str(self.source), initial_size=100, initial_depth=100,
                                        root=self.root / name, metadata={"method": "ppo", "training_seed": 0,
                                                                        "run_id": 0},
                                        num_steps=1, exporter=exporter)


class PersistentArchiveTest(ArchiveFixture, TestCase):
    def test_admission_ties_retirement_and_reference(self):
        archive = self.archive()
        def record(s, d, a):
            return archive.record_terminal(initial_size=100, initial_depth=100,
                                           final_size=s, final_depth=d, actions=[a])
        self.assertTrue(record(80, 95, 0))
        self.assertTrue(record(95, 80, 1))
        self.assertTrue(record(90, 90, 2))  # dominates neither existing coordinate
        self.assertTrue(record(90, 90, 3))  # new serialized circuit at a tie
        self.assertTrue(record(90, 90, 3))  # repeat occurrence, no new artifact
        self.assertEqual(len(archive.artifacts), 4)
        self.assertEqual(archive.repeats, 1)
        self.assertFalse(record(96, 96, 4))
        self.assertTrue(record(70, 70, 5))
        self.assertTrue(record(110, 60, 6))
        archive.finalize()
        self.assertEqual([(p.size, p.depth) for p in archive.points], [(70, 70), (110, 60)])
        self.assertEqual(len(list((archive.root / "aig").glob("*.aig"))), 6)
        self.assertEqual(archive.snapshot()["active_generated_artifact_count"], 2)
        self.assertEqual(archive.attempted_trajectories, 8)
        self.assertTrue(json.loads((archive.root / "manifest.json").read_text())["complete"])
        self.assertEqual(len((archive.root / "training_terminals.jsonl").read_text().splitlines()), 8)
        with self.assertRaises(FileExistsError):
            self.archive()

    def test_failed_export_does_not_commit_front(self):
        archive = self.archive()
        archive.exporter = mock.Mock(side_effect=RuntimeError("export failed"))
        with self.assertRaisesRegex(RuntimeError, "export failed"):
            archive.record_terminal(initial_size=100, initial_depth=100,
                                    final_size=70, final_depth=70, actions=[0])
        self.assertEqual(archive.attempted_trajectories, 0)
        self.assertEqual(archive.nondominated_count, 1)
        self.assertFalse(json.loads((archive.root / "status.json").read_text())["complete"])

    def test_adapters_all_four_algorithms_and_calibration(self):
        for algorithm in ("reinforce", "drills_a2c", "ppo", "gflownet_tb"):
            tracker = TrainingDiscoveryTracker(initial_metrics={str(self.source): (100, 100)},
                archive_options={"root": self.root / algorithm, "num_steps": 2, "exporter": fake_export})
            values = dict(file_path=str(self.source), initial_size=100, initial_depth=100,
                          final_size=80, final_depth=90)
            if algorithm == "reinforce":
                trajectory = {**values, "actions_applied": [0, 1], "terminal": True}
            else:
                trajectory = SimpleNamespace(**values)
                steps = [SimpleNamespace(action=0, done=False), SimpleNamespace(action=1, done=True)]
                setattr(trajectory, "transitions" if algorithm == "ppo" else "steps", steps)
            record_training_trajectory(tracker, trajectory,
                                       origin="calibration" if algorithm == "gflownet_tb" else "training")
            archive = next(iter(tracker.archives.values()))
            self.assertEqual(next(iter(archive.artifacts.values()))["actions"], [0, 1])
            self.assertEqual(archive.attempted_trajectories, 1)
            with self.assertRaises(ValueError):
                record_training_trajectory(tracker, trajectory, origin="evaluation")
            tracker.finalize()

    def test_nonterminal_rejected(self):
        tracker = TrainingDiscoveryTracker(initial_metrics={str(self.source): (100, 100)},
            archive_options={"root": self.root / "adapter", "num_steps": 2, "exporter": fake_export})
        with self.assertRaises(ValueError):
            record_training_trajectory(tracker, dict(file_path=str(self.source), initial_size=100,
                initial_depth=100, final_size=90, final_depth=90, actions_applied=[0], terminal=False))


class DiversityMetricsTest(TestCase):
    def test_mapper_command_and_k_validation(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.blif"
            def run(command, **kwargs):
                self.assertIn("if -K 4", command[2])
                self.assertEqual(kwargs["timeout"], 7)
                path.write_text("# timestamp\n.model test\n.inputs a\n.outputs a\n.end\n")
                return SimpleNamespace(returncode=0, stdout="nd = 4 lev = 2", stderr="")
            with mock.patch("src.sampling_diversity.subprocess.run", side_effect=run):
                self.assertEqual(map_to_lut(abc_path=Path("abc"), aig_path=Path("input.aig"),
                                           lut_path=path, timeout_seconds=7, k=4), (4, 2))
            self.assertTrue(path.read_text().startswith(".model canonical"))
            with self.assertRaises(ValueError):
                map_to_lut(abc_path=Path("abc"), aig_path=Path("input.aig"),
                           lut_path=path, timeout_seconds=7, k=1)

    def test_hypervolume_and_counts(self):
        self.assertAlmostEqual(hypervolume([(80, 90), (90, 80), (110, 60)], (100, 100)), .03)
        self.assertEqual(hypervolume([], (100, 100)), 0)
        with self.assertRaises(ValueError):
            hypervolume([], (0, 100))
        rows = [dict(sample_index=i, aig_size=80, aig_depth=90, aig_sha256=str(i), actions=[i],
                     lut_size=5, lut_depth=2, lut_sha256="same", lut_path="winner.blif") for i in range(2)]
        metrics = population_metrics(rows, (100, 100), permutations=1)
        self.assertEqual(metrics["coordinate_count"], 1)
        self.assertEqual(metrics["unique_aig_artifacts"], 2)
        self.assertEqual(metrics["unique_lut_artifacts"], 1)
        self.assertEqual(metrics["front_generated_artifact_count"], 2)
        self.assertIsNone(metrics["rank_correlation"]["size"])
        self.assertEqual(metrics["quality"]["lut"]["best_size_depth_product"]["selected_sample_index"], 0)
        empty = population_metrics([], (100, 100), include_reference=True)
        self.assertEqual(empty["front_coordinate_count"], 1)
        self.assertEqual(empty["front_generated_artifact_count"], 0)
        self.assertIsNone(empty["quality"]["lut"]["best_size"])

    def test_aggregation_averages_within_models_first(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            groups = []
            for training_seed, values in ((0, [0, 10]), (1, [20, 40])):
                for evaluation_seed, value in enumerate(values):
                    groups.append({"metadata": {"method": "ppo", "circuit": "a", "training_seed": training_seed,
                                                 "evaluation_seed": evaluation_seed},
                                   "population": "final_sampling", "sample_budget": 10,
                                   "metrics": {"hypervolume": value}})
            write_json(path, {"complete": True, "groups": groups})
            result = aggregate([path])["groups"][0]["metrics"]["hypervolume"]
            self.assertEqual(result, {"mean": 17.5, "std": 12.5, "training_seed_count": 2})
            with self.assertRaises(ValueError):
                aggregate([path, path])


class MappingEvaluationTest(ArchiveFixture, TestCase):
    def test_retired_best_mapping_resume_and_identity(self):
        archive = self.archive()
        for index, point in enumerate(((80, 90), (70, 70))):
            archive.record_terminal(initial_size=100, initial_depth=100, final_size=point[0],
                                    final_depth=point[1], actions=[index])
        archive.finalize()
        abc = self.root / "abc"
        abc.write_text("fake executable")
        abc.chmod(0o700)
        def mapper(**kwargs):
            actions, _ = json.loads(kwargs["aig_path"].read_text())
            size = 1 if actions == [0] else 5
            kwargs["lut_path"].write_text(f"lut {size}")
            self.assertEqual(kwargs["k"], 4)
            return size, 2
        manifest = archive.root / "manifest.json"
        with mock.patch("src.diversity_evaluation.map_to_lut", side_effect=mapper) as mapping:
            result = evaluate([manifest], output_dir=self.root / "eval", abc_path=abc, k=4,
                              milestone_interval=1, permutations=1)
            self.assertTrue(result["complete"])
            historical = next(g for g in result["groups"] if g["population"] == "training_archive")
            self.assertEqual(historical["metrics"]["quality"]["lut"]["best_size_depth_product"]["value"], 2)
            final = next(g for g in result["groups"] if g["population"] == "training_final_front")
            self.assertEqual(final["metrics"]["quality"]["lut"]["best_size_depth_product"]["value"], 10)
            self.assertEqual(mapping.call_count, 3)  # two generated + original
            evaluate([manifest], output_dir=self.root / "eval", abc_path=abc, k=4,
                     milestone_interval=1, permutations=1, resume=True)
            self.assertEqual(mapping.call_count, 3)
            with self.assertRaises(ValueError):
                evaluate([manifest], output_dir=self.root / "eval", abc_path=abc, k=6, resume=True, milestone_interval=1, permutations=1)
            abc.write_text("changed mapper")
            with self.assertRaises(ValueError):
                evaluate([manifest], output_dir=self.root / "eval", abc_path=abc, k=4, resume=True, milestone_interval=1, permutations=1)

    def test_final_prefixes_duplicates_and_failed_resume(self):
        archive = self.archive()
        archive.record_terminal(initial_size=100, initial_depth=100, final_size=80, final_depth=90, actions=[0])
        archive.finalize()
        manifest_path = archive.root / "samples.json"
        manifest = json.loads((archive.root / "manifest.json").read_text())
        manifest["kind"] = "final_sampling"
        manifest["metadata"]["evaluation_seed"] = 0
        record = manifest["records"][0]
        manifest["records"] = [{**record, "sample_index": i} for i in range(200)]
        write_json(manifest_path, manifest)
        abc = self.root / "abc"
        abc.write_text("abc")
        abc.chmod(0o700)
        with mock.patch("src.diversity_evaluation.map_to_lut", side_effect=TimeoutError("timeout")):
            failed = evaluate([manifest_path], output_dir=self.root / "eval", abc_path=abc, permutations=1)
            self.assertFalse(failed["complete"])
            self.assertIsNone(failed["groups"][0]["metrics"]["quality"]["lut"]["best_size"])
        def mapper(**kwargs):
            kwargs["lut_path"].write_text("same lut")
            return 3, 2
        with mock.patch("src.diversity_evaluation.map_to_lut", side_effect=mapper) as mapping:
            result = evaluate([manifest_path], output_dir=self.root / "eval", abc_path=abc, resume=True, permutations=1)
            self.assertTrue(result["complete"])
            self.assertEqual(mapping.call_count, 2)
            self.assertEqual([g["metrics"]["occurrence_count"] for g in result["groups"]], [10, 50, 100, 200])
            self.assertEqual(result["groups"][-1]["metrics"]["unique_aig_artifacts"], 1)
