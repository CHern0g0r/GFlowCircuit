from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import yaml

from src.exploration_analysis import (
    analyze_stage,
    bootstrap_mean_ci,
    hypervolume_min_2d,
    pareto_front_min,
    select_settings,
)
from src.exploration_protocol import ExplorationProtocol, ProtocolError, Setting
from src.exploration_tuning import _validate_attempt, build_stage_manifest, run_task


def _approved_protocol() -> ExplorationProtocol:
    protocol = ExplorationProtocol.load()
    protocol.data["prerequisites"]["gfn_optimizer_health"] = {
        "status": "approved",
        "evidence_path": "/artifact/health/report.md",
        "approved_by": "tester",
        "approved_at": "2026-08-18T00:00:00Z",
    }
    return protocol


class ProtocolExpansionTest(TestCase):
    def test_pending_health_gate_blocks_gfn_stages(self) -> None:
        protocol = ExplorationProtocol.load()
        with self.assertRaisesRegex(ProtocolError, "not approved"):
            protocol.validate_health_gate("smoke")
        protocol.validate_health_gate("entropy_grid")

    def test_static_stage_counts_and_task_ids(self) -> None:
        protocol = _approved_protocol()
        expected = {
            "smoke": (5, 5),
            "coarse": (15, 90),
            "gfn_start": (5, 30),
            "entropy_grid": (12, 72),
            "pcn_seeds": (3, 18),
        }
        for stage, (setting_count, task_count) in expected.items():
            settings = protocol.settings_for_stage(stage)
            tasks = protocol.build_tasks(stage, settings)
            self.assertEqual(len(settings), setting_count)
            self.assertEqual(len(tasks), task_count)
            self.assertEqual(len({task.task_id for task in tasks}), task_count)
            self.assertTrue(all(" " not in task.task_id for task in tasks))

    def test_gfn_dynamic_grids_follow_previous_noncontrol(self) -> None:
        protocol = _approved_protocol()
        winner = next(
            setting for setting in protocol.settings_for_stage("gfn_start")
            if setting.setting_id == "start_0p25"
        )
        floor = protocol.settings_for_stage(
            "gfn_floor",
            selections={"gflownet": {"best_noncontrol": winner.to_dict()}},
        )
        self.assertEqual(len(floor), 4)
        self.assertEqual(
            {setting.overrides.get("tb.exploration_epsilon_end") for setting in floor if not setting.control},
            {0.0, 0.01, 0.05},
        )
        floor_winner = next(setting for setting in floor if setting.setting_id == "floor_0p01")
        schedules = protocol.settings_for_stage(
            "gfn_schedule",
            selections={"gflownet": {"best_noncontrol": floor_winner.to_dict()}},
        )
        self.assertEqual(len(schedules), 6)
        self.assertEqual(
            {
                (
                    setting.overrides["tb.exploration_warmup_episodes"],
                    setting.overrides["tb.exploration_decay_episodes"],
                )
                for setting in schedules if not setting.control
            },
            {(0, 25), (5, 15), (5, 25), (5, 40), (20, 20)},
        )

    def test_entropy_neighbors_and_confirmation_controls(self) -> None:
        protocol = _approved_protocol()
        grid = protocol.settings_for_stage("entropy_grid")
        selections = {}
        for algorithm in ("reinforce", "ppo", "drills"):
            selected = next(
                setting for setting in grid
                if setting.algorithm == algorithm and math.isclose(setting.exploration_rank, 0.001)
            )
            selections[algorithm] = {"selected": selected.to_dict()}
        neighbors = protocol.settings_for_stage("entropy_neighbors", selections=selections)
        for algorithm in selections:
            values = {
                round(float(next(iter(setting.overrides.values()))), 10)
                for setting in neighbors if setting.algorithm == algorithm
            }
            self.assertIn(round(0.001 / 3.0, 10), values)
            self.assertIn(0.003, values)

    def test_budget_translation(self) -> None:
        protocol = _approved_protocol()
        tasks = protocol.build_tasks("coarse", protocol.settings_for_stage("coarse"))
        episodes = {task.algorithm: task.episodes for task in tasks}
        self.assertEqual(episodes["gflownet"], 50)
        self.assertEqual(episodes["reinforce"], 200)
        self.assertEqual(episodes["ppo"], 50)
        self.assertEqual(episodes["drills"], 50)
        self.assertEqual(episodes["pcn"], 200)


class AnalysisTest(TestCase):
    def test_front_and_strict_hypervolume(self) -> None:
        points = [(0.8, 0.95), (0.9, 0.9), (0.95, 0.95), (0.8, 0.95)]
        self.assertEqual(pareto_front_min(points), [(0.8, 0.95), (0.9, 0.9)])
        self.assertTrue(math.isclose(hypervolume_min_2d(points), 0.015, abs_tol=1e-12))
        self.assertEqual(hypervolume_min_2d([(0.8, 1.0)]), 0.0)

    def test_selection_zero_hv_and_tie_breakers(self) -> None:
        low = Setting("algo", "low", {"x": 0.0}, exploration_rank=0.0, control=True)
        high = Setting("algo", "high", {"x": 1.0}, exploration_rank=1.0)
        rows = []
        for circuit in ("C1355", "dalu"):
            rows.extend(
                [
                    {
                        "algorithm": "algo",
                        "setting_id": "low",
                        "circuit": circuit,
                        "mean_hypervolume": 0.0,
                        "mean_product_improvement": 0.2,
                    },
                    {
                        "algorithm": "algo",
                        "setting_id": "high",
                        "circuit": circuit,
                        "mean_hypervolume": 0.0,
                        "mean_product_improvement": 0.2,
                    },
                ]
            )
        selected = select_settings(rows, [low, high], circuits=["C1355", "dalu"], tie_threshold=0.02)
        self.assertEqual(selected["algo"]["selected"]["setting_id"], "low")
        self.assertEqual(selected["algo"]["best_noncontrol"]["setting_id"], "high")

    def test_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_mean_ci([1.0, 2.0, 3.0], repetitions=1000, seed=20260818)
        second = bootstrap_mean_ci([1.0, 2.0, 3.0], repetitions=1000, seed=20260818)
        self.assertEqual(first, second)

    def test_stage_analysis_writes_reproducible_selection_artifacts(self) -> None:
        protocol = _approved_protocol()
        settings = protocol.settings_for_stage("gfn_start")
        tasks = protocol.build_tasks("gfn_start", settings)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "state": "prepared",
                "protocol_hash": protocol.protocol_hash,
                "project_commit": "abc123",
                "tasks": [task.to_dict() for task in tasks],
            }
            (root / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for task in tasks:
                attempt = root / "tasks" / task.task_id / "attempt_001"
                train = attempt / "train"
                train.mkdir(parents=True)
                is_target = task.setting.setting_id == "start_0p1"
                size = 90 if is_target else 95
                depth = 18 if is_target else 19
                with (train / "points.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["circuit", "run_id", "size", "depth"])
                    writer.writeheader()
                    writer.writerow({"circuit": task.circuit_path, "run_id": "", "size": 100, "depth": 20})
                    for _ in range(task.evaluation_samples):
                        writer.writerow({"circuit": task.circuit_path, "run_id": 0, "size": size, "depth": depth})
                status = {
                    "task_id": task.task_id,
                    "state": "complete",
                    "attempt_dir": str(attempt),
                    "wall_time_seconds": 1.0,
                }
                (attempt / "attempt.json").write_text(json.dumps(status), encoding="utf-8")
                (attempt.parent / "task_status.json").write_text(json.dumps(status), encoding="utf-8")
            result = analyze_stage(protocol=protocol, stage="gfn_start", artifact_root=root, settings=settings)
            self.assertEqual(result["algorithms"]["gflownet"]["selected"]["setting_id"], "start_0p1")
            for name in (
                "per_seed_metrics.csv",
                "setting_summary.csv",
                "pooled_front.csv",
                "discovery_curves.csv",
                "selection.json",
                "report.md",
            ):
                self.assertTrue((root / name).is_file(), name)


class RunnerTest(TestCase):
    def _fake_command(self, task, attempt_dir):
        calls = {"count": 0}

        def run(command, *, cwd, env, stdout_path, stderr_path):
            calls["count"] += 1
            stdout_path.write_text("ok\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            train_dir = attempt_dir / "train"
            if calls["count"] == 1:
                config = {
                    "seed": task.seed,
                    "episodes": task.episodes,
                    "num_steps": 20,
                    "available_actions": [0, 1, 2, 3, 4, 5, 6],
                    "paper_mode": {"num_runs": 1, "infer_rollouts": 1},
                }
                for path, value in {**task.setting.overrides, **task.fixed_overrides}.items():
                    current = config
                    components = path.split(".")
                    for component in components[:-1]:
                        current = current.setdefault(component, {})
                    current[components[-1]] = value
                (train_dir / ".hydra").mkdir(parents=True)
                (train_dir / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
                checkpoint = train_dir / "saved_models" / "run_0" / "last.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
                (train_dir / f"{task.report_algorithm}_report.json").write_text(
                    json.dumps({"algorithm": task.report_algorithm}), encoding="utf-8"
                )
            else:
                with (train_dir / "points.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["circuit", "run_id", "size", "depth"])
                    writer.writeheader()
                    writer.writerow({"circuit": task.circuit_path, "run_id": "", "size": 100, "depth": 20})
                    for _ in range(task.evaluation_samples):
                        writer.writerow({"circuit": task.circuit_path, "run_id": 0, "size": 90, "depth": 19})
            return 0

        return run

    def test_complete_task_is_reused_and_invalid_result_gets_new_attempt(self) -> None:
        protocol = _approved_protocol()
        task = protocol.build_tasks("smoke", protocol.settings_for_stage("smoke"))[1]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            attempt1 = root / "tasks" / task.task_id / "attempt_001"
            with patch("src.exploration_tuning._run_command", side_effect=self._fake_command(task, attempt1)):
                first = run_task(
                    task,
                    artifact_root=root,
                    protocol=protocol,
                    project_commit="abc123",
                    device="0",
                    python_executable="python",
                )
            self.assertEqual(first["state"], "complete")
            with patch("src.exploration_tuning._run_command") as command:
                second = run_task(
                    task,
                    artifact_root=root,
                    protocol=protocol,
                    project_commit="abc123",
                    device="0",
                    python_executable="python",
                )
            self.assertEqual(second["attempt_dir"], str(attempt1))
            command.assert_not_called()

            (attempt1 / "train" / "points.csv").write_text("bad\n", encoding="utf-8")
            attempt2 = root / "tasks" / task.task_id / "attempt_002"
            with patch("src.exploration_tuning._run_command", side_effect=self._fake_command(task, attempt2)):
                third = run_task(
                    task,
                    artifact_root=root,
                    protocol=protocol,
                    project_commit="abc123",
                    device="0",
                    python_executable="python",
                )
            self.assertEqual(third["attempt_dir"], str(attempt2))
            self.assertTrue(attempt1.exists())

    def test_manifest_reuses_matching_dependency_task(self) -> None:
        protocol = _approved_protocol()
        settings = protocol.settings_for_stage("gfn_start")
        tasks = protocol.build_tasks("gfn_start", settings)
        control = next(task for task in tasks if task.setting.control)
        with TemporaryDirectory() as directory:
            base = Path(directory)
            dependency = base / "dependency"
            source_attempt = dependency / "tasks" / control.task_id / "attempt_001"
            (source_attempt / "train").mkdir(parents=True)
            (source_attempt / "train" / "points.csv").write_text("x\n", encoding="utf-8")
            dependency_manifest = {
                "state": "complete",
                "protocol_hash": protocol.protocol_hash,
                "project_commit": "abc123",
                "tasks": [control.to_dict()],
            }
            (dependency / "stage_manifest.json").write_text(json.dumps(dependency_manifest), encoding="utf-8")
            status_path = dependency / "tasks" / control.task_id / "task_status.json"
            status_path.write_text(
                json.dumps({"state": "complete", "attempt_dir": str(source_attempt)}), encoding="utf-8"
            )
            target = base / "target"
            build_stage_manifest(
                protocol=protocol,
                stage="gfn_start",
                artifact_root=target,
                artifact_base=base,
                project_commit="abc123",
                settings=settings,
                dependency_roots=[dependency],
            )
            reused = json.loads(
                (target / "tasks" / control.task_id / "task_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reused["state"], "reused")
            self.assertEqual(reused["source_attempt"], str(source_attempt))

    def test_attempt_validation_rejects_missing_checkpoint_wrong_config_and_samples(self) -> None:
        protocol = _approved_protocol()
        task = protocol.build_tasks("smoke", protocol.settings_for_stage("smoke"))[1]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "tasks" / task.task_id / "attempt_001"
            with patch("src.exploration_tuning._run_command", side_effect=self._fake_command(task, attempt)):
                run_task(
                    task,
                    artifact_root=root,
                    protocol=protocol,
                    project_commit="abc123",
                    device="0",
                    python_executable="python",
                )
            checkpoint = attempt / "train" / "saved_models" / "run_0" / "last.pt"
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError, "one checkpoint"):
                _validate_attempt(task, attempt, protocol)
            checkpoint.write_bytes(b"checkpoint")

            config_path = attempt / "train" / ".hydra" / "config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["seed"] = 999
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config mismatch for seed"):
                _validate_attempt(task, attempt, protocol)
            config["seed"] = task.seed
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            points_path = attempt / "train" / "points.csv"
            rows = points_path.read_text(encoding="utf-8").splitlines()
            points_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                _validate_attempt(task, attempt, protocol)
