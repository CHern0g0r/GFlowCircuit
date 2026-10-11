import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from src import campaign_workers as w
from src.circuit_artifacts import sha256, write_json
from src.diversity_campaign import DEFAULT_PROTOCOL, load_protocol, smoke_protocol, validate


class WorkersTest(unittest.TestCase):
    def test_smoke_keeps_production_unchanged(self):
        production = load_protocol(DEFAULT_PROTOCOL)
        reduced = smoke_protocol(production)
        self.assertEqual(validate(reduced)["tasks"], 32)
        self.assertEqual(production["baseline"]["common"]["training_seeds"], list(range(10)))
        self.assertEqual(production["baseline"]["common"]["training_trajectories"], 800)

    def test_inventory_corruption_and_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "artifact.aig"
            file.write_bytes(b"aig")
            files = w.inventory(root, [file])
            w.verify_inventory(root, files)
            file.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                w.verify_inventory(root, files)
            with self.assertRaises(ValueError):
                w.verify_inventory(root, {"../escape": "digest"})
            file.unlink()
            with self.assertRaises(FileNotFoundError):
                w.verify_inventory(root, files)

    def test_lock_contention(self):
        with TemporaryDirectory() as tmp:
            lock = Path(tmp) / "method.lock"
            with w.method_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with w.method_lock(lock):
                        pass
            with w.method_lock(lock):
                pass

    def test_newest_matching_complete_attempt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, identity, complete in [(0, "same", True), (1, "same", False), (2, "changed", True)]:
                write_json(root / f"attempt_{index:03d}" / "status.json", {
                    "identity": identity, "training_complete": complete, "sampling_complete": complete})
            with mock.patch.object(w, "validate_inputs") as check:
                result = w.select_attempt(root, "same")
                self.assertEqual(result[0].name, "attempt_000")
                check.assert_called_once()
                self.assertIsNone(w.select_attempt(root, "missing"))
            with mock.patch.object(w, "validate_inputs", side_effect=ValueError("corrupt")):
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    w.select_attempt(root, "same")

    def test_mapping_completion_checks_inventory(self):
        with TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            write_json(attempt / "diversity/status.json", {"complete": True, "failed_count": 0})
            write_json(attempt / "diversity/metrics.json", {"complete": True, "failures": []})
            status = {"complete": True, "mapping_complete": True,
                      "mapping_files": w.inventory(attempt, [attempt / "diversity"])}
            self.assertTrue(w.mapping_complete(attempt, status))
            (attempt / "diversity/metrics.json").unlink()
            with self.assertRaises(FileNotFoundError):
                w.mapping_complete(attempt, status)

    def run_stub_supervisor(self, tmp, *, stage="train-sample", fail=None, stop=False):
        protocol = load_protocol(DEFAULT_PROTOCOL)
        selected = ["C1355", "C5315", "adder"]
        active, launched, finished = set(), [], set()
        peak = 0
        class Process:
            next_pid = 10000
            def __init__(self, circuit):
                nonlocal peak
                Process.next_pid += 1
                self.pid = Process.next_pid
                self.circuit, self.polls = circuit, 0
                active.add(circuit)
                peak = max(peak, len(active))
            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                active.discard(self.circuit)
                finished.add(self.circuit)
                return 7 if self.circuit == fail else 0
            def wait(self):
                return self.poll()
        def popen(command, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            if stage == "mapping":
                self.assertIn("resume-mapping", command)
                self.assertNotIn("run-task", command)
                circuit = Path(command[command.index("--attempt") + 1]).parent.name.split("_", 1)[1]
            else:
                self.assertIn("train-sample", command)
                self.assertEqual(command[command.index("--device") + 1], "cuda")
                circuit = command[command.index("--circuit") + 1]
            launched.append(circuit)
            return Process(circuit)
        def select(root, identity):
            circuit = root.name.split("_", 1)[1]
            if stage == "mapping" or circuit in finished:
                return root / "attempt_000", {"complete": True}
            return None
        def sleep(_):
            if stop:
                os.kill(os.getpid(), signal.SIGTERM)
        with mock.patch.object(w, "attempt_identity", return_value={}), \
                mock.patch.object(w, "select_attempt", side_effect=select), \
                mock.patch.object(w, "mapping_complete", side_effect=lambda a, s: a.parent.name.split("_", 1)[1] in finished), \
                mock.patch.object(w.subprocess, "Popen", side_effect=popen), \
                mock.patch.object(w.time, "sleep", side_effect=sleep), \
                mock.patch.object(w, "stop_processes", side_effect=lambda active: [stream.close() for _, stream in active.values()]):
            result = w.run_method(protocol, protocol_path=DEFAULT_PROTOCOL, method="reinforce", stage=stage,
                                  workers=2, circuits=list(reversed(selected)), artifact_root=Path(tmp),
                                  abc_path=Path("abc"))
        return result, launched, peak

    def test_bounded_workers_order_and_failure_isolation(self):
        with TemporaryDirectory() as tmp:
            result, launched, peak = self.run_stub_supervisor(tmp, fail="C1355")
            self.assertEqual(launched, ["C1355", "C5315", "adder"])
            self.assertEqual(peak, 2)
            self.assertFalse(result["complete"])
            self.assertEqual(result["circuits"]["C1355"]["exit_code"], 7)
            self.assertEqual(result["circuits"]["adder"]["state"], "complete")
            self.assertEqual(len({r["log"] for r in result["circuits"].values()}), 3)

    def test_mapping_workers_and_termination(self):
        with TemporaryDirectory() as tmp:
            result, _, peak = self.run_stub_supervisor(tmp, stage="mapping")
            self.assertTrue(result["complete"])
            self.assertEqual(peak, 2)
        with TemporaryDirectory() as tmp:
            result, launched, _ = self.run_stub_supervisor(tmp, stop=True)
            self.assertFalse(result["complete"])
            self.assertEqual(len(launched), 2)
            self.assertTrue(all(r["state"] == "interrupted" for r in result["circuits"].values()))

    def test_completed_training_skips_and_missing_mapping_fails(self):
        protocol = load_protocol(DEFAULT_PROTOCOL)
        with TemporaryDirectory() as tmp, mock.patch.object(w, "attempt_identity", return_value={}), \
                mock.patch.object(w, "select_attempt", return_value=(Path(tmp), {})), \
                mock.patch.object(w.subprocess, "Popen") as start:
            result = w.run_method(protocol, protocol_path=DEFAULT_PROTOCOL, method="ppo", stage="train-sample",
                                  workers=2, circuits=None, artifact_root=Path(tmp), abc_path=Path("abc"))
            self.assertEqual(len(result["circuits"]), 8)
            self.assertTrue(result["complete"])
            start.assert_not_called()
        with TemporaryDirectory() as tmp, mock.patch.object(w, "attempt_identity", return_value={}), \
                mock.patch.object(w, "select_attempt", return_value=None):
            result = w.run_method(protocol, protocol_path=DEFAULT_PROTOCOL, method="ppo", stage="mapping",
                                  workers=2, circuits=["C1355"], artifact_root=Path(tmp), abc_path=Path("abc"))
            self.assertFalse(result["complete"])

    def test_process_group_cleanup(self):
        with TemporaryDirectory() as tmp:
            stream = (Path(tmp) / "log").open("w")
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                       start_new_session=True, stdout=stream)
            w.stop_processes({"circuit": (process, stream)})
            self.assertIsNotNone(process.poll())
            self.assertTrue(stream.closed)

    def test_supervisor_import_does_not_load_native_runtimes(self):
        subprocess.run([sys.executable, "-c", "import sys; import src.campaign_workers; import src.diversity_evaluation; "
                        "assert 'torch' not in sys.modules and 'pyspiel' not in sys.modules"], check=True)


class StageTest(unittest.TestCase):
    def test_training_device_and_fresh_attempt_on_failure(self):
        from types import SimpleNamespace
        from src.diversity_campaign import run_task
        protocol = smoke_protocol(load_protocol(DEFAULT_PROTOCOL))
        sampler = SimpleNamespace(sample_paired_evaluation_seed_from_paths=mock.Mock())
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with mock.patch.dict(sys.modules, {"src.sample_exp": sampler, "torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                run_task(protocol, method="reinforce", circuit="C1355", artifact_root=Path("unused"),
                         abc_path=Path("abc"), device="cuda", stage="train-sample")
            with TemporaryDirectory() as tmp, mock.patch.object(w, "attempt_identity", return_value={}), \
                    mock.patch("src.diversity_campaign.subprocess.run", side_effect=RuntimeError("training failed")) as run:
                for index in range(2):
                    with self.assertRaisesRegex(RuntimeError, "training failed"):
                        run_task(protocol, method="reinforce", circuit="C1355", artifact_root=Path(tmp),
                                 abc_path=Path("abc"), device="cpu", stage="train-sample")
                    attempt = Path(tmp) / "reinforce_C1355" / f"attempt_{index:03d}"
                    status = json.loads((attempt / "status.json").read_text())
                    self.assertFalse(status["training_complete"])
                    self.assertFalse(status["mapping_complete"])
                    self.assertIn("training_device=cpu", run.call_args.args[0])
                sampler.sample_paired_evaluation_seed_from_paths.assert_not_called()

    def test_population_validation_and_hashes(self):
        from src.diversity_campaign import task_config
        with TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            protocol = smoke_protocol(load_protocol(DEFAULT_PROTOCOL))
            identity = {"method": "reinforce", "circuit": "C1355", "circuit_sha256": "source"}
            train = attempt / "train"
            config = train / ".hydra/config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("config")
            checkpoint = train / "saved_models/run_0/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("checkpoint")
            (train / task_config(protocol, "reinforce", "C1355")[2]).write_text("{}")
            paths = [train / "pareto_archives/run_0/C1355/manifest.json",
                     attempt / "final_sampling/run_0/seed_0/manifest.json"]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True)
                aig = path.parent / "reference.aig"
                aig.write_bytes(b"aig")
                row = {"aig_path": aig.name, "aig_sha256": sha256(aig)}
                data = {"complete": True, "kind": "final_sampling" if index else "training_archive",
                        "reference": row, "metadata": {"method": "reinforce", "source_sha256": "source",
                        "run_id": 0, "training_seed": 0, "evaluation_seed": 0,
                        "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(config)},
                        "records": [{**row, "sample_index": j} for j in range(4)] if index else [],
                        "snapshot": {"local_trajectory": 4}}
                write_json(path, data)
            for suffix in ("", "_n2", "_n4"):
                (attempt / f"samples_seed_0{suffix}.csv").write_text("sample_id\n")
            status = {"training_complete": True, "sampling_complete": True, "protocol": protocol,
                      "identity": identity}
            w.seal_inputs(attempt, status)
            w.validate_inputs(attempt, status)
            data = json.loads(paths[1].read_text())
            data["records"].pop()
            write_json(paths[1], data)
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                w.validate_inputs(attempt, status)
            with self.assertRaisesRegex(ValueError, "Incomplete ordered"):
                w.validate_inputs(attempt, status, verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
