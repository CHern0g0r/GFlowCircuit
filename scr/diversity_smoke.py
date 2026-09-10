"""Real four-method smoke test, including archive-on/off learning equivalence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy  # Load the environment's BLAS runtime before PyTorch.
import torch

from src.circuit_artifacts import circuit_metrics, write_json
from src.diversity_evaluation import evaluate
from src.sample_exp import sample_paired_evaluation_seed_from_paths
from src.sampling_diversity import _abc_quote, _resolve_abc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--abc-path", required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    abc = _resolve_abc(args.abc_path)
    circuit = repo / "data/hdl-benchmarks/mcnc/Combinational/blif/C1355.blif"
    methods = {
        "reinforce": ("baseline_configs/reinforce", []),
        "drills": ("baseline_configs/drills", ["algorithm.drills.trajectories_per_episode=2"]),
        "ppo": ("baseline_configs/ppo", ["algorithm.ppo.rollout_steps=4", "algorithm.ppo.ppo_epochs=1",
                                           "algorithm.ppo.minibatch_size=4"]),
        "gflownet_tb": ("tb_zhuDOP_baseline_backbone", ["tb.trajectories_per_episode=2", "tb.calibration_trajectories=2"]),
    }
    summary = {}
    for method, (config, extra) in methods.items():
        for enabled in (False, True):
            train = root / method / ("archive" if enabled else "legacy")
            train.mkdir(parents=True)
            # Seed before model construction, identically in both comparisons.
            command = [sys.executable, "-c",
                       "import numpy,torch,runpy; torch.manual_seed(123); runpy.run_module('src.run',run_name='__main__')",
                       "--config-name", config, "dataset_cfg=cfg/data/zhu2020/C1355.yaml",
                       "seed_training_rng=true", "training_device=cpu", "num_steps=2", "episodes=2", "eval_every=2", "paper_mode.num_runs=1",
                       "paper_mode.infer_rollouts=1", "logging.tensorboard=false",
                       "discovery_metrics.emit_every_trajectories=1",
                       f"discovery_metrics.archive_enabled={str(enabled).lower()}",
                       f"hydra.run.dir={train}", f"output_dir={train}", *extra]
            with (train / "smoke.log").open("w") as log:
                subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env={**os.environ, "OMP_NUM_THREADS": "1"})
        paths = [root / method / mode / "saved_models/run_0/last.pt" for mode in ("legacy", "archive")]
        left, right = (torch.load(p, map_location="cpu", weights_only=False) for p in paths)
        for key in ("policy_state_dict", "value_state_dict"):
            if key in left:
                if not all(torch.equal(value, right[key][name]) for name, value in left[key].items()):
                    raise AssertionError(f"Archive capture changed {method} {key}")
        train = root / method / "archive"
        archives = list((train / "pareto_archives").glob("run_*/*/manifest.json"))
        archive = json.loads(archives[0].read_text())
        expected_count = 2 if method == "reinforce" else 4
        assert archive["snapshot"]["local_trajectory"] == expected_count
        sample_paired_evaluation_seed_from_paths(
            config_path=train / ".hydra/config.yaml", saved_models_dir=train / "saved_models",
            circuit_path=circuit, method=method, circuit_name="C1355", num_samples=4,
            evaluation_seed=0, device=torch.device("cpu"), num_steps=2, gflownet_batch_size=2,
            artifact_root=root / method / "samples")
        manifests = archives + list((root / method / "samples").glob("run_*/seed_*/manifest.json"))
        report = evaluate(manifests, output_dir=root / method / "evaluation", abc_path=abc,
                          budgets=[2, 4], milestone_interval=1, permutations=2)
        assert report["complete"]
        records = json.loads((root / method / "evaluation/mapping_records.json").read_text())
        checked = set()
        for record in records:
            assert circuit_metrics(Path(record["aig_path"]), 2) == (record["aig_size"], record["aig_depth"])
            for artifact in (record["aig_path"], record["lut_path"]):
                if artifact in checked:
                    continue
                command = f"cec -n {_abc_quote(circuit)} {_abc_quote(Path(artifact))}"
                result = subprocess.run([str(abc), "-c", command], capture_output=True, text=True, timeout=120)
                if result.returncode or "Networks are equivalent" not in result.stdout:
                    raise AssertionError(f"Equivalence failed for {artifact}: {result.stdout} {result.stderr}")
                checked.add(artifact)
        summary[method] = {"learning_unchanged": True, "training_trajectories": expected_count,
                           "saved_training_artifacts": len(archive["records"]),
                           "sample_occurrences": 4, "equivalence_checks": len(checked), "complete": True}
        write_json(root / "summary.json", summary)
        print(method, summary[method], flush=True)


if __name__ == "__main__":
    main()
