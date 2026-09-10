"""Fresh four-method training, paired sampling, and persistent LUT evaluation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.circuit_artifacts import sha256, write_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "cfg/exp/diversity_evaluation/protocol.yaml"


def load_protocol(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if data["version"] != 1 or data["methods"] != ["reinforce", "drills", "ppo", "gflownet_tb"]:
        raise ValueError("Unsupported diversity campaign")
    base_path, gfn_path = (REPO / data[k] for k in ("baseline_protocol", "gflownet_protocol"))
    base = yaml.safe_load(base_path.read_text())
    gfn = yaml.safe_load(gfn_path.read_text())
    if base["common"]["sample_budgets"] != gfn["common"]["sample_budgets"]:
        raise ValueError("Methods must share evaluation budgets")
    for key in ("training_seeds", "evaluation_seeds"):
        if base["common"][key] != list(range(10)) or gfn["common"][key] != list(range(10)):
            raise ValueError("Campaign requires ten paired training/evaluation seeds")
    if base["common"]["sample_budgets"] != [10, 50, 100, 200]:
        raise ValueError("Campaign requires nested budgets [10, 50, 100, 200]")
    if not data["archive"]["enabled"] or data["archive"]["emit_every_trajectories"] <= 0:
        raise ValueError("Campaign requires archive capture")
    if not 2 <= data["mapping"]["k"] <= 32 or data["mapping"]["timeout_seconds"] <= 0:
        raise ValueError("Invalid mapping configuration")
    if list(base["circuits"]) != list(gfn["circuits"]):
        raise ValueError("Methods must share circuit ordering")
    for circuit in base["circuits"].values():
        for field in ("dataset_cfg", "circuit_path"):
            if not (REPO / circuit[field]).is_file():
                raise FileNotFoundError(circuit[field])
    data.update(baseline=base, gflownet=gfn, protocol_sha256=sha256(path),
                source_protocol_sha256={str(p): sha256(p) for p in (base_path, gfn_path)})
    return data


def smoke_protocol(protocol: dict) -> dict:
    """Explicit test-only reduction; production protocol loading stays strict."""
    import copy
    protocol = copy.deepcopy(protocol)
    protocol["smoke"] = True
    protocol["archive"]["emit_every_trajectories"] = 1
    protocol["mapping"]["curve_permutations"] = 2
    for key in ("baseline", "gflownet"):
        protocol[key]["common"].update(training_seeds=[0], evaluation_seeds=[0],
                                      sample_budgets=[2, 4], max_samples_per_seed=4,
                                      training_trajectories=4, num_steps=2)
    return protocol


def smoke_overrides(method: str) -> dict:
    extra = {"num_steps": 2, "episodes": 4 if method == "reinforce" else 2,
             "eval_every": 2, "paper_mode.num_runs": 1, "paper_mode.infer_rollouts": 1,
             "logging.tensorboard": False}
    if method == "drills":
        extra["algorithm.drills.trajectories_per_episode"] = 2
    elif method == "ppo":
        extra.update({"algorithm.ppo.rollout_steps": 4, "algorithm.ppo.ppo_epochs": 1,
                      "algorithm.ppo.minibatch_size": 4})
    elif method == "gflownet_tb":
        extra.update({"tb.trajectories_per_episode": 2, "tb.calibration_trajectories": 2})
    return extra


def task_config(protocol: dict, method: str, circuit: str) -> tuple[str, dict, str]:
    if method not in protocol["methods"] or circuit not in protocol["baseline"]["circuits"]:
        raise ValueError("Unknown method/circuit")
    if method == "gflownet_tb":
        settings = protocol["gflownet"]["training"]
        return str(Path(settings["config"]).relative_to("cfg").with_suffix("")), settings["expected"], settings["report_file"]
    settings = protocol["baseline"]["methods"][method]
    return settings["config_name"], {**protocol["baseline"]["common"]["expected"], **settings["expected"]}, settings["report_file"]


def overrides(protocol: dict, method: str, circuit: str, train_dir: Path) -> list[str]:
    result = [f"dataset_cfg={protocol['baseline']['circuits'][circuit]['dataset_cfg']}",
            f"run_name=diversity_{method}_{circuit}", f"hydra.run.dir={train_dir}",
            f"output_dir={train_dir}", "discovery_metrics.enabled=true",
            "seed_training_rng=true",
            "discovery_metrics.archive_enabled=true",
            f"discovery_metrics.emit_every_trajectories={protocol['archive']['emit_every_trajectories']}"]
    if protocol.get("smoke"):
        result += [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in smoke_overrides(method).items()]
    return result


def validate(protocol: dict) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    count = 0
    with initialize_config_dir(config_dir=str(REPO / "cfg"), version_base=None):
        for method in protocol["methods"]:
            for circuit in protocol["baseline"]["circuits"]:
                config, expected, _ = task_config(protocol, method, circuit)
                cfg = compose(config_name=config, overrides=overrides(protocol, method, circuit, Path("/tmp/diversity-validation")))
                expected = {**expected, **(smoke_overrides(method) if protocol.get("smoke") else {})}
                for key, value in expected.items():
                    if OmegaConf.select(cfg, key) != value:
                        raise ValueError(f"{method}/{circuit}: {key} expected {value}, got {OmegaConf.select(cfg, key)}")
                count += 1
    return {"valid": True, "tasks": count, "training_seeds_per_task": len(protocol["baseline"]["common"]["training_seeds"]),
            "max_samples_per_evaluation_seed": protocol["baseline"]["common"]["max_samples_per_seed"]}


def run_task(protocol: dict, *, method: str, circuit: str, artifact_root: Path,
             abc_path: Path, device: str = "cpu", python: str = sys.executable,
             stage: str = "all") -> Path:
    from src.campaign_workers import attempt_identity, seal_inputs
    from src.sample_exp import sample_paired_evaluation_seed_from_paths
    import torch

    if stage not in {"all", "train-sample"}:
        raise ValueError("Invalid task stage")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    common = protocol["baseline"]["common"]
    config, _, report_file = task_config(protocol, method, circuit)
    task_root = artifact_root.resolve() / f"{method}_{circuit}"
    task_root.mkdir(parents=True, exist_ok=True)
    index = 0
    while (task_root / f"attempt_{index:03d}").exists():
        index += 1
    attempt = task_root / f"attempt_{index:03d}"
    attempt.mkdir()
    train_dir = attempt / "train"
    command = [python, "-m", "src.run", "--config-name", config, *overrides(protocol, method, circuit, train_dir), f"training_device={device}"]
    status = {"complete": False, "training_complete": False, "sampling_complete": False,
              "mapping_complete": False, "command": command, "protocol": protocol,
              "identity": attempt_identity(protocol, method, circuit, abc_path, device),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(attempt / "status.json", status)
    try:
        start = time.monotonic()
        with (attempt / "train.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        status["training_seconds"] = time.monotonic() - start
        checkpoints = sorted((train_dir / "saved_models").glob("run_*/last.pt"))
        archives = sorted((train_dir / "pareto_archives").glob("run_*/*/manifest.json"))
        if len(checkpoints) != len(common["training_seeds"]) or len(archives) != len(common["training_seeds"]) or not (train_dir / report_file).is_file():
            raise ValueError("Incomplete trained checkpoints and circuit archives")
        for path in archives:
            archive = json.loads(path.read_text())
            if not archive["complete"] or archive["snapshot"]["local_trajectory"] != common["training_trajectories"]:
                raise ValueError(f"Incomplete training budget/archive: {path}")
        status["training_complete"] = True
        write_json(attempt / "status.json", status)
        start = time.monotonic()
        for seed in protocol["baseline"]["common"]["evaluation_seeds"]:
            frame = sample_paired_evaluation_seed_from_paths(
                config_path=train_dir / ".hydra/config.yaml", saved_models_dir=train_dir / "saved_models",
                circuit_path=REPO / protocol["baseline"]["circuits"][circuit]["circuit_path"],
                method=method, circuit_name=circuit, num_samples=common["max_samples_per_seed"], evaluation_seed=seed,
                device=torch.device(device), num_steps=common["num_steps"],
                gflownet_batch_size=protocol["gflownet"]["common"]["gflownet_batch_size"],
                artifact_root=attempt / "final_sampling")
            frame.to_csv(attempt / f"samples_seed_{seed}.csv", index=False)
            for budget in protocol["baseline"]["common"]["sample_budgets"]:
                frame.loc[frame.sample_id < budget].to_csv(attempt / f"samples_seed_{seed}_n{budget}.csv", index=False)
        status["sampling_seconds"] = time.monotonic() - start
        status["sampling_complete"] = True
        seal_inputs(attempt, status)
        write_json(attempt / "status.json", status)
        if stage == "all":
            status = resume_mapping(attempt, abc_path)
    except Exception as exc:
        if "input_files" not in status:
            status["sampling_complete"] = False
        status["error"] = str(exc)
        raise
    finally:
        write_json(attempt / "status.json", status)
    return attempt


def resume_mapping(attempt: Path, abc_path: Path) -> dict:
    from src.diversity_evaluation import evaluate

    status_path = attempt / "status.json"
    status = json.loads(status_path.read_text())
    if not status["training_complete"] or not status["sampling_complete"]:
        raise ValueError("Only mapping can resume; interrupted training/sampling needs a fresh attempt")
    protocol = status["protocol"]
    manifests = sorted((attempt / "train/pareto_archives").glob("run_*/*/manifest.json"))
    manifests += sorted((attempt / "final_sampling").glob("run_*/seed_*/manifest.json"))
    try:
        if "identity" in status:
            from src.campaign_workers import validate_inputs
            validate_inputs(attempt, status)
        result = evaluate(manifests, output_dir=attempt / "diversity", abc_path=abc_path,
                          k=protocol["mapping"]["k"], timeout_seconds=protocol["mapping"]["timeout_seconds"],
                          budgets=protocol["baseline"]["common"]["sample_budgets"],
                          permutations=protocol["mapping"]["curve_permutations"],
                          milestone_interval=protocol["archive"]["emit_every_trajectories"],
                          resume=(attempt / "diversity/identity.json").exists())
        status["mapping_complete"] = status["complete"] = result["complete"]
        if not result["complete"]:
            raise RuntimeError("Mapping remains incomplete")
        status.pop("error", None)
        if "identity" in status:
            from src.campaign_workers import inventory
            status["mapping_files"] = inventory(attempt, [attempt / "diversity"])
    except Exception as exc:
        status["mapping_complete"] = status["complete"] = False
        status["error"] = str(exc)
        raise
    finally:
        write_json(status_path, status)
    return status


def sample_checkpoints(*, experiment: Path, circuit: Path, output_dir: Path,
                       evaluation_seeds: list[int], num_samples: int, device: str) -> None:
    from src.sample_exp import sample_paired_evaluation_seed
    from omegaconf import OmegaConf
    import torch

    cfg = OmegaConf.load(experiment / ".hydra/config.yaml")
    method = str(OmegaConf.select(cfg, "algorithm.name") or "reinforce")
    for seed in evaluation_seeds:
        sample_paired_evaluation_seed(
            experiment_dir=experiment, circuit_path=circuit.resolve(), method=method,
            circuit_name=circuit.stem, num_samples=num_samples, evaluation_seed=seed,
            device=torch.device(device), artifact_root=output_dir, gflownet_batch_size=20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--smoke", action="store_true", help="Use fixed reduced budgets for validation only")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("validate")
    run = subs.add_parser("run-task")
    run.add_argument("--method", required=True)
    run.add_argument("--circuit", required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--abc-path", type=Path, required=True)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--stage", choices=["all", "train-sample"], default="all")
    method = subs.add_parser("run-method")
    method.add_argument("--method", required=True)
    method.add_argument("--stage", choices=["train-sample", "mapping"], required=True)
    method.add_argument("--workers", type=int, default=2)
    method.add_argument("--circuits", nargs="+")
    method.add_argument("--artifact-root", type=Path, required=True)
    method.add_argument("--abc-path", type=Path, required=True)
    method.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    resume = subs.add_parser("resume-mapping")
    resume.add_argument("--attempt", type=Path, required=True)
    resume.add_argument("--abc-path", type=Path, required=True)
    sample = subs.add_parser("sample-checkpoints")
    sample.add_argument("--experiment", type=Path, required=True)
    sample.add_argument("--circuit", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(10)))
    sample.add_argument("--num-samples", type=int, default=200)
    sample.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "resume-mapping":
        resume_mapping(args.attempt.resolve(), args.abc_path)
        return
    if args.command == "sample-checkpoints":
        sample_checkpoints(experiment=args.experiment, circuit=args.circuit, output_dir=args.output_dir,
                           evaluation_seeds=args.evaluation_seeds, num_samples=args.num_samples, device=args.device)
        return
    protocol = load_protocol(args.protocol)
    if args.smoke:
        protocol = smoke_protocol(protocol)
    print(json.dumps(validate(protocol)))
    if args.command == "run-method":
        from src.campaign_workers import run_method
        result = run_method(protocol, protocol_path=args.protocol.resolve(), method=args.method,
                            stage=args.stage, workers=args.workers, circuits=args.circuits,
                            artifact_root=args.artifact_root, abc_path=args.abc_path, device=args.device)
        raise SystemExit(0 if result["complete"] else 1)
    if args.command == "run-task":
        print(run_task(protocol, method=args.method, circuit=args.circuit,
                       artifact_root=args.artifact_root, abc_path=args.abc_path, device=args.device, stage=args.stage))


if __name__ == "__main__":
    main()
