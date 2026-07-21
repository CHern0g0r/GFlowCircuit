from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.algorithms.gflownet_tb.exact import (
    BehaviorMode,
    ExactGateConfig,
    ExactRunResult,
    ExactTrainingConfig,
    ExactTree,
    Prefix,
    TabularPrefixPolicy,
    build_synthetic_tree,
    train_exact_case,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True), encoding="utf-8")


def _sequence_text(prefix: Prefix) -> str:
    return " ".join(str(action) for action in prefix)


def _write_distribution(
    *,
    path: Path,
    tree: ExactTree,
    result: ExactRunResult,
    terminal_metadata: dict[Prefix, dict[str, float | int]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation = result.final_evaluation
    metadata_fields = [
        "final_size",
        "final_depth",
        "improvement_raw",
        "improvement_clipped",
        "terminal_reward",
        "log_pb",
    ]
    fields = [
        "sequence",
        "log_reward",
        "target_probability",
        "predicted_probability",
        "log_pf",
        "tb_residual",
        *metadata_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prefix in tree.terminal_prefixes:
            log_reward = tree.terminal_log_rewards[prefix]
            log_pf = evaluation.log_pf_by_terminal[prefix]
            row: dict[str, Any] = {
                "sequence": _sequence_text(prefix),
                "log_reward": log_reward,
                "target_probability": tree.target_terminal_probs[prefix],
                "predicted_probability": evaluation.predicted_terminal_probs[prefix],
                "log_pf": log_pf,
                "tb_residual": float(result.final_evaluation.metrics.log_z + log_pf - log_reward),
            }
            if terminal_metadata is not None:
                row.update(terminal_metadata[prefix])
            writer.writerow(row)


def _run_case(
    *,
    case_name: str,
    tree: ExactTree,
    policy: torch.nn.Module,
    probability_fn: Any,
    behavior: BehaviorMode,
    training_config: ExactTrainingConfig,
    gates: ExactGateConfig,
    output_dir: Path,
    metrics_path: Path,
    terminal_metadata: dict[Prefix, dict[str, float | int]] | None = None,
) -> tuple[ExactRunResult, dict[str, Any]]:
    def record(metrics: Any) -> None:
        row = {
            "case": case_name,
            "tree": tree.name,
            "behavior": behavior,
            **metrics.to_dict(),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        print(json.dumps(_json_safe(row), sort_keys=True), flush=True)

    result = train_exact_case(
        tree=tree,
        policy=policy,
        probability_fn=probability_fn,
        behavior=behavior,
        config=training_config,
        gates=gates,
        on_evaluation=record,
    )
    distribution_path = output_dir / "distributions" / f"{case_name}.csv"
    _write_distribution(
        path=distribution_path,
        tree=tree,
        result=result,
        terminal_metadata=terminal_metadata,
    )
    checkpoint_path = output_dir / "checkpoints" / f"{case_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "case": case_name,
            "tree_summary": tree.summary(),
            "behavior": behavior,
            "training_config": asdict(training_config),
            "gates": asdict(gates),
            "completed_updates": result.completed_updates,
            "passed": result.passed,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": result.optimizer_state_dict,
            "final_metrics": result.final_evaluation.metrics.to_dict(),
        },
        checkpoint_path,
    )
    summary = {
        "case": case_name,
        "tree": tree.name,
        "behavior": behavior,
        "passed": result.passed,
        "completed_updates": result.completed_updates,
        "first_passing_update": result.first_passing_update,
        "consecutive_passes": result.consecutive_passes,
        "numerical_failure": result.numerical_failure,
        "final_metrics": result.final_evaluation.metrics.to_dict(),
        "checkpoint": str(checkpoint_path),
        "terminal_distribution": str(distribution_path),
    }
    return result, summary


def _reset_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tabular_probability_fn(policy: TabularPrefixPolicy) -> Any:
    return policy.probabilities


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _compose_project_config(config_name: str) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = str((_repo_root() / "cfg").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name)


def _tb_value(cfg: Any, key: str, default: Any) -> Any:
    from omegaconf import OmegaConf

    tb_cfg = OmegaConf.select(cfg, "tb")
    if tb_cfg is None:
        tb_cfg = OmegaConf.select(cfg, "algorithm.tb")
    value = OmegaConf.select(tb_cfg, key) if tb_cfg is not None else None
    return default if value is None else value


def _run_experiment(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    training_config = ExactTrainingConfig(
        batch_size=args.batch_size,
        max_updates=args.max_updates,
        eval_every=args.eval_every,
        required_consecutive_passes=3,
        learning_rate=0.001,
        log_z_learning_rate=0.01,
        seed=args.seed,
    )
    gates = ExactGateConfig()
    resolved_config: dict[str, Any] = {
        "mode": args.mode,
        "config_name": args.config_name,
        "circuit": str(args.circuit),
        "output_dir": str(output_dir),
        "device": args.device,
        "training": asdict(training_config),
        "gates": asdict(gates),
    }
    tree_summaries: dict[str, Any] = {}
    case_summaries: list[dict[str, Any]] = []
    tabular_results: list[ExactRunResult] = []
    neural_results: list[ExactRunResult] = []

    if args.mode in ("synthetic", "all"):
        synthetic_tree = build_synthetic_tree(depth=4)
        if len(synthetic_tree.terminal_log_rewards) != 41:
            raise RuntimeError(
                f"synthetic tree has {len(synthetic_tree.terminal_log_rewards)} terminals, expected 41"
            )
        tree_summaries["synthetic"] = synthetic_tree.summary()
        for behavior in ("on_policy", "epsilon_0.5", "uniform"):
            _reset_torch_seed(args.seed)
            policy = TabularPrefixPolicy(synthetic_tree).to(torch.device("cpu"))
            result, summary = _run_case(
                case_name=f"synthetic_tabular_{behavior}",
                tree=synthetic_tree,
                policy=policy,
                probability_fn=_tabular_probability_fn(policy),
                behavior=behavior,
                training_config=training_config,
                gates=gates,
                output_dir=output_dir,
                metrics_path=metrics_path,
            )
            tabular_results.append(result)
            case_summaries.append(summary)

    if args.mode in ("circuit", "all"):
        from omegaconf import OmegaConf

        from src.algorithms.gflownet_tb.circuit_exact import (
            analyze_observation_aliases,
            enumerate_circuit_tree,
            neural_probability_fn,
        )
        from src.algorithms.gflownet_tb.factory import build_tb_policy
        from src.models import reward_class_factory
        from src.utils import normalize_available_actions

        cfg = _compose_project_config(args.config_name)
        # ``cfg/default.yaml`` normally gets this value from Hydra's runtime.
        # This standalone CLI composes the config without launching a Hydra job,
        # so bind it explicitly before resolving the config snapshot.
        cfg.output_dir = str(output_dir)
        resolved_config["project_config"] = OmegaConf.to_container(cfg, resolve=True)
        reward_cfg = OmegaConf.to_container(cfg.reward, resolve=True)
        if not isinstance(reward_cfg, dict):
            raise TypeError("reward config must resolve to a mapping")
        reward_class = reward_class_factory(reward_cfg)
        available_actions_raw = OmegaConf.select(cfg, "available_actions")
        available_actions = None if available_actions_raw is None else [int(v) for v in available_actions_raw]

        circuit_path = args.circuit
        if not circuit_path.is_absolute():
            circuit_path = (_repo_root() / circuit_path).resolve()
        if not circuit_path.is_file():
            raise FileNotFoundError(f"circuit file not found: {circuit_path}")
        resolved_config["circuit"] = str(circuit_path)

        enumeration = enumerate_circuit_tree(
            circuit_path=circuit_path,
            horizon=4,
            reward_class=reward_class,
            reward_alpha=float(_tb_value(cfg, "reward_alpha", 4.0)),
            reward_eps=float(_tb_value(cfg, "reward_eps", 1e-8)),
            reward_improvement_clip=float(_tb_value(cfg, "reward_improvement_clip", 2.0)),
            available_actions=available_actions,
            require_constant_branching=7,
        )
        circuit_tree = enumeration.tree
        if len(circuit_tree.nodes) != 400:
            raise RuntimeError(f"circuit tree has {len(circuit_tree.nodes)} nonterminals, expected 400")
        if len(circuit_tree.terminal_log_rewards) != 2_401:
            raise RuntimeError(
                f"circuit tree has {len(circuit_tree.terminal_log_rewards)} terminals, expected 2401"
            )
        tree_summaries["circuit"] = {
            **circuit_tree.summary(),
            "circuit_path": enumeration.circuit_path,
            "initial_size": enumeration.initial_size,
            "initial_depth": enumeration.initial_depth,
        }

        alias_report = analyze_observation_aliases(circuit_tree)
        _write_json(output_dir / "observation_collisions.json", alias_report)

        _reset_torch_seed(args.seed)
        tabular_policy = TabularPrefixPolicy(circuit_tree).to(torch.device("cpu"))
        tabular_result, tabular_summary = _run_case(
            case_name="circuit_tabular_epsilon_0.5",
            tree=circuit_tree,
            policy=tabular_policy,
            probability_fn=_tabular_probability_fn(tabular_policy),
            behavior="epsilon_0.5",
            training_config=training_config,
            gates=gates,
            output_dir=output_dir,
            metrics_path=metrics_path,
            terminal_metadata=enumeration.terminal_metadata,
        )
        tabular_results.append(tabular_result)
        case_summaries.append(tabular_summary)

        root_observation = circuit_tree.nodes[()].payload
        obs_dim = int(root_observation.obs_tensor.numel())
        node_dim = int(root_observation.graph.x.shape[1])
        edge_attr = root_observation.graph.edge_attr
        edge_dim = int(edge_attr.shape[1]) if edge_attr.dim() > 1 else 1
        num_actions = int(circuit_tree.num_actions)
        normalized_actions = normalize_available_actions(available_actions, num_actions)
        device = _resolve_device(args.device)
        _reset_torch_seed(args.seed)
        neural_policy = build_tb_policy(
            cfg,
            obs_dim=obs_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_actions=num_actions,
            available_actions=normalized_actions,
        ).to(device)
        neural_result, neural_summary = _run_case(
            case_name="circuit_contract_epsilon_0.5",
            tree=circuit_tree,
            policy=neural_policy,
            probability_fn=neural_probability_fn(tree=circuit_tree, policy=neural_policy),
            behavior="epsilon_0.5",
            training_config=training_config,
            gates=gates,
            output_dir=output_dir,
            metrics_path=metrics_path,
            terminal_metadata=enumeration.terminal_metadata,
        )
        neural_results.append(neural_result)
        neural_summary["observation_collision_groups"] = alias_report["num_collision_groups"]
        neural_summary["conflicting_observation_groups"] = alias_report["num_conflicting_groups"]
        case_summaries.append(neural_summary)

    _write_json(output_dir / "resolved_config.json", resolved_config)
    _write_json(output_dir / "tree_summary.json", tree_summaries)

    implementation_pass = bool(tabular_results) and all(result.passed for result in tabular_results)
    representation_pass: bool | None = None
    if neural_results:
        representation_pass = all(result.passed for result in neural_results)

    if not implementation_pass:
        exit_code = 2
        status = "implementation_failure"
    elif representation_pass is False:
        exit_code = 3
        status = "representation_failure"
    else:
        exit_code = 0
        status = "success"
    summary = {
        "status": status,
        "exit_code": exit_code,
        "implementation_pass": implementation_pass,
        "representation_pass": representation_pass,
        "cases": case_summaries,
        "metrics": str(metrics_path),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact correctness tests for the TB GFlowNet.")
    parser.add_argument("--mode", choices=("synthetic", "circuit", "all"), default="all")
    parser.add_argument("--config-name", default="tb_zhuDOP")
    parser.add_argument(
        "--circuit",
        type=Path,
        default=Path("data/hdl-benchmarks/mcnc/Combinational/blif/bc0.blif"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-updates", type=int, default=20_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run_experiment(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "execution_failure",
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_dir / "summary.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
