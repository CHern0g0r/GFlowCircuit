from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.algorithms.gflownet_tb.exact import ExactTree, Prefix
from src.algorithms.gflownet_tb.supervised_exact import (
    ConditionalEvaluation,
    ConditionalGateConfig,
    ExactConditionalTargets,
    SchedulerMode,
    SupervisedRunResult,
    SupervisedTrainingConfig,
    build_exact_conditional_targets,
    classify_supervised_runs,
    train_supervised_case,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
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


def _reset_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _compose_project_config(config_name: str) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str((_repo_root() / "cfg").resolve())):
        return compose(config_name=config_name)


def _tb_value(cfg: Any, key: str, default: Any) -> Any:
    from omegaconf import OmegaConf

    tb_cfg = OmegaConf.select(cfg, "tb")
    if tb_cfg is None:
        tb_cfg = OmegaConf.select(cfg, "algorithm.tb")
    value = OmegaConf.select(tb_cfg, key) if tb_cfg is not None else None
    return default if value is None else value


def _state_dict_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().to(device="cpu").contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(str(tuple(value.shape)).encode("utf-8"))
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in module.state_dict().items()
    }


def _write_target_conditionals(
    path: Path,
    *,
    targets: ExactConditionalTargets,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prefix",
        "depth",
        "action",
        "legal",
        "target_probability",
        "target_entropy",
        "subtree_log_flow",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, prefix in enumerate(targets.prefixes):
            for action in range(targets.num_actions):
                writer.writerow(
                    {
                        "prefix": _sequence_text(prefix),
                        "depth": int(targets.depths[row].item()),
                        "action": action,
                        "legal": bool(targets.legal_mask[row, action].item()),
                        "target_probability": float(targets.probabilities[row, action].item()),
                        "target_entropy": float(targets.target_entropies[row].item()),
                        "subtree_log_flow": float(targets.subtree_log_flows[row].item()),
                    }
                )


def _write_conditional_predictions(
    path: Path,
    *,
    targets: ExactConditionalTargets,
    evaluation: ConditionalEvaluation,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "prefix",
        "depth",
        "action",
        "legal",
        "target_probability",
        "predicted_probability",
        "absolute_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, prefix in enumerate(targets.prefixes):
            for action in range(targets.num_actions):
                target = float(targets.probabilities[row, action].item())
                predicted = float(evaluation.predicted_conditionals[row, action].item())
                writer.writerow(
                    {
                        "prefix": _sequence_text(prefix),
                        "depth": int(targets.depths[row].item()),
                        "action": action,
                        "legal": bool(targets.legal_mask[row, action].item()),
                        "target_probability": target,
                        "predicted_probability": predicted,
                        "absolute_error": abs(predicted - target),
                    }
                )


def _write_terminal_distribution(
    path: Path,
    *,
    tree: ExactTree,
    evaluation: ConditionalEvaluation,
    terminal_metadata: dict[Prefix, dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    terminal_evaluation = evaluation.terminal_evaluation
    log_z = evaluation.metrics.terminal.log_z
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prefix in tree.terminal_prefixes:
            log_reward = tree.terminal_log_rewards[prefix]
            log_pf = terminal_evaluation.log_pf_by_terminal[prefix]
            writer.writerow(
                {
                    "sequence": _sequence_text(prefix),
                    "log_reward": log_reward,
                    "target_probability": tree.target_terminal_probs[prefix],
                    "predicted_probability": terminal_evaluation.predicted_terminal_probs[prefix],
                    "log_pf": log_pf,
                    "tb_residual": log_z + log_pf - log_reward,
                    **terminal_metadata[prefix],
                }
            )


def _write_checkpoint(
    path: Path,
    *,
    checkpoint_kind: str,
    run_name: str,
    state_dict: dict[str, torch.Tensor],
    training_config: SupervisedTrainingConfig,
    gates: ConditionalGateConfig,
    tree: ExactTree,
    metrics: dict[str, Any],
    initialization_checksum: str,
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_kind": checkpoint_kind,
            "run_name": run_name,
            "tree_summary": tree.summary(),
            "training_config": asdict(training_config),
            "gates": asdict(gates),
            "initialization_checksum": initialization_checksum,
            "policy_state_dict": state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "metrics": metrics,
        },
        path,
    )


def _run_one(
    *,
    seed: int,
    variant: SchedulerMode,
    policy: torch.nn.Module,
    prediction_fn: Any,
    tree: ExactTree,
    targets: ExactConditionalTargets,
    terminal_metadata: dict[Prefix, dict[str, float | int]],
    training_config: SupervisedTrainingConfig,
    gates: ConditionalGateConfig,
    output_dir: Path,
    metrics_path: Path,
    initialization_checksum: str,
) -> tuple[SupervisedRunResult, dict[str, Any]]:
    run_name = f"seed_{seed}_{variant}"

    def record(evaluation: ConditionalEvaluation) -> None:
        row = {
            "run": run_name,
            "seed": seed,
            "variant": variant,
            **evaluation.metrics.to_dict(),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        print(json.dumps(_json_safe(row), sort_keys=True), flush=True)

    result = train_supervised_case(
        tree=tree,
        targets=targets,
        policy=policy,
        prediction_fn=prediction_fn,
        config=training_config,
        gates=gates,
        on_evaluation=record,
    )
    conditional_path = output_dir / "conditional_predictions" / f"{run_name}.csv"
    terminal_path = output_dir / "terminal_distributions" / f"{run_name}.csv"
    best_checkpoint = output_dir / "checkpoints" / f"{run_name}_best.pt"
    final_checkpoint = output_dir / "checkpoints" / f"{run_name}_final.pt"
    _write_conditional_predictions(
        conditional_path,
        targets=targets,
        evaluation=result.final_evaluation,
    )
    _write_terminal_distribution(
        terminal_path,
        tree=tree,
        evaluation=result.final_evaluation,
        terminal_metadata=terminal_metadata,
    )
    _write_checkpoint(
        best_checkpoint,
        checkpoint_kind="best",
        run_name=run_name,
        state_dict=result.best_policy_state_dict,
        training_config=training_config,
        gates=gates,
        tree=tree,
        metrics=result.best_evaluation.metrics.to_dict(),
        initialization_checksum=initialization_checksum,
        optimizer_state_dict=result.best_optimizer_state_dict,
        scheduler_state_dict=result.best_scheduler_state_dict,
    )
    _write_checkpoint(
        final_checkpoint,
        checkpoint_kind="final",
        run_name=run_name,
        state_dict=result.final_policy_state_dict,
        training_config=training_config,
        gates=gates,
        tree=tree,
        metrics=result.final_evaluation.metrics.to_dict(),
        initialization_checksum=initialization_checksum,
        optimizer_state_dict=result.optimizer_state_dict,
        scheduler_state_dict=result.scheduler_state_dict,
    )
    summary = {
        "run": run_name,
        "seed": seed,
        "variant": variant,
        "passed": result.passed,
        "completed_updates": result.completed_updates,
        "optimizer_updates": result.completed_updates,
        "prefix_presentations": result.prefix_presentations,
        "first_passing_update": result.first_passing_update,
        "consecutive_passes": result.consecutive_passes,
        "best_update": result.best_update,
        "wall_time_seconds": result.final_evaluation.metrics.wall_time_seconds,
        "initialization_checksum": initialization_checksum,
        "numerical_failure": result.numerical_failure,
        "initial_metrics": result.initial_evaluation.metrics.to_dict(),
        "best_metrics": result.best_evaluation.metrics.to_dict(),
        "final_metrics": result.final_evaluation.metrics.to_dict(),
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "conditional_predictions": str(conditional_path),
        "terminal_distribution": str(terminal_path),
    }
    return result, summary


def _run_experiment(args: argparse.Namespace) -> int:
    from omegaconf import OmegaConf

    from src.algorithms.gflownet_tb.circuit_exact import (
        analyze_observation_aliases,
        enumerate_circuit_tree,
        neural_prediction_fn,
    )
    from src.algorithms.gflownet_tb.factory import build_tb_policy
    from src.models import reward_class_factory
    from src.utils import normalize_available_actions

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    seeds = [int(seed) for seed in args.seeds]
    variants = list(dict.fromkeys(args.variants))
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    if not seeds or not variants:
        raise ValueError("at least one seed and optimizer variant are required")
    device = _resolve_device(args.device)
    circuit_path = args.circuit
    if not circuit_path.is_absolute():
        circuit_path = (_repo_root() / circuit_path).resolve()
    if not circuit_path.is_file():
        raise FileNotFoundError(f"circuit file not found: {circuit_path}")

    cfg = _compose_project_config(args.config_name)
    cfg.output_dir = str(output_dir)
    reward_cfg = OmegaConf.to_container(cfg.reward, resolve=True)
    if not isinstance(reward_cfg, dict):
        raise TypeError("reward config must resolve to a mapping")
    reward_class = reward_class_factory(reward_cfg)
    available_actions_raw = OmegaConf.select(cfg, "available_actions")
    available_actions = (
        None
        if available_actions_raw is None
        else [int(value) for value in available_actions_raw]
    )

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
    tree = enumeration.tree
    if len(tree.nodes) != 400:
        raise RuntimeError(f"bc0 tree has {len(tree.nodes)} nonterminal prefixes, expected 400")
    if len(tree.terminal_log_rewards) != 2_401:
        raise RuntimeError(f"bc0 tree has {len(tree.terminal_log_rewards)} terminals, expected 2401")
    targets = build_exact_conditional_targets(tree)
    if targets.num_prefixes != 400 or targets.num_actions != 7:
        raise RuntimeError(
            f"supervised target shape is {(targets.num_prefixes, targets.num_actions)}, expected (400, 7)"
        )

    resolved_config = {
        "config_name": args.config_name,
        "circuit": str(circuit_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "seeds": seeds,
        "variants": variants,
        "learning_rate": args.learning_rate,
        "cosine_min_lr": args.cosine_min_lr,
        "prefix_batch_size": args.prefix_batch_size,
        "eval_every": args.eval_every,
        "max_updates": args.max_updates,
        "required_consecutive_passes": 3,
        "project_config": OmegaConf.to_container(cfg, resolve=True),
        "gates": asdict(ConditionalGateConfig()),
    }
    _write_json(output_dir / "resolved_config.json", resolved_config)
    _write_json(
        output_dir / "tree_summary.json",
        {
            **tree.summary(),
            "circuit_path": enumeration.circuit_path,
            "initial_size": enumeration.initial_size,
            "initial_depth": enumeration.initial_depth,
            "num_supervised_prefixes": targets.num_prefixes,
            "mean_target_entropy": targets.target_entropy,
            "reconstructed_target_probability_mass": sum(
                targets.reconstructed_terminal_probs.values()
            ),
        },
    )
    _write_json(output_dir / "observation_collisions.json", analyze_observation_aliases(tree))
    _write_target_conditionals(output_dir / "target_conditionals.csv", targets=targets)

    root_observation = tree.nodes[()].payload
    obs_dim = int(root_observation.obs_tensor.numel())
    node_dim = int(root_observation.graph.x.shape[1])
    edge_attr = root_observation.graph.edge_attr
    edge_dim = int(edge_attr.shape[1]) if edge_attr.dim() > 1 else 1
    normalized_actions = normalize_available_actions(available_actions, tree.num_actions)

    def build_policy() -> torch.nn.Module:
        return build_tb_policy(
            cfg,
            obs_dim=obs_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_actions=tree.num_actions,
            available_actions=normalized_actions,
        ).to(device)

    gates = ConditionalGateConfig()
    initialization_checksums: dict[str, Any] = {}
    run_results: list[SupervisedRunResult] = []
    run_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        _reset_torch_seed(seed)
        initial_policy = build_policy()
        initial_state = _cpu_state_dict(initial_policy)
        initial_checksum = _state_dict_checksum(initial_state)
        variant_checksums: dict[str, str] = {}
        del initial_policy

        for variant in variants:
            _reset_torch_seed(seed)
            policy = build_policy()
            policy.load_state_dict(initial_state, strict=True)
            loaded_checksum = _state_dict_checksum(_cpu_state_dict(policy))
            if loaded_checksum != initial_checksum:
                raise RuntimeError(
                    f"{variant} policy for seed {seed} did not receive the paired initialization"
                )
            variant_checksums[variant] = loaded_checksum
            training_config = SupervisedTrainingConfig(
                learning_rate=float(args.learning_rate),
                cosine_min_lr=float(args.cosine_min_lr),
                max_updates=int(args.max_updates),
                eval_every=int(args.eval_every),
                required_consecutive_passes=3,
                prefix_batch_size=int(args.prefix_batch_size),
                scheduler=variant,
                seed=seed,
            )
            result, summary = _run_one(
                seed=seed,
                variant=variant,
                policy=policy,
                prediction_fn=neural_prediction_fn(tree=tree, policy=policy),
                tree=tree,
                targets=targets,
                terminal_metadata=enumeration.terminal_metadata,
                training_config=training_config,
                gates=gates,
                output_dir=output_dir,
                metrics_path=metrics_path,
                initialization_checksum=initial_checksum,
            )
            run_results.append(result)
            run_summaries.append(summary)
            del policy
            if device.type == "cuda":
                torch.cuda.empty_cache()

        initialization_checksums[str(seed)] = {
            "base": initial_checksum,
            "variants": variant_checksums,
            "matched": all(value == initial_checksum for value in variant_checksums.values()),
        }
        _write_json(output_dir / "initialization_checksums.json", initialization_checksums)
    _write_json(output_dir / "initialization_checksums.json", initialization_checksums)

    status, exit_code, pass_counts = classify_supervised_runs(
        run_results,
        seeds=seeds,
        variants=variants,
    )
    if status == "execution_failure":
        fit_interpretation = "execution_failure"
    elif "fixed" in pass_counts and pass_counts["fixed"] == len(seeds):
        fit_interpretation = "fixed_recipe_fits"
    elif "cosine" in pass_counts and pass_counts["cosine"] == len(seeds):
        fit_interpretation = "cosine_only_fits"
    elif any(result.passed for result in run_results):
        fit_interpretation = "isolated_passes"
    else:
        fit_interpretation = "neither_recipe_fits"
    summary = {
        "status": status,
        "exit_code": exit_code,
        "fit_interpretation": fit_interpretation,
        "pass_counts": pass_counts,
        "num_seeds": len(seeds),
        "num_runs": len(run_results),
        "all_initializations_paired": all(
            row["matched"] for row in initialization_checksums.values()
        ),
        "runs": run_summaries,
        "metrics": str(metrics_path),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), flush=True)
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the bc0 exact conditional policy with supervised learning."
    )
    parser.add_argument("--config-name", default="tb_zhuDOP")
    parser.add_argument(
        "--circuit",
        type=Path,
        default=Path("data/hdl-benchmarks/mcnc/Combinational/blif/bc0.blif"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--variants",
        choices=("fixed", "cosine"),
        nargs="+",
        default=["fixed", "cosine"],
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cosine-min-lr", type=float, default=1e-5)
    parser.add_argument("--prefix-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-updates", type=int, default=20_000)
    parser.add_argument("--device", default="cuda")
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
