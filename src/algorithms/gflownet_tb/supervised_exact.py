from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

import torch
from torch import nn

from src.algorithms.gflownet_tb.exact import (
    ExactEvaluation,
    ExactGateConfig,
    ExactMetrics,
    ExactTree,
    Prefix,
    evaluate_exact_policy,
)
from src.algorithms.gflownet_tb.optim import build_policy_optimizer

SchedulerMode = Literal["fixed", "cosine"]
PredictionFn = Callable[[list[Prefix]], tuple[torch.Tensor, torch.Tensor]]
EvaluationCallback = Callable[["ConditionalEvaluation"], None]


@dataclass(frozen=True)
class ExactConditionalTargets:
    prefixes: tuple[Prefix, ...]
    probabilities: torch.Tensor
    legal_mask: torch.Tensor
    depths: torch.Tensor
    target_entropies: torch.Tensor
    target_entropy: float
    subtree_log_flows: torch.Tensor
    reconstructed_terminal_probs: dict[Prefix, float]

    @property
    def num_prefixes(self) -> int:
        return len(self.prefixes)

    @property
    def num_actions(self) -> int:
        return int(self.probabilities.shape[1])


@dataclass(frozen=True)
class ConditionalGateConfig:
    normalization_error: float = 1e-6
    illegal_probability: float = 1e-6
    mean_prefix_tv: float = 0.005
    max_prefix_tv: float = 0.02
    exact: ExactGateConfig = field(default_factory=ExactGateConfig)


@dataclass(frozen=True)
class ConditionalMetrics:
    update: int
    prefix_presentations: int
    learning_rate: float
    wall_time_seconds: float
    supervised_loss: float
    mean_prefix_tv: float
    max_prefix_tv: float
    mean_conditional_kl: float
    max_conditional_kl: float
    max_conditional_probability_error: float
    depth_metrics: dict[int, dict[str, float | int]]
    max_action_normalization_error: float
    max_illegal_probability: float
    logits_finite: bool
    probabilities_finite: bool
    loss_finite: bool
    gradients_finite: bool
    parameters_finite: bool
    finite: bool
    terminal: ExactMetrics
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionalEvaluation:
    metrics: ConditionalMetrics
    predicted_conditionals: torch.Tensor
    terminal_evaluation: ExactEvaluation


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    learning_rate: float = 0.001
    cosine_min_lr: float = 1e-5
    max_updates: int = 20_000
    eval_every: int = 100
    required_consecutive_passes: int = 3
    prefix_batch_size: int = 128
    scheduler: SchedulerMode = "fixed"
    seed: int = 0


@dataclass
class SupervisedRunResult:
    variant: SchedulerMode
    seed: int
    passed: bool
    completed_updates: int
    prefix_presentations: int
    first_passing_update: int | None
    consecutive_passes: int
    initial_evaluation: ConditionalEvaluation
    final_evaluation: ConditionalEvaluation
    best_evaluation: ConditionalEvaluation
    best_update: int
    evaluations: list[ConditionalMetrics]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    best_optimizer_state_dict: dict[str, Any]
    best_scheduler_state_dict: dict[str, Any] | None
    best_policy_state_dict: dict[str, torch.Tensor]
    final_policy_state_dict: dict[str, torch.Tensor]
    numerical_failure: str | None = None


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in module.state_dict().items()
    }


def build_exact_conditional_targets(
    tree: ExactTree,
    *,
    tolerance: float = 1e-10,
) -> ExactConditionalTargets:
    """Materialize and independently validate all exact conditional targets."""
    tolerance = float(tolerance)
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    prefixes = tuple(tree.nonterminal_prefixes)
    prefix_to_row = {prefix: row for row, prefix in enumerate(prefixes)}
    probabilities = torch.zeros(
        (len(prefixes), tree.num_actions),
        dtype=torch.float64,
    )
    legal_mask = torch.zeros_like(probabilities, dtype=torch.bool)
    depths = torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.long)
    subtree_log_flows = torch.tensor(
        [tree.subtree_log_flows[prefix] for prefix in prefixes],
        dtype=torch.float64,
    )

    for row, prefix in enumerate(prefixes):
        legal_actions = tree.nodes[prefix].legal_actions
        targets = tree.target_action_probs(prefix)
        for action in legal_actions:
            probabilities[row, action] = float(targets[action])
            legal_mask[row, action] = True
        legal_mass = float(probabilities[row, legal_mask[row]].sum().item())
        if abs(legal_mass - 1.0) > tolerance:
            raise ValueError(
                f"target probabilities at prefix {prefix} sum to {legal_mass}, expected 1"
            )
        if bool(torch.any(probabilities[row, ~legal_mask[row]] != 0.0).item()):
            raise ValueError(f"target probabilities assign mass to an illegal action at {prefix}")

    positive_targets = probabilities.clamp_min(torch.finfo(torch.float64).tiny)
    target_entropies = -(probabilities * positive_targets.log()).sum(dim=1)
    reconstructed: dict[Prefix, float] = {}
    for terminal in tree.terminal_prefixes:
        probability = 1.0
        for depth, action in enumerate(terminal):
            prefix = terminal[:depth]
            probability *= float(probabilities[prefix_to_row[prefix], action].item())
        reconstructed[terminal] = probability
        target_probability = tree.target_terminal_probs[terminal]
        if abs(probability - target_probability) > tolerance:
            raise ValueError(
                "exact conditionals do not reconstruct terminal probability for "
                f"{terminal}: {probability} != {target_probability}"
            )

    reconstructed_mass = sum(reconstructed.values())
    if abs(reconstructed_mass - 1.0) > tolerance:
        raise ValueError(
            f"reconstructed terminal probabilities sum to {reconstructed_mass}, expected 1"
        )

    return ExactConditionalTargets(
        prefixes=prefixes,
        probabilities=probabilities,
        legal_mask=legal_mask,
        depths=depths,
        target_entropies=target_entropies,
        target_entropy=float(target_entropies.mean().item()),
        subtree_log_flows=subtree_log_flows,
        reconstructed_terminal_probs=reconstructed,
    )


def _validate_prediction_shapes(
    *,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    batch_size: int,
    num_actions: int,
) -> None:
    expected = (int(batch_size), int(num_actions))
    if tuple(logits.shape) != expected:
        raise ValueError(f"prediction_fn returned logits shape {tuple(logits.shape)}, expected {expected}")
    if tuple(probabilities.shape) != expected:
        raise ValueError(
            f"prediction_fn returned probabilities shape {tuple(probabilities.shape)}, expected {expected}"
        )


def _probability_fn(prediction_fn: PredictionFn) -> Callable[[list[Prefix]], torch.Tensor]:
    def probabilities(prefixes: list[Prefix]) -> torch.Tensor:
        return prediction_fn(prefixes)[1]

    return probabilities


def evaluate_conditional_policy(
    *,
    tree: ExactTree,
    targets: ExactConditionalTargets,
    policy: nn.Module,
    prediction_fn: PredictionFn,
    update: int,
    prefix_presentations: int = 0,
    learning_rate: float = 0.0,
    wall_time_seconds: float = 0.0,
    gates: ConditionalGateConfig | None = None,
    loss_finite: bool = True,
    gradients_finite: bool = True,
    parameters_finite: bool = True,
    evaluation_batch_size: int = 256,
) -> ConditionalEvaluation:
    gates = gates or ConditionalGateConfig()
    evaluation_batch_size = int(evaluation_batch_size)
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    if tuple(targets.prefixes) != tuple(tree.nonterminal_prefixes):
        raise ValueError("conditional targets are not aligned with the tree prefix order")

    was_training = policy.training
    policy.eval()
    predicted_chunks: list[torch.Tensor] = []
    logits_finite = True
    probabilities_finite = True
    try:
        with torch.no_grad():
            for start in range(0, targets.num_prefixes, evaluation_batch_size):
                batch_prefixes = list(targets.prefixes[start : start + evaluation_batch_size])
                logits, probabilities = prediction_fn(batch_prefixes)
                _validate_prediction_shapes(
                    logits=logits,
                    probabilities=probabilities,
                    batch_size=len(batch_prefixes),
                    num_actions=tree.num_actions,
                )
                logits_finite = logits_finite and bool(torch.isfinite(logits).all().item())
                probabilities_finite = probabilities_finite and bool(
                    torch.isfinite(probabilities).all().item()
                )
                predicted_chunks.append(
                    probabilities.detach().to(device="cpu", dtype=torch.float64)
                )
    finally:
        policy.train(was_training)

    predicted = torch.cat(predicted_chunks, dim=0)
    legal_values = predicted.masked_fill(~targets.legal_mask, 0.0)
    legal_mass = legal_values.sum(dim=1)
    normalization_errors = (legal_mass - 1.0).abs()
    illegal_values = predicted.masked_fill(targets.legal_mask, 0.0).abs()
    max_normalization_error = float(normalization_errors.max().item())
    max_illegal_probability = float(illegal_values.max().item())

    tiny = torch.finfo(torch.float64).tiny
    safe_predicted = predicted.clamp_min(tiny)
    safe_targets = targets.probabilities.clamp_min(tiny)
    cross_entropy_by_prefix = -(
        targets.probabilities * safe_predicted.log()
    ).sum(dim=1)
    kl_by_prefix = (
        targets.probabilities * (safe_targets.log() - safe_predicted.log())
    ).sum(dim=1)
    tv_by_prefix = 0.5 * (predicted - targets.probabilities).abs().sum(dim=1)
    action_errors = (predicted - targets.probabilities).abs()

    depth_metrics: dict[int, dict[str, float | int]] = {}
    for depth in sorted(set(int(value) for value in targets.depths.tolist())):
        rows = targets.depths == depth
        depth_metrics[depth] = {
            "num_prefixes": int(rows.sum().item()),
            "mean_prefix_tv": float(tv_by_prefix[rows].mean().item()),
            "max_prefix_tv": float(tv_by_prefix[rows].max().item()),
            "mean_conditional_kl": float(kl_by_prefix[rows].mean().item()),
            "max_conditional_kl": float(kl_by_prefix[rows].max().item()),
            "max_conditional_probability_error": float(action_errors[rows].max().item()),
        }

    terminal_evaluation = evaluate_exact_policy(
        tree=tree,
        policy=policy,
        probability_fn=_probability_fn(prediction_fn),
        update=update,
        gates=gates.exact,
        evaluation_batch_size=evaluation_batch_size,
    )
    supervised_loss = float(cross_entropy_by_prefix.mean().item())
    finite = bool(
        logits_finite
        and probabilities_finite
        and loss_finite
        and gradients_finite
        and parameters_finite
        and math.isfinite(supervised_loss)
        and bool(torch.isfinite(kl_by_prefix).all().item())
        and bool(torch.isfinite(tv_by_prefix).all().item())
        and terminal_evaluation.metrics.finite
    )
    passed = bool(
        finite
        and max_normalization_error <= gates.normalization_error
        and max_illegal_probability <= gates.illegal_probability
        and float(tv_by_prefix.mean().item()) <= gates.mean_prefix_tv
        and float(tv_by_prefix.max().item()) <= gates.max_prefix_tv
        and terminal_evaluation.metrics.passed
    )
    metrics = ConditionalMetrics(
        update=int(update),
        prefix_presentations=int(prefix_presentations),
        learning_rate=float(learning_rate),
        wall_time_seconds=float(wall_time_seconds),
        supervised_loss=supervised_loss,
        mean_prefix_tv=float(tv_by_prefix.mean().item()),
        max_prefix_tv=float(tv_by_prefix.max().item()),
        mean_conditional_kl=float(kl_by_prefix.mean().item()),
        max_conditional_kl=float(kl_by_prefix.max().item()),
        max_conditional_probability_error=float(action_errors.max().item()),
        depth_metrics=depth_metrics,
        max_action_normalization_error=max_normalization_error,
        max_illegal_probability=max_illegal_probability,
        logits_finite=logits_finite,
        probabilities_finite=probabilities_finite,
        loss_finite=bool(loss_finite),
        gradients_finite=bool(gradients_finite),
        parameters_finite=bool(parameters_finite),
        finite=finite,
        terminal=terminal_evaluation.metrics,
        passed=passed,
    )
    return ConditionalEvaluation(
        metrics=metrics,
        predicted_conditionals=predicted,
        terminal_evaluation=terminal_evaluation,
    )


def maximum_normalized_gate_violation(
    metrics: ConditionalMetrics,
    gates: ConditionalGateConfig | None = None,
) -> float:
    gates = gates or ConditionalGateConfig()
    if not metrics.finite:
        return math.inf
    exact = metrics.terminal
    ratios = [
        metrics.max_action_normalization_error / gates.normalization_error,
        metrics.max_illegal_probability / gates.illegal_probability,
        metrics.mean_prefix_tv / gates.mean_prefix_tv,
        metrics.max_prefix_tv / gates.max_prefix_tv,
        exact.log_z_error / gates.exact.log_z_error,
        exact.rms_residual / gates.exact.rms_residual,
        exact.max_abs_residual / gates.exact.max_abs_residual,
        exact.total_variation / gates.exact.total_variation,
        exact.max_probability_error / gates.exact.max_probability_error,
        exact.probability_mass_error / gates.exact.probability_mass_error,
    ]
    return max(0.0, max(ratios) - 1.0)


def _validate_training_config(config: SupervisedTrainingConfig) -> None:
    if config.scheduler not in ("fixed", "cosine"):
        raise ValueError(f"unsupported scheduler: {config.scheduler}")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.cosine_min_lr < 0.0 or config.cosine_min_lr > config.learning_rate:
        raise ValueError("cosine_min_lr must be between zero and learning_rate")
    if config.max_updates <= 0 or config.eval_every <= 0 or config.prefix_batch_size <= 0:
        raise ValueError("max_updates, eval_every, and prefix_batch_size must be positive")
    if config.required_consecutive_passes <= 0:
        raise ValueError("required_consecutive_passes must be positive")


def train_supervised_case(
    *,
    tree: ExactTree,
    targets: ExactConditionalTargets,
    policy: nn.Module,
    prediction_fn: PredictionFn,
    config: SupervisedTrainingConfig | None = None,
    gates: ConditionalGateConfig | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> SupervisedRunResult:
    config = config or SupervisedTrainingConfig()
    gates = gates or ConditionalGateConfig()
    _validate_training_config(config)
    if targets.num_prefixes != len(tree.nodes):
        raise ValueError("the number of supervised targets does not match the tree")
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    log_z = getattr(policy, "log_z", None)
    if not isinstance(log_z, nn.Parameter):
        raise TypeError("policy.log_z must be a torch.nn.Parameter")
    with torch.no_grad():
        log_z.copy_(torch.as_tensor(tree.true_log_z, dtype=log_z.dtype, device=log_z.device))
    log_z.requires_grad_(False)
    expected_log_z = log_z.detach().clone()

    optimizer = build_policy_optimizer(policy, learning_rate=config.learning_rate)
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.max_updates),
            eta_min=float(config.cosine_min_lr),
        )

    started_at = time.monotonic()
    evaluations: list[ConditionalMetrics] = []

    def learning_rate() -> float:
        return float(optimizer.param_groups[0]["lr"])

    latest_evaluation = evaluate_conditional_policy(
        tree=tree,
        targets=targets,
        policy=policy,
        prediction_fn=prediction_fn,
        update=0,
        prefix_presentations=0,
        learning_rate=learning_rate(),
        wall_time_seconds=time.monotonic() - started_at,
        gates=gates,
        evaluation_batch_size=config.prefix_batch_size,
    )
    initial_evaluation = latest_evaluation
    evaluations.append(latest_evaluation.metrics)
    if on_evaluation is not None:
        on_evaluation(latest_evaluation)

    best_evaluation = latest_evaluation
    best_state = _clone_state_dict(policy)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_scheduler_state = None if scheduler is None else copy.deepcopy(scheduler.state_dict())
    best_key = (
        maximum_normalized_gate_violation(latest_evaluation.metrics, gates),
        latest_evaluation.metrics.mean_conditional_kl,
    )
    consecutive_passes = 1 if latest_evaluation.metrics.passed else 0
    first_passing_update = 0 if latest_evaluation.metrics.passed else None
    completed_updates = 0
    numerical_failure: str | None = None
    last_loss_finite = True
    last_gradients_finite = True
    last_parameters_finite = True

    policy.train()
    for update in range(1, int(config.max_updates) + 1):
        optimizer_stepped = False
        try:
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for start in range(0, targets.num_prefixes, int(config.prefix_batch_size)):
                stop = min(start + int(config.prefix_batch_size), targets.num_prefixes)
                batch_prefixes = list(targets.prefixes[start:stop])
                logits, probabilities = prediction_fn(batch_prefixes)
                _validate_prediction_shapes(
                    logits=logits,
                    probabilities=probabilities,
                    batch_size=len(batch_prefixes),
                    num_actions=tree.num_actions,
                )
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError(f"logits are not finite at update {update}")
                if not bool(torch.isfinite(probabilities).all().item()):
                    raise FloatingPointError(f"probabilities are not finite at update {update}")
                batch_targets = targets.probabilities[start:stop].to(
                    device=probabilities.device,
                    dtype=probabilities.dtype,
                )
                tiny = torch.finfo(probabilities.dtype).tiny
                chunk_mean_loss = -(
                    batch_targets * probabilities.clamp_min(tiny).log()
                ).sum(dim=1).mean()
                chunk_weight = len(batch_prefixes) / targets.num_prefixes
                chunk_loss = chunk_mean_loss * chunk_weight
                if not bool(torch.isfinite(chunk_loss).item()):
                    raise FloatingPointError(f"supervised loss is not finite at update {update}")
                chunk_loss.backward()
                accumulated_loss += float(chunk_loss.detach().to(device="cpu").item())

            last_loss_finite = math.isfinite(accumulated_loss)
            last_gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in policy.parameters()
            )
            if not last_gradients_finite:
                raise FloatingPointError(f"gradient is not finite at update {update}")
            optimizer.step()
            optimizer_stepped = True
            if scheduler is not None:
                scheduler.step()
            last_parameters_finite = all(
                bool(torch.isfinite(parameter).all().item()) for parameter in policy.parameters()
            )
            if not last_parameters_finite:
                raise FloatingPointError(f"parameter is not finite at update {update}")
            if log_z.requires_grad or not torch.equal(log_z.detach(), expected_log_z):
                raise FloatingPointError(f"log_z changed during supervised training at update {update}")
        except FloatingPointError as exc:
            numerical_failure = str(exc)
            completed_updates = update if optimizer_stepped else update - 1
            break

        completed_updates = update
        if update % int(config.eval_every) != 0 and update != int(config.max_updates):
            continue

        latest_evaluation = evaluate_conditional_policy(
            tree=tree,
            targets=targets,
            policy=policy,
            prediction_fn=prediction_fn,
            update=update,
            prefix_presentations=update * targets.num_prefixes,
            learning_rate=learning_rate(),
            wall_time_seconds=time.monotonic() - started_at,
            gates=gates,
            loss_finite=last_loss_finite,
            gradients_finite=last_gradients_finite,
            parameters_finite=last_parameters_finite,
            evaluation_batch_size=config.prefix_batch_size,
        )
        evaluations.append(latest_evaluation.metrics)
        if on_evaluation is not None:
            on_evaluation(latest_evaluation)

        candidate_key = (
            maximum_normalized_gate_violation(latest_evaluation.metrics, gates),
            latest_evaluation.metrics.mean_conditional_kl,
        )
        if candidate_key < best_key:
            best_key = candidate_key
            best_evaluation = latest_evaluation
            best_state = _clone_state_dict(policy)
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scheduler_state = (
                None if scheduler is None else copy.deepcopy(scheduler.state_dict())
            )

        if latest_evaluation.metrics.passed:
            consecutive_passes += 1
            if first_passing_update is None:
                first_passing_update = update
        else:
            consecutive_passes = 0
        if consecutive_passes >= int(config.required_consecutive_passes):
            break

    if latest_evaluation.metrics.update != completed_updates or numerical_failure is not None:
        latest_evaluation = evaluate_conditional_policy(
            tree=tree,
            targets=targets,
            policy=policy,
            prediction_fn=prediction_fn,
            update=completed_updates,
            prefix_presentations=completed_updates * targets.num_prefixes,
            learning_rate=learning_rate(),
            wall_time_seconds=time.monotonic() - started_at,
            gates=gates,
            loss_finite=last_loss_finite and numerical_failure is None,
            gradients_finite=last_gradients_finite and numerical_failure is None,
            parameters_finite=last_parameters_finite,
            evaluation_batch_size=config.prefix_batch_size,
        )
        if not evaluations or evaluations[-1].update != completed_updates:
            evaluations.append(latest_evaluation.metrics)
            if on_evaluation is not None:
                on_evaluation(latest_evaluation)
        candidate_key = (
            maximum_normalized_gate_violation(latest_evaluation.metrics, gates),
            latest_evaluation.metrics.mean_conditional_kl,
        )
        if candidate_key < best_key:
            best_evaluation = latest_evaluation
            best_state = _clone_state_dict(policy)
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scheduler_state = (
                None if scheduler is None else copy.deepcopy(scheduler.state_dict())
            )

    passed = bool(
        numerical_failure is None
        and consecutive_passes >= int(config.required_consecutive_passes)
        and latest_evaluation.metrics.passed
    )
    return SupervisedRunResult(
        variant=config.scheduler,
        seed=int(config.seed),
        passed=passed,
        completed_updates=completed_updates,
        prefix_presentations=completed_updates * targets.num_prefixes,
        first_passing_update=first_passing_update,
        consecutive_passes=consecutive_passes,
        initial_evaluation=initial_evaluation,
        final_evaluation=latest_evaluation,
        best_evaluation=best_evaluation,
        best_update=best_evaluation.metrics.update,
        evaluations=evaluations,
        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
        scheduler_state_dict=None if scheduler is None else copy.deepcopy(scheduler.state_dict()),
        best_optimizer_state_dict=best_optimizer_state,
        best_scheduler_state_dict=best_scheduler_state,
        best_policy_state_dict=best_state,
        final_policy_state_dict=_clone_state_dict(policy),
        numerical_failure=numerical_failure,
    )


def classify_supervised_runs(
    results: list[SupervisedRunResult],
    *,
    seeds: list[int] | tuple[int, ...],
    variants: list[SchedulerMode] | tuple[SchedulerMode, ...],
) -> tuple[str, int, dict[str, int]]:
    expected_seeds = {int(seed) for seed in seeds}
    expected_variants = tuple(variants)
    counts = {
        variant: sum(result.passed for result in results if result.variant == variant)
        for variant in expected_variants
    }
    observed = {(result.variant, result.seed) for result in results}
    expected = {(variant, seed) for variant in expected_variants for seed in expected_seeds}
    if observed != expected or any(result.numerical_failure is not None for result in results):
        return "execution_failure", 1, counts
    if any(counts[variant] == len(expected_seeds) for variant in expected_variants):
        return "robust_representability", 0, counts
    if any(result.passed for result in results):
        return "representable_but_optimizer_sensitive", 3, counts
    return "representability_not_demonstrated", 4, counts


__all__ = [
    "ConditionalEvaluation",
    "ConditionalGateConfig",
    "ConditionalMetrics",
    "ExactConditionalTargets",
    "PredictionFn",
    "SchedulerMode",
    "SupervisedRunResult",
    "SupervisedTrainingConfig",
    "build_exact_conditional_targets",
    "classify_supervised_runs",
    "evaluate_conditional_policy",
    "maximum_normalized_gate_violation",
    "train_supervised_case",
]
