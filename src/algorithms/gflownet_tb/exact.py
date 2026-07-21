from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

import torch
from torch import nn

from src.algorithms.gflownet_tb.behavior import epsilon_mixed_probs
from src.algorithms.gflownet_tb.loss import trajectory_balance_loss
from src.algorithms.gflownet_tb.optim import build_tb_optimizer

Prefix = tuple[int, ...]
BehaviorMode = Literal["on_policy", "epsilon_0.5", "uniform"]
ProbabilityFn = Callable[[list[Prefix]], torch.Tensor]
EvaluationCallback = Callable[["ExactMetrics"], None]


def _logsumexp(values: list[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


@dataclass(frozen=True)
class ExactNode:
    prefix: Prefix
    legal_actions: tuple[int, ...]
    payload: Any = field(default=None, compare=False, repr=False)


@dataclass
class ExactTree:
    depth: int
    num_actions: int
    nodes: dict[Prefix, ExactNode]
    terminal_log_rewards: dict[Prefix, float]
    name: str = "exact_tree"
    subtree_log_flows: dict[Prefix, float] = field(init=False)
    target_terminal_probs: dict[Prefix, float] = field(init=False)

    def __post_init__(self) -> None:
        self.depth = int(self.depth)
        self.num_actions = int(self.num_actions)
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {self.num_actions}")
        if () not in self.nodes:
            raise ValueError("tree must contain the root prefix")
        if not self.terminal_log_rewards:
            raise ValueError("tree must contain terminal rewards")

        for prefix, node in self.nodes.items():
            if prefix != node.prefix:
                raise ValueError(f"node key {prefix} does not match node prefix {node.prefix}")
            if len(prefix) >= self.depth:
                raise ValueError(f"nonterminal prefix has invalid depth: {prefix}")
            if not node.legal_actions:
                raise ValueError(f"nonterminal prefix has no legal actions: {prefix}")
            if len(set(node.legal_actions)) != len(node.legal_actions):
                raise ValueError(f"duplicate legal actions at prefix {prefix}: {node.legal_actions}")
            for action in node.legal_actions:
                if action < 0 or action >= self.num_actions:
                    raise ValueError(f"invalid action {action} at prefix {prefix}")
                child = prefix + (int(action),)
                if child not in self.nodes and child not in self.terminal_log_rewards:
                    raise ValueError(f"missing child {child} from prefix {prefix}")

        for prefix, log_reward in self.terminal_log_rewards.items():
            if len(prefix) != self.depth:
                raise ValueError(f"terminal prefix {prefix} does not have depth {self.depth}")
            if not math.isfinite(float(log_reward)):
                raise ValueError(f"terminal log reward is not finite for {prefix}: {log_reward}")

        subtree_log_flows = {
            prefix: float(log_reward) for prefix, log_reward in self.terminal_log_rewards.items()
        }
        for level in range(self.depth - 1, -1, -1):
            for prefix in sorted((p for p in self.nodes if len(p) == level)):
                node = self.nodes[prefix]
                child_flows = [subtree_log_flows[prefix + (action,)] for action in node.legal_actions]
                subtree_log_flows[prefix] = _logsumexp(child_flows)
        self.subtree_log_flows = subtree_log_flows
        true_log_z = subtree_log_flows[()]
        self.target_terminal_probs = {
            prefix: math.exp(log_reward - true_log_z)
            for prefix, log_reward in self.terminal_log_rewards.items()
        }

        target_mass = sum(self.target_terminal_probs.values())
        if abs(target_mass - 1.0) > 1e-10:
            raise ValueError(f"target terminal probabilities sum to {target_mass}, expected 1")

    @property
    def true_log_z(self) -> float:
        return self.subtree_log_flows[()]

    @property
    def nonterminal_prefixes(self) -> list[Prefix]:
        return sorted(self.nodes, key=lambda prefix: (len(prefix), prefix))

    @property
    def terminal_prefixes(self) -> list[Prefix]:
        return sorted(self.terminal_log_rewards)

    def target_action_probs(self, prefix: Prefix) -> dict[int, float]:
        node = self.nodes[prefix]
        parent_flow = self.subtree_log_flows[prefix]
        return {
            action: math.exp(self.subtree_log_flows[prefix + (action,)] - parent_flow)
            for action in node.legal_actions
        }

    def summary(self) -> dict[str, Any]:
        branching = [len(node.legal_actions) for node in self.nodes.values()]
        rewards = [math.exp(log_reward) for log_reward in self.terminal_log_rewards.values()]
        return {
            "name": self.name,
            "depth": self.depth,
            "num_actions": self.num_actions,
            "num_nonterminal_prefixes": len(self.nodes),
            "num_terminal_sequences": len(self.terminal_log_rewards),
            "min_branching": min(branching),
            "max_branching": max(branching),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "reward_ratio": max(rewards) / min(rewards),
            "true_log_z": self.true_log_z,
            "target_probability_mass": sum(self.target_terminal_probs.values()),
        }


def build_synthetic_tree(*, depth: int = 4) -> ExactTree:
    """Build the deterministic nonuniform tree used by the exact TB test."""
    depth = int(depth)
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")

    nodes: dict[Prefix, ExactNode] = {}
    frontier: list[Prefix] = [()]
    for level in range(depth):
        next_frontier: list[Prefix] = []
        for prefix in frontier:
            if not prefix:
                legal_actions = (0, 1, 2)
            else:
                legal_actions = ((0, 1), (0, 1, 2), (1, 2))[sum(prefix) % 3]
            nodes[prefix] = ExactNode(prefix=prefix, legal_actions=legal_actions)
            next_frontier.extend(prefix + (action,) for action in legal_actions)
        frontier = next_frontier

    terminals = sorted(frontier)
    denominator = max(1, len(terminals) - 1)
    terminal_log_rewards = {
        prefix: math.log(100.0) * rank / denominator
        for rank, prefix in enumerate(terminals)
    }
    return ExactTree(
        depth=depth,
        num_actions=3,
        nodes=nodes,
        terminal_log_rewards=terminal_log_rewards,
        name=f"synthetic_depth_{depth}",
    )


class TabularPrefixPolicy(nn.Module):
    def __init__(self, tree: ExactTree) -> None:
        super().__init__()
        self.num_actions = int(tree.num_actions)
        self._prefixes = tree.nonterminal_prefixes
        self._prefix_to_index = {prefix: idx for idx, prefix in enumerate(self._prefixes)}
        self._legal_actions = {prefix: tree.nodes[prefix].legal_actions for prefix in self._prefixes}
        self.logits = nn.Parameter(torch.zeros(len(self._prefixes), self.num_actions, dtype=torch.float32))
        self.log_z = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        return self.log_z.device

    def probabilities(self, prefixes: list[Prefix]) -> torch.Tensor:
        if not prefixes:
            return torch.empty((0, self.num_actions), dtype=self.logits.dtype, device=self.device)
        indices = torch.tensor(
            [self._prefix_to_index[prefix] for prefix in prefixes],
            dtype=torch.long,
            device=self.device,
        )
        selected_logits = self.logits.index_select(0, indices)
        masked_logits = torch.full_like(selected_logits, -torch.inf)
        for row_idx, prefix in enumerate(prefixes):
            legal_idx = torch.tensor(self._legal_actions[prefix], dtype=torch.long, device=self.device)
            masked_logits[row_idx, legal_idx] = selected_logits[row_idx, legal_idx]
        return torch.softmax(masked_logits, dim=-1)


@dataclass(frozen=True)
class ExactGateConfig:
    log_z_error: float = 0.05
    rms_residual: float = 0.05
    max_abs_residual: float = 0.15
    total_variation: float = 0.02
    max_probability_error: float = 0.01
    probability_mass_error: float = 1e-6


@dataclass(frozen=True)
class ExactMetrics:
    update: int
    loss: float
    log_z: float
    true_log_z: float
    log_z_error: float
    rms_residual: float
    max_abs_residual: float
    total_variation: float
    max_probability_error: float
    predicted_probability_mass: float
    probability_mass_error: float
    max_action_normalization_error: float
    max_illegal_probability: float
    finite: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactEvaluation:
    metrics: ExactMetrics
    predicted_terminal_probs: dict[Prefix, float]
    log_pf_by_terminal: dict[Prefix, float]


@dataclass(frozen=True)
class ExactTrainingConfig:
    batch_size: int = 64
    max_updates: int = 20_000
    eval_every: int = 100
    required_consecutive_passes: int = 3
    learning_rate: float = 0.001
    log_z_learning_rate: float = 0.01
    seed: int = 0


@dataclass
class ExactRunResult:
    behavior: BehaviorMode
    passed: bool
    completed_updates: int
    first_passing_update: int | None
    consecutive_passes: int
    evaluations: list[ExactMetrics]
    final_evaluation: ExactEvaluation
    optimizer_state_dict: dict[str, Any]
    numerical_failure: str | None = None


def _probability_tables(
    *,
    tree: ExactTree,
    probability_fn: ProbabilityFn,
    evaluation_batch_size: int = 256,
) -> tuple[dict[Prefix, list[float]], float, float, bool]:
    tables: dict[Prefix, list[float]] = {}
    max_normalization_error = 0.0
    max_illegal_probability = 0.0
    finite = True
    prefixes = tree.nonterminal_prefixes

    with torch.no_grad():
        for start in range(0, len(prefixes), int(evaluation_batch_size)):
            batch_prefixes = prefixes[start : start + int(evaluation_batch_size)]
            probs = probability_fn(batch_prefixes)
            if probs.shape != (len(batch_prefixes), tree.num_actions):
                raise ValueError(
                    "probability_fn returned shape "
                    f"{tuple(probs.shape)}, expected {(len(batch_prefixes), tree.num_actions)}"
                )
            probs_cpu = probs.detach().to(dtype=torch.float64, device="cpu")
            finite = finite and bool(torch.isfinite(probs_cpu).all().item())
            for row_idx, prefix in enumerate(batch_prefixes):
                row = probs_cpu[row_idx]
                legal = set(tree.nodes[prefix].legal_actions)
                legal_mass = float(sum(float(row[action].item()) for action in legal))
                illegal_values = [float(row[action].item()) for action in range(tree.num_actions) if action not in legal]
                max_normalization_error = max(max_normalization_error, abs(legal_mass - 1.0))
                if illegal_values:
                    max_illegal_probability = max(max_illegal_probability, max(abs(v) for v in illegal_values))
                tables[prefix] = [float(value) for value in row.tolist()]
    return tables, max_normalization_error, max_illegal_probability, finite


def evaluate_exact_policy(
    *,
    tree: ExactTree,
    policy: nn.Module,
    probability_fn: ProbabilityFn,
    update: int,
    gates: ExactGateConfig | None = None,
    evaluation_batch_size: int = 256,
) -> ExactEvaluation:
    gates = gates or ExactGateConfig()
    was_training = policy.training
    policy.eval()
    try:
        tables, normalization_error, illegal_probability, probabilities_finite = _probability_tables(
            tree=tree,
            probability_fn=probability_fn,
            evaluation_batch_size=evaluation_batch_size,
        )
    finally:
        policy.train(was_training)

    log_pf_by_prefix: dict[Prefix, float] = {(): 0.0}
    finite = probabilities_finite
    for prefix in tree.nonterminal_prefixes:
        parent_log_pf = log_pf_by_prefix[prefix]
        row = tables[prefix]
        for action in tree.nodes[prefix].legal_actions:
            probability = row[action]
            if not math.isfinite(probability) or probability <= 0.0:
                child_log_pf = -math.inf
                finite = False
            else:
                child_log_pf = parent_log_pf + math.log(probability)
            log_pf_by_prefix[prefix + (action,)] = child_log_pf

    log_z = float(getattr(policy, "log_z").detach().to(dtype=torch.float64, device="cpu").item())
    residuals: list[float] = []
    predicted_terminal_probs: dict[Prefix, float] = {}
    probability_errors: list[float] = []
    for prefix in tree.terminal_prefixes:
        log_pf = log_pf_by_prefix[prefix]
        log_reward = tree.terminal_log_rewards[prefix]
        residual = log_z + log_pf - log_reward
        predicted = math.exp(log_pf) if math.isfinite(log_pf) else 0.0
        residuals.append(residual)
        predicted_terminal_probs[prefix] = predicted
        probability_errors.append(abs(predicted - tree.target_terminal_probs[prefix]))

    finite = finite and math.isfinite(log_z) and all(math.isfinite(value) for value in residuals)
    if finite:
        mean_square = sum(value * value for value in residuals) / len(residuals)
        rms_residual = math.sqrt(mean_square)
        max_abs_residual = max(abs(value) for value in residuals)
        loss = mean_square
    else:
        rms_residual = math.inf
        max_abs_residual = math.inf
        loss = math.inf

    predicted_mass = sum(predicted_terminal_probs.values())
    total_variation = 0.5 * sum(
        abs(predicted_terminal_probs[prefix] - tree.target_terminal_probs[prefix])
        for prefix in tree.terminal_prefixes
    )
    max_probability_error = max(probability_errors)
    log_z_error = abs(log_z - tree.true_log_z)
    probability_mass_error = abs(predicted_mass - 1.0)
    passed = bool(
        finite
        and log_z_error <= gates.log_z_error
        and rms_residual <= gates.rms_residual
        and max_abs_residual <= gates.max_abs_residual
        and total_variation <= gates.total_variation
        and max_probability_error <= gates.max_probability_error
        and probability_mass_error <= gates.probability_mass_error
        and normalization_error <= gates.probability_mass_error
        and illegal_probability <= gates.probability_mass_error
    )
    metrics = ExactMetrics(
        update=int(update),
        loss=loss,
        log_z=log_z,
        true_log_z=tree.true_log_z,
        log_z_error=log_z_error,
        rms_residual=rms_residual,
        max_abs_residual=max_abs_residual,
        total_variation=total_variation,
        max_probability_error=max_probability_error,
        predicted_probability_mass=predicted_mass,
        probability_mass_error=probability_mass_error,
        max_action_normalization_error=normalization_error,
        max_illegal_probability=illegal_probability,
        finite=finite,
        passed=passed,
    )
    return ExactEvaluation(
        metrics=metrics,
        predicted_terminal_probs=predicted_terminal_probs,
        log_pf_by_terminal={prefix: log_pf_by_prefix[prefix] for prefix in tree.terminal_prefixes},
    )


def sample_exact_batch(
    *,
    tree: ExactTree,
    probability_fn: ProbabilityFn,
    batch_size: int,
    behavior: BehaviorMode,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, list[Prefix]]:
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    prefixes: list[Prefix] = [() for _ in range(int(batch_size))]
    log_pf_terms: list[torch.Tensor] = []

    for _ in range(tree.depth):
        policy_probs = probability_fn(prefixes)
        if not bool(torch.isfinite(policy_probs).all().item()):
            raise FloatingPointError("policy probabilities contain NaN or Inf")
        legal_rows = [list(tree.nodes[prefix].legal_actions) for prefix in prefixes]
        if behavior == "on_policy":
            behavior_probs = policy_probs.detach()
        elif behavior == "epsilon_0.5":
            behavior_probs = epsilon_mixed_probs(policy_probs.detach(), legal_rows, 0.5)
        elif behavior == "uniform":
            behavior_probs = epsilon_mixed_probs(policy_probs.detach(), legal_rows, 1.0)
        else:
            raise ValueError(f"Unsupported behavior mode: {behavior}")

        actions = torch.multinomial(behavior_probs, num_samples=1, generator=generator).squeeze(1)
        selected_policy_probs = policy_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        if bool(torch.any(selected_policy_probs <= 0.0).item()):
            raise FloatingPointError("sampled action has nonpositive learned-policy probability")
        log_pf_terms.append(torch.log(selected_policy_probs))
        action_values = [int(action) for action in actions.detach().cpu().tolist()]
        prefixes = [prefix + (action,) for prefix, action in zip(prefixes, action_values)]

    log_pf = torch.stack(log_pf_terms, dim=0).sum(dim=0)
    log_rewards = torch.tensor(
        [tree.terminal_log_rewards[prefix] for prefix in prefixes],
        dtype=log_pf.dtype,
        device=log_pf.device,
    )
    return log_pf, log_rewards, prefixes


def train_exact_case(
    *,
    tree: ExactTree,
    policy: nn.Module,
    probability_fn: ProbabilityFn,
    behavior: BehaviorMode,
    config: ExactTrainingConfig | None = None,
    gates: ExactGateConfig | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> ExactRunResult:
    config = config or ExactTrainingConfig()
    gates = gates or ExactGateConfig()
    if config.batch_size <= 0 or config.max_updates <= 0 or config.eval_every <= 0:
        raise ValueError("batch_size, max_updates, and eval_every must be positive")
    if config.required_consecutive_passes <= 0:
        raise ValueError("required_consecutive_passes must be positive")

    log_z = getattr(policy, "log_z", None)
    if not isinstance(log_z, nn.Parameter):
        raise TypeError("policy.log_z must be a torch.nn.Parameter")
    device = log_z.device
    policy.train()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed))
    optimizer = build_tb_optimizer(
        policy,
        learning_rate=config.learning_rate,
        log_z_learning_rate=config.log_z_learning_rate,
    )

    evaluations: list[ExactMetrics] = []
    latest_evaluation = evaluate_exact_policy(
        tree=tree,
        policy=policy,
        probability_fn=probability_fn,
        update=0,
        gates=gates,
    )
    evaluations.append(latest_evaluation.metrics)
    if on_evaluation is not None:
        on_evaluation(latest_evaluation.metrics)

    consecutive_passes = 1 if latest_evaluation.metrics.passed else 0
    first_passing_update = 0 if latest_evaluation.metrics.passed else None
    completed_updates = 0
    numerical_failure: str | None = None

    for update in range(1, int(config.max_updates) + 1):
        try:
            optimizer.zero_grad(set_to_none=True)
            log_pf, log_rewards, _ = sample_exact_batch(
                tree=tree,
                probability_fn=probability_fn,
                batch_size=config.batch_size,
                behavior=behavior,
                generator=generator,
            )
            loss = trajectory_balance_loss(
                log_z=log_z,
                log_pf_sums=log_pf,
                log_rewards=log_rewards,
                log_pb_sums=torch.zeros_like(log_pf),
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"TB loss is not finite at update {update}")
            loss.backward()
            gradients_finite = all(
                param.grad is None or bool(torch.isfinite(param.grad).all().item())
                for param in policy.parameters()
            )
            if not gradients_finite:
                raise FloatingPointError(f"gradient is not finite at update {update}")
            optimizer.step()
        except FloatingPointError as exc:
            numerical_failure = str(exc)
            completed_updates = update - 1
            latest_evaluation = evaluate_exact_policy(
                tree=tree,
                policy=policy,
                probability_fn=probability_fn,
                update=completed_updates,
                gates=gates,
            )
            break

        completed_updates = update
        if update % int(config.eval_every) != 0 and update != int(config.max_updates):
            continue

        latest_evaluation = evaluate_exact_policy(
            tree=tree,
            policy=policy,
            probability_fn=probability_fn,
            update=update,
            gates=gates,
        )
        evaluations.append(latest_evaluation.metrics)
        if on_evaluation is not None:
            on_evaluation(latest_evaluation.metrics)
        if latest_evaluation.metrics.passed:
            consecutive_passes += 1
            if first_passing_update is None:
                first_passing_update = update
        else:
            consecutive_passes = 0
        if consecutive_passes >= int(config.required_consecutive_passes):
            break

    if latest_evaluation.metrics.update != completed_updates:
        latest_evaluation = evaluate_exact_policy(
            tree=tree,
            policy=policy,
            probability_fn=probability_fn,
            update=completed_updates,
            gates=gates,
        )
        evaluations.append(latest_evaluation.metrics)
        if on_evaluation is not None:
            on_evaluation(latest_evaluation.metrics)

    passed = bool(
        numerical_failure is None
        and consecutive_passes >= int(config.required_consecutive_passes)
        and latest_evaluation.metrics.passed
    )
    return ExactRunResult(
        behavior=behavior,
        passed=passed,
        completed_updates=completed_updates,
        first_passing_update=first_passing_update,
        consecutive_passes=consecutive_passes,
        evaluations=evaluations,
        final_evaluation=latest_evaluation,
        optimizer_state_dict=optimizer.state_dict(),
        numerical_failure=numerical_failure,
    )


__all__ = [
    "BehaviorMode",
    "ExactEvaluation",
    "ExactGateConfig",
    "ExactMetrics",
    "ExactNode",
    "ExactRunResult",
    "ExactTrainingConfig",
    "ExactTree",
    "Prefix",
    "TabularPrefixPolicy",
    "build_synthetic_tree",
    "evaluate_exact_policy",
    "sample_exact_batch",
    "train_exact_case",
]
