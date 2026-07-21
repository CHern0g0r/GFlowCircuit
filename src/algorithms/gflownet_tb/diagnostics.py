"""Dependency-light diagnostics for the active Trajectory Balance baseline.

The production trainer deliberately does not depend on this module.  It contains
only numerical decompositions and sequence-aware discovery helpers used by the
isolated active-baseline experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def derive_seed(base_seed: int, stream_name: str) -> int:
    """Derive a stable positive 63-bit seed without Python's randomized hash."""
    payload = f"gflowcircuit-active-baseline-v1\0{int(base_seed)}\0{stream_name}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << 63) - 1)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_dict_checksum(state_dict: Mapping[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def _as_float64(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return torch.as_tensor(list(values), dtype=torch.float64).reshape(-1)


def _finite(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all().item())


def _quantile_dict(values: torch.Tensor) -> dict[str, float]:
    names = ("p00", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p100")
    if values.numel() == 0:
        return {name: math.nan for name in names}
    q = torch.quantile(values, torch.tensor(_QUANTILES, dtype=torch.float64))
    return {name: float(item) for name, item in zip(names, q.tolist(), strict=True)}


def residual_diagnostics(
    *,
    log_z: float | torch.Tensor,
    log_pf: torch.Tensor | Sequence[float],
    log_r: torch.Tensor | Sequence[float],
    log_pb: torch.Tensor | Sequence[float],
) -> dict[str, Any]:
    """Decompose TB MSE into global-offset bias and centered variance."""
    pf = _as_float64(log_pf)
    reward = _as_float64(log_r)
    pb = _as_float64(log_pb)
    if not (pf.numel() == reward.numel() == pb.numel()) or pf.numel() == 0:
        raise ValueError("log_pf, log_r, and log_pb must have the same non-zero length")
    z = float(log_z.detach().cpu().item()) if isinstance(log_z, torch.Tensor) else float(log_z)
    residual = z + pf - reward - pb
    mean = residual.mean()
    centered = residual - mean
    mse = residual.square().mean()
    centered_mse = centered.square().mean()
    std = centered_mse.sqrt()
    squared_bias = mean.square()
    eps = 1e-12
    bias_fraction = squared_bias / (mse + eps)
    removed = (mse - centered_mse) / (mse + eps)
    target = (reward + pb - pf).mean()
    finite = _finite(torch.cat((pf, reward, pb, residual))) and math.isfinite(z)
    return {
        "count": int(pf.numel()),
        "finite": finite,
        "learned_log_z": z,
        "analytic_log_z_target": float(target),
        "log_z_target_gap": float(z - target),
        "learned_log_z_mse": float(mse),
        "centered_mse": float(centered_mse),
        "centered_rms": float(std),
        "residual_mean": float(mean),
        "residual_population_variance": float(centered_mse),
        "residual_rms": float(mse.sqrt()),
        "residual_std": float(std),
        "residual_quantiles": _quantile_dict(residual),
        "squared_bias": float(squared_bias),
        "bias_fraction": float(bias_fraction),
        "standardized_bias": float(mean.abs() / (std + eps)),
        "recentered_mse_reduction": float(removed),
        "log_pf": distribution_summary(pf),
        "log_pb": distribution_summary(pb),
        "log_r": distribution_summary(reward),
    }


def distribution_summary(values: torch.Tensor | Sequence[float]) -> dict[str, Any]:
    tensor = _as_float64(values)
    if tensor.numel() == 0:
        return {"count": 0, "finite": True, "mean": math.nan, "std": math.nan, "quantiles": {}}
    return {
        "count": int(tensor.numel()),
        "finite": _finite(tensor),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "quantiles": _quantile_dict(tensor),
    }


def regression_diagnostics(
    log_pf: torch.Tensor | Sequence[float],
    target: torch.Tensor | Sequence[float],
) -> dict[str, float]:
    x = _as_float64(target)
    y = _as_float64(log_pf)
    if x.numel() != y.numel() or x.numel() == 0:
        raise ValueError("regression arrays must have the same non-zero length")
    xc = x - x.mean()
    yc = y - y.mean()
    x_var = xc.square().mean()
    y_var = yc.square().mean()
    slope = (xc * yc).mean() / (x_var + 1e-12)
    corr = (xc * yc).mean() / (x_var.sqrt() * y_var.sqrt() + 1e-12)
    centered_rmse = ((yc - xc).square().mean()).sqrt()
    return {
        "slope": float(slope),
        "correlation": float(corr),
        "centered_rmse": float(centered_rmse),
    }


def parameter_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squares: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().double().square().sum().cpu())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def parameter_snapshot(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def parameter_update_norm(
    before: Sequence[torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> float:
    after = list(parameters)
    if len(before) != len(after):
        raise ValueError("parameter snapshot length changed")
    total = 0.0
    for old, new in zip(before, after, strict=True):
        total += float((new.detach().cpu().double() - old.double()).square().sum())
    return math.sqrt(total)


def module_is_finite(module: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in module.state_dict().values())


def optimizer_is_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
                return False
    return True


@dataclass(frozen=True)
class SequenceRecord:
    index: int
    actions: tuple[int, ...]
    initial_size: int
    initial_depth: int
    final_size: int
    final_depth: int
    comparable_return: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["actions"] = list(self.actions)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequenceRecord":
        return cls(
            index=int(value["index"]),
            actions=tuple(int(action) for action in value["actions"]),
            initial_size=int(value["initial_size"]),
            initial_depth=int(value["initial_depth"]),
            final_size=int(value["final_size"]),
            final_depth=int(value["final_depth"]),
            comparable_return=float(value["comparable_return"]),
        )


class SequenceArchive:
    """Retain action sequences while computing HV on unique feasible endpoints."""

    def __init__(self, *, circuit: str, initial_size: int, initial_depth: int) -> None:
        if initial_size <= 0 or initial_depth <= 0:
            raise ValueError("initial circuit metrics must be positive")
        self.circuit = str(circuit)
        self.initial_size = int(initial_size)
        self.initial_depth = int(initial_depth)
        self.records: list[SequenceRecord] = []

    def record(
        self,
        *,
        actions: Sequence[int],
        initial_size: int,
        initial_depth: int,
        final_size: int,
        final_depth: int,
        comparable_return: float,
    ) -> None:
        if (int(initial_size), int(initial_depth)) != (self.initial_size, self.initial_depth):
            raise ValueError("archive initial circuit metrics changed")
        self.records.append(
            SequenceRecord(
                index=len(self.records) + 1,
                actions=tuple(int(action) for action in actions),
                initial_size=int(initial_size),
                initial_depth=int(initial_depth),
                final_size=int(final_size),
                final_depth=int(final_depth),
                comparable_return=float(comparable_return),
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "circuit": self.circuit,
            "initial_size": self.initial_size,
            "initial_depth": self.initial_depth,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "SequenceArchive":
        archive = cls(
            circuit=str(value["circuit"]),
            initial_size=int(value["initial_size"]),
            initial_depth=int(value["initial_depth"]),
        )
        archive.records = [SequenceRecord.from_dict(item) for item in value["records"]]
        return archive

    def snapshot(self) -> dict[str, Any]:
        return archive_metrics(
            self.records,
            initial_size=self.initial_size,
            initial_depth=self.initial_depth,
        )


def nondominated_endpoints(
    records: Sequence[SequenceRecord],
    *,
    initial_size: int,
    initial_depth: int,
) -> list[tuple[int, int]]:
    feasible = {(int(initial_size), int(initial_depth))}
    feasible.update(
        (record.final_size, record.final_depth)
        for record in records
        if 0 <= record.final_size <= initial_size and 0 <= record.final_depth <= initial_depth
    )
    result: list[tuple[int, int]] = []
    for point in sorted(feasible):
        if any(
            other != point
            and other[0] <= point[0]
            and other[1] <= point[1]
            for other in feasible
        ):
            continue
        result.append(point)
    return sorted(result)


def normalized_hypervolume(
    endpoints: Sequence[tuple[int, int]],
    *,
    initial_size: int,
    initial_depth: int,
) -> float:
    current_y = 1.0
    volume = 0.0
    for size, depth in sorted(endpoints):
        x = float(size) / float(initial_size)
        y = float(depth) / float(initial_depth)
        if y < current_y:
            volume += max(0.0, 1.0 - x) * (current_y - y)
            current_y = y
    return float(volume)


def archive_metrics(
    records: Sequence[SequenceRecord],
    *,
    initial_size: int,
    initial_depth: int,
) -> dict[str, Any]:
    endpoints = nondominated_endpoints(
        records, initial_size=initial_size, initial_depth=initial_depth
    )
    sequences = {record.actions for record in records}
    terminal_points = {(record.final_size, record.final_depth) for record in records}
    returns = [record.comparable_return for record in records]
    best_size = min((record.final_size for record in records), default=initial_size)
    best_depth = min((record.final_depth for record in records), default=initial_depth)
    return {
        "trajectory_count": len(records),
        "distinct_sequences": len(sequences),
        "distinct_terminal_points": len(terminal_points),
        "nondominated_count": len(endpoints),
        "nondominated_endpoints": [list(point) for point in endpoints],
        "hypervolume": normalized_hypervolume(
            endpoints, initial_size=initial_size, initial_depth=initial_depth
        ),
        "mean_product_improvement": float(np.mean(returns)) if returns else 0.0,
        "best_size_reduction": int(initial_size - best_size),
        "best_depth_reduction": int(initial_depth - best_depth),
    }


def nested_search_metrics(
    records: Sequence[SequenceRecord],
    *,
    initial_size: int,
    initial_depth: int,
    budgets: Sequence[int] = (1, 2, 5, 10, 20, 50),
) -> dict[str, Any]:
    if not records:
        raise ValueError("search records must be non-empty")
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        if budget <= 0 or budget > len(records):
            raise ValueError(f"search budget {budget} is outside 1..{len(records)}")
        row = {"n": int(budget), **archive_metrics(
            records[:budget], initial_size=initial_size, initial_depth=initial_depth
        )}
        rows.append(row)
    x = np.log2(np.asarray([row["n"] for row in rows], dtype=np.float64))
    y = np.asarray([row["hypervolume"] for row in rows], dtype=np.float64)
    width = float(x[-1] - x[0])
    if width > 0.0:
        trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc = float(trapezoid(y, x) / width)
    else:
        auc = float(y[-1])
    return {"budgets": rows, "log2_n_hypervolume_auc": auc, **rows[-1]}


@dataclass(frozen=True)
class HealthThresholds:
    log_z_target_gap: float = 0.5
    bias_fraction: float = 0.05
    standardized_bias: float = 0.25
    gradient_p99_median_ratio: float = 20.0
    clipping_rate: float = 0.05
    collapse_fraction: float = 0.95
    normalization_error: float = 1e-6
    illegal_probability: float = 1e-6


def health_gates(
    *,
    validation: Mapping[str, Any],
    gradient_p99_median_ratio: float,
    clipping_enabled: bool,
    clipping_rate: float,
    thresholds: HealthThresholds = HealthThresholds(),
) -> dict[str, Any]:
    residual = validation["residual"]
    policy = validation["policy"]
    checks = {
        "finite": bool(validation.get("finite", False)),
        "legal_probability_normalization": float(policy["max_normalization_error"])
        <= thresholds.normalization_error,
        "illegal_probability": float(policy["max_illegal_probability"])
        <= thresholds.illegal_probability,
        "log_z_target_gap": abs(float(residual["log_z_target_gap"]))
        <= thresholds.log_z_target_gap,
        "bias_fraction": float(residual["bias_fraction"]) <= thresholds.bias_fraction,
        "standardized_bias": float(residual["standardized_bias"])
        <= thresholds.standardized_bias,
        "gradient_p99_median_ratio": float(gradient_p99_median_ratio)
        <= thresholds.gradient_p99_median_ratio,
        "clipping_rate": (not clipping_enabled) or float(clipping_rate) < thresholds.clipping_rate,
        "collapse_fraction": float(policy["collapse_fraction"])
        <= thresholds.collapse_fraction,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "thresholds": asdict(thresholds),
    }


__all__ = [
    "HealthThresholds",
    "SequenceArchive",
    "SequenceRecord",
    "archive_metrics",
    "canonical_sha256",
    "derive_seed",
    "distribution_summary",
    "health_gates",
    "module_is_finite",
    "nested_search_metrics",
    "nondominated_endpoints",
    "normalized_hypervolume",
    "optimizer_is_finite",
    "parameter_gradient_norm",
    "parameter_snapshot",
    "parameter_update_norm",
    "regression_diagnostics",
    "residual_diagnostics",
    "state_dict_checksum",
]
