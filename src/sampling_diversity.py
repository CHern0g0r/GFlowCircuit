from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ACTION_NAMES = {
    0: "balance",
    1: "rewrite",
    2: "refactor",
    3: "rewrite -z",
    4: "refactor -z",
    5: "resub fast",
    6: "resub strong",
}
STATS_COLUMNS = [
    "circuit",
    "algorithm",
    "run_id",
    "training_seed",
    "sampling_seed",
    "checkpoint",
    "sample_index",
    "action_sequence",
    "aig_path",
    "aig_sha256",
    "aig_size",
    "aig_depth",
    "lut_path",
    "lut_sha256",
    "lut_size",
    "lut_depth",
]
_RUN_RE = re.compile(r"^run_(\d+)$")
_ABC_NODE_RE = re.compile(r"\bnd\s*=\s*(\d+)\b")
_ABC_LEVEL_RE = re.compile(r"\blev\s*=\s*(\d+)\b")


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "circuit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pandas as pd

    temporary = path.with_suffix(path.suffix + ".tmp")
    frame = pd.DataFrame(rows, columns=STATS_COLUMNS)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def discover_checkpoints(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    roots = [path]
    if (path / "saved_models").is_dir():
        roots.insert(0, path / "saved_models")

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.glob("run_*/last.pt"))
    if not candidates:
        candidates.extend(path.glob("*.pt"))

    def checkpoint_sort_key(candidate: Path) -> tuple[int, str]:
        match = _RUN_RE.match(candidate.parent.name)
        return (int(match.group(1)) if match else 0, str(candidate))

    unique = sorted(
        {candidate.resolve() for candidate in candidates}, key=checkpoint_sort_key
    )
    if not unique:
        raise FileNotFoundError(
            "No checkpoints found. Expected a checkpoint file, *.pt files, "
            f"or run_*/last.pt below: {path}"
        )
    return unique


def _load_circuits(dataset_cfg: Path, repo_root: Path) -> list[Path]:
    data = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Dataset config must contain a mapping: {dataset_cfg}")
    if "path" not in data:
        raise KeyError(f"Dataset config has no 'path': {dataset_cfg}")

    root_value = Path(str(data["path"])).expanduser()
    if root_value.is_absolute():
        circuit_root = root_value
    else:
        repo_candidate = repo_root / root_value
        config_candidate = dataset_cfg.parent / root_value
        circuit_root = repo_candidate if repo_candidate.exists() else config_candidate

    file_format = str(data.get("format", "aig")).lstrip(".")
    names = data.get("files")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Dataset config contains no circuit files: {dataset_cfg}")

    circuits: list[Path] = []
    for value in names:
        name = str(value)
        candidate = circuit_root / name
        if not candidate.suffix:
            candidate = candidate.with_suffix(f".{file_format}")
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Circuit listed in {dataset_cfg} does not exist: {candidate}"
            )
        circuits.append(candidate)
    return circuits


def action_sequence_string(actions: Sequence[int]) -> str:
    return ";".join(
        ACTION_NAMES.get(int(action), f"action_{int(action)}") for action in actions
    )


def _entropy_from_counts(counts: Iterable[int]) -> float:
    values = np.asarray([int(value) for value in counts if int(value) > 0], dtype=float)
    if values.size == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def unique_recipe_fraction_curve(
    recipes: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> list[dict[str, float | int]]:
    if not recipes:
        return []
    if permutations <= 0:
        raise ValueError("permutations must be positive")

    recipe_array = np.asarray(list(recipes), dtype=object)
    rng = np.random.default_rng(int(seed))
    curves = np.empty((int(permutations), len(recipes)), dtype=float)
    for permutation_index in range(int(permutations)):
        order = rng.permutation(len(recipes))
        seen: set[str] = set()
        for sample_index, recipe_index in enumerate(order):
            seen.add(str(recipe_array[recipe_index]))
            curves[permutation_index, sample_index] = len(seen) / float(
                sample_index + 1
            )

    return [
        {
            "sample_budget": index + 1,
            "mean": float(curves[:, index].mean()),
            "p05": float(np.quantile(curves[:, index], 0.05)),
            "p95": float(np.quantile(curves[:, index], 0.95)),
        }
        for index in range(len(recipes))
    ]


def recipe_diversity_metrics(
    action_sequences: Sequence[Sequence[int]],
    *,
    curve_permutations: int,
    seed: int,
) -> dict[str, Any]:
    recipes = [action_sequence_string(actions) for actions in action_sequences]
    recipe_counts = Counter(recipes)
    horizon = max((len(actions) for actions in action_sequences), default=0)
    per_position: list[dict[str, float | int]] = []
    for position in range(horizon):
        active_actions = [
            int(actions[position])
            for actions in action_sequences
            if position < len(actions)
        ]
        per_position.append(
            {
                "position": position,
                "active_samples": len(active_actions),
                "entropy_nats": _entropy_from_counts(Counter(active_actions).values()),
            }
        )

    return {
        "unique_recipe_fraction_curve": unique_recipe_fraction_curve(
            recipes,
            permutations=int(curve_permutations),
            seed=int(seed),
        ),
        "recipe_entropy_nats": _entropy_from_counts(recipe_counts.values()),
        "action_entropy_nats": {
            "per_position": per_position,
            "mean": (
                float(np.mean([row["entropy_nats"] for row in per_position]))
                if per_position
                else 0.0
            ),
        },
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have the same length")
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.ptp(left_ranks) == 0.0 or np.ptp(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def pareto_front(
    rows: Sequence[Mapping[str, Any]], *, size_key: str, depth_key: str
) -> list[dict[str, Any]]:
    coordinates: dict[tuple[int, int], list[int]] = {}
    for row in rows:
        coordinate = (int(row[size_key]), int(row[depth_key]))
        coordinates.setdefault(coordinate, []).append(int(row["sample_index"]))

    front: list[dict[str, Any]] = []
    for size, depth in sorted(coordinates):
        dominated = any(
            other_size <= size
            and other_depth <= depth
            and (other_size < size or other_depth < depth)
            for other_size, other_depth in coordinates
        )
        if not dominated:
            front.append(
                {
                    "size": size,
                    "depth": depth,
                    "sample_indices": sorted(coordinates[(size, depth)]),
                }
            )
    return front


def _best_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    tie_keys: Sequence[str],
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row[value_key]),
            *(int(row[key]) for key in tie_keys),
            int(row["sample_index"]),
        ),
    )
    best_value = int(ordered[0][value_key])
    return {
        "value": best_value,
        "sample_indices": [
            int(row["sample_index"])
            for row in rows
            if int(row[value_key]) == best_value
        ],
        "selected_sample_index": int(ordered[0]["sample_index"]),
    }


def compute_group_metrics(
    rows: Sequence[Mapping[str, Any]],
    action_sequences: Sequence[Sequence[int]],
    *,
    curve_permutations: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute diversity metrics for an empty sample set")
    if len(rows) != len(action_sequences):
        raise ValueError("Rows and action sequences must have the same length")

    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["aig_product"] = int(item["aig_size"]) * int(item["aig_depth"])
        item["lut_product"] = int(item["lut_size"]) * int(item["lut_depth"])
        enriched.append(item)

    aig_pairs = {(int(row["aig_size"]), int(row["aig_depth"])) for row in enriched}
    lut_pairs = {(int(row["lut_size"]), int(row["lut_depth"])) for row in enriched}

    return {
        "recipe_diversity": recipe_diversity_metrics(
            action_sequences,
            curve_permutations=int(curve_permutations),
            seed=int(seed),
        ),
        "aig_diversity": {"unique_size_depth_pairs": len(aig_pairs)},
        "lut_diversity": {"unique_size_depth_pairs": len(lut_pairs)},
        "quality": {
            "aig": {
                "best_size": _best_value(
                    enriched,
                    value_key="aig_size",
                    tie_keys=("aig_depth",),
                ),
                "best_depth": _best_value(
                    enriched,
                    value_key="aig_depth",
                    tie_keys=("aig_size",),
                ),
                "best_size_depth_product": _best_value(
                    enriched,
                    value_key="aig_product",
                    tie_keys=("aig_size", "aig_depth"),
                ),
            },
            "lut": {
                "best_size": _best_value(
                    enriched,
                    value_key="lut_size",
                    tie_keys=("lut_depth",),
                ),
                "best_depth": _best_value(
                    enriched,
                    value_key="lut_depth",
                    tie_keys=("lut_size",),
                ),
                "best_size_depth_product": _best_value(
                    enriched,
                    value_key="lut_product",
                    tie_keys=("lut_size", "lut_depth"),
                ),
            },
            "lut_pareto_front": pareto_front(
                enriched,
                size_key="lut_size",
                depth_key="lut_depth",
            ),
            "aig_to_lut_rank_correlation": {
                "size": spearman_rank_correlation(
                    [int(row["aig_size"]) for row in enriched],
                    [int(row["lut_size"]) for row in enriched],
                ),
                "depth": spearman_rank_correlation(
                    [int(row["aig_depth"]) for row in enriched],
                    [int(row["lut_depth"]) for row in enriched],
                ),
            },
        },
    }


def parse_abc_lut_stats(output: str) -> tuple[int, int]:
    size_matches = _ABC_NODE_RE.findall(output)
    depth_matches = _ABC_LEVEL_RE.findall(output)
    if not size_matches or not depth_matches:
        raise ValueError(f"Could not parse LUT size/depth from ABC output:\n{output}")
    return int(size_matches[-1]), int(depth_matches[-1])


def _abc_quote(path: Path) -> str:
    value = str(path)
    if any(character in value for character in ('"', ";", "\n", "\r")):
        raise ValueError(f"ABC path contains an unsupported character: {path}")
    return f'"{value}"'


def _canonicalize_blif(path: Path) -> None:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith(".model "):
            line = ".model canonical"
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def map_to_lut6(
    *,
    abc_path: Path,
    aig_path: Path,
    lut_path: Path,
    timeout_seconds: float,
) -> tuple[int, int]:
    lut_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"read {_abc_quote(aig_path)}; "
        "strash; if -K 6; print_stats; "
        f"write_blif {_abc_quote(lut_path)}"
    )
    completed = subprocess.run(
        [str(abc_path), "-c", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
    )
    output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise RuntimeError(
            f"ABC failed with exit code {completed.returncode} while mapping {aig_path}:\n{output}"
        )
    if not lut_path.is_file():
        raise RuntimeError(
            f"ABC did not create mapped LUT network: {lut_path}\n{output}"
        )
    lut_size, lut_depth = parse_abc_lut_stats(output)
    _canonicalize_blif(lut_path)
    return lut_size, lut_depth


def _resolve_abc(value: str) -> Path:
    supplied = Path(value).expanduser()
    if supplied.parent != Path(".") or supplied.is_absolute():
        resolved = supplied.resolve()
    else:
        executable = shutil.which(value)
        if executable is None:
            raise FileNotFoundError(f"ABC executable was not found: {value}")
        resolved = Path(executable).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"ABC path is not an executable file: {resolved}")
    return resolved


def _checkpoint_metadata(checkpoint_path: Path) -> tuple[int, int]:
    import torch

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {checkpoint_path}")
    inferred_run = (
        int(_RUN_RE.match(checkpoint_path.parent.name).group(1))
        if _RUN_RE.match(checkpoint_path.parent.name)
        else 0
    )
    return int(payload.get("run_idx", inferred_run)), int(
        payload.get("seed", inferred_run)
    )


def _config_for_checkpoint(checkpoint_path: Path, explicit_config: Path | None) -> Path:
    if explicit_config is not None:
        return explicit_config
    inferred = checkpoint_path.parents[2] / ".hydra" / "config.yaml"
    if not inferred.is_file():
        raise FileNotFoundError(
            f"Could not infer training config for {checkpoint_path}. "
            "Pass --model-config explicitly."
        )
    return inferred


def _replay_and_save_aig(
    *,
    circuit_path: Path,
    actions: Sequence[int],
    aig_path: Path,
) -> tuple[int, int]:
    import pyspiel

    if not hasattr(pyspiel, "save_circuit"):
        raise RuntimeError("The loaded pyspiel build does not expose save_circuit")
    game = pyspiel.load_game(
        "circuit",
        {"num_steps": len(actions), "file_path": str(circuit_path)},
    )
    state = game.new_initial_state()
    for action in actions:
        legal_actions = state.legal_actions()
        if int(action) not in legal_actions:
            raise ValueError(
                f"Cannot replay illegal action {action} for {circuit_path}"
            )
        state.apply_action(int(action))
    observation = state.observation_tensor(0)
    aig_path.parent.mkdir(parents=True, exist_ok=True)
    if int(pyspiel.save_circuit(state, str(aig_path))) != 1:
        raise RuntimeError(f"Failed to save terminal AIG: {aig_path}")
    return int(observation[2]), int(observation[3])


def run_pipeline(
    *,
    dataset_cfg: Path,
    checkpoint: Path,
    num_samples: int,
    abc_path: Path,
    output_dir: Path,
    device_name: str | None,
    sampling_seed: int,
    model_config: Path | None,
    num_steps: int | None,
    curve_permutations: int,
    abc_timeout_seconds: float,
    gflownet_batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    import torch

    from src.sample_exp import _sample_trajectories
    from src.test import _get_algorithm_name, _load_cfg

    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(curve_permutations) <= 0:
        raise ValueError("curve_permutations must be positive")
    if float(abc_timeout_seconds) <= 0.0:
        raise ValueError("abc_timeout_seconds must be positive")
    if int(gflownet_batch_size) <= 0:
        raise ValueError("gflownet_batch_size must be positive")
    dataset_cfg = dataset_cfg.expanduser().resolve()
    if not dataset_cfg.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {dataset_cfg}")
    checkpoints = discover_checkpoints(checkpoint)
    repo_root = Path(__file__).resolve().parents[1]
    circuits = _load_circuits(dataset_cfg, repo_root)

    output_dir = output_dir.expanduser().resolve()
    stats_path = output_dir / "stats.csv"
    metrics_path = output_dir / "metrics.json"
    if not overwrite and (stats_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"{output_dir} already contains stats.csv or metrics.json; pass --overwrite to replace them"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    all_rows: list[dict[str, Any]] = []
    metric_groups: list[dict[str, Any]] = []

    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        run_id, training_seed = _checkpoint_metadata(checkpoint_path)
        config_path = _config_for_checkpoint(checkpoint_path, model_config)
        cfg = _load_cfg(config_path)
        algorithm = _get_algorithm_name(cfg)
        resolved_num_steps = int(
            num_steps if num_steps is not None else cfg["num_steps"]
        )

        for circuit_index, circuit_path in enumerate(circuits):
            circuit_name = _safe_component(circuit_path.stem)
            group_root = (
                output_dir
                / circuit_name
                / (
                    f"model_{checkpoint_index:03d}_run_{run_id:03d}"
                    f"_seed_{training_seed}"
                )
            )
            aig_dir = group_root / "aig"
            lut_dir = group_root / "lut"
            evaluation_seed = int(sampling_seed) + circuit_index * 10_000
            sampled = _sample_trajectories(
                checkpoint_path=checkpoint_path,
                cfg=cfg,
                circuit_path=circuit_path,
                num_steps=resolved_num_steps,
                num_samples=int(num_samples),
                device=device,
                seed=evaluation_seed,
                pcn_sampling_mode="target",
                pcn_zero_variance_jitter=0.05,
                gflownet_batch_size=int(gflownet_batch_size),
                include_actions=True,
            )
            if len(sampled) != int(num_samples):
                raise RuntimeError(
                    f"Expected {num_samples} samples from {checkpoint_path}, got {len(sampled)}"
                )

            group_rows: list[dict[str, Any]] = []
            group_actions: list[list[int]] = []
            for sample_index, sample in enumerate(sampled):
                actions_value = sample.get("actions")
                if not isinstance(actions_value, list):
                    raise TypeError("Sampler result does not contain an action list")
                actions = [int(action) for action in actions_value]
                aig_path = aig_dir / f"{sample_index:06d}.aig"
                lut_path = lut_dir / f"{sample_index:06d}.blif"
                aig_size, aig_depth = _replay_and_save_aig(
                    circuit_path=circuit_path,
                    actions=actions,
                    aig_path=aig_path,
                )
                if aig_size != int(sample["size"]) or aig_depth != int(sample["depth"]):
                    raise RuntimeError(
                        "Replayed AIG metrics do not match sampled metrics for "
                        f"{checkpoint_path}, {circuit_path}, sample {sample_index}"
                    )
                lut_size, lut_depth = map_to_lut6(
                    abc_path=abc_path,
                    aig_path=aig_path,
                    lut_path=lut_path,
                    timeout_seconds=abc_timeout_seconds,
                )
                row = {
                    "circuit": str(circuit_path),
                    "algorithm": algorithm,
                    "run_id": run_id,
                    "training_seed": training_seed,
                    "sampling_seed": evaluation_seed,
                    "checkpoint": str(checkpoint_path),
                    "sample_index": sample_index,
                    "action_sequence": action_sequence_string(actions),
                    "aig_path": str(aig_path.relative_to(output_dir)),
                    "aig_sha256": _sha256(aig_path),
                    "aig_size": aig_size,
                    "aig_depth": aig_depth,
                    "lut_path": str(lut_path.relative_to(output_dir)),
                    "lut_sha256": _sha256(lut_path),
                    "lut_size": lut_size,
                    "lut_depth": lut_depth,
                }
                group_rows.append(row)
                group_actions.append(actions)
                all_rows.append(row)

            group_metrics = compute_group_metrics(
                group_rows,
                group_actions,
                curve_permutations=int(curve_permutations),
                seed=evaluation_seed,
            )
            metric_groups.append(
                {
                    "circuit": str(circuit_path),
                    "algorithm": algorithm,
                    "run_id": run_id,
                    "training_seed": training_seed,
                    "sampling_seed": evaluation_seed,
                    "checkpoint": str(checkpoint_path),
                    "num_samples": len(group_rows),
                    "recipe_horizon": resolved_num_steps,
                    **group_metrics,
                }
            )
            _atomic_csv(stats_path, all_rows)
            _atomic_json(
                metrics_path,
                {
                    "schema_version": 1,
                    "inputs": {
                        "dataset_cfg": str(dataset_cfg),
                        "checkpoint": str(checkpoint.expanduser().resolve()),
                        "num_samples_per_checkpoint_and_circuit": int(num_samples),
                        "abc_path": str(abc_path),
                        "lut_size": 6,
                        "sampling_seed": int(sampling_seed),
                        "curve_permutations": int(curve_permutations),
                        "gflownet_batch_size": int(gflownet_batch_size),
                    },
                    "groups": metric_groups,
                },
            )

    return {
        "stats_path": str(stats_path),
        "metrics_path": str(metrics_path),
        "rows": len(all_rows),
        "groups": len(metric_groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample trained policies and evaluate AIG and 6-LUT diversity.",
    )
    parser.add_argument(
        "--dataset-cfg", required=True, help="Circuit dataset YAML config."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file, saved_models directory, or experiment output directory.",
    )
    parser.add_argument(
        "--num-samples",
        required=True,
        type=int,
        help="Samples per checkpoint and circuit.",
    )
    parser.add_argument(
        "--abc-path", required=True, help="Path or command name for the ABC executable."
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--device", default=None, help="Torch device; defaults to CUDA when available."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base inference sampling seed."
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Training Hydra config; inferred from the checkpoint location by default.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the training recipe horizon.",
    )
    parser.add_argument(
        "--curve-permutations",
        type=int,
        default=100,
        help="Random permutations used for the unique-recipe fraction curve.",
    )
    parser.add_argument(
        "--abc-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for mapping one AIG.",
    )
    parser.add_argument(
        "--gflownet-batch-size",
        type=int,
        default=32,
        help="Number of GFlowNet trajectories sampled in one inference batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace stats.csv and metrics.json if they already exist.",
    )
    args = parser.parse_args()

    abc_path = _resolve_abc(str(args.abc_path))
    result = run_pipeline(
        dataset_cfg=Path(args.dataset_cfg),
        checkpoint=Path(args.checkpoint),
        num_samples=int(args.num_samples),
        abc_path=abc_path,
        output_dir=Path(args.output_dir),
        device_name=args.device,
        sampling_seed=int(args.seed),
        model_config=(
            Path(args.model_config).expanduser().resolve()
            if args.model_config
            else None
        ),
        num_steps=args.num_steps,
        curve_permutations=int(args.curve_permutations),
        abc_timeout_seconds=float(args.abc_timeout_seconds),
        gflownet_batch_size=int(args.gflownet_batch_size),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
