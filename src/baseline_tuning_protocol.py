"""Declarative task expansion for the baseline training-tuning campaign."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import yaml


PROTOCOL_PATH = Path("cfg/exp/baseline_tuning/protocol.yaml")
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(value: object) -> str:
    return _SAFE_COMPONENT.sub("-", str(value)).strip("-")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineSetting:
    algorithm: str
    setting_id: str
    profile_id: str
    training_trajectories: int
    overrides: dict[str, Any] = field(default_factory=dict)
    role: str = "candidate"
    control: bool = False
    source_stage: str = ""
    fragment: str | None = None
    exploration_rank: float = 0.0

    @property
    def key(self) -> str:
        return canonical_json({"algorithm": self.algorithm, "overrides": self.overrides})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BaselineSetting":
        return cls(
            algorithm=str(value["algorithm"]),
            setting_id=str(value["setting_id"]),
            profile_id=str(value.get("profile_id", value["setting_id"])),
            training_trajectories=int(value["training_trajectories"]),
            overrides=dict(value.get("overrides", {})),
            role=str(value.get("role", "candidate")),
            control=bool(value.get("control", False)),
            source_stage=str(value.get("source_stage", "")),
        )


@dataclass(frozen=True)
class BaselineTask:
    stage: str
    task_id: str
    algorithm: str
    report_algorithm: str
    setting: BaselineSetting
    circuit: str
    dataset_cfg: str
    circuit_path: str
    seed: int
    base_config: str
    episodes: int
    training_trajectories: int
    evaluation_samples: int
    fixed_overrides: dict[str, Any]
    profile_id: str
    role: str

    @property
    def reuse_key(self) -> str:
        return canonical_json(
            {
                "algorithm": self.algorithm,
                "base_config": self.base_config,
                "profile": self.setting.key,
                "circuit": self.circuit,
                "seed": self.seed,
                "episodes": self.episodes,
                "training_trajectories": self.training_trajectories,
                "evaluation_samples": self.evaluation_samples,
                "fixed_overrides": self.fixed_overrides,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["reuse_key"] = self.reuse_key
        return output


class BaselineTuningProtocol:
    def __init__(self, *, path: Path, data: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.repo_root = self.path.parents[3]
        self.data = dict(data)
        self.protocol_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.validate_structure()

    @classmethod
    def load(cls, path: Path | str = PROTOCOL_PATH) -> "BaselineTuningProtocol":
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        with resolved.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, Mapping):
            raise ProtocolError("protocol root must be a mapping")
        return cls(path=resolved, data=data)

    @property
    def stages(self) -> Mapping[str, Any]:
        return self.data["stages"]

    @property
    def algorithms(self) -> Mapping[str, Any]:
        return self.data["algorithms"]

    def validate_structure(self) -> None:
        required = {"version", "campaign", "common", "circuits", "algorithms", "stages", "martin"}
        missing = required - set(self.data)
        if missing:
            raise ProtocolError(f"protocol missing keys: {sorted(missing)}")
        if int(self.data["version"]) != 1:
            raise ProtocolError("only baseline-tuning protocol version 1 is supported")
        common = self.data["common"]
        grid = [int(value) for value in common["budget_grid"]]
        if grid != sorted(set(grid)) or any(value <= 0 for value in grid):
            raise ProtocolError("common.budget_grid must be strictly increasing and positive")
        extension = int(common["conditional_extension_trajectory"])
        if extension <= grid[-1]:
            raise ProtocolError("conditional extension must exceed the budget grid")
        for circuit, cfg in self.data["circuits"].items():
            for key in ("dataset_cfg", "circuit_path"):
                path = self.repo_root / str(cfg[key])
                if not path.is_file():
                    raise ProtocolError(f"{circuit} {key} does not exist: {path}")
        for algorithm, cfg in self.algorithms.items():
            base = self.repo_root / "cfg" / f"{cfg['base_config']}.yaml"
            if not base.is_file():
                raise ProtocolError(f"{algorithm} base config does not exist: {base}")
            trajectories = int(cfg["trajectories_per_episode"])
            if trajectories <= 0:
                raise ProtocolError(f"{algorithm} trajectories_per_episode must be positive")
            profiles = cfg.get("profiles", [])
            ids = [str(profile["id"]) for profile in profiles]
            if len(ids) < 2 or len(ids) != len(set(ids)):
                raise ProtocolError(f"{algorithm} must define at least two unique profiles")
            if sum(bool(profile.get("control", False)) for profile in profiles) != 1:
                raise ProtocolError(f"{algorithm} must define exactly one control profile")
            fixed = dict(cfg.get("fixed_overrides", {}))
            for profile in profiles:
                overlap = set(fixed) & set(profile.get("overrides", {}))
                if overlap:
                    raise ProtocolError(
                        f"{algorithm}/{profile['id']} overrides frozen fields: {sorted(overlap)}"
                    )
            frozen = dict(cfg.get("frozen_exploration", {}))
            for path, expected in frozen.items():
                if path in {"source", "status"}:
                    continue
                if fixed.get(path) != expected:
                    raise ProtocolError(
                        f"{algorithm} frozen exploration {path} must equal fixed override {expected}"
                    )
            if algorithm == "drills" and fixed.get("algorithm.drills.trajectories_per_episode") != trajectories:
                raise ProtocolError("DRiLLS trajectory translation must be pinned in fixed_overrides")
            if algorithm == "ppo":
                rollout_steps = int(fixed.get("algorithm.ppo.rollout_steps", 0))
                if rollout_steps != trajectories * int(common["num_steps"]):
                    raise ProtocolError("PPO rollout_steps must exactly match the trajectory translation")
            for budget in [20, int(common["screen_training_trajectories"]), *grid, extension]:
                if budget % trajectories:
                    raise ProtocolError(f"budget {budget} is not divisible for {algorithm}")
        for stage, cfg in self.stages.items():
            dependencies = [str(value) for value in cfg.get("dependencies", [])]
            for dependency in dependencies:
                if dependency not in self.stages:
                    raise ProtocolError(f"stage {stage} has unknown dependency {dependency}")
            for circuit in cfg["circuits"]:
                if circuit not in self.data["circuits"]:
                    raise ProtocolError(f"stage {stage} has unknown circuit {circuit}")
            algorithm = cfg.get("algorithm")
            if algorithm is not None and algorithm not in self.algorithms:
                raise ProtocolError(f"stage {stage} has unknown algorithm {algorithm}")
            profile_ids = (
                {
                    str(profile["id"])
                    for profile in self.algorithms[str(algorithm)]["profiles"]
                }
                if algorithm is not None
                else set()
            )
            explicit_profiles = cfg.get("profiles")
            if explicit_profiles is not None:
                if cfg.get("kind") != "budget" or algorithm is None:
                    raise ProtocolError(
                        f"stage {stage} explicit profiles require a budget algorithm"
                    )
                if dependencies:
                    raise ProtocolError(
                        f"stage {stage} cannot combine explicit profiles with dependencies"
                    )
                if not isinstance(explicit_profiles, list):
                    raise ProtocolError(f"stage {stage} profiles must be a list")
                selected_profiles = [str(value) for value in explicit_profiles]
                if len(selected_profiles) != 2 or len(set(selected_profiles)) != 2:
                    raise ProtocolError(
                        f"stage {stage} must declare exactly two distinct profiles"
                    )
                if not set(selected_profiles) <= profile_ids:
                    raise ProtocolError(
                        f"stage {stage} profiles contains an unknown profile"
                    )
            elif cfg.get("kind") == "budget" and len(dependencies) != 1:
                raise ProtocolError(
                    f"stage {stage} budget requires one screen dependency or explicit profiles"
                )
            raw_stage_interactions = cfg.get("interaction_profiles", [])
            if not isinstance(raw_stage_interactions, list):
                raise ProtocolError(f"stage {stage} interaction_profiles must be a list")
            stage_interactions = {str(value) for value in raw_stage_interactions}
            if stage_interactions:
                if cfg.get("kind") != "screen" or algorithm is None:
                    raise ProtocolError(
                        f"stage {stage} interaction_profiles requires a screen algorithm"
                    )
                if not stage_interactions <= profile_ids:
                    raise ProtocolError(
                        f"stage {stage} interaction_profiles contains an unknown profile"
                    )
            paired_contrasts = cfg.get("paired_contrasts", [])
            if not isinstance(paired_contrasts, list):
                raise ProtocolError(f"stage {stage} paired_contrasts must be a list")
            if paired_contrasts:
                if cfg.get("kind") != "screen" or algorithm is None:
                    raise ProtocolError(
                        f"stage {stage} paired_contrasts requires a screen algorithm"
                    )
                if any(not isinstance(value, Mapping) for value in paired_contrasts):
                    raise ProtocolError(
                        f"stage {stage} paired_contrasts entries must be mappings"
                    )
                contrast_ids = [str(value.get("id", "")) for value in paired_contrasts]
                if (
                    any(not value or _safe(value) != value for value in contrast_ids)
                    or len(contrast_ids) != len(set(contrast_ids))
                ):
                    raise ProtocolError(
                        f"stage {stage} paired contrast ids must be unique and safe"
                    )
                for contrast in paired_contrasts:
                    terms = contrast.get("terms", {})
                    if not isinstance(terms, Mapping) or len(terms) < 2:
                        raise ProtocolError(
                            f"stage {stage} contrast {contrast['id']} requires at least two terms"
                        )
                    if not set(terms) <= profile_ids:
                        raise ProtocolError(
                            f"stage {stage} contrast {contrast['id']} contains an unknown profile"
                        )
                    weights = list(terms.values())
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) == 0.0
                        for value in weights
                    ):
                        raise ProtocolError(
                            f"stage {stage} contrast {contrast['id']} has an invalid weight"
                        )
                    if not math.isclose(
                        sum(float(value) for value in weights), 0.0, abs_tol=1e-12
                    ):
                        raise ProtocolError(
                            f"stage {stage} contrast {contrast['id']} weights must sum to zero"
                        )
            factorial = cfg.get("factorial")
            if factorial is not None:
                if cfg.get("kind") != "screen" or algorithm is None:
                    raise ProtocolError(
                        f"stage {stage} factorial metadata requires a screen algorithm"
                    )
                factors = [str(value) for value in factorial.get("factors", [])]
                if len(factors) < 2 or len(factors) != len(set(factors)):
                    raise ProtocolError(f"stage {stage} factorial factors must be unique")
                raw_cells = factorial.get("cells", {})
                if not isinstance(raw_cells, Mapping):
                    raise ProtocolError(f"stage {stage} factorial cells must be a mapping")
                if set(raw_cells) != profile_ids:
                    raise ProtocolError(
                        f"stage {stage} factorial cells must map exactly all algorithm profiles"
                    )
                cells: list[tuple[int, ...]] = []
                for profile_id, raw_cell in raw_cells.items():
                    if not isinstance(raw_cell, list) or len(raw_cell) != len(factors):
                        raise ProtocolError(
                            f"stage {stage} factorial cell {profile_id} has the wrong dimension"
                        )
                    if any(value not in {0, 1} for value in raw_cell):
                        raise ProtocolError(
                            f"stage {stage} factorial cell {profile_id} must contain only 0/1"
                        )
                    cell = tuple(int(value) for value in raw_cell)
                    cells.append(cell)
                expected_cells = set(product((0, 1), repeat=len(factors)))
                if set(cells) != expected_cells or len(cells) != len(expected_cells):
                    raise ProtocolError(
                        f"stage {stage} factorial cells are incomplete or duplicated"
                    )
                interactions = {str(value) for value in factorial.get("interaction_profiles", [])}
                expected_interactions = {
                    str(profile_id)
                    for profile_id, cell in raw_cells.items()
                    if sum(cell) >= 2
                }
                if interactions != expected_interactions:
                    raise ProtocolError(
                        f"stage {stage} interaction_profiles must identify all interaction cells"
                    )
        jobs = self.data["martin"]["job_names"]
        if set(jobs) != set(self.stages):
            raise ProtocolError("martin.job_names must contain exactly all stages")
        for value in jobs.values():
            if _safe(value) != value:
                raise ProtocolError(f"unsafe Martin job name: {value}")

    def dependency_stages(self, stage: str, *, transitive: bool = True) -> list[str]:
        if stage not in self.stages:
            raise ProtocolError(f"unknown stage: {stage}")
        direct = [str(value) for value in self.stages[stage].get("dependencies", [])]
        if not transitive:
            return direct
        output: list[str] = []
        for dependency in direct:
            for ancestor in self.dependency_stages(dependency, transitive=True):
                if ancestor not in output:
                    output.append(ancestor)
            if dependency not in output:
                output.append(dependency)
        return output

    def artifact_root(self, stage: str, *, artifact_base: Path | None = None) -> Path:
        base = artifact_base or Path(self.data["martin"]["artifact_base"])
        return base / str(self.data["martin"]["job_names"][stage])

    def _profile(self, algorithm: str, profile_id: str) -> Mapping[str, Any]:
        matches = [profile for profile in self.algorithms[algorithm]["profiles"] if profile["id"] == profile_id]
        if len(matches) != 1:
            raise ProtocolError(f"unknown profile: {algorithm}/{profile_id}")
        return matches[0]

    def profile_setting(
        self,
        *,
        algorithm: str,
        profile_id: str,
        training_trajectories: int,
        source_stage: str,
        role: str = "candidate",
        setting_id: str | None = None,
    ) -> BaselineSetting:
        raw = self._profile(algorithm, profile_id)
        return BaselineSetting(
            algorithm=algorithm,
            setting_id=setting_id or f"{profile_id}__b{int(training_trajectories)}",
            profile_id=profile_id,
            training_trajectories=int(training_trajectories),
            overrides=dict(raw.get("overrides", {})),
            role=role,
            control=bool(raw.get("control", False)),
            source_stage=source_stage,
        )

    @staticmethod
    def _algorithm_selection(payloads: Mapping[str, Any], stage: str, algorithm: str) -> Mapping[str, Any]:
        try:
            return payloads[stage]["algorithms"][algorithm]
        except (KeyError, TypeError) as exc:
            raise ProtocolError(f"selection unavailable for {stage}/{algorithm}") from exc

    def budget_profiles(
        self,
        stage: str,
        *,
        payloads: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Resolve the two profiles for dependency-driven or standalone budget stages."""
        cfg = self.stages[stage]
        if str(cfg.get("kind")) != "budget":
            raise ProtocolError(f"stage is not a budget stage: {stage}")
        explicit = cfg.get("profiles")
        if explicit is not None:
            return [str(value) for value in explicit]
        algorithm = str(cfg["algorithm"])
        screen_stage = str(cfg["dependencies"][0])
        selected = self._algorithm_selection(payloads or {}, screen_stage, algorithm)
        return [str(value) for value in selected["top_profiles"]]

    def settings_for_stage(
        self,
        stage: str,
        *,
        payloads: Mapping[str, Any] | None = None,
    ) -> list[BaselineSetting]:
        payloads = payloads or {}
        cfg = self.stages[stage]
        kind = str(cfg["kind"])
        common = self.data["common"]
        if kind == "smoke":
            output = []
            for algorithm in cfg["algorithms"]:
                control = next(profile for profile in self.algorithms[algorithm]["profiles"] if profile.get("control"))
                output.append(
                    self.profile_setting(
                        algorithm=algorithm,
                        profile_id=str(control["id"]),
                        training_trajectories=20,
                        source_stage=stage,
                        role="smoke",
                    )
                )
            return output
        algorithm = str(cfg.get("algorithm", ""))
        if kind == "screen":
            return [
                self.profile_setting(
                    algorithm=algorithm,
                    profile_id=str(profile["id"]),
                    training_trajectories=int(common["screen_training_trajectories"]),
                    source_stage=stage,
                    role="screen",
                )
                for profile in self.algorithms[algorithm]["profiles"]
            ]
        if kind == "budget":
            profiles = self.budget_profiles(stage, payloads=payloads)
            return [
                self.profile_setting(
                    algorithm=algorithm,
                    profile_id=profile,
                    training_trajectories=int(budget),
                    source_stage=stage,
                    role="budget_curve",
                )
                for profile in profiles
                for budget in common["budget_grid"]
            ]
        if kind == "extend":
            budget_stage = str(cfg["dependencies"][0])
            selected = self._algorithm_selection(payloads, budget_stage, algorithm)
            if selected["status"] != "extend_required":
                return []
            return [
                self.profile_setting(
                    algorithm=algorithm,
                    profile_id=str(profile),
                    training_trajectories=int(common["conditional_extension_trajectory"]),
                    source_stage=stage,
                    role="conditional_extension",
                )
                for profile in selected["top_profiles"]
            ]
        if kind == "confirm":
            extend_stage = str(cfg["dependencies"][0])
            selected = self._algorithm_selection(payloads, extend_stage, algorithm)
            if selected["status"] != "selected":
                raise ProtocolError(
                    f"{algorithm} has no selectable budget after extension: {selected.get('status')}"
                )
            budget = int(selected["selected_budget"])
            successor = int(selected["successor_budget"])
            winner = str(selected["selected_profile"])
            runner_up = str(selected["runner_up_profile"])
            return [
                self.profile_setting(
                    algorithm=algorithm,
                    profile_id=winner,
                    training_trajectories=budget,
                    source_stage=stage,
                    role="selected",
                    setting_id=f"selected__{winner}__b{budget}",
                ),
                self.profile_setting(
                    algorithm=algorithm,
                    profile_id=runner_up,
                    training_trajectories=budget,
                    source_stage=stage,
                    role="runner_up",
                    setting_id=f"runner_up__{runner_up}__b{budget}",
                ),
                self.profile_setting(
                    algorithm=algorithm,
                    profile_id=winner,
                    training_trajectories=successor,
                    source_stage=stage,
                    role="successor",
                    setting_id=f"successor__{winner}__b{successor}",
                ),
            ]
        if kind == "common":
            confirmed: dict[str, Mapping[str, Any]] = {}
            for dependency in cfg["dependencies"]:
                payload = payloads[str(dependency)]
                for name, selection in payload["algorithms"].items():
                    if selection.get("status") != "confirmed":
                        raise ProtocolError(f"{name} confirmation did not pass")
                    confirmed[str(name)] = selection
            budgets = sorted(int(value["selected_budget"]) for value in confirmed.values())
            common_budget = int(median(budgets))
            grid = [int(value) for value in common["budget_grid"]] + [
                int(common["conditional_extension_trajectory"])
            ]
            common_budget = next((value for value in grid if value >= common_budget), grid[-1])
            return [
                self.profile_setting(
                    algorithm=algorithm_name,
                    profile_id=str(confirmed[algorithm_name]["selected_profile"]),
                    training_trajectories=common_budget,
                    source_stage=stage,
                    role="common_budget",
                    setting_id=f"common__{confirmed[algorithm_name]['selected_profile']}__b{common_budget}",
                )
                for algorithm_name in cfg["algorithms"]
            ]
        raise ProtocolError(f"unsupported stage kind: {kind}")

    def build_tasks(self, stage: str, settings: Iterable[BaselineSetting]) -> list[BaselineTask]:
        cfg = self.stages[stage]
        kind = str(cfg["kind"])
        seeds = [0] if kind == "smoke" else (
            list(self.data["common"]["confirmation_seeds"])
            if kind in {"confirm", "common"}
            else list(self.data["common"]["screen_seeds"])
        )
        evaluation_samples = 2 if kind == "smoke" else int(self.data["common"]["evaluation_samples"])
        tasks: list[BaselineTask] = []
        for setting in settings:
            algorithm_cfg = self.algorithms[setting.algorithm]
            trajectories_per_episode = int(algorithm_cfg["trajectories_per_episode"])
            if setting.training_trajectories % trajectories_per_episode:
                raise ProtocolError(f"budget is not divisible for {setting.algorithm}")
            episodes = setting.training_trajectories // trajectories_per_episode
            for circuit in cfg["circuits"]:
                circuit_cfg = self.data["circuits"][circuit]
                for seed in seeds:
                    task_id = "__".join(
                        _safe(value)
                        for value in (
                            setting.algorithm,
                            setting.role,
                            setting.profile_id,
                            f"b{setting.training_trajectories}",
                            circuit,
                            f"seed{int(seed)}",
                        )
                    )
                    tasks.append(
                        BaselineTask(
                            stage=stage,
                            task_id=task_id,
                            algorithm=setting.algorithm,
                            report_algorithm=str(algorithm_cfg["report_algorithm"]),
                            setting=setting,
                            circuit=str(circuit),
                            dataset_cfg=str(circuit_cfg["dataset_cfg"]),
                            circuit_path=str(circuit_cfg["circuit_path"]),
                            seed=int(seed),
                            base_config=str(algorithm_cfg["base_config"]),
                            episodes=episodes,
                            training_trajectories=setting.training_trajectories,
                            evaluation_samples=evaluation_samples,
                            fixed_overrides=dict(algorithm_cfg.get("fixed_overrides", {})),
                            profile_id=setting.profile_id,
                            role=setting.role,
                        )
                    )
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ProtocolError(f"stage {stage} generated duplicate task IDs")
        return tasks


__all__ = [
    "BaselineSetting",
    "BaselineTask",
    "BaselineTuningProtocol",
    "PROTOCOL_PATH",
    "ProtocolError",
    "canonical_json",
]
