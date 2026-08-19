"""Declarative protocol and task expansion for exploration tuning."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROTOCOL_PATH = Path("cfg/exp/exploration_tuning/protocol.yaml")
ENTROPY_ALGORITHMS = ("reinforce", "ppo", "drills")
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(value: object) -> str:
    return _SAFE_COMPONENT.sub("-", str(value)).strip("-")


def _number_id(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def _gfn_exploration_rank(start: float, end: float, warmup: int, decay: int, episodes: int = 50) -> float:
    warmup = min(int(episodes), max(0, int(warmup)))
    decay = min(max(0, int(episodes) - warmup), max(0, int(decay)))
    consolidation = max(0, int(episodes) - warmup - decay)
    total = warmup * float(start) + decay * (float(start) + float(end)) / 2.0 + consolidation * float(end)
    return total / max(1, int(episodes))


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Setting:
    algorithm: str
    setting_id: str
    overrides: dict[str, Any] = field(default_factory=dict)
    fragment: str | None = None
    exploration_rank: float = 0.0
    control: bool = False
    source_stage: str = ""

    @property
    def key(self) -> str:
        return canonical_json({"algorithm": self.algorithm, "overrides": self.overrides})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Setting":
        return cls(
            algorithm=str(value["algorithm"]),
            setting_id=str(value["setting_id"]),
            overrides=dict(value.get("overrides", {})),
            fragment=str(value["fragment"]) if value.get("fragment") else None,
            exploration_rank=float(value.get("exploration_rank", 0.0)),
            control=bool(value.get("control", False)),
            source_stage=str(value.get("source_stage", "")),
        )


@dataclass(frozen=True)
class ExperimentTask:
    stage: str
    task_id: str
    algorithm: str
    report_algorithm: str
    setting: Setting
    circuit: str
    dataset_cfg: str
    circuit_path: str
    seed: int
    base_config: str
    episodes: int
    training_trajectories: int
    evaluation_samples: int
    fixed_overrides: dict[str, Any]

    @property
    def reuse_key(self) -> str:
        return canonical_json(
            {
                "algorithm": self.algorithm,
                "base_config": self.base_config,
                "setting": self.setting.key,
                "circuit": self.circuit,
                "seed": self.seed,
                "episodes": self.episodes,
                "training_trajectories": self.training_trajectories,
                "evaluation_samples": self.evaluation_samples,
                "fixed_overrides": self.fixed_overrides,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reuse_key"] = self.reuse_key
        return out


class ProtocolError(ValueError):
    pass


class ExplorationProtocol:
    def __init__(self, *, path: Path, data: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.repo_root = self.path.parents[3]
        self.data = dict(data)
        self.protocol_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.validate_structure()

    @classmethod
    def load(cls, path: Path | str = PROTOCOL_PATH) -> "ExplorationProtocol":
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

    def validate_structure(self) -> None:
        required = {
            "version", "campaign", "prerequisites", "common", "budgets",
            "circuits", "algorithms", "refinement", "stages", "martin",
        }
        missing = required - set(self.data)
        if missing:
            raise ProtocolError(f"protocol missing keys: {sorted(missing)}")
        if int(self.data["version"]) != 1:
            raise ProtocolError("only exploration protocol version 1 is supported")
        for stage, cfg in self.stages.items():
            for dependency in cfg.get("dependencies", []):
                if dependency not in self.stages:
                    raise ProtocolError(f"stage {stage} has unknown dependency {dependency}")
            budget = str(cfg["budget"])
            if budget not in self.data["budgets"]:
                raise ProtocolError(f"stage {stage} has unknown budget {budget}")
            for circuit in cfg["circuits"]:
                if circuit not in self.data["circuits"]:
                    raise ProtocolError(f"stage {stage} has unknown circuit {circuit}")
        for circuit, cfg in self.data["circuits"].items():
            for key in ("dataset_cfg", "circuit_path"):
                path = self.repo_root / str(cfg[key])
                if not path.is_file():
                    raise ProtocolError(f"{circuit} {key} does not exist: {path}")
        for algorithm, cfg in self.data["algorithms"].items():
            base = self.repo_root / "cfg" / f"{cfg['base_config']}.yaml"
            if not base.is_file():
                raise ProtocolError(f"{algorithm} base config does not exist: {base}")
            trajectories = int(cfg["trajectories_per_episode"])
            if trajectories <= 0:
                raise ProtocolError(f"{algorithm} trajectories_per_episode must be positive")
            for setting in cfg["coarse"]:
                if setting.get("fragment"):
                    self._fragment_overrides(str(setting["fragment"]))
        gfn = self.data["algorithms"]["gflownet"]
        source = gfn.get("optimizer_source", {})
        required_source = {
            "branch", "commit", "status", "optimizer", "policy_learning_rate",
            "log_z_learning_rate", "trajectories_per_update",
        }
        missing_source = required_source - set(source)
        if missing_source:
            raise ProtocolError(f"gflownet optimizer_source missing keys: {sorted(missing_source)}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(source["commit"])):
            raise ProtocolError("gflownet optimizer_source.commit must be a full Git SHA")
        expected_overrides = {
            "learning_rate": float(source["policy_learning_rate"]),
            "tb.log_z_learning_rate": float(source["log_z_learning_rate"]),
            "tb.trajectories_per_episode": int(source["trajectories_per_update"]),
        }
        actual_overrides = gfn.get("fixed_overrides", {})
        for path, expected in expected_overrides.items():
            if actual_overrides.get(path) != expected:
                raise ProtocolError(
                    f"gflownet fixed override {path} must match optimizer_source: {expected}"
                )
        jobs = self.data["martin"]["job_names"]
        if set(jobs) != set(self.stages):
            raise ProtocolError("martin.job_names must contain exactly all stages")
        for job_name in jobs.values():
            if _safe(job_name) != job_name:
                raise ProtocolError(f"unsafe Martin job name: {job_name}")

    def dependency_stages(self, stage: str, *, transitive: bool = True) -> list[str]:
        if stage not in self.stages:
            raise ProtocolError(f"unknown stage: {stage}")
        direct = [str(value) for value in self.stages[stage].get("dependencies", [])]
        if not transitive:
            return direct
        out: list[str] = []
        for dependency in direct:
            for ancestor in self.dependency_stages(dependency, transitive=True):
                if ancestor not in out:
                    out.append(ancestor)
            if dependency not in out:
                out.append(dependency)
        return out

    def artifact_root(self, stage: str, *, artifact_base: Path | None = None) -> Path:
        base = artifact_base or Path(self.data["martin"]["artifact_base"])
        return base / str(self.data["martin"]["job_names"][stage])

    def _fragment_overrides(self, fragment: str) -> dict[str, Any]:
        path = self.repo_root / "cfg" / "exp" / "exploration_tuning" / f"{fragment}.yaml"
        if not path.is_file():
            raise ProtocolError(f"exploration fragment does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ProtocolError(f"exploration fragment must be a mapping: {path}")
        return _flatten({key: value for key, value in data.items() if key != "@package"})

    def coarse_settings(self, algorithm: str, *, source_stage: str = "coarse") -> list[Setting]:
        cfg = self.data["algorithms"][algorithm]
        out: list[Setting] = []
        for raw in cfg["coarse"]:
            fragment = str(raw["fragment"]) if raw.get("fragment") else None
            overrides = self._fragment_overrides(fragment) if fragment else dict(raw.get("overrides", {}))
            out.append(
                Setting(
                    algorithm=algorithm,
                    setting_id=str(raw["id"]),
                    overrides=overrides,
                    fragment=fragment,
                    exploration_rank=float(raw.get("exploration_rank", 0.0)),
                    control=bool(raw.get("control", False)),
                    source_stage=source_stage,
                )
            )
        return out

    def control_setting(self, algorithm: str) -> Setting:
        controls = [setting for setting in self.coarse_settings(algorithm) if setting.control]
        if len(controls) != 1:
            raise ProtocolError(f"{algorithm} must define exactly one coarse control")
        return controls[0]

    def _selection_setting(self, selections: Mapping[str, Any], algorithm: str, field: str) -> Setting:
        try:
            raw = selections[algorithm][field]
        except (KeyError, TypeError) as exc:
            raise ProtocolError(f"selection for {algorithm}.{field} is unavailable") from exc
        return Setting.from_dict(raw)

    def settings_for_stage(self, stage: str, *, selections: Mapping[str, Any] | None = None) -> list[Setting]:
        selections = selections or {}
        if stage == "smoke":
            return [
                Setting(
                    **{
                        **self.control_setting(algorithm).to_dict(),
                        "setting_id": f"smoke_{self.control_setting(algorithm).setting_id}",
                        "source_stage": stage,
                    }
                )
                for algorithm in self.data["algorithms"]
            ]
        if stage == "coarse":
            return [
                setting
                for algorithm in self.data["algorithms"]
                for setting in self.coarse_settings(algorithm, source_stage=stage)
            ]
        if stage == "gfn_start":
            cfg = self.data["refinement"][stage]
            settings = [self.control_setting("gflownet")]
            for start in cfg["epsilon_start"]:
                settings.append(
                    Setting(
                        algorithm="gflownet",
                        setting_id=f"start_{_number_id(start)}",
                        overrides={
                            "tb.exploration_epsilon_enabled": True,
                            "tb.exploration_epsilon_start": float(start),
                            "tb.exploration_epsilon_end": float(cfg["epsilon_end"]),
                            "tb.exploration_warmup_episodes": int(cfg["warmup"]),
                            "tb.exploration_decay_episodes": int(cfg["decay"]),
                        },
                        exploration_rank=_gfn_exploration_rank(
                            float(start), float(cfg["epsilon_end"]), int(cfg["warmup"]), int(cfg["decay"])
                        ),
                        source_stage=stage,
                    )
                )
            return settings
        if stage == "gfn_floor":
            winner = self._selection_setting(selections, "gflownet", "best_noncontrol")
            start = float(winner.overrides["tb.exploration_epsilon_start"])
            base_cfg = self.data["refinement"]["gfn_start"]
            settings = [self.control_setting("gflownet")]
            for end in self.data["refinement"][stage]["epsilon_end"]:
                settings.append(
                    Setting(
                        algorithm="gflownet",
                        setting_id=f"floor_{_number_id(end)}",
                        overrides={
                            "tb.exploration_epsilon_enabled": True,
                            "tb.exploration_epsilon_start": start,
                            "tb.exploration_epsilon_end": float(end),
                            "tb.exploration_warmup_episodes": int(base_cfg["warmup"]),
                            "tb.exploration_decay_episodes": int(base_cfg["decay"]),
                        },
                        exploration_rank=_gfn_exploration_rank(
                            start, float(end), int(base_cfg["warmup"]), int(base_cfg["decay"])
                        ),
                        source_stage=stage,
                    )
                )
            return settings
        if stage == "gfn_schedule":
            winner = self._selection_setting(selections, "gflownet", "best_noncontrol")
            start = float(winner.overrides["tb.exploration_epsilon_start"])
            end = float(winner.overrides["tb.exploration_epsilon_end"])
            settings = [self.control_setting("gflownet")]
            for warmup, decay in self.data["refinement"][stage]["schedules"]:
                settings.append(
                    Setting(
                        algorithm="gflownet",
                        setting_id=f"schedule_w{int(warmup)}_d{int(decay)}",
                        overrides={
                            "tb.exploration_epsilon_enabled": True,
                            "tb.exploration_epsilon_start": start,
                            "tb.exploration_epsilon_end": end,
                            "tb.exploration_warmup_episodes": int(warmup),
                            "tb.exploration_decay_episodes": int(decay),
                        },
                        exploration_rank=_gfn_exploration_rank(start, end, int(warmup), int(decay)),
                        source_stage=stage,
                    )
                )
            return settings
        if stage == "entropy_grid":
            settings: list[Setting] = []
            for algorithm in ENTROPY_ALGORITHMS:
                path = str(self.data["algorithms"][algorithm]["entropy_path"])
                for beta in self.data["refinement"][stage]["beta"]:
                    settings.append(
                        Setting(
                            algorithm=algorithm,
                            setting_id=f"entropy_{_number_id(beta)}",
                            overrides={path: float(beta)},
                            exploration_rank=float(beta),
                            control=float(beta) == 0.0,
                            source_stage=stage,
                        )
                    )
            return settings
        if stage == "entropy_neighbors":
            settings = self.settings_for_stage("entropy_grid")
            cfg = self.data["refinement"][stage]
            for algorithm in ENTROPY_ALGORITHMS:
                winner = self._selection_setting(selections, algorithm, "selected")
                beta = float(next(iter(winner.overrides.values())))
                values = cfg["zero_neighbors"] if beta == 0.0 else [beta / float(cfg["factor"]), beta * float(cfg["factor"])]
                path = str(self.data["algorithms"][algorithm]["entropy_path"])
                for value in values:
                    settings.append(
                        Setting(
                            algorithm=algorithm,
                            setting_id=f"entropy_{_number_id(value)}",
                            overrides={path: float(value)},
                            exploration_rank=float(value),
                            source_stage=stage,
                        )
                    )
            return _dedupe_settings(settings)
        if stage == "pcn_seeds":
            settings = []
            for count in self.data["refinement"][stage]["random_seed_episodes"]:
                settings.append(
                    Setting(
                        algorithm="pcn",
                        setting_id=f"seeds_{int(count)}_noise_none",
                        overrides={
                            "pcn.random_seed_episodes": int(count),
                            "pcn.target_noise_scale": 0.0,
                            "pcn.target_min_sigma": 0.0,
                        },
                        exploration_rank=float(count),
                        control=int(count) == 32,
                        source_stage=stage,
                    )
                )
            return settings
        if stage == "pcn_noise":
            seed_winner = self._selection_setting(selections, "pcn", "selected")
            count = int(seed_winner.overrides["pcn.random_seed_episodes"])
            settings = []
            for scale, sigma in self.data["refinement"][stage]["pairs"]:
                settings.append(
                    Setting(
                        algorithm="pcn",
                        setting_id=f"seeds_{count}_noise_{_number_id(scale)}_sigma_{_number_id(sigma)}",
                        overrides={
                            "pcn.random_seed_episodes": count,
                            "pcn.target_noise_scale": float(scale),
                            "pcn.target_min_sigma": float(sigma),
                        },
                        exploration_rank=float(count) + float(scale),
                        control=count == 32 and float(scale) == 0.0,
                        source_stage=stage,
                    )
                )
            return settings
        if stage == "pcn_interaction":
            ranked_seeds = selections.get("pcn_seed_ranked")
            ranked_noise = selections.get("pcn_noise_ranked")
            if not isinstance(ranked_seeds, list) or not isinstance(ranked_noise, list):
                raise ProtocolError("PCN interaction requires ranked seed and noise selections")
            seed_values = [int(Setting.from_dict(item).overrides["pcn.random_seed_episodes"]) for item in ranked_seeds[:2]]
            pairs = [
                (
                    float(Setting.from_dict(item).overrides["pcn.target_noise_scale"]),
                    float(Setting.from_dict(item).overrides["pcn.target_min_sigma"]),
                )
                for item in ranked_noise[:2]
            ]
            settings = []
            for count in seed_values:
                for scale, sigma in pairs:
                    settings.append(
                        Setting(
                            algorithm="pcn",
                            setting_id=f"seeds_{count}_noise_{_number_id(scale)}_sigma_{_number_id(sigma)}",
                            overrides={
                                "pcn.random_seed_episodes": count,
                                "pcn.target_noise_scale": scale,
                                "pcn.target_min_sigma": sigma,
                            },
                            exploration_rank=float(count) + scale,
                            control=count == 32 and scale == 0.0,
                            source_stage=stage,
                        )
                    )
            return _dedupe_settings(settings)
        if stage == "confirmation":
            settings = []
            for algorithm in self.data["algorithms"]:
                selected = self._selection_setting(selections, algorithm, "selected")
                best_noncontrol = self._selection_setting(selections, algorithm, "best_noncontrol")
                control = self.control_setting(algorithm)
                opponent = best_noncontrol if selected.control else control
                for role, setting in (("selected", selected), ("comparator", opponent)):
                    settings.append(
                        Setting(
                            **{
                                **setting.to_dict(),
                                "setting_id": f"{role}_{setting.setting_id}",
                                "source_stage": stage,
                            }
                        )
                    )
            return settings
        raise ProtocolError(f"unsupported stage: {stage}")

    def build_tasks(self, stage: str, settings: Iterable[Setting]) -> list[ExperimentTask]:
        stage_cfg = self.stages[stage]
        budget_name = str(stage_cfg["budget"])
        budget = self.data["budgets"][budget_name]
        tasks: list[ExperimentTask] = []
        for setting in settings:
            algorithm_cfg = self.data["algorithms"][setting.algorithm]
            trajectories = int(algorithm_cfg["trajectories_per_episode"])
            training_trajectories = int(budget["training_trajectories"])
            if training_trajectories % trajectories != 0:
                raise ProtocolError(f"budget is not divisible for {setting.algorithm}")
            episodes = training_trajectories // trajectories
            fixed_overrides = dict(algorithm_cfg.get("fixed_overrides", {}))
            if stage == "smoke" and setting.algorithm == "pcn":
                fixed_overrides.update(
                    {
                        "pcn.random_seed_episodes": 8,
                        "pcn.collect_episodes_per_iter": 4,
                        "pcn.train_updates_per_iter": 1,
                    }
                )
            for circuit in stage_cfg["circuits"]:
                circuit_cfg = self.data["circuits"][circuit]
                for seed in budget["seeds"]:
                    task_id = "__".join(
                        _safe(value) for value in (setting.algorithm, setting.setting_id, circuit, f"seed{int(seed)}")
                    )
                    tasks.append(
                        ExperimentTask(
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
                            training_trajectories=training_trajectories,
                            evaluation_samples=int(budget["evaluation_samples"]),
                            fixed_overrides=fixed_overrides,
                        )
                    )
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ProtocolError(f"stage {stage} generated duplicate task IDs")
        return tasks


def _dedupe_settings(settings: Iterable[Setting]) -> list[Setting]:
    out: list[Setting] = []
    seen: set[str] = set()
    for setting in settings:
        if setting.key in seen:
            continue
        seen.add(setting.key)
        out.append(setting)
    return out


__all__ = [
    "ENTROPY_ALGORITHMS",
    "ExperimentTask",
    "ExplorationProtocol",
    "PROTOCOL_PATH",
    "ProtocolError",
    "Setting",
    "canonical_json",
]
