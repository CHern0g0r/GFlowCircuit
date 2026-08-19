from __future__ import annotations

from unittest import TestCase

from src.baseline_tuning_analysis import confirm_selection, select_budget
from src.baseline_tuning_protocol import BaselineTuningProtocol


def _summary(algorithm: str, profile: str, budget: int, circuit: str, hv: float) -> dict:
    return {
        "algorithm": algorithm,
        "setting_id": f"{profile}__b{budget}",
        "profile_id": profile,
        "role": "budget_curve",
        "training_trajectories": budget,
        "circuit": circuit,
        "mean_hypervolume": hv,
        "mean_product_improvement": hv,
    }


def _seed(
    algorithm: str,
    profile: str,
    budget: int,
    circuit: str,
    seed: int,
    hv: float,
    role: str = "budget_curve",
) -> dict:
    return {
        "algorithm": algorithm,
        "profile_id": profile,
        "training_trajectories": budget,
        "circuit": circuit,
        "seed": seed,
        "hypervolume": hv,
        "role": role,
    }


def test_protocol_expands_equal_trajectory_tasks() -> None:
    protocol = BaselineTuningProtocol.load()
    smoke = protocol.build_tasks("smoke", protocol.settings_for_stage("smoke"))
    assert len(smoke) == 3
    assert {task.training_trajectories for task in smoke} == {20}
    assert {task.evaluation_samples for task in smoke} == {2}

    screen_payload = {
        "screen_reinforce": {
            "algorithms": {"reinforce": {"top_profiles": ["control", "policy_lr_low"]}}
        }
    }
    screen_settings = protocol.settings_for_stage("screen_reinforce", payloads={"smoke": {}})
    screen_tasks = protocol.build_tasks("screen_reinforce", screen_settings)
    assert len(screen_settings) == 7
    assert len(screen_tasks) == 42
    assert sum(task.training_trajectories for task in screen_tasks) == 33_600
    assert {task.evaluation_samples for task in screen_tasks} == {50}

    budget_settings = protocol.settings_for_stage("budget_reinforce", payloads=screen_payload)
    budget_tasks = protocol.build_tasks("budget_reinforce", budget_settings)
    assert len(budget_settings) == 10
    assert len(budget_tasks) == 60
    assert sum(task.training_trajectories for task in budget_tasks) == 74_400


def test_conditional_extension_confirmation_and_common_budget() -> None:
    protocol = BaselineTuningProtocol.load()
    no_extension = {
        "budget_reinforce": {
            "algorithms": {
                "reinforce": {"status": "selected", "top_profiles": ["control", "policy_lr_low"]}
            }
        }
    }
    assert protocol.settings_for_stage("extend_reinforce", payloads=no_extension) == []

    extension = {
        "budget_reinforce": {
            "algorithms": {
                "reinforce": {"status": "extend_required", "top_profiles": ["control", "policy_lr_low"]}
            }
        }
    }
    extension_settings = protocol.settings_for_stage("extend_reinforce", payloads=extension)
    assert len(extension_settings) == 2
    assert {setting.training_trajectories for setting in extension_settings} == {6400}

    confirm_payload = {
        "extend_reinforce": {
            "algorithms": {
                "reinforce": {
                    "status": "selected",
                    "selected_profile": "control",
                    "runner_up_profile": "policy_lr_low",
                    "selected_budget": 400,
                    "successor_budget": 800,
                }
            }
        }
    }
    confirm_settings = protocol.settings_for_stage("confirm_reinforce", payloads=confirm_payload)
    confirm_tasks = protocol.build_tasks("confirm_reinforce", confirm_settings)
    assert len(confirm_settings) == 3
    assert len(confirm_tasks) == 60
    assert {task.seed for task in confirm_tasks} == set(range(10))

    common_payloads = {}
    for stage, algorithm, budget, profile in (
        ("confirm_reinforce", "reinforce", 400, "control"),
        ("confirm_drills", "drills", 800, "control"),
        ("confirm_ppo", "ppo", 1600, "control"),
    ):
        common_payloads[stage] = {
            "algorithms": {
                algorithm: {
                    "status": "confirmed",
                    "selected_profile": profile,
                    "selected_budget": budget,
                }
            }
        }
    common_settings = protocol.settings_for_stage("common_budget", payloads=common_payloads)
    assert len(common_settings) == 3
    assert {setting.training_trajectories for setting in common_settings} == {800}
    assert len(protocol.build_tasks("common_budget", common_settings)) == 60


def test_budget_rule_selects_smallest_near_maximum_plateau() -> None:
    protocol = BaselineTuningProtocol.load()
    summaries = []
    seeds = []
    values = {
        "control": {200: 0.080, 400: 0.099, 800: 0.100},
        "policy_lr_low": {200: 0.070, 400: 0.090, 800: 0.094},
    }
    jitter = {
        200: [0.0, 0.0, 0.0],
        400: [-0.001, 0.001, 0.0],
        800: [0.0, -0.001, 0.001],
    }
    for profile, curve in values.items():
        for budget, hv in curve.items():
            for circuit in ("C1355", "dalu"):
                summaries.append(_summary("reinforce", profile, budget, circuit, hv))
                for seed, delta in enumerate(jitter[budget]):
                    seeds.append(_seed("reinforce", profile, budget, circuit, seed, hv + delta))
    result = select_budget(
        protocol,
        "budget_reinforce",
        summaries,
        seeds,
        top_profiles=["control", "policy_lr_low"],
    )["reinforce"]
    assert result["status"] == "selected"
    assert result["selected_profile"] == "control"
    assert result["selected_budget"] == 400
    assert result["successor_budget"] == 800


def test_budget_rule_requires_extension_for_improving_curve() -> None:
    protocol = BaselineTuningProtocol.load()
    summaries = []
    seeds = []
    for profile, offset in (("control", 0.0), ("policy_lr_low", -0.02)):
        for budget, hv in ((200, 0.04), (400, 0.06), (800, 0.08)):
            value = hv + offset
            for circuit in ("C1355", "dalu"):
                summaries.append(_summary("reinforce", profile, budget, circuit, value))
                for seed in range(3):
                    seeds.append(_seed("reinforce", profile, budget, circuit, seed, value))
    result = select_budget(
        protocol,
        "budget_reinforce",
        summaries,
        seeds,
        top_profiles=["control", "policy_lr_low"],
    )["reinforce"]
    assert result["status"] == "extend_required"


def test_confirmation_accepts_equivalent_successor_and_better_profile() -> None:
    protocol = BaselineTuningProtocol.load()
    payload = {
        "extend_reinforce": {
            "algorithms": {
                "reinforce": {
                    "status": "selected",
                    "selected_profile": "control",
                    "runner_up_profile": "policy_lr_low",
                    "selected_budget": 400,
                    "successor_budget": 800,
                }
            }
        }
    }
    settings = protocol.settings_for_stage("confirm_reinforce", payloads=payload)
    summaries = []
    seeds = []
    role_values = {
        "selected": ("control", 400, 0.100),
        "runner_up": ("policy_lr_low", 400, 0.090),
        "successor": ("control", 800, 0.101),
    }
    for role, (profile, budget, hv) in role_values.items():
        for circuit in ("C1355", "dalu"):
            row = _summary("reinforce", profile, budget, circuit, hv)
            row["role"] = role
            summaries.append(row)
            for seed in range(10):
                delta = 0.001 if role == "successor" and seed % 2 == 0 else (-0.001 if role == "successor" else 0.0)
                seed_hv = 0.100 + delta if role == "successor" else hv
                seeds.append(_seed("reinforce", profile, budget, circuit, seed, seed_hv, role=role))
    result = confirm_selection(
        protocol,
        "confirm_reinforce",
        settings,
        summaries,
        seeds,
    )["reinforce"]
    assert result["status"] == "confirmed"
    assert result["profile_gate"] is True
    assert result["budget_gate"] is True


class BaselineTuningTest(TestCase):
    """Expose the assertion-style cases to the repository's unittest runner."""

    def test_protocol_matrix(self) -> None:
        test_protocol_expands_equal_trajectory_tasks()

    def test_conditional_stages(self) -> None:
        test_conditional_extension_confirmation_and_common_budget()

    def test_plateau_selection(self) -> None:
        test_budget_rule_selects_smallest_near_maximum_plateau()

    def test_unresolved_curve_extension(self) -> None:
        test_budget_rule_requires_extension_for_improving_curve()

    def test_confirmation_gate(self) -> None:
        test_confirmation_accepts_equivalent_successor_and_better_profile()
