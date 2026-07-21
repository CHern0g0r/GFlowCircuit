from __future__ import annotations

from typing import Any

__all__ = [
    "TBGFlowNetPolicy",
    "TBGFlowNetTrainer",
    "build_policy_optimizer",
    "build_tb_optimizer",
    "build_tb_policy",
    "epsilon_mixed_probs",
]


def __getattr__(name: str) -> Any:
    """Load OpenSpiel-dependent modules only when their exports are requested."""
    if name == "TBGFlowNetPolicy":
        from .policy import TBGFlowNetPolicy

        return TBGFlowNetPolicy
    if name == "TBGFlowNetTrainer":
        from .trainer import TBGFlowNetTrainer

        return TBGFlowNetTrainer
    if name == "build_tb_policy":
        from .factory import build_tb_policy

        return build_tb_policy
    if name == "build_tb_optimizer":
        from .optim import build_tb_optimizer

        return build_tb_optimizer
    if name == "build_policy_optimizer":
        from .optim import build_policy_optimizer

        return build_policy_optimizer
    if name == "epsilon_mixed_probs":
        from .behavior import epsilon_mixed_probs

        return epsilon_mixed_probs
    raise AttributeError(name)
