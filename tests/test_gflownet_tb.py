from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from torch import nn

pyspiel_stub = types.ModuleType("pyspiel")
pyspiel_stub.State = object
sys.modules.setdefault("pyspiel", pyspiel_stub)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.gflownet_tb.policy import TBGFlowNetPolicy
from src.algorithms.gflownet_tb.trainer import _build_tb_optimizer


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)


class TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


def _policy() -> TBGFlowNetPolicy:
    return TBGFlowNetPolicy(encoder=TinyEncoder(), head=TinyHead(), num_actions=2)


class GFlowNetTBTest(unittest.TestCase):
    def test_optimizer_uses_separate_log_z_learning_rate(self) -> None:
        policy = _policy()

        optimizer = _build_tb_optimizer(
            policy,
            learning_rate=1e-3,
            log_z_learning_rate=1e-2,
        )

        self.assertEqual(len(optimizer.param_groups), 2)
        policy_group, log_z_group = optimizer.param_groups
        self.assertEqual(policy_group["lr"], 1e-3)
        self.assertEqual(log_z_group["lr"], 1e-2)

        policy_param_ids = {id(param) for param in policy_group["params"]}
        log_z_param_ids = {id(param) for param in log_z_group["params"]}
        encoder_param_ids = {id(param) for param in policy.encoder.parameters()}
        head_param_ids = {id(param) for param in policy.head.parameters()}

        self.assertEqual(log_z_param_ids, {id(policy.log_z)})
        self.assertNotIn(id(policy.log_z), policy_param_ids)
        self.assertTrue(encoder_param_ids.issubset(policy_param_ids))
        self.assertTrue(head_param_ids.issubset(policy_param_ids))

    def test_optimizer_rejects_nonpositive_learning_rates(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _build_tb_optimizer(
                _policy(),
                learning_rate=0.0,
                log_z_learning_rate=1e-2,
            )

        with self.assertRaisesRegex(ValueError, "log_z_learning_rate"):
            _build_tb_optimizer(
                _policy(),
                learning_rate=1e-3,
                log_z_learning_rate=-1e-2,
            )


if __name__ == "__main__":
    unittest.main()
