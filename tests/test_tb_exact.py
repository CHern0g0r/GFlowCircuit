from __future__ import annotations

import math
import unittest

import torch

from src.algorithms.gflownet_tb.exact import (
    ExactTrainingConfig,
    TabularPrefixPolicy,
    build_synthetic_tree,
    evaluate_exact_policy,
    sample_exact_batch,
    train_exact_case,
)
from src.algorithms.gflownet_tb.optim import build_tb_optimizer


def _set_exact_tabular_policy(policy: TabularPrefixPolicy, tree) -> None:
    with torch.no_grad():
        for prefix in tree.nonterminal_prefixes:
            row_idx = policy._prefix_to_index[prefix]
            targets = tree.target_action_probs(prefix)
            for action, probability in targets.items():
                policy.logits[row_idx, action] = math.log(probability)
        policy.log_z.fill_(tree.true_log_z)


class ExactTreeTest(unittest.TestCase):
    def test_synthetic_tree_shape_reward_range_and_mass(self) -> None:
        tree = build_synthetic_tree(depth=4)

        self.assertEqual(len(tree.terminal_log_rewards), 41)
        self.assertEqual({len(node.legal_actions) for node in tree.nodes.values()}, {2, 3})
        rewards = [math.exp(value) for value in tree.terminal_log_rewards.values()]
        self.assertAlmostEqual(min(rewards), 1.0)
        self.assertAlmostEqual(max(rewards), 100.0)
        self.assertAlmostEqual(max(rewards) / min(rewards), 100.0)
        self.assertAlmostEqual(sum(tree.target_terminal_probs.values()), 1.0, places=12)

    def test_subtree_flows_define_normalized_target_actions(self) -> None:
        tree = build_synthetic_tree(depth=4)

        for prefix in tree.nonterminal_prefixes:
            target = tree.target_action_probs(prefix)
            self.assertEqual(set(target), set(tree.nodes[prefix].legal_actions))
            self.assertAlmostEqual(sum(target.values()), 1.0, places=12)

    def test_perfect_policy_has_zero_exact_error(self) -> None:
        tree = build_synthetic_tree(depth=4)
        policy = TabularPrefixPolicy(tree)
        _set_exact_tabular_policy(policy, tree)

        evaluation = evaluate_exact_policy(
            tree=tree,
            policy=policy,
            probability_fn=policy.probabilities,
            update=0,
        )

        self.assertTrue(evaluation.metrics.passed)
        self.assertLess(evaluation.metrics.rms_residual, 1e-5)
        self.assertLess(evaluation.metrics.total_variation, 1e-6)
        self.assertLess(evaluation.metrics.max_probability_error, 1e-6)

    def test_perturbed_log_z_fails_residual_gates(self) -> None:
        tree = build_synthetic_tree(depth=4)
        policy = TabularPrefixPolicy(tree)
        _set_exact_tabular_policy(policy, tree)
        with torch.no_grad():
            policy.log_z.add_(1.0)

        evaluation = evaluate_exact_policy(
            tree=tree,
            policy=policy,
            probability_fn=policy.probabilities,
            update=0,
        )

        self.assertFalse(evaluation.metrics.passed)
        self.assertAlmostEqual(evaluation.metrics.log_z_error, 1.0, places=5)
        self.assertAlmostEqual(evaluation.metrics.rms_residual, 1.0, places=5)
        self.assertLess(evaluation.metrics.total_variation, 1e-6)

    def test_off_policy_behavior_is_scored_under_learned_policy(self) -> None:
        tree = build_synthetic_tree(depth=1)
        learned = torch.tensor([0.8, 0.15, 0.05], dtype=torch.float32)
        for behavior in ("epsilon_0.5", "uniform"):
            with self.subTest(behavior=behavior):
                policy = TabularPrefixPolicy(tree)
                with torch.no_grad():
                    policy.logits[0].copy_(learned.log())
                generator = torch.Generator(device="cpu")
                generator.manual_seed(11)

                log_pf, _, prefixes = sample_exact_batch(
                    tree=tree,
                    probability_fn=policy.probabilities,
                    batch_size=128,
                    behavior=behavior,
                    generator=generator,
                )

                expected = torch.tensor(
                    [math.log(float(learned[prefix[0]].item())) for prefix in prefixes],
                    dtype=log_pf.dtype,
                )
                self.assertTrue(torch.allclose(log_pf.detach().cpu(), expected, atol=1e-6))
                self.assertEqual(set(prefix[0] for prefix in prefixes), {0, 1, 2})

    def test_perturbed_policy_changes_distribution_metrics(self) -> None:
        tree = build_synthetic_tree(depth=4)
        policy = TabularPrefixPolicy(tree)
        _set_exact_tabular_policy(policy, tree)
        perfect = evaluate_exact_policy(
            tree=tree,
            policy=policy,
            probability_fn=policy.probabilities,
            update=0,
        )
        with torch.no_grad():
            policy.logits[0, 0].add_(1.0)

        perturbed = evaluate_exact_policy(
            tree=tree,
            policy=policy,
            probability_fn=policy.probabilities,
            update=0,
        )

        self.assertGreater(perturbed.metrics.total_variation, perfect.metrics.total_variation)
        self.assertGreater(
            perturbed.metrics.max_probability_error,
            perfect.metrics.max_probability_error,
        )
        self.assertGreater(perturbed.metrics.rms_residual, perfect.metrics.rms_residual)

    def test_public_optimizer_separates_log_z(self) -> None:
        tree = build_synthetic_tree(depth=2)
        policy = TabularPrefixPolicy(tree)

        optimizer = build_tb_optimizer(policy, learning_rate=1e-3, log_z_learning_rate=1e-2)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual({id(value) for value in optimizer.param_groups[1]["params"]}, {id(policy.log_z)})
        self.assertNotIn(id(policy.log_z), {id(value) for value in optimizer.param_groups[0]["params"]})

    def test_reduced_synthetic_tree_converges(self) -> None:
        tree = build_synthetic_tree(depth=2)
        policy = TabularPrefixPolicy(tree)

        result = train_exact_case(
            tree=tree,
            policy=policy,
            probability_fn=policy.probabilities,
            behavior="uniform",
            config=ExactTrainingConfig(
                batch_size=256,
                max_updates=5_000,
                eval_every=100,
                required_consecutive_passes=1,
                learning_rate=0.01,
                log_z_learning_rate=0.05,
                seed=7,
            ),
        )

        self.assertTrue(result.passed, result.final_evaluation.metrics.to_dict())


if __name__ == "__main__":
    unittest.main()
