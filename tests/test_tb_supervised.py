from __future__ import annotations

import contextlib
import copy
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.algorithms.gflownet_tb.exact import TabularPrefixPolicy, build_synthetic_tree
from src.algorithms.gflownet_tb.optim import build_policy_optimizer
from src.algorithms.gflownet_tb.supervised_exact import (
    ConditionalGateConfig,
    SupervisedTrainingConfig,
    build_exact_conditional_targets,
    classify_supervised_runs,
    evaluate_conditional_policy,
    train_supervised_case,
)


def _tabular_prediction_fn(policy: TabularPrefixPolicy):
    def predict(prefixes):
        indices = torch.tensor(
            [policy._prefix_to_index[prefix] for prefix in prefixes],
            dtype=torch.long,
            device=policy.device,
        )
        return policy.logits.index_select(0, indices), policy.probabilities(prefixes)

    return predict


def _set_exact_policy(policy: TabularPrefixPolicy, tree) -> None:
    with torch.no_grad():
        for prefix in tree.nonterminal_prefixes:
            row = policy._prefix_to_index[prefix]
            for action, probability in tree.target_action_probs(prefix).items():
                policy.logits[row, action] = math.log(probability)
        policy.log_z.fill_(tree.true_log_z)


class ExactConditionalTargetTest(unittest.TestCase):
    def test_targets_are_normalized_legal_and_reconstruct_terminals(self) -> None:
        tree = build_synthetic_tree(depth=4)
        targets = build_exact_conditional_targets(tree)

        self.assertEqual(targets.num_prefixes, len(tree.nodes))
        self.assertTrue(
            torch.allclose(
                (targets.probabilities * targets.legal_mask).sum(dim=1),
                torch.ones(targets.num_prefixes, dtype=torch.float64),
                atol=1e-12,
                rtol=0.0,
            )
        )
        self.assertEqual(
            int(torch.count_nonzero(targets.probabilities.masked_fill(targets.legal_mask, 0.0))),
            0,
        )
        for terminal, target_probability in tree.target_terminal_probs.items():
            self.assertAlmostEqual(
                targets.reconstructed_terminal_probs[terminal],
                target_probability,
                places=12,
            )

    def test_perfect_predictions_pass_local_and_terminal_gates(self) -> None:
        tree = build_synthetic_tree(depth=4)
        targets = build_exact_conditional_targets(tree)
        policy = TabularPrefixPolicy(tree)
        _set_exact_policy(policy, tree)

        evaluation = evaluate_conditional_policy(
            tree=tree,
            targets=targets,
            policy=policy,
            prediction_fn=_tabular_prediction_fn(policy),
            update=0,
        )

        self.assertTrue(evaluation.metrics.passed, evaluation.metrics.to_dict())
        self.assertLess(evaluation.metrics.mean_conditional_kl, 1e-6)
        self.assertLess(evaluation.metrics.mean_prefix_tv, 1e-6)
        self.assertLess(evaluation.metrics.terminal.total_variation, 1e-6)

    def test_perturbing_one_prefix_changes_local_and_descendant_probabilities(self) -> None:
        tree = build_synthetic_tree(depth=4)
        targets = build_exact_conditional_targets(tree)
        policy = TabularPrefixPolicy(tree)
        _set_exact_policy(policy, tree)
        perfect = evaluate_conditional_policy(
            tree=tree,
            targets=targets,
            policy=policy,
            prediction_fn=_tabular_prediction_fn(policy),
            update=0,
        )
        with torch.no_grad():
            policy.logits[policy._prefix_to_index[()], 0].add_(1.0)

        perturbed = evaluate_conditional_policy(
            tree=tree,
            targets=targets,
            policy=policy,
            prediction_fn=_tabular_prediction_fn(policy),
            update=1,
        )

        self.assertGreater(perturbed.metrics.mean_conditional_kl, perfect.metrics.mean_conditional_kl)
        self.assertGreater(perturbed.metrics.max_prefix_tv, perfect.metrics.max_prefix_tv)
        self.assertGreater(
            perturbed.metrics.terminal.total_variation,
            perfect.metrics.terminal.total_variation,
        )


class SupervisedTrainingTest(unittest.TestCase):
    def test_policy_optimizer_excludes_log_z(self) -> None:
        policy = TabularPrefixPolicy(build_synthetic_tree(depth=2))
        optimizer = build_policy_optimizer(policy, learning_rate=1e-3)

        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertNotIn(id(policy.log_z), optimized)
        self.assertIn(id(policy.logits), optimized)

    def test_chunked_and_unchunked_full_batch_updates_match(self) -> None:
        tree = build_synthetic_tree(depth=2)
        targets = build_exact_conditional_targets(tree)
        reference = TabularPrefixPolicy(tree)
        initial_state = copy.deepcopy(reference.state_dict())
        policies = []
        for prefix_batch_size in (2, targets.num_prefixes):
            policy = TabularPrefixPolicy(tree)
            policy.load_state_dict(initial_state)
            train_supervised_case(
                tree=tree,
                targets=targets,
                policy=policy,
                prediction_fn=_tabular_prediction_fn(policy),
                config=SupervisedTrainingConfig(
                    learning_rate=0.01,
                    max_updates=1,
                    eval_every=1,
                    required_consecutive_passes=3,
                    prefix_batch_size=prefix_batch_size,
                    scheduler="fixed",
                ),
            )
            policies.append(policy)

        self.assertTrue(torch.allclose(policies[0].logits, policies[1].logits, atol=1e-7, rtol=0.0))

    def test_log_z_remains_exact_and_reduced_case_converges(self) -> None:
        tree = build_synthetic_tree(depth=2)
        targets = build_exact_conditional_targets(tree)
        policy = TabularPrefixPolicy(tree)

        result = train_supervised_case(
            tree=tree,
            targets=targets,
            policy=policy,
            prediction_fn=_tabular_prediction_fn(policy),
            config=SupervisedTrainingConfig(
                learning_rate=0.05,
                max_updates=2_000,
                eval_every=25,
                required_consecutive_passes=1,
                prefix_batch_size=3,
                scheduler="fixed",
                seed=7,
            ),
        )

        self.assertTrue(result.passed, result.final_evaluation.metrics.to_dict())
        self.assertFalse(policy.log_z.requires_grad)
        self.assertEqual(float(policy.log_z.item()), float(torch.tensor(tree.true_log_z).item()))

    def test_cosine_scheduler_reaches_configured_minimum(self) -> None:
        tree = build_synthetic_tree(depth=2)
        targets = build_exact_conditional_targets(tree)
        policy = TabularPrefixPolicy(tree)

        result = train_supervised_case(
            tree=tree,
            targets=targets,
            policy=policy,
            prediction_fn=_tabular_prediction_fn(policy),
            config=SupervisedTrainingConfig(
                learning_rate=0.01,
                cosine_min_lr=1e-4,
                max_updates=2,
                eval_every=1,
                required_consecutive_passes=3,
                prefix_batch_size=3,
                scheduler="cosine",
            ),
        )

        self.assertAlmostEqual(result.final_evaluation.metrics.learning_rate, 1e-4)
        self.assertIsNotNone(result.scheduler_state_dict)

    def test_fixed_and_cosine_can_load_identical_initialization(self) -> None:
        tree = build_synthetic_tree(depth=2)
        torch.manual_seed(5)
        base = TabularPrefixPolicy(tree)
        initial = copy.deepcopy(base.state_dict())
        loaded = []
        for _ in ("fixed", "cosine"):
            policy = TabularPrefixPolicy(tree)
            policy.load_state_dict(initial)
            loaded.append(policy.state_dict())

        for name in initial:
            self.assertTrue(torch.equal(loaded[0][name], loaded[1][name]))

    def test_summary_classification_and_exit_codes(self) -> None:
        seeds = [0, 1, 2]
        variants = ["fixed", "cosine"]

        def rows(passing, failure=None):
            return [
                SimpleNamespace(
                    seed=seed,
                    variant=variant,
                    passed=(variant, seed) in passing,
                    numerical_failure=failure,
                )
                for variant in variants
                for seed in seeds
            ]

        self.assertEqual(
            classify_supervised_runs(
                rows({("fixed", seed) for seed in seeds}),
                seeds=seeds,
                variants=variants,
            )[:2],
            ("robust_representability", 0),
        )
        self.assertEqual(
            classify_supervised_runs(rows({("cosine", 1)}), seeds=seeds, variants=variants)[:2],
            ("representable_but_optimizer_sensitive", 3),
        )
        self.assertEqual(
            classify_supervised_runs(rows(set()), seeds=seeds, variants=variants)[:2],
            ("representability_not_demonstrated", 4),
        )
        self.assertEqual(
            classify_supervised_runs(rows(set(), "NaN"), seeds=seeds, variants=variants)[:2],
            ("execution_failure", 1),
        )

    def test_run_artifacts_and_checkpoints_are_written(self) -> None:
        from src.experiments.tb_supervised import (
            _cpu_state_dict,
            _run_one,
            _state_dict_checksum,
        )

        tree = build_synthetic_tree(depth=2)
        targets = build_exact_conditional_targets(tree)
        policy = TabularPrefixPolicy(tree)
        checksum = _state_dict_checksum(_cpu_state_dict(policy))
        terminal_metadata = {
            prefix: {
                "final_size": 0,
                "final_depth": 0,
                "improvement_raw": 0.0,
                "improvement_clipped": 0.0,
                "terminal_reward": math.exp(tree.terminal_log_rewards[prefix]),
                "log_pb": 0.0,
            }
            for prefix in tree.terminal_prefixes
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_path = output_dir / "metrics.jsonl"
            metrics_path.write_text("", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                result, summary = _run_one(
                    seed=0,
                    variant="fixed",
                    policy=policy,
                    prediction_fn=_tabular_prediction_fn(policy),
                    tree=tree,
                    targets=targets,
                    terminal_metadata=terminal_metadata,
                    training_config=SupervisedTrainingConfig(
                        max_updates=1,
                        eval_every=1,
                        required_consecutive_passes=3,
                        prefix_batch_size=3,
                        scheduler="fixed",
                    ),
                    gates=ConditionalGateConfig(),
                    output_dir=output_dir,
                    metrics_path=metrics_path,
                    initialization_checksum=checksum,
                )

            self.assertEqual(result.completed_updates, 1)
            self.assertGreater(metrics_path.stat().st_size, 0)
            for key in (
                "best_checkpoint",
                "final_checkpoint",
                "conditional_predictions",
                "terminal_distribution",
            ):
                self.assertTrue(Path(summary[key]).is_file(), summary[key])


if __name__ == "__main__":
    unittest.main()
