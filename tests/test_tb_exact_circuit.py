from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

try:
    import pyspiel  # noqa: F401
except ImportError:
    pyspiel = None


@unittest.skipIf(pyspiel is None, "OpenSpiel Python bindings are unavailable")
class ExactCircuitIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.algorithms.gflownet_tb.circuit_exact import enumerate_circuit_tree
        from src.models.rewards import DiffOfProductReward

        repo_root = Path(__file__).resolve().parents[1]
        cls.circuit_path = repo_root / "data/hdl-benchmarks/mcnc/Combinational/blif/bc0.blif"
        try:
            cls.enumeration = enumerate_circuit_tree(
                circuit_path=cls.circuit_path,
                horizon=4,
                reward_class=DiffOfProductReward,
                reward_alpha=4.0,
                reward_eps=1e-8,
                reward_improvement_clip=2.0,
                available_actions=list(range(7)),
                require_constant_branching=7,
            )
        except Exception as exc:
            raise unittest.SkipTest(f"OpenSpiel circuit game is unavailable: {exc}") from exc

    def test_bc0_tree_counts_and_backward_probability(self) -> None:
        tree = self.enumeration.tree

        self.assertEqual(len(tree.nodes), 400)
        self.assertEqual(len(tree.terminal_log_rewards), 2_401)
        self.assertTrue(all(len(node.legal_actions) == 7 for node in tree.nodes.values()))
        self.assertTrue(all(row["log_pb"] == 0.0 for row in self.enumeration.terminal_metadata.values()))

    def test_terminal_rewards_use_production_transform(self) -> None:
        from src.algorithms.gflownet_tb.sampler import _finish_rollout, _new_rollout
        from src.models.rewards import DiffOfProductReward

        class SamplerPolicy(torch.nn.Module):
            def __init__(self, num_actions: int) -> None:
                super().__init__()
                self.num_actions = int(num_actions)
                self.log_z = torch.nn.Parameter(torch.tensor(0.0))

        policy = SamplerPolicy(self.enumeration.tree.num_actions)

        prefixes = self.enumeration.tree.terminal_prefixes
        for prefix in (prefixes[0], prefixes[len(prefixes) // 2], prefixes[-1]):
            row = self.enumeration.terminal_metadata[prefix]
            rollout = _new_rollout(
                file_path=str(self.circuit_path),
                num_steps=4,
                policy=policy,
                reward_class=DiffOfProductReward,
                available_actions=list(range(7)),
            )
            for action in prefix:
                rollout.state.apply_action(action)
            sampled = _finish_rollout(
                rollout=rollout,
                policy=policy,
                reward_class=DiffOfProductReward,
                reward_alpha=4.0,
                reward_eps=1e-8,
                reward_improvement_clip=2.0,
                available_actions=list(range(7)),
            )
            self.assertAlmostEqual(float(row["log_reward"]), sampled.log_reward)
            self.assertAlmostEqual(float(row["terminal_reward"]), sampled.terminal_reward)

    def test_supervised_targets_cover_all_prefixes_and_reconstruct_terminals(self) -> None:
        from src.algorithms.gflownet_tb.supervised_exact import build_exact_conditional_targets

        tree = self.enumeration.tree
        targets = build_exact_conditional_targets(tree)

        self.assertEqual(targets.num_prefixes, 400)
        self.assertEqual(targets.num_actions, 7)
        self.assertEqual(tuple(targets.prefixes), tuple(tree.nonterminal_prefixes))
        self.assertTrue(torch.all(targets.legal_mask).item())
        for terminal in tree.terminal_prefixes:
            self.assertAlmostEqual(
                targets.reconstructed_terminal_probs[terminal],
                tree.target_terminal_probs[terminal],
                places=10,
            )

    def test_short_contract_policy_run_is_finite_and_writes_artifacts(self) -> None:
        from hydra import compose, initialize_config_dir

        from src.algorithms.gflownet_tb.circuit_exact import neural_probability_fn
        from src.algorithms.gflownet_tb.exact import ExactGateConfig, ExactTrainingConfig
        from src.algorithms.gflownet_tb.factory import build_tb_policy
        from src.experiments.tb_exact import _run_case

        tree = self.enumeration.tree
        root_observation = tree.nodes[()].payload
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str((repo_root / "cfg").resolve())):
            cfg = compose(config_name="tb_zhuDOP")
        edge_attr = root_observation.graph.edge_attr
        policy = build_tb_policy(
            cfg,
            obs_dim=int(root_observation.obs_tensor.numel()),
            node_dim=int(root_observation.graph.x.shape[1]),
            edge_dim=int(edge_attr.shape[1]) if edge_attr.dim() > 1 else 1,
            num_actions=tree.num_actions,
            available_actions=list(range(7)),
        ).to(torch.device("cpu"))

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_path = output_dir / "metrics.jsonl"
            metrics_path.write_text("", encoding="utf-8")
            result, _ = _run_case(
                case_name="integration_smoke",
                tree=tree,
                policy=policy,
                probability_fn=neural_probability_fn(tree=tree, policy=policy),
                behavior="epsilon_0.5",
                training_config=ExactTrainingConfig(
                    batch_size=2,
                    max_updates=1,
                    eval_every=1,
                    required_consecutive_passes=3,
                    seed=0,
                ),
                gates=ExactGateConfig(),
                output_dir=output_dir,
                metrics_path=metrics_path,
                terminal_metadata=self.enumeration.terminal_metadata,
            )

            self.assertTrue(result.final_evaluation.metrics.finite)
            self.assertTrue((output_dir / "checkpoints/integration_smoke.pt").is_file())
            self.assertTrue((output_dir / "distributions/integration_smoke.csv").is_file())
            self.assertGreater(metrics_path.stat().st_size, 0)

    def test_short_supervised_contract_run_writes_reloadable_artifacts(self) -> None:
        from hydra import compose, initialize_config_dir

        from src.algorithms.gflownet_tb.circuit_exact import neural_prediction_fn
        from src.algorithms.gflownet_tb.factory import build_tb_policy
        from src.algorithms.gflownet_tb.supervised_exact import (
            ConditionalGateConfig,
            SupervisedTrainingConfig,
            build_exact_conditional_targets,
            evaluate_conditional_policy,
        )
        from src.experiments.tb_supervised import (
            _cpu_state_dict,
            _run_one,
            _state_dict_checksum,
        )

        tree = self.enumeration.tree
        targets = build_exact_conditional_targets(tree)
        root_observation = tree.nodes[()].payload
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str((repo_root / "cfg").resolve())):
            cfg = compose(config_name="tb_zhuDOP")
        edge_attr = root_observation.graph.edge_attr

        def build_policy():
            return build_tb_policy(
                cfg,
                obs_dim=int(root_observation.obs_tensor.numel()),
                node_dim=int(root_observation.graph.x.shape[1]),
                edge_dim=int(edge_attr.shape[1]) if edge_attr.dim() > 1 else 1,
                num_actions=tree.num_actions,
                available_actions=list(range(7)),
            ).to(torch.device("cpu"))

        torch.manual_seed(0)
        policy = build_policy()
        checksum = _state_dict_checksum(_cpu_state_dict(policy))
        training_config = SupervisedTrainingConfig(
            max_updates=1,
            eval_every=1,
            required_consecutive_passes=3,
            prefix_batch_size=128,
            scheduler="fixed",
            seed=0,
        )
        gates = ConditionalGateConfig()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics_path = output_dir / "metrics.jsonl"
            metrics_path.write_text("", encoding="utf-8")
            result, summary = _run_one(
                seed=0,
                variant="fixed",
                policy=policy,
                prediction_fn=neural_prediction_fn(tree=tree, policy=policy),
                tree=tree,
                targets=targets,
                terminal_metadata=self.enumeration.terminal_metadata,
                training_config=training_config,
                gates=gates,
                output_dir=output_dir,
                metrics_path=metrics_path,
                initialization_checksum=checksum,
            )

            self.assertTrue(result.final_evaluation.metrics.finite)
            for path in (
                summary["best_checkpoint"],
                summary["final_checkpoint"],
                summary["conditional_predictions"],
                summary["terminal_distribution"],
            ):
                self.assertTrue(Path(path).is_file(), path)
            self.assertGreater(metrics_path.stat().st_size, 0)

            try:
                checkpoint = torch.load(
                    summary["final_checkpoint"],
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(summary["final_checkpoint"], map_location="cpu")
            reloaded_policy = build_policy()
            reloaded_policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
            reloaded = evaluate_conditional_policy(
                tree=tree,
                targets=targets,
                policy=reloaded_policy,
                prediction_fn=neural_prediction_fn(tree=tree, policy=reloaded_policy),
                update=result.completed_updates,
                gates=gates,
            )
            self.assertAlmostEqual(
                reloaded.metrics.mean_prefix_tv,
                result.final_evaluation.metrics.mean_prefix_tv,
                places=7,
            )
            self.assertAlmostEqual(
                reloaded.metrics.terminal.total_variation,
                result.final_evaluation.metrics.terminal.total_variation,
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
