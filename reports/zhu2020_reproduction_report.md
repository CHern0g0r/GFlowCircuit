# Zhu 2020 Reproduction Analysis

This report analyzes the completed size-objective reproduction against Zhu et al. Table 2 RL-1. Saved JSON reports and saved Hydra configs are treated as the source of truth.

## Diagnosis

- Selected 10 benchmark reports.
- Protocol config audit failures: 0.
- Source audit failures: 0.
- Baseline/toolchain drift detected for 9 benchmark(s).
- RL-vs-paper differences should be interpreted after checking baseline drift.

## Provenance

| circuit | json | hydra | run_count | duplicates |
| --- | --- | --- | --- | --- |
| apex1 | zhu2020_apex1_reproduce_78666.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/11:26_zhu2020_apex1_reproduce_78666/.hydra/config.yaml | 10 |  |
| bc0 | zhu2020_bc0_reproduce_78666.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/12:02_zhu2020_bc0_reproduce_78666/.hydra/config.yaml | 10 |  |
| c1355 | zhu2020_c1355_reproduce_78666.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/12:28_zhu2020_c1355_reproduce_78666/.hydra/config.yaml | 10 |  |
| c5315 | zhu2020_c5315_reproduce_78666.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/12:50_zhu2020_c5315_reproduce_78666/.hydra/config.yaml | 10 |  |
| c6288 | zhu2020_c6288_reproduce_78666.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/13:24_zhu2020_c6288_reproduce_78666/.hydra/config.yaml | 10 |  |
| c7552 | zhu2020_c7552_reproduce_78668.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/11:30_zhu2020_c7552_reproduce_78668/.hydra/config.yaml | 10 |  |
| dalu | zhu2020_dalu_reproduce_78668.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/12:13_zhu2020_dalu_reproduce_78668/.hydra/config.yaml | 10 |  |
| i10 | zhu2020_i10_reproduce_78668.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/12:44_zhu2020_i10_reproduce_78668/.hydra/config.yaml | 10 | zhu2020_i10_reproduce_76786.json |
| k2 | zhu2020_k2_reproduce_78668.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/13:22_zhu2020_k2_reproduce_78668/.hydra/config.yaml | 10 |  |
| mainpla | zhu2020_mainpla_reproduce_78668.json | /Users/fedor.chernogorskii/workspace/circ/GFlowCircuit/outputs/13:53_zhu2020_mainpla_reproduce_78668/.hydra/config.yaml | 10 |  |

## Protocol Config Audit

| check | expected | actual | status |
| --- | --- | --- | --- |
| algorithm | 'reinforce' | all='reinforce' | pass |
| reward.type | 'zhu_size' | all='zhu_size' | pass |
| num_steps | 20 | all=20 | pass |
| episodes | 200 | all=200 | pass |
| gamma | 0.9 | all=0.9 | pass |
| policy_learning_rate | 0.0008 | all=0.0008 | pass |
| value_learning_rate | 0.003 | all=0.003 | pass |
| baseline | 'zhu_resyn2' | all='zhu_resyn2' | pass |
| paper_mode.per_circuit_mode | True | all=True | pass |
| paper_mode.num_runs | 10 | all=10 | pass |
| paper_mode.infer_rollouts | 10 | all=10 | pass |
| entropy_beta | 0.0 | all=0.0 | pass |
| clip_grad_norm_policy | None | all=None | pass |
| clip_grad_norm_value | None | all=None | pass |
| normalize_returns | False | all=False | pass |
| available_actions | [0, 1, 2, 3, 4] | all=[0, 1, 2, 3, 4] | pass |
| encoder.type | 'hybrid' | all='hybrid' | pass |
| encoder.graph.type | 'zhu_gcn' | all='zhu_gcn' | pass |
| encoder.vector.source | 'zhu10' | all='zhu10' | pass |
| value.input | 'zhu10' | all='zhu10' | pass |

## Source Audit

| check | expected | actual | status |
| --- | --- | --- | --- |
| RESYN2_ACTION_SEQUENCE | [0, 1, 2, 0, 1, 3, 0, 4, 3, 0] | [0, 1, 2, 0, 1, 3, 0, 4, 3, 0] | pass |
| resyn2_inf_stops_after_5_unchanged | unchanged_runs >= 5 | found | pass |
| evaluation_selects_best_final_return | max(candidates, key=lambda r: float(r["final_return"])) | found | pass |
| zhu_baseline_applied_to_reward | baseline == "zhu_resyn2" | found | pass |
| zhu_reward_subtracts_baseline | return gain - self.baseline_scale * self.baseline_per_step | found | pass |

## TensorBoard Cross-Check

| circuit | status | checked_points | max_abs_error | note |
| --- | --- | --- | --- | --- |
| all | skipped | 0 |  | tensorboard is not installed |

## Baseline Comparison

| circuit | ours_init | paper_init | ours_r2_1 | paper_r2_1 | ours_r2_2 | paper_r2_2 | ours_r2_inf | paper_r2_inf | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apex1 | 2655 | 2665 | 2002 | 1999 | 1965 | 1966 | 1929 | 1941 | toolchain_or_benchmark_drift:initial_size,resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| bc0 | 1592 | 1592 | 919 | 933 | 875 | 899 | 847 | 875 | toolchain_or_benchmark_drift:resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| c1355 | 504 | 504 | 387 | 390 | 387 | 390 | 387 | 390 | toolchain_or_benchmark_drift:resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| c5315 | 1773 | 1780 | 1304 | 1306 | 1293 | 1295 | 1289 | 1294 | toolchain_or_benchmark_drift:initial_size,resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| c6288 | 2337 | 2337 | 1870 | 1870 | 1870 | 1870 | 1870 | 1870 | match |
| c7552 | 2074 | 2093 | 1408 | 1469 | 1369 | 1416 | 1338 | 1398 | toolchain_or_benchmark_drift:initial_size,resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| dalu | 1371 | 1371 | 1083 | 1106 | 1077 | 1103 | 1077 | 1103 | toolchain_or_benchmark_drift:resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| i10 | 2675 | 2675 | 1816 | 1829 | 1789 | 1804 | 1762 | 1789 | toolchain_or_benchmark_drift:resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| k2 | 1998 | 1998 | 1201 | 1234 | 1148 | 1186 | 1085 | 1145 | toolchain_or_benchmark_drift:resyn2_1_size,resyn2_2_size,resyn2_inf_size |
| mainpla | 5346 | 5346 | 3676 | 3678 | 3578 | 3583 | 3492 | 3504 | toolchain_or_benchmark_drift:resyn2_1_size,resyn2_2_size,resyn2_inf_size |

## RL-1 Result Comparison

| circuit | n | ours_size_mean | ours_size_std | paper_rl1_size | delta_size | norm_delta | ours_depth_mean | paper_rl1_depth | win_vs_r2_2 | win_vs_r2_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apex1 | 10 | 1916 | 6.885 | 1922 | -5.3 | -0.001989 | 21.3 | 19.2 | 1 | 1 |
| bc0 | 10 | 820.6 | 7.697 | 819.4 | 1.2 | 0.0007538 | 21.5 | 18.6 | 1 | 1 |
| c1355 | 10 | 386 | 0 | 386.2 | -0.2 | -0.0003968 | 21.8 | 17.6 | 1 | 1 |
| c5315 | 10 | 1326 | 19.06 | 1337 | -11.9 | -0.006685 | 29.3 | 27.2 | 0 | 0 |
| c6288 | 10 | 1870 | 0 | 1870 | 0 | 0 | 89 | 88 | 0 | 0 |
| c7552 | 10 | 1336 | 13.77 | 1395 | -58.9 | -0.02814 | 52.3 | 27.4 | 1 | 0.5 |
| dalu | 10 | 1020 | 14.02 | 1040 | -20 | -0.01459 | 45.8 | 33.2 | 1 | 1 |
| i10 | 10 | 1735 | 29.68 | 1730 | 5.2 | 0.001944 | 53.2 | 40.3 | 1 | 0.7 |
| k2 | 10 | 1062 | 13.9 | 1128 | -66.2 | -0.03313 | 22 | 19.8 | 1 | 0.9 |
| mainpla | 10 | 3451 | 20.87 | 3438 | 12.2 | 0.002282 | 28.9 | 25 | 1 | 1 |

## Notes

- The actual run settings are `entropy_beta=0.0`, no gradient clipping, and no return normalization.
- `i10` has duplicate JSON reports; the selected report is the one with a matching saved Hydra directory and highest reproduce id.
- TensorBoard is supporting evidence only; final JSON evaluation may be a fresh stochastic best-of-10 evaluation after the last logged training evaluation.
