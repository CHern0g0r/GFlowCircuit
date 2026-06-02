# Zhu et al. 2020 Reproduction Report

This report summarizes the reproduction of the Zhu et al. 2020 size-objective experiment using the `GFlowCircuit` implementation and the JSON reports in `output/`.

The comparison target is Table 2 from the paper, specifically the `RL-1` setting. `RL-1` is the reward setting focused on reducing the number of AIG nodes. This matches the reproduction configuration that uses `reward.type: zhu_size`.

## Experiment Protocol

The reproduced runs match the main paper protocol:

| item | reproduced setting |
| --- | --- |
| algorithm | REINFORCE |
| objective | size optimization, `zhu_size` |
| circuits | 10 Zhu/MCNC benchmark circuits |
| runs per circuit | 10 |
| episodes per run | 200 |
| synthesis operations per episode | 20 |
| final inference | best of 10 sampled sequences |
| discount factor | `gamma = 0.9` |
| policy learning rate | `8e-4` |
| value learning rate | `3e-3` |
| entropy bonus | `0.0` |
| gradient clipping | disabled |
| return normalization | disabled |
| action set | `balance`, `rewrite`, `refactor`, `rewrite -z`, `refactor -z` |

The selected output files are the latest available reproduction reports per circuit. For `i10`, two reports are present; this report uses `zhu2020_i10_reproduce_78668.json`.

## Main Result

On the primary metric, final node count, the reproduction is close to the paper. Across the 10 circuits:

- 6 circuits are better than the paper `RL-1` mean node count.
- 3 circuits are worse than the paper `RL-1` mean node count.
- 1 circuit is equal.
- Mean absolute normalized size delta is about `0.90%` of the paper initial size.
- Worst normalized size delta is `3.31%`, on `k2`, where this reproduction is better than the paper.

This is a good reproduction of the paper's size result. The main caveat is that the local resyn2 baselines do not match the paper on 9 of 10 circuits, so exact paper comparability is limited by benchmark/toolchain drift.

## RL-1 Size Comparison

Negative size delta means this implementation achieved a smaller circuit than the paper `RL-1` result.

| circuit | runs | ours final size mean | ours std | paper RL-1 size | size delta | normalized delta | ours final depth mean | paper RL-1 depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `apex1` | 10 | 1916.3 | 6.9 | 1921.6 | -5.3 | -0.20% | 21.3 | 19.2 |
| `bc0` | 10 | 820.6 | 7.7 | 819.4 | +1.2 | +0.08% | 21.5 | 18.6 |
| `c1355` | 10 | 386.0 | 0.0 | 386.2 | -0.2 | -0.04% | 21.8 | 17.6 |
| `c5315` | 10 | 1325.5 | 19.1 | 1337.4 | -11.9 | -0.67% | 29.3 | 27.2 |
| `c6288` | 10 | 1870.0 | 0.0 | 1870.0 | 0.0 | 0.00% | 89.0 | 88.0 |
| `c7552` | 10 | 1336.5 | 13.8 | 1395.4 | -58.9 | -2.81% | 52.3 | 27.4 |
| `dalu` | 10 | 1019.8 | 14.0 | 1039.8 | -20.0 | -1.46% | 45.8 | 33.2 |
| `i10` | 10 | 1735.4 | 29.7 | 1730.2 | +5.2 | +0.19% | 53.2 | 40.3 |
| `k2` | 10 | 1062.2 | 13.9 | 1128.4 | -66.2 | -3.31% | 22.0 | 19.8 |
| `mainpla` | 10 | 3450.6 | 20.9 | 3438.4 | +12.2 | +0.23% | 28.9 | 25.0 |

The size results are very close on most circuits. The cases where this implementation is worse than the paper are small differences:

- `bc0`: `+1.2` nodes on average
- `i10`: `+5.2` nodes on average
- `mainpla`: `+12.2` nodes on average

Those are all under `0.25%` of the paper initial size.

The largest differences are in the favorable direction:

- `k2`: `-66.2` nodes, `-3.31%`
- `c7552`: `-58.9` nodes, `-2.81%`
- `dalu`: `-20.0` nodes, `-1.46%`

## Baseline Comparison

Before interpreting RL-vs-paper differences, the deterministic resyn2 baselines must be checked. This is where the reproduction is weakest.

| circuit | ours initial | paper initial | ours resyn2-2 | paper resyn2-2 | ours resyn2-inf | paper resyn2-inf | baseline status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `apex1` | 2655 | 2665 | 1965 | 1966 | 1929 | 1941 | drift |
| `bc0` | 1592 | 1592 | 875 | 899 | 847 | 875 | drift |
| `c1355` | 504 | 504 | 387 | 390 | 387 | 390 | drift |
| `c5315` | 1773 | 1780 | 1293 | 1295 | 1289 | 1294 | drift |
| `c6288` | 2337 | 2337 | 1870 | 1870 | 1870 | 1870 | match |
| `c7552` | 2074 | 2093 | 1369 | 1416 | 1338 | 1398 | drift |
| `dalu` | 1371 | 1371 | 1077 | 1103 | 1077 | 1103 | drift |
| `i10` | 2675 | 2675 | 1789 | 1804 | 1762 | 1789 | drift |
| `k2` | 1998 | 1998 | 1148 | 1186 | 1085 | 1145 | drift |
| `mainpla` | 5346 | 5346 | 3578 | 3583 | 3492 | 3504 | drift |

Only `c6288` has matching deterministic baselines. The other circuits differ in at least one of initial size, resyn2-2, or resyn2-inf.

This is important: if resyn2 gives different numbers, then the local ABC/OpenSpiel/benchmark environment is not identical to the one used for the paper. That does not imply the RL implementation is wrong, but it does mean the experiment is not an exact bit-for-bit reproduction of the paper setup.

## Performance Versus Local resyn2

The learned agent usually beats the local `resyn2-2` baseline by final size:

| circuit | local resyn2-2 | ours final size mean | win rate vs resyn2-2 | win rate vs resyn2-inf |
| --- | ---: | ---: | ---: | ---: |
| `apex1` | 1965 | 1916.3 | 1.0 | 1.0 |
| `bc0` | 875 | 820.6 | 1.0 | 1.0 |
| `c1355` | 387 | 386.0 | 1.0 | 1.0 |
| `c5315` | 1293 | 1325.5 | 0.0 | 0.0 |
| `c6288` | 1870 | 1870.0 | 0.0 | 0.0 |
| `c7552` | 1369 | 1336.5 | 1.0 | 0.5 |
| `dalu` | 1077 | 1019.8 | 1.0 | 1.0 |
| `i10` | 1789 | 1735.4 | 1.0 | 0.7 |
| `k2` | 1148 | 1062.2 | 1.0 | 0.9 |
| `mainpla` | 3578 | 3450.6 | 1.0 | 1.0 |

The weak cases are:

- `c5315`: learned policy is worse than local resyn2-2 and resyn2-inf.
- `c6288`: learned policy ties the deterministic baseline and does not improve.

For the remaining 8 circuits, the policy reliably beats local resyn2-2 by size.

## Depth Interpretation

Depth is not the primary objective in this reproduction. The paper's `RL-1` setting optimizes node count, not depth. Still, depth is useful as a diagnostic.

The reproduction often has worse depth than the paper `RL-1` result:

- `c7552`: ours `52.3`, paper `27.4`
- `dalu`: ours `45.8`, paper `33.2`
- `i10`: ours `53.2`, paper `40.3`

This should not be treated as a failure of size-objective reproduction by itself. It does show that the size-only policy can trade depth for node reduction. For future multi-objective experiments, final QoR and depth should be reported alongside final size.

## Assessment

The reproduction is successful on the main size metric, but not exact at the environment/baseline level.

What looks good:

- The protocol matches the paper's `RL-1` setup.
- Mean final size is close to the paper on almost every circuit.
- The reproduction is worse than the paper on only 3 circuits, and those gaps are very small.
- The implementation beats local resyn2-2 on 8 of 10 circuits.

What limits exact reproducibility:

- Deterministic baselines differ from the paper on 9 of 10 circuits.
- Some initial circuit sizes differ from the paper.
- Depth behavior differs substantially on several circuits.

The most likely cause is benchmark/toolchain drift: different BLIF/AIG inputs, ABC/OpenSpiel implementation details, ABC version differences, or command semantics. Since deterministic resyn2 results already differ, RL result differences cannot be attributed cleanly to the learning implementation alone.

## Conclusion

The code reproduces the qualitative result of Zhu et al.: a learned sequence policy can match or improve the size results of fixed resyn2-style heuristics under the same action budget.

Numerically, the final node counts are close to the paper's `RL-1` results. The reproduction is not exact because the deterministic resyn2 baselines do not match the paper for most circuits. Therefore, this should be described as a close reproduction under a slightly different toolchain/benchmark environment, not as an exact reproduction of the original experimental environment.

The next step for stronger reproducibility is to make the deterministic baselines match first. If initial sizes, resyn2-1, resyn2-2, and resyn2-inf match the paper, then RL-vs-paper comparisons will become much more meaningful.
