# Multi-Objective Reward Experiments on 6 Circuits

This report analyzes the multi-objective reward experiments stored under `outputs/`.

The batch covers 6 circuits:

```text
c1355, bc0, apex1, c5315, dalu, k2
```

It compares 4 method/reward variants:


| variant            | algorithm                   | reward                                     |
| ------------------ | --------------------------- | ------------------------------------------ |
| `REINFORCE-Linear` | REINFORCE                   | `linear`, `c_size=0.5`, `c_depth=0.5`      |
| `REINFORCE-DOP`    | REINFORCE                   | `diff_of_product`, `c_size=1`, `c_depth=1` |
| `TB-Linear`        | GFlowNet Trajectory Balance | `linear`, `c_size=0.5`, `c_depth=0.5`      |
| `TB-DOP`           | GFlowNet Trajectory Balance | `diff_of_product`, `c_size=1`, `c_depth=1` |


All reports have 10 runs, 200 episodes, 20 synthesis operations per episode, and `paper_mode.infer_rollouts=10`. This fixes the comparability issue from the earlier batch where TB used fewer inference rollouts than REINFORCE.

## Executive Summary

The main conclusion is that **Trajectory Balance is best for finding smaller circuits, but REINFORCE is best for balanced size-depth quality**.

- `TB-Linear` has the best mean final size on 3/6 circuits.
- `TB-DOP` has the best mean final size on 1/6 circuits.
- REINFORCE variants have the best mean final depth on all 6 circuits.
- Best QoR (`final_size * final_depth`) is achieved by REINFORCE on all 6 circuits:
  - `REINFORCE-Linear`: best QoR on 4/6 circuits.
  - `REINFORCE-DOP`: best QoR on 2/6 circuits.
- TB variants are useful as the **size-extreme side of the Pareto frontier**, but they frequently pay for smaller size with worse depth.

Therefore, if the goal is a **balanced front in size and depth**, the best current choice is not a pure TB run. The most informative front comes from combining:

```text
REINFORCE-Linear + REINFORCE-DOP + TB-Linear/TB-DOP size-extreme candidates
```

If only one algorithm family can be used, choose **REINFORCE** for balanced QoR. If the goal is pure size reduction, choose **TB**, especially `TB-Linear`.

## Aggregate Metrics

These are means over the 6 circuits. Since every variant covers the same circuits, these group averages are directly comparable.


| variant            | mean final size | mean final depth | mean final QoR | avg win rate vs resyn2-2 | avg normalized improvement vs resyn2-2 |
| ------------------ | --------------- | ---------------- | -------------- | ------------------------ | -------------------------------------- |
| `REINFORCE-DOP`    | 1152.1          | 21.0             | 24366.0        | 0.25                     | -0.0166                                |
| `REINFORCE-Linear` | 1150.3          | 21.0             | 24352.8        | 0.22                     | -0.0154                                |
| `TB-DOP`           | 1127.5          | 23.2             | 25971.5        | 0.57                     | -0.0007                                |
| `TB-Linear`        | 1120.4          | 23.3             | 26005.3        | 0.68                     | +0.0031                                |


Interpretation:

- TB gives the smallest average size.
- REINFORCE gives substantially better average depth.
- REINFORCE gives better average QoR.
- TB has higher win rate versus `resyn2-2` by size, but this metric is size-only and does not capture depth penalties.

## Per-Circuit Results


| circuit | variant            | mean size | mean depth | mean QoR | win vs resyn2-2 | normalized improvement vs resyn2-2 |
| ------- | ------------------ | --------- | ---------- | -------- | --------------- | ---------------------------------- |
| `apex1` | `REINFORCE-DOP`    | 1995.6    | 17.0       | 33925.2  | 0.2             | -0.0115                            |
| `apex1` | `REINFORCE-Linear` | 1990.2    | 17.2       | 34220.4  | 0.1             | -0.0095                            |
| `apex1` | `TB-DOP`           | 1963.1    | 17.6       | 34537.9  | 0.6             | +0.0007                            |
| `apex1` | `TB-Linear`        | 1949.8    | 17.9       | 34892.7  | 0.8             | +0.0057                            |
| `bc0`   | `REINFORCE-DOP`    | 847.0     | 18.7       | 15842.5  | 1.0             | +0.0176                            |
| `bc0`   | `REINFORCE-Linear` | 848.0     | 18.2       | 15432.7  | 1.0             | +0.0170                            |
| `bc0`   | `TB-DOP`           | 860.5     | 18.7       | 16082.4  | 0.7             | +0.0091                            |
| `bc0`   | `TB-Linear`        | 857.5     | 18.2       | 15607.7  | 0.9             | +0.0110                            |
| `c1355` | `REINFORCE-DOP`    | 388.1     | 17.7       | 6868.7   | 0.1             | -0.0022                            |
| `c1355` | `REINFORCE-Linear` | 387.9     | 17.7       | 6865.8   | 0.0             | -0.0018                            |
| `c1355` | `TB-DOP`           | 386.1     | 18.8       | 7258.5   | 0.9             | +0.0018                            |
| `c1355` | `TB-Linear`        | 386.4     | 18.8       | 7264.2   | 0.7             | +0.0012                            |
| `c5315` | `REINFORCE-DOP`    | 1328.3    | 27.6       | 36657.5  | 0.1             | -0.0199                            |
| `c5315` | `REINFORCE-Linear` | 1324.4    | 27.6       | 36555.0  | 0.2             | -0.0177                            |
| `c5315` | `TB-DOP`           | 1343.0    | 27.6       | 37062.5  | 0.0             | -0.0282                            |
| `c5315` | `TB-Linear`        | 1332.5    | 27.8       | 37043.0  | 0.1             | -0.0223                            |
| `dalu`  | `REINFORCE-DOP`    | 1160.6    | 31.9       | 37023.8  | 0.0             | -0.0610                            |
| `dalu`  | `REINFORCE-Linear` | 1151.2    | 31.9       | 36723.0  | 0.0             | -0.0541                            |
| `dalu`  | `TB-DOP`           | 1051.6    | 42.8       | 44986.0  | 0.9             | +0.0185                            |
| `dalu`  | `TB-Linear`        | 1040.0    | 43.3       | 45025.2  | 1.0             | +0.0270                            |
| `k2`    | `REINFORCE-DOP`    | 1193.3    | 13.3       | 15878.0  | 0.1             | -0.0227                            |
| `k2`    | `REINFORCE-Linear` | 1200.0    | 13.6       | 16320.0  | 0.0             | -0.0260                            |
| `k2`    | `TB-DOP`           | 1160.4    | 13.7       | 15901.6  | 0.3             | -0.0062                            |
| `k2`    | `TB-Linear`        | 1156.0    | 14.0       | 16199.2  | 0.6             | -0.0040                            |


## Pareto Front by Circuit

A variant is considered Pareto non-dominated if no other variant has both lower or equal size and lower or equal depth, with at least one strict improvement.


| circuit | non-dominated variants                                     | best size          | best depth                                      | best QoR           |
| ------- | ---------------------------------------------------------- | ------------------ | ----------------------------------------------- | ------------------ |
| `apex1` | `TB-Linear`, `TB-DOP`, `REINFORCE-Linear`, `REINFORCE-DOP` | `TB-Linear`        | `REINFORCE-DOP`                                 | `REINFORCE-DOP`    |
| `bc0`   | `REINFORCE-DOP`, `REINFORCE-Linear`                        | `REINFORCE-DOP`    | `REINFORCE-Linear`                              | `REINFORCE-Linear` |
| `c1355` | `TB-DOP`, `REINFORCE-Linear`                               | `TB-DOP`           | `REINFORCE-DOP` / `REINFORCE-Linear`            | `REINFORCE-Linear` |
| `c5315` | `REINFORCE-Linear`                                         | `REINFORCE-Linear` | `REINFORCE-DOP` / `REINFORCE-Linear` / `TB-DOP` | `REINFORCE-Linear` |
| `dalu`  | `TB-Linear`, `TB-DOP`, `REINFORCE-Linear`                  | `TB-Linear`        | `REINFORCE-DOP` / `REINFORCE-Linear`            | `REINFORCE-Linear` |
| `k2`    | `TB-Linear`, `TB-DOP`, `REINFORCE-DOP`                     | `TB-Linear`        | `REINFORCE-DOP`                                 | `REINFORCE-DOP`    |


Front appearances:


| variant            | Pareto front appearances |
| ------------------ | ------------------------ |
| `REINFORCE-Linear` | 5                        |
| `TB-DOP`           | 4                        |
| `REINFORCE-DOP`    | 3                        |
| `TB-Linear`        | 3                        |


This is the key result for "balanced front" behavior:

- `REINFORCE-Linear` appears on the most fronts and has best QoR most often.
- `TB-DOP` and `TB-Linear` are valuable front points when size is prioritized.
- `REINFORCE-DOP` is valuable when depth or QoR is prioritized.

## Circuit-Level Interpretation

### `apex1`

All four variants are non-dominated. TB variants produce smaller circuits, while REINFORCE variants preserve better depth and QoR. This is a true tradeoff circuit.

Best balanced choice: `REINFORCE-DOP`.

### `bc0`

REINFORCE dominates TB. TB has neither the best size nor the best depth. `REINFORCE-Linear` gives the best QoR.

Best balanced choice: `REINFORCE-Linear`.

### `c1355`

TB finds the smallest size, but REINFORCE keeps much better depth and QoR. Since size is nearly saturated on this circuit, the depth penalty matters more.

Best balanced choice: `REINFORCE-Linear`.

### `c5315`

`REINFORCE-Linear` is the clear winner. It has the best size and best QoR, and its depth is tied with the best observed depth. TB does not add a useful Pareto point here.

Best balanced choice: `REINFORCE-Linear`.

### `dalu`

This is the clearest size-depth tradeoff. TB reduces size dramatically:

- `TB-Linear`: 1040.0 size
- `REINFORCE-Linear`: 1151.2 size

But TB depth is much worse:

- `TB-Linear`: 43.3 depth
- `REINFORCE-Linear`: 31.9 depth

QoR strongly favors REINFORCE despite TB's size advantage.

Best balanced choice: `REINFORCE-Linear`.

### `k2`

TB variants give smaller size, but REINFORCE-DOP gives the best depth and best QoR. This is another tradeoff circuit.

Best balanced choice: `REINFORCE-DOP`.

## Which Algorithm Is Best for a Balanced Size-Depth Front?

There are two different answers depending on what "best" means.

### If "balanced" means best single operating point

Use **REINFORCE**, not TB.

Best QoR is always achieved by a REINFORCE variant:


| best QoR variant   | circuits |
| ------------------ | -------- |
| `REINFORCE-Linear` | 4        |
| `REINFORCE-DOP`    | 2        |
| TB variants        | 0        |


`REINFORCE-Linear` is the best default balanced method. It wins QoR on `bc0`, `c1355`, `c5315`, and `dalu`.

`REINFORCE-DOP` should be kept because it wins QoR on `apex1` and `k2`, and it often gives the best depth.

### If "balanced front" means a set of diverse Pareto candidates

Use both algorithm families:

```text
REINFORCE-Linear, REINFORCE-DOP, TB-Linear, TB-DOP
```

TB contributes the low-size side of the front on `apex1`, `c1355`, `dalu`, and `k2`. REINFORCE contributes the low-depth and best-QoR side.

The practical recommendation is:

1. Always run `REINFORCE-Linear`.
2. Run `REINFORCE-DOP` when depth and QoR matter.
3. Run one TB variant when you want a size-extreme candidate.
4. Prefer `TB-Linear` for size minimization; prefer `TB-DOP` if you want a slightly more conservative TB point.

## Conclusions

1. **REINFORCE is currently the best balanced algorithm family.**
  It consistently produces better depth and better QoR than TB.
2. **TB is better at finding smaller circuits, but it is not balanced yet.**
  TB often improves size but gives up too much depth. This is most visible on `dalu`.
3. `**REINFORCE-Linear` is the strongest default.**
  It has the best QoR on 4/6 circuits and appears on 5/6 Pareto fronts.
4. `**REINFORCE-DOP` is still important.**
  It wins QoR on `apex1` and `k2`, and it usually gives the best depth.
5. **TB variants are useful for frontier expansion, not for final balanced QoR.**
  They should be treated as low-size candidates that expand the Pareto set, not as replacements for REINFORCE.
6. **Size-only win rate is misleading for this experiment.**
  TB has higher win rates versus resyn2-2, but REINFORCE has better QoR. Multi-objective experiments should report size, depth, and QoR together.

## Recommended Next Experiments

For the next iteration:

1. Keep all 6 circuits; this subset is informative.
2. Compare using Pareto front and QoR, not only final size.
3. Add a scalarized "front quality" metric, such as hypervolume over normalized size/depth.
4. If runtime is constrained, run:

```text
REINFORCE-Linear, REINFORCE-DOP, TB-Linear
```

This gives the best QoR methods plus the strongest low-size TB candidate.

1. Consider modifying TB reward or selection so depth penalties are stronger. The current TB behavior suggests it can discover size reductions but does not adequately preserve depth.

