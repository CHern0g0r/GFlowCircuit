# Circuit Subset Recommendation

Running all 10 Zhu/MCNC circuits for every experiment iteration is expensive. For routine development, use a smaller subset that still exposes the main behaviors seen in the current REINFORCE and Trajectory Balance results.

## Recommended Standard Subset

Use these 5 circuits:

```text
c1355, bc0, apex1, c5315, dalu
```

| circuit | Why include it |
| --- | --- |
| `c1355` | Small and nearly saturated. Good sanity check: most methods should land close in size, so bad runs show up as instability or poor depth/QoR. |
| `bc0` | Small but still discriminative. It distinguishes reward choices better than `c1355` while remaining cheap enough for frequent runs. |
| `apex1` | Medium-size continuity benchmark. It appears across many existing runs and is useful for comparing new batches to old ones. |
| `c5315` | Hard case. Most methods struggle here, so it helps detect methods that only work on easy circuits. |
| `dalu` | Important TB stress case. TB DifferenceOfProduct reduced size strongly but hurt depth/QoR, so this circuit exposes the key multi-objective tradeoff. |

This subset keeps one easy/saturated circuit, one small discriminative circuit, one medium continuity circuit, one hard circuit, and one circuit that exposes the TB size-depth failure mode.

## Smaller Tiers

For very quick debugging:

```text
c1355, c5315
```

For fast but still informative runs:

```text
c1355, bc0, c5315, dalu
```

For the normal iteration loop:

```text
c1355, bc0, apex1, c5315, dalu
```

For final reporting:

```text
all 10 circuits
```

## Optional 6th Circuit

If runtime allows, add:

```text
k2
```

`k2` is useful because TB DifferenceOfProduct also improves size while hurting depth/QoR. It gives a second example of the same tradeoff seen on `dalu`, reducing the risk of overfitting conclusions to one circuit.

## Circuits to Skip During Routine Iteration

Do not include `c6288` in the routine subset unless you specifically need a flat-control benchmark. In these results, it is mostly saturated around the resyn2 baseline and gives limited signal for the cost.

Avoid `mainpla` for routine iteration because it is the largest circuit and likely expensive. Keep it for final validation.

## Comparability Requirement

Use the same subset and the same evaluation budget across algorithms. In the current batch, REINFORCE used `infer_rollouts=10`, while TB used `infer_rollouts=4`; that makes direct TB-vs-REINFORCE comparison conservative for TB but not strictly fair.

For future subset experiments, set the same `infer_rollouts` for both algorithms, preferably:

```yaml
paper_mode:
  infer_rollouts: 10
```

If runtime is the limiting factor, reduce both algorithms to the same smaller value rather than giving them different inference budgets.
