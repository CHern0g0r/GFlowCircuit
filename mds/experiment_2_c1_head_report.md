# Experiment 2 report: active baseline with C1_head

Date: 2026-08-03

## Status

The experiment is complete and scientifically conclusive. All six full runs
(`bc0` and `dalu`, seeds 0--2) finished without numerical failures or stderr.
The runs used the C1_head backbone, policy learning rate `0.001`, resolved
`logZ` learning rate `0.01`, and checkpoints at 200, 400, and 800 training
trajectories.

Provenance:

- preflight job: `16626`;
- full-run array: `16628` (scheduler tasks `16628`--`16633`);
- aggregate diagnosis: `16645`;
- project commit: `82ca6ef2a41e246fe5eecf51244b2213eccc4a27`;
- source-tree SHA-256:
  `94f469986e769482609d1c81c2a02fbbb1d68fff4e0dffdda7bd9b7953a0b0a6`.

## Results

The global-offset hypothesis is supported. At 800 trajectories, the pooled
median bias fraction is `0.997182`; analytic recentering removes nearly all
validation MSE in every circuit/stratum:

| Circuit | `fixed_uniform` | `fresh_on_policy` |
| --- | ---: | ---: |
| `bc0` | 0.997333 | 0.996551 |
| `dalu` | 0.997161 | 0.996610 |

The aggregate undertraining rule is also met, but only through `dalu` search
quality. From 400 to 800 trajectories, `dalu` best-of-N AUC rises by 19.92%;
`bc0` rises by only 3.90%. Archive-hypervolume gains (`0.000446` and
`0.002542`) remain below the `0.005` threshold, and centered residual RMS gets
worse on both circuits rather than improving.

The active baseline is not healthy at 800 trajectories. All 12 combinations
of circuit, seed, and validation stratum fail `log_z_target_gap`,
`bias_fraction`, and `standardized_bias`. Other execution-level checks remain
finite and the report contains no numerical-failure signal.

## Interpretation and decision

C1_head changes neither the diagnosis nor the next experimental step: the
dominant endpoint error is a common residual offset attributable to `logZ`,
not an unrecoverable run failure. More training may still improve search on
`dalu`, but the mixed learning signal and universal calibration-gate failures
do not justify treating a budget extension as the remedy. Proceed to
Experiment 3 (calibrated `logZ` initialization), retaining this active
configuration as its control.

The canonical remote report bundle is
`/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-active-baseline/tb-active-baseline-report-v3/diagnosis`.
