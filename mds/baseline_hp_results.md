# Baseline training hyperparameter and trajectory-budget results

## Executive decision

The completed compact screens and trajectory-budget experiments select 800
training trajectories for REINFORCE and DRiLLS-A2C, and 400 for PPO. None of
the selected profiles needs the conditional 6,400-trajectory extension.

| Method | Selected profile | Selected budget | Next tested budget | Successor-minus-selected normalized HV gain (95% CI) | 6,400 test? |
| --- | --- | ---: | ---: | ---: | --- |
| REINFORCE | `policy_lr_high` | 800 | 1,600 | -0.0164 [-0.0348, -0.0041] | No |
| DRiLLS-A2C | `lr_low_value_high` | 800 | 1,600 | -0.0749 [-0.1339, -0.0212] | No |
| PPO | `epochs_high` | 400 | 800 | -0.0304 [-0.0861, +0.0060] | No |

For every selected configuration, its budget is within 2% of the best
two-circuit score observed in the applicable curve, and the paired bootstrap
does not establish that the next doubling is better (`CI95_low <= 0`). These
are the two protocol gates for an eligible plateau. The extension is required
only when no tested budget is eligible, so running 6,400 trajectories would
not be supported by the stopping rule.

Two subsequent interaction screens motivated follow-up decisions that are now
resolved by the fresh DRiLLS and PPO budget curves:

| Follow-up screen | Winner | Runner-up | Combined profile in top two? | Decision |
| --- | --- | --- | --- | --- |
| DRiLLS-A2C LR × gamma × value loss | `lr_low_value_high` | `lr_low_long_credit_value_high` | Yes, both | Combined-profile curve completed; select `lr_low_value_high` at 800 and proceed to confirmation |
| PPO epochs × clipping | `clip_low` | `epochs_low` | No | Three-profile curve completed; select `epochs_high` at 400 and retain `clip_low` as runner-up |

The fresh DRiLLS curve resolves the issue that the old `long_credit` budget
curve could not answer for a combined profile. The fresh PPO curve resolves
the single-factor `clip_low` ambiguity without promoting either antagonistic
epoch/clip combination. All screen and budget results still use only three
training seeds and are not a substitute for ten-seed confirmation.

## Experimental contract and validity

The initial campaign used project commit
`cd32b58ac705118cc7024e8b29d2ab54dbb71648` and protocol hash
`d741b16c335400e9d68a51bb4729300730d1f2901000bf45bba35380eec889b3`.
Exploration was frozen before this campaign: REINFORCE entropy beta `0.01`,
DRiLLS-A2C entropy beta `0.0003`, and PPO entropy beta `0.03`. Thus, the results
below compare only the remaining baseline training hyperparameters.

The interaction campaigns used project commit
`5449881f6da23c47051f128b294e63f7dc600cbf`. The DRiLLS protocol hash was
`39d034095e46ed8dd628641664b7b8602c891278629a137b05f490da0128f7e1` and
the PPO protocol hash was
`0a7cd710dcada90b20d7ac81e9313e40aaf1d6efbf1cae0a3d4ce01009ba54ff`.
They retained the same circuits, seeds, 800-trajectory budget, and 50-sample
evaluation contract.

The DRiLLS combined-profile budget campaign used project commit
`16f18be5a6c5b3d970330da177ffb1eb9f6c1e2e` and protocol hash
`aa784919e573dfa62e5f9bf01a74030f9a641710e31ef92d10e0e2c2927fd7a1`.
It reran both interaction-screen finalists at every budget rather than reusing
the earlier 800-trajectory artifacts.

The PPO clip-ambiguity budget campaign used the same project commit and
protocol hash
`7ce26fe03cc7381c4ffdc176077ec6c532f00e6c272b939b3252d2e618680ee2`.
It freshly reran `clip_low`, `epochs_low`, and `epochs_high` at every budget.

The test circuits were `C1355` (smaller structured logic) and `dalu` (medium
ALU/datapath). Every setting used training seeds 0, 1, and 2. The primary
metric is the mean per-training-seed hypervolume (HV) from exactly 50 paired
post-training samples using evaluation seed 42. Reported `±` values are the
standard deviation across the three training seeds. Pooled-front HV was kept
as a descriptive output but was not used for selection.

The selection score is computed within each method and stage: for each circuit,
mean HV is divided by the best mean HV in that stage, then the two circuit
ratios are averaged. It is useful for comparing configurations of the same
method, but it must not be treated as a cross-method effect size.
Screen ranks follow the protocol's 0.02 practical-tie band and its minimum-
circuit-score/product-improvement tie breaks, so they are not always in strict
descending order of the displayed average selection score.

| Stage | SLURM job | State | Wall time | Peak host memory | Task outcome |
| --- | ---: | --- | ---: | ---: | --- |
| Smoke | 20701 | completed, `0:0` | 0:00:47 | 4.8 GiB | 3/3 complete |
| REINFORCE screen | 20706 | completed, `0:0` | 2:16:48 | 6.1 GiB | 42/42 complete |
| DRiLLS-A2C screen | 20707 | completed, `0:0` | 1:42:00 | 6.1 GiB | 42/42 complete |
| PPO screen | 20725 | completed, `0:0` | 2:38:02 | 6.5 GiB | 48/48 complete |
| REINFORCE budget curve | 20729 | completed, `0:0` | 4:17:58 | 6.1 GiB | 48 complete + 12 reused |
| DRiLLS-A2C budget curve | 20749 | completed, `0:0` | 3:16:22 | 6.1 GiB | 48 complete + 12 reused |
| PPO budget curve | 20751 | completed, `0:0` | 4:44:29 | 6.5 GiB | 48 complete + 12 reused |
| DRiLLS-A2C interaction screen | 20862 | completed, `0:0` | 1:52:02 | 6.1 GiB | 48/48 complete |
| PPO epoch/clip interaction screen | 20863 | completed, `0:0` | 2:00:27 | 6.6 GiB | 36/36 complete |
| DRiLLS-A2C combined-profile budget curve | 21428 | completed, `0:0` | 3:39:24 | 6.8 GiB | 60/60 complete |
| PPO clip-ambiguity budget curve | 21430 | completed, `0:0` | 8:03:13 | 6.5 GiB | 90/90 complete |

The reused tasks in the original budget campaigns are the matching
800-trajectory screen runs for the two finalists (two circuits × three seeds ×
two profiles). The final DRiLLS and PPO campaigns instead reran all 60 and 90
tasks, respectively. Both manifests have state `complete`; all training and
sampling subprocesses exited zero, every task produced exactly 50 evaluation
samples, and both jobs' scheduler stdout and stderr logs are empty.

## REINFORCE

### Hyperparameter screen at 800 trajectories

The control configuration used policy learning rate `8e-4`, value learning
rate `3e-3`, gamma `0.9`, no return normalization, and no policy/value gradient
clipping. Each non-control profile changed one of these choices.

| Rank | Profile | Change from control | C1355 HV | dalu HV | Selection score | Mean product improvement | Mean task time (min) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `policy_lr_high` | policy LR `2e-3` | 0.081044 ± 0.000000 | 0.041971 ± 0.000758 | 1.0000 | 0.3783 | 12.11 |
| 2 | `control` | — | 0.081044 ± 0.000000 | 0.041944 ± 0.000629 | 0.9997 | 0.3737 | 12.44 |
| 3 | `long_credit` | gamma `0.99` | 0.080891 ± 0.000062 | 0.039603 ± 0.003027 | 0.9708 | 0.3610 | 12.37 |
| 4 | `value_lr_high` | value LR `1e-2` | 0.078017 ± 0.004281 | 0.037234 ± 0.003608 | 0.9249 | 0.3439 | 12.04 |
| 5 | `value_lr_low` | value LR `1e-3` | 0.080688 ± 0.000401 | 0.037171 ± 0.002764 | 0.9406 | 0.3618 | 12.36 |
| 6 | `policy_lr_low` | policy LR `3e-4` | 0.080866 ± 0.000036 | 0.031746 ± 0.002204 | 0.8771 | 0.3543 | 12.56 |
| 7 | `normalized_returns` | normalize returns | 0.079747 ± 0.000108 | 0.026939 ± 0.004558 | 0.8129 | 0.3722 | 11.98 |

`policy_lr_high` and `control` advanced. They were effectively tied: the score
difference was only 0.00033, both saturated C1355, and their dalu means differed
by only 0.000027. Raising the policy learning rate therefore gave a very small
screen advantage rather than a decisive separation. Lowering the policy rate
or normalizing returns substantially damaged dalu HV. Changing the value
learning rate and increasing gamma were intermediate.

### Trajectory-budget curve

| Profile | Trajectories | C1355 HV | dalu HV | Selection score | Mean product improvement | Optimizer updates | Mean task time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `policy_lr_high` | 200 | 0.080917 ± 0.000036 | 0.035011 ± 0.000373 | 0.9163 | 0.3336 | 4,000 | 3.77 |
| `policy_lr_high` | 400 | 0.080993 ± 0.000072 | 0.039193 ± 0.003373 | 0.9666 | 0.3626 | 8,000 | 6.72 |
| `policy_lr_high` | 800 | 0.081044 ± 0.000000 | 0.041971 ± 0.000758 | 1.0000 | 0.3783 | 16,000 | 12.11 |
| `policy_lr_high` | 1,600 | 0.080230 ± 0.000589 | 0.040992 ± 0.000524 | 0.9833 | 0.3850 | 32,000 | 23.65 |
| `policy_lr_high` | 3,200 | 0.080891 ± 0.000216 | 0.036852 ± 0.002902 | 0.9381 | 0.3893 | 64,000 | 46.51 |
| `control` | 200 | 0.080942 ± 0.000072 | 0.038324 ± 0.004388 | 0.9559 | 0.3194 | 4,000 | 3.87 |
| `control` | 400 | 0.080942 ± 0.000072 | 0.039783 ± 0.003104 | 0.9733 | 0.3497 | 8,000 | 6.95 |
| `control` | 800 | 0.081044 ± 0.000000 | 0.041944 ± 0.000629 | 0.9997 | 0.3737 | 16,000 | 12.44 |
| `control` | 1,600 | 0.081044 ± 0.000000 | 0.041305 ± 0.000354 | 0.9921 | 0.3952 | 32,000 | 23.74 |
| `control` | 3,200 | 0.081044 ± 0.000000 | 0.040929 ± 0.000691 | 0.9876 | 0.3975 | 64,000 | 46.00 |

The higher policy rate improves much more sharply from 200 to 800 trajectories,
where it attains the global best score. Its 1,600 successor is 1.67% lower, and
the paired normalized difference is significantly negative. The control is
better at 200 and 400 and slightly more stable after 800, but it does not beat
the selected profile at the selected budget. The protocol therefore selects
`policy_lr_high` at 800, with `control` as the runner-up and 1,600 as the
confirmation successor.

Mean product improvement continues to increase at larger budgets even when
50-sample HV declines. This is not a contradiction: product improvement is a
scalar average, while HV rewards the quality and coverage of the sampled
two-objective front. HV remains the predeclared primary metric.

## DRiLLS-A2C

### Hyperparameter screen at 800 trajectories

The control used learning rate `1e-3`, gamma `0.9`, value-loss coefficient
`0.5`, raw advantages, and no gradient clipping. DRiLLS used four trajectories
per episode throughout.

| Rank | Profile | Change from control | C1355 HV | dalu HV | Selection score | Mean product improvement | Mean task time (min) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `long_credit` | gamma `0.99` | 0.081019 ± 0.000036 | 0.031426 ± 0.003662 | 0.9732 | 0.3440 | 9.35 |
| 2 | `learning_rate_low` | learning rate `3e-4` | 0.074939 ± 0.004209 | 0.033205 ± 0.000582 | 0.9625 | 0.3032 | 10.04 |
| 3 | `value_loss_high` | value coefficient `1.0` | 0.080993 ± 0.000072 | 0.029516 ± 0.000175 | 0.9443 | 0.3483 | 9.75 |
| 4 | `normalized_advantages` | normalize advantages | 0.077788 ± 0.004281 | 0.029384 ± 0.002426 | 0.9225 | 0.3568 | 9.28 |
| 5 | `value_loss_low` | value coefficient `0.25` | 0.080739 ± 0.000330 | 0.029134 ± 0.000723 | 0.9370 | 0.3519 | 8.39 |
| 6 | `control` | — | 0.080128 ± 0.000648 | 0.028995 ± 0.000332 | 0.9311 | 0.3670 | 8.05 |
| 7 | `learning_rate_high` | learning rate `3e-3` | 0.080128 ± 0.000648 | 0.026981 ± 0.000865 | 0.9008 | 0.3619 | 9.84 |

`long_credit` and `learning_rate_low` advanced. Their strengths differ by
circuit: gamma `0.99` is much stronger on C1355, while the lower learning rate
has the best dalu HV. The 0.0107 score gap is inside the practical 0.02 tie
band. The control has the highest mean product improvement and shortest time,
but its weaker dalu HV prevents it from winning under the primary metric.
Neither advantage normalization nor a high learning rate helped.

### Trajectory-budget curve

| Profile | Trajectories | C1355 HV | dalu HV | Selection score | Mean product improvement | Optimizer updates | Mean task time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_credit` | 200 | 0.074990 ± 0.004173 | 0.028078 ± 0.005403 | 0.8855 | 0.2807 | 50 | 3.05 |
| `long_credit` | 400 | 0.077890 ± 0.004353 | 0.030218 ± 0.002969 | 0.9356 | 0.3311 | 100 | 5.75 |
| `long_credit` | 800 | 0.081019 ± 0.000036 | 0.031426 ± 0.003662 | 0.9731 | 0.3440 | 200 | 9.35 |
| `long_credit` | 1,600 | 0.081044 ± 0.000000 | 0.027717 ± 0.001196 | 0.9174 | 0.3582 | 400 | 17.35 |
| `long_credit` | 3,200 | 0.074150 ± 0.003881 | 0.022979 ± 0.002247 | 0.8035 | 0.3668 | 800 | 35.70 |
| `learning_rate_low` | 200 | 0.074990 ± 0.004227 | 0.018283 ± 0.008604 | 0.7380 | 0.2379 | 50 | 2.82 |
| `learning_rate_low` | 400 | 0.071963 ± 0.000000 | 0.027578 ± 0.000277 | 0.8592 | 0.2556 | 100 | 4.93 |
| `learning_rate_low` | 800 | 0.074939 ± 0.004209 | 0.033205 ± 0.000582 | 0.9623 | 0.3032 | 200 | 10.04 |
| `learning_rate_low` | 1,600 | 0.080790 ± 0.000360 | 0.030808 ± 0.001756 | 0.9623 | 0.3452 | 400 | 17.15 |
| `learning_rate_low` | 3,200 | 0.081044 ± 0.000000 | 0.028023 ± 0.000963 | 0.9220 | 0.3600 | 800 | 33.32 |

Both profiles improve through 800, but neither benefits in HV from continuing
beyond it. `long_credit` drops 5.72% in selection score at 1,600 and 17.46%
relative to its 800 score by 3,200, mostly because dalu deteriorates. The low
learning rate is almost exactly flat from 800 to 1,600, then declines. The
protocol selects `long_credit` at 800, `learning_rate_low` as runner-up, and
1,600 as the confirmation successor.

The DRiLLS result is another example where scalar product improvement keeps
rising while sampled Pareto-front HV falls. More training trajectories do not
translate into better 50-sample coverage here.

### Learning-rate, credit-horizon, and value-loss interaction screen

The follow-up screen reran the control and three single-factor profiles and
completed the full binary factorial over learning rate (`1e-3`/`3e-4`), gamma
(`0.9`/`0.99`), and value-loss coefficient (`0.5`/`1.0`). All other settings
were fixed: raw advantages, no gradient clipping, entropy beta `0.0003`, four
trajectories per episode, 800 training trajectories, and 50 evaluation
samples.

| Rank | Profile | Learning rate | Gamma | Value coefficient | C1355 HV | dalu HV | Selection score | Mean product improvement | Mean task time (min) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `lr_low_value_high` | 3e-4 | 0.9 | 1.0 | 0.077915 ± 0.004155 | 0.036109 ± 0.003418 | 0.9652 | 0.3060 | 8.98 |
| 2 | `lr_low_long_credit_value_high` | 3e-4 | 0.99 | 1.0 | 0.074990 ± 0.004173 | 0.037324 ± 0.003413 | 0.9634 | 0.2999 | 9.37 |
| 3 | `learning_rate_low` | 3e-4 | 0.9 | 0.5 | 0.080891 ± 0.000125 | 0.032232 ± 0.000762 | 0.9316 | 0.3154 | 8.37 |
| 4 | `value_loss_high` | 1e-3 | 0.9 | 1.0 | 0.080764 ± 0.000396 | 0.029085 ± 0.000225 | 0.8887 | 0.3572 | 8.51 |
| 5 | `control` | 1e-3 | 0.9 | 0.5 | 0.080688 ± 0.000504 | 0.028828 ± 0.000309 | 0.8848 | 0.3595 | 7.39 |
| 6 | `long_credit_value_high` | 1e-3 | 0.99 | 1.0 | 0.080891 ± 0.000000 | 0.028710 ± 0.000600 | 0.8845 | 0.3438 | 8.92 |
| 7 | `lr_low_long_credit` | 3e-4 | 0.99 | 0.5 | 0.074914 ± 0.008400 | 0.029829 ± 0.002577 | 0.8625 | 0.3035 | 10.96 |
| 8 | `long_credit` | 1e-3 | 0.99 | 0.5 | 0.080917 ± 0.000180 | 0.027988 ± 0.000723 | 0.8749 | 0.3555 | 9.24 |

Both top profiles combine the lower learning rate with the higher value-loss
coefficient. They sacrifice some C1355 HV but produce large dalu gains, which
raises their balanced selection scores. Adding gamma `0.99` produces the best
dalu mean, but slightly lowers the overall score relative to
`lr_low_value_high`.

The paired factorial effects below are changes in circuit-normalized sampled
HV over the six circuit/seed blocks. Positive values favor the named factor or
combination.

| Factorial contrast | Mean effect | Bootstrap 95% CI |
| --- | ---: | ---: |
| Low learning rate | +0.04746 | [-0.03068, +0.12225] |
| Long credit | -0.02125 | [-0.04979, +0.00465] |
| High value-loss coefficient | +0.03696 | [-0.01125, +0.08619] |
| Low LR × long credit | -0.02842 | [-0.09734, +0.02842] |
| Low LR × high value loss | +0.06049 | [-0.02021, +0.14580] |
| Long credit × high value loss | +0.03648 | [-0.00458, +0.09943] |
| Low LR × long credit × high value loss | +0.06174 | [-0.04528, +0.19036] |

No DRiLLS factorial interval excludes zero, so these effect estimates are
explanatory rather than confirmatory. The largest positive two-way estimate is
low learning rate × high value-loss coefficient, consistent with the winning
profile; the gamma `0.99` main effect is mildly negative. Under the predeclared
ranking gate, both requested interaction profiles enter the top two. At that
point, DRiLLS confirmation was postponed pending a new trajectory-budget curve
for `lr_low_value_high` and `lr_low_long_credit_value_high`; the completed
`long_credit` curve could not be transferred to either combined profile.

### Combined-profile trajectory-budget curve

The follow-up campaign reran both interaction-screen winners at 200, 400, 800,
1,600, and 3,200 trajectories under one immutable protocol. Each cell again
contains both circuits and three training seeds, with exactly 50 evaluation
samples per task.

| Profile | Trajectories | C1355 HV | dalu HV | Selection score | Mean product improvement | Optimizer updates | Mean task time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lr_low_value_high` | 200 | 0.077991 ± 0.004209 | 0.027648 ± 0.000307 | 0.8792 | 0.2390 | 50 | 2.96 |
| `lr_low_value_high` | 400 | 0.077991 ± 0.004263 | 0.027474 ± 0.000305 | 0.8767 | 0.2565 | 100 | 5.17 |
| `lr_low_value_high` | 800 | 0.080891 ± 0.000062 | 0.033677 ± 0.001282 | 0.9840 | 0.3121 | 200 | 8.66 |
| `lr_low_value_high` | 1,600 | 0.077890 ± 0.004137 | 0.029732 ± 0.001270 | 0.9086 | 0.3444 | 400 | 20.73 |
| `lr_low_value_high` | 3,200 | 0.081044 ± 0.000000 | 0.027314 ± 0.002516 | 0.8933 | 0.3537 | 800 | 28.60 |
| `lr_low_long_credit_value_high` | 200 | 0.075015 ± 0.004209 | 0.027780 ± 0.000434 | 0.8628 | 0.2509 | 50 | 2.79 |
| `lr_low_long_credit_value_high` | 400 | 0.075041 ± 0.004245 | 0.022806 ± 0.003401 | 0.7913 | 0.2604 | 100 | 5.06 |
| `lr_low_long_credit_value_high` | 800 | 0.072039 ± 0.000000 | 0.034726 ± 0.004991 | 0.9444 | 0.3020 | 200 | 8.65 |
| `lr_low_long_credit_value_high` | 1,600 | 0.077864 ± 0.004119 | 0.032309 ± 0.000960 | 0.9456 | 0.3461 | 400 | 16.87 |
| `lr_low_long_credit_value_high` | 3,200 | 0.078042 ± 0.004245 | 0.029190 ± 0.001778 | 0.9018 | 0.3542 | 800 | 34.98 |

The successor tests below use paired circuit/seed blocks. The reported gain is
the successor minus the current budget in circuit-normalized sampled HV. A row
is eligible only when its selection score is within 2% of the global curve
best and the bootstrap interval does not establish that the next doubling is
better. The final 3,200 point has no tested successor and therefore cannot be
an eligible stopping point in this campaign.

| Profile | Budget → successor | Selection score | Within 2% of global best? | Paired normalized gain (95% CI) | Successor established better? | Eligible? |
| --- | ---: | ---: | --- | ---: | --- | --- |
| `lr_low_value_high` | 200 → 400 | 0.8792 | No | -0.0031 [-0.0567, +0.0520] | No | No |
| `lr_low_value_high` | 400 → 800 | 0.8767 | No | +0.1065 [+0.0399, +0.1689] | Yes | No |
| `lr_low_value_high` | 800 → 1,600 | 0.9840 | Yes | -0.0749 [-0.1339, -0.0212] | No | **Yes** |
| `lr_low_value_high` | 1,600 → 3,200 | 0.9086 | No | -0.0189 [-0.1196, +0.0559] | No | No |
| `lr_low_long_credit_value_high` | 200 → 400 | 0.8628 | No | -0.0877 [-0.2058, +0.0224] | No | No |
| `lr_low_long_credit_value_high` | 400 → 800 | 0.7913 | No | +0.1241 [-0.0228, +0.3033] | No | No |
| `lr_low_long_credit_value_high` | 800 → 1,600 | 0.9444 | No | +0.0071 [-0.0921, +0.0801] | No | No |
| `lr_low_long_credit_value_high` | 1,600 → 3,200 | 0.9456 | No | -0.0457 [-0.1017, -0.0069] | No | No |

Only `lr_low_value_high` at 800 satisfies both gates. It has the curve's best
selection score, and increasing its budget to 1,600 significantly reduces
rather than improves normalized HV. The protocol therefore selects
`lr_low_value_high` at 800 trajectories, names
`lr_low_long_credit_value_high` as the runner-up at that budget, and names
1,600 as the winner's confirmation successor. The status is `selected`, not
`extend_required`, so no 6,400-trajectory DRiLLS experiment is warranted.

As in the original curve, mean product improvement rises with budget while
sampled HV peaks earlier. In particular, the winning profile's mean product
improvement grows from 0.3121 at 800 to 0.3537 at 3,200, but dalu HV falls from
0.033677 to 0.027314. This reinforces that scalar improvement cannot replace
the predeclared Pareto-front coverage metric.

## PPO

### Hyperparameter screen at 800 trajectories

The control used learning rate `1e-3`, gamma `0.9`, 20 PPO epochs, minibatch
size 64, clipping epsilon `0.2`, value-loss coefficient `0.5`, normalized
advantages, GAE lambda `0.95`, and no gradient clipping. PPO used rollout
length 80 throughout.

| Rank | Profile | Change from control | C1355 HV | dalu HV | Selection score | Mean product improvement | Mean task time (min) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `epochs_high` | 40 PPO epochs | 0.080205 ± 0.000601 | 0.040770 ± 0.001111 | 0.9929 | 0.4226 | 16.52 |
| 2 | `epochs_low` | 10 PPO epochs | 0.079874 ± 0.000864 | 0.040929 ± 0.000592 | 0.9928 | 0.3908 | 10.54 |
| 3 | `clip_low` | clipping epsilon `0.1` | 0.081044 ± 0.000000 | 0.039269 ± 0.003043 | 0.9797 | 0.4189 | 12.14 |
| 4 | `long_credit` | gamma `0.99` | 0.079925 ± 0.000236 | 0.038866 ± 0.002746 | 0.9679 | 0.4033 | 11.61 |
| 5 | `learning_rate_low` | learning rate `3e-4` | 0.079594 ± 0.000000 | 0.038366 ± 0.004318 | 0.9597 | 0.3888 | 11.65 |
| 6 | `control` | — | 0.080179 ± 0.000624 | 0.038227 ± 0.003373 | 0.9617 | 0.4169 | 12.12 |
| 7 | `clip_high` | clipping epsilon `0.3` | 0.079696 ± 0.000144 | 0.037206 ± 0.003263 | 0.9462 | 0.3959 | 13.73 |
| 8 | `learning_rate_high` | learning rate `3e-3` | 0.080408 ± 0.000605 | 0.025605 ± 0.018292 | 0.8089 | 0.3964 | 12.90 |

`epochs_high` and `epochs_low` advanced and are virtually tied in selection
score (difference 0.00009). Forty epochs gives higher product improvement but
requires four times as many optimizer updates and is about 57% slower per task
than ten epochs at the same trajectory count. The lower clipping epsilon is a
credible third profile, but its weaker minimum circuit-relative score loses
the tie break. A learning rate of `3e-3` is clearly unstable on dalu, as shown
by both its low mean and very large standard deviation.

### Trajectory-budget curve

| Profile | Trajectories | C1355 HV | dalu HV | Selection score | Mean product improvement | Optimizer updates | Mean task time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `epochs_high` | 200 | 0.080255 ± 0.000579 | 0.033504 ± 0.000493 | 0.9044 | 0.3755 | 4,000 | 4.81 |
| `epochs_high` | 400 | 0.081044 ± 0.000000 | 0.040096 ± 0.000247 | 0.9898 | 0.3880 | 8,000 | 8.63 |
| `epochs_high` | 800 | 0.080205 ± 0.000601 | 0.040770 ± 0.001111 | 0.9929 | 0.4226 | 16,000 | 16.52 |
| `epochs_high` | 1,600 | 0.080561 ± 0.000684 | 0.038477 ± 0.003072 | 0.9671 | 0.4049 | 32,000 | 31.78 |
| `epochs_high` | 3,200 | 0.080255 ± 0.000599 | 0.040311 ± 0.000940 | 0.9876 | 0.3928 | 64,000 | 61.59 |
| `epochs_low` | 200 | 0.074634 ± 0.004560 | 0.030801 ± 0.001403 | 0.8367 | 0.3599 | 1,000 | 2.99 |
| `epochs_low` | 400 | 0.080026 ± 0.000307 | 0.034448 ± 0.004863 | 0.9145 | 0.3708 | 2,000 | 5.22 |
| `epochs_low` | 800 | 0.079874 ± 0.000864 | 0.040929 ± 0.000592 | 0.9928 | 0.3908 | 4,000 | 10.54 |
| `epochs_low` | 1,600 | 0.079187 ± 0.000288 | 0.039679 ± 0.000164 | 0.9733 | 0.4165 | 8,000 | 18.04 |
| `epochs_low` | 3,200 | 0.079187 ± 0.000288 | 0.033948 ± 0.000723 | 0.9033 | 0.4146 | 16,000 | 43.54 |

The 40-epoch profile rises sharply by 400 trajectories and is already within
about 1% of the method's global best there. The formal rule nevertheless
selects 800 because it has the stronger eligible selection score. Its 1,600
successor is 2.60% lower and is not significantly better in the paired test.
The 10-epoch profile also peaks at 800 and then declines, sharply on dalu at
3,200. The protocol selects `epochs_high` at 800, with `epochs_low` as the
runner-up and 1,600 as the confirmation successor.

The near tie has a meaningful compute tradeoff. If confirmation shows no
robust HV advantage for 40 epochs, the 10-epoch runner-up would be attractive
for compute efficiency. The current protocol correctly carries both profiles
into confirmation rather than replacing the primary-metric winner early.
This was the result of the original two-profile curve; its confirmation choice
is superseded by the fresh three-profile clip-ambiguity curve below.

### Epoch-count and clipping interaction screen

The follow-up screen reran the control, both epoch changes, and `clip_low`, then
tested the two requested combinations. Learning rate `1e-3`, gamma `0.9`,
minibatch size 64, value-loss coefficient `0.5`, normalized advantages, GAE
lambda `0.95`, no gradient clipping, entropy beta `0.03`, and rollout length 80
remained fixed.

| Rank | Profile | PPO epochs | Clipping epsilon | C1355 HV | dalu HV | Selection score | Mean product improvement | Mean task time (min) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `clip_low` | 20 | 0.1 | 0.081044 ± 0.000000 | 0.040603 ± 0.000716 | 1.0000 | 0.4198 | 12.18 |
| 2 | `epochs_low` | 10 | 0.2 | 0.079619 ± 0.000036 | 0.040117 ± 0.000625 | 0.9852 | 0.3958 | 10.09 |
| 3 | `epochs_low_clip_low` | 10 | 0.1 | 0.079594 ± 0.000000 | 0.038547 ± 0.003189 | 0.9657 | 0.3892 | 10.42 |
| 4 | `epochs_high_clip_low` | 40 | 0.1 | 0.080688 ± 0.000504 | 0.037366 ± 0.003133 | 0.9579 | 0.4184 | 15.41 |
| 5 | `epochs_high` | 40 | 0.2 | 0.080662 ± 0.000540 | 0.036435 ± 0.003716 | 0.9463 | 0.4148 | 15.05 |
| 6 | `control` | 20 | 0.2 | 0.079187 ± 0.000288 | 0.035442 ± 0.003601 | 0.9250 | 0.4030 | 12.00 |

Neither combined profile enters the top two. Lower clipping by itself ranks
first, while the 10-epoch profile remains the best epoch-only change and is
about 5 minutes faster per task than the 40-epoch profiles.

| Paired contrast on circuit-normalized sampled HV | Mean effect | Bootstrap 95% CI |
| --- | ---: | ---: |
| Low epochs × low clip interaction | -0.09450 | [-0.20886, -0.00492] |
| High epochs × low clip interaction | -0.06340 | [-0.20454, +0.06422] |
| `epochs_low_clip_low` − `epochs_low` | -0.01949 | [-0.07612, +0.01488] |
| `epochs_low_clip_low` − `clip_low` | -0.03427 | [-0.09621, +0.00979] |
| `epochs_high_clip_low` − `epochs_high` | +0.01162 | [-0.06305, +0.09141] |
| `epochs_high_clip_low` − `clip_low` | -0.04206 | [-0.10493, +0.01508] |

The low-epochs × low-clip interaction is antagonistic and its interval excludes
zero: the two individually useful changes do not stack additively. The high-
epochs interaction is also estimated as negative, but remains uncertain. None
of the four direct combined-versus-parent comparisons excludes zero.

Under the predeclared interaction gate, no combined-profile budget curve was
warranted. The single-factor `clip_low` result was nevertheless strong enough
to justify the separate three-profile trajectory-budget curve reported below.

### Clip-ambiguity trajectory-budget curve

The final PPO campaign freshly compared `clip_low`, `epochs_low`, and
`epochs_high` at 200, 400, 800, 1,600, and 3,200 trajectories. Thus, all three
profiles share one project commit, protocol, circuit/seed matrix, and
50-sample evaluation contract; no earlier 800-trajectory artifacts were
reused.

| Profile | Trajectories | C1355 HV | dalu HV | Selection score | Mean product improvement | Optimizer updates | Mean task time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clip_low` | 200 | 0.080739 ± 0.000432 | 0.038304 ± 0.004067 | 0.9609 | 0.3732 | 2,000 | 3.83 |
| `clip_low` | 400 | 0.079747 ± 0.000216 | 0.040888 ± 0.006090 | 0.9860 | 0.3843 | 4,000 | 6.30 |
| `clip_low` | 800 | 0.080077 ± 0.000684 | 0.040923 ± 0.000845 | 0.9885 | 0.4205 | 8,000 | 12.46 |
| `clip_low` | 1,600 | 0.080357 ± 0.000971 | 0.039964 ± 0.000756 | 0.9786 | 0.4252 | 16,000 | 24.10 |
| `clip_low` | 3,200 | 0.080917 ± 0.000180 | 0.038672 ± 0.002688 | 0.9665 | 0.4158 | 32,000 | 48.84 |
| `epochs_low` | 200 | 0.080077 ± 0.000288 | 0.030676 ± 0.001251 | 0.8647 | 0.3657 | 1,000 | 3.26 |
| `epochs_low` | 400 | 0.080103 ± 0.000666 | 0.035011 ± 0.004451 | 0.9172 | 0.3730 | 2,000 | 5.49 |
| `epochs_low` | 800 | 0.079619 ± 0.000036 | 0.034816 ± 0.003490 | 0.9119 | 0.3858 | 4,000 | 11.55 |
| `epochs_low` | 1,600 | 0.080662 ± 0.000540 | 0.035970 ± 0.002640 | 0.9323 | 0.4112 | 8,000 | 18.89 |
| `epochs_low` | 3,200 | 0.079670 ± 0.000971 | 0.032642 ± 0.002867 | 0.8859 | 0.4133 | 16,000 | 47.83 |
| `epochs_high` | 200 | 0.081044 ± 0.000000 | 0.036553 ± 0.002210 | 0.9417 | 0.3955 | 4,000 | 4.34 |
| `epochs_high` | 400 | 0.079594 ± 0.000432 | 0.041381 ± 0.000378 | 0.9911 | 0.4084 | 8,000 | 8.07 |
| `epochs_high` | 800 | 0.079874 ± 0.000864 | 0.038693 ± 0.002681 | 0.9603 | 0.4169 | 16,000 | 16.17 |
| `epochs_high` | 1,600 | 0.081044 ± 0.000000 | 0.038081 ± 0.003184 | 0.9601 | 0.4134 | 32,000 | 31.16 |
| `epochs_high` | 3,200 | 0.080357 ± 0.000971 | 0.040638 ± 0.001006 | 0.9868 | 0.4178 | 64,000 | 61.67 |

The paired successor checks use the same eligibility rule as the other budget
curves. The gain is successor minus current budget in circuit-normalized
sampled HV.

| Profile | Budget → successor | Selection score | Within 2% of global best? | Paired normalized gain (95% CI) | Eligible? |
| --- | ---: | ---: | --- | ---: | --- |
| `clip_low` | 200 → 400 | 0.9609 | No | +0.0207 [-0.0647, +0.1106] | No |
| `clip_low` | 400 → 800 | 0.9860 | Yes | +0.0024 [-0.0627, +0.0688] | Yes |
| `clip_low` | 800 → 1,600 | 0.9885 | Yes | -0.0097 [-0.0335, +0.0129] | Yes |
| `clip_low` | 1,600 → 3,200 | 0.9786 | Yes | -0.0122 [-0.0498, +0.0104] | Yes |
| `epochs_low` | 200 → 400 | 0.8647 | No | +0.0529 [-0.0053, +0.1520] | No |
| `epochs_low` | 400 → 800 | 0.9172 | No | -0.0053 [-0.1250, +0.1043] | No |
| `epochs_low` | 800 → 1,600 | 0.9119 | No | +0.0210 [-0.0517, +0.1126] | No |
| `epochs_low` | 1,600 → 3,200 | 0.9323 | No | -0.0481 [-0.0886, -0.0082] | No |
| `epochs_high` | 200 → 400 | 0.9417 | No | +0.0488 [-0.0085, +0.1097] | No |
| `epochs_high` | 400 → 800 | 0.9911 | Yes | -0.0304 [-0.0861, +0.0060] | **Yes** |
| `epochs_high` | 800 → 1,600 | 0.9603 | No | -0.0002 [-0.0702, +0.0717] | No |
| `epochs_high` | 1,600 → 3,200 | 0.9601 | No | +0.0265 [-0.0085, +0.0756] | No |

The formal resolver selects `epochs_high` at 400 because it has the highest
selection score among eligible plateaus (`0.9911`). Its 800-trajectory
successor is 3.1% lower in score, and the paired interval provides no evidence
that the extra trajectories improve normalized HV. At the selected budget,
`clip_low` is the runner-up (`0.9860`) and `epochs_low` is far behind
(`0.9172`), so the low-epoch profile is retired from confirmation.

There is an important practical-tie nuance. `clip_low` at 800 appears first in
the generated broad ranking because its score is within the 0.02 tie band and
it has stronger minimum-circuit and product-improvement tie breaks. Budget
selection, however, is predeclared to choose the numerically highest eligible
selection score and to choose the runner-up at that same budget. Accordingly,
the confirmation set is `epochs_high` at 400, `clip_low` at 400, and the
selected profile's 800-trajectory successor. The curve status is `selected`,
not `extend_required`; PPO does not need a 6,400-trajectory test.

## Same-budget 800-trajectory cross-method snapshot

The table holds the training budget at 800 trajectories to permit a descriptive
sample-efficiency snapshot. PPO's selected deployment budget is 400, so its
800 row is shown only for this common-budget comparison. The planned common-
budget stage with ten paired seeds remains the confirmatory cross-method test.
The DRiLLS and PPO rows use their fresh final-curve reruns.

| Method/profile | C1355 HV | dalu HV | Mean product improvement | Mean task time (min) |
| --- | ---: | ---: | ---: | ---: |
| REINFORCE `policy_lr_high` | 0.081044 ± 0.000000 | 0.041971 ± 0.000758 | 0.3783 | 12.11 |
| DRiLLS-A2C `lr_low_value_high` | 0.080891 ± 0.000062 | 0.033677 ± 0.001282 | 0.3121 | 8.66 |
| PPO `epochs_high` | 0.079874 ± 0.000864 | 0.038693 ± 0.002681 | 0.4169 | 16.17 |

C1355 is nearly saturated by all three methods, so dalu provides most of the
separation. REINFORCE has the strongest 50-sample HV on both circuits (with a
negligible C1355 margin), PPO is close on dalu and has the strongest mean
product improvement, and DRiLLS is fastest per task but has materially lower
dalu HV. Equal trajectories establish sample efficiency, not wall-clock
speed: update counts and per-task runtimes differ substantially by algorithm.

## Final recommendations

- **REINFORCE:** proceed to ten-seed confirmation with `policy_lr_high` at 800
  trajectories, `control` at 800, and the selected profile at 1,600. Its
  original curve does not justify a 6,400-trajectory run.
- **DRiLLS-A2C:** proceed to ten-seed confirmation with
  `lr_low_value_high` at 800 trajectories,
  `lr_low_long_credit_value_high` at 800, and the selected profile at 1,600.
  The combined-profile curve returns `selected`, not `extend_required`, and
  therefore does not justify a 6,400-trajectory run.
- **PPO:** proceed to ten-seed confirmation with `epochs_high` at 400
  trajectories, `clip_low` at 400, and `epochs_high` at 800 as the successor.
  Retire `epochs_low` and both combined epoch/clip profiles. The three-profile
  curve returns `selected` and does not justify a 6,400-trajectory run.

Thus, no currently selected baseline profile needs 6,400 trajectories, and
neither PPO interaction profile warrants further budget testing. Both the
DRiLLS combined-profile question and PPO clipping ambiguity are resolved; the
next stage for all three methods is ten-seed confirmation.

## Final selected baseline hyperparameter configurations

These are the primary configurations selected by the completed three-seed
screening and budget protocols. They are the best current choices for baseline
training; the planned ten-seed stage should confirm robustness rather than
reopen hyperparameter search.

| Method | Selected profile | Training trajectories | Selected hyperparameters |
| --- | --- | ---: | --- |
| REINFORCE | `policy_lr_high` | 800 | policy LR `2e-3`; value LR `3e-3`; gamma `0.9`; entropy beta `0.01`; raw returns; no policy/value gradient clipping; one trajectory per episode |
| DRiLLS-A2C | `lr_low_value_high` | 800 | learning rate `3e-4`; gamma `0.9`; value-loss coefficient `1.0`; entropy beta `0.0003`; raw advantages; no gradient clipping; four trajectories per episode |
| PPO | `epochs_high` | 400 | learning rate `1e-3`; gamma `0.9`; 40 PPO epochs; clipping epsilon `0.2`; minibatch size `64`; value-loss coefficient `0.5`; normalized advantages; GAE lambda `0.95`; no gradient clipping; rollout length `80`; four trajectories per episode; entropy beta `0.03` |

All three use `zhu_resyn2` with baseline scale `1.0`, 20 synthesis steps, and
actions `[0, 1, 2, 3, 4, 5, 6]`. Final evaluation remains exactly 50 samples
with evaluation seed 42.

For confirmation, pair each primary configuration with the runner-up at the
same selected budget and the primary profile at its tested successor budget:

| Method | Primary | Same-budget runner-up | Successor check |
| --- | --- | --- | --- |
| REINFORCE | `policy_lr_high`, 800 | `control`, 800 | `policy_lr_high`, 1,600 |
| DRiLLS-A2C | `lr_low_value_high`, 800 | `lr_low_long_credit_value_high`, 800 | `lr_low_value_high`, 1,600 |
| PPO | `epochs_high`, 400 | `clip_low`, 400 | `epochs_high`, 800 |
