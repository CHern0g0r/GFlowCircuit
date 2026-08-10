# Experiment 5: trajectory-budget selection report

## Decision

Experiment 5 **rejects the finite-common-budget-within-cap hypothesis**. Neither
`bc0` nor `dalu` has an eligible trajectory budget, so there is no common
`B*`. In particular, 6,400 trajectories must not be selected as a fallback:
the protocol treats that checkpoint as confirmation-only because no
12,800-trajectory successor was run.

The rejection is scientific rather than operational. All ten training runs
completed successfully and the aggregate report returned its documented exit
code `3`, meaning a complete matrix with no selectable budget. The main result
is:

- training-archive hypervolume and best-of-`N` AUC are nearly saturated at the
  upper end of the tested range;
- `bc0` is close to an RMS plateau from 3,200 to 6,400 trajectories, but the
  five-seed bootstrap interval is too wide to establish it;
- `dalu` is not close to an RMS plateau and improves by roughly 18--21% over
  the same doubling;
- normalization health remains intermittent on both circuits, with all
  failures caused by the bias-fraction and/or standardized-bias gates.

The decision-complete output is therefore:

| Circuit | Smallest eligible budget | Result |
|:--|--:|:--|
| `bc0` | none | no candidate through 3,200 |
| `dalu` | none | no candidate through 3,200 |
| Common `B*` | none | reject finite budget within the 6,400 cap |

The prescribed next action is to extend the trajectory cap or test Experiment
6 and then repeat the full trajectory-budget curve. Experiment 6 should not be
treated as automatically justified, however: its protocol assumes healthy
`logZ`, while this experiment still has intermittent normalization-health
failures.

## Configuration and execution integrity

The run used the Experiment 4 descriptive control:

- initialization: `zcal`;
- `logZ` learning rate: `0.01`;
- policy learning rate: `0.001`;
- circuits: `bc0`, `dalu`;
- seeds: 0--4;
- milestones: 200, 400, 800, 1,600, 3,200, 6,400 trajectories;
- epsilon schedule: the Experiment 4 decay through trajectory 800, followed
  by a clamp at `0.01`;
- source commit: `2f1f2d369c9404d9f62c71a190f36fb391d3c444`.

This remains the least-bad Experiment 4 cross-circuit baseline, not a healthy
Experiment 4 winner.

| Stage | SLURM job | State | Exit | Observed runtime | Peak host RSS |
|:--|--:|:--|:--|--:|--:|
| preflight, `bc0` | `18246_0` | completed | `0:0` | 00:43:21 | 2,106,700 KiB |
| preflight, `dalu` | `18246_1` | completed | `0:0` | 00:16:57 | 2,186,952 KiB |
| training, `bc0`, five seeds | `18258_0`--`18258_4` | 5/5 completed | `0:0` | 03:31:03--03:33:55 per seed | at most 2,422,944 KiB |
| training, `dalu`, five seeds | `18258_5`--`18258_9` | 5/5 completed | `0:0` | 01:24:05--01:24:29 per seed | at most 2,280,788 KiB |
| aggregate decision | `18261` | scientific rejection | `3:0` | 00:00:44 | 2,775,796 KiB |

Every seed independently records:

| Integrity item | Observed value |
|:--|--:|
| complete run summary | yes, 10/10 |
| training trajectories | 6,400 |
| training presentations | 6,400 |
| optimizer updates | 1,600 |
| final archive trajectories | 6,400 |
| diagnostic validation rollouts | 1,324 |
| completed milestones | 200, 400, 800, 1,600, 3,200, 6,400 |
| numerical failure | none |

These checks also show that diagnostic evaluation did not enter the training
archive or trajectory budget.

## Milestone comparison

The health columns give the number of seeds passing every health gate in the
fixed-uniform (`F`) and fresh-on-policy (`U`) strata. RMS values are means over
five seeds. Archive hypervolume and search AUC are also five-seed means; they
are seed-level metrics and therefore do not differ by validation stratum.

| Circuit | Budget | F health | U health | F centered RMS | U centered RMS | Archive HV | Search AUC |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `bc0` | 200 | 1/5 | 2/5 | 0.320306 | 0.325280 | 0.234272 | 0.202994 |
| `bc0` | 400 | 4/5 | 5/5 | 0.229964 | 0.236139 | 0.238159 | 0.206239 |
| `bc0` | 800 | 4/5 | 5/5 | 0.242901 | 0.239962 | 0.243338 | 0.205527 |
| `bc0` | 1,600 | 3/5 | 4/5 | 0.207253 | 0.217977 | 0.245096 | 0.204074 |
| `bc0` | 3,200 | 4/5 | 5/5 | 0.133387 | 0.138317 | 0.245891 | 0.207379 |
| `bc0` | 6,400 | 4/5 | 5/5 | 0.124472 | 0.127574 | 0.246178 | 0.205935 |
| `dalu` | 200 | 4/5 | 2/5 | 0.458386 | 0.485714 | 0.032577 | 0.011348 |
| `dalu` | 400 | 4/5 | 4/5 | 0.389137 | 0.385683 | 0.036845 | 0.013161 |
| `dalu` | 800 | 4/5 | 5/5 | 0.423099 | 0.422586 | 0.038612 | 0.012695 |
| `dalu` | 1,600 | 3/5 | 4/5 | 0.336707 | 0.340073 | 0.040375 | 0.012965 |
| `dalu` | 3,200 | 5/5 | 5/5 | 0.284846 | 0.282364 | 0.043234 | 0.013254 |
| `dalu` | 6,400 | 5/5 | 3/5 | 0.236597 | 0.232627 | 0.043464 | 0.013291 |

The mean curves are not monotone: both circuits regress from 400 to 800, and
`bc0` health also deteriorates again at 1,600. This is consistent with the
oscillatory behavior already identified for the Experiment 4 control. The
large `bc0` RMS improvement between 1,600 and 3,200 is therefore not evidence
that a stable plateau was reached earlier.

## Paired budget-selection gates

For each seed and validation stratum, the RMS reduction is
`(RMS_B - RMS_2B) / abs(RMS_B)`. A negative value means that RMS worsened. The
RMS gate requires both the median to be below 5% and the deterministic paired
bootstrap 95% upper endpoint to be below 10%, separately in each stratum.
`F med/high` and `U med/high` below report those two percentages.

| Circuit | B -> 2B | Health | F med / high | U med / high | abs dHV | HV gate | rel dAUC | AUC gate | Eligible |
|:--|:--|:--:|--:|--:|--:|:--:|--:|:--:|:--:|
| `bc0` | 200 -> 400 | fail | 28.14% / 54.60% | 34.02% / 48.97% | 0.003886 | pass | +1.60% | pass | no |
| `bc0` | 400 -> 800 | fail | -3.63% / 47.35% | -6.47% / 43.70% | 0.005179 | **fail** | -0.35% | pass | no |
| `bc0` | 800 -> 1,600 | fail | 8.21% / 50.67% | -2.94% / 52.46% | 0.001759 | pass | -0.71% | pass | no |
| `bc0` | 1,600 -> 3,200 | fail | 37.06% / 57.61% | 38.67% / 55.41% | 0.000794 | pass | +1.62% | pass | no |
| `bc0` | 3,200 -> 6,400 | fail | 1.31% / 34.05% | 4.20% / 32.18% | 0.000288 | pass | -0.70% | pass | no |
| `dalu` | 200 -> 400 | fail | 17.97% / 25.68% | 22.63% / 37.97% | 0.004268 | pass | +15.97% | **fail** | no |
| `dalu` | 400 -> 800 | fail | -7.00% / 10.36% | -10.39% / 0.13% | 0.001767 | pass | -3.54% | pass | no |
| `dalu` | 800 -> 1,600 | fail | 24.06% / 39.81% | 23.28% / 38.01% | 0.001763 | pass | +2.13% | pass | no |
| `dalu` | 1,600 -> 3,200 | fail | 14.74% / 28.76% | 14.37% / 33.90% | 0.002859 | pass | +2.23% | pass | no |
| `dalu` | 3,200 -> 6,400 | fail | 17.55% / 23.64% | 20.52% / 25.77% | 0.000229 | pass | +0.28% | pass | no |

No row passes either the all-seed health requirement or the two-stratum RMS
criterion. The only isolated search-quality gate failures are the `bc0`
400-to-800 hypervolume change and the `dalu` 200-to-400 AUC gain, so search
quality is not the limiting factor at larger budgets.

The closest potential candidate is `bc0` at 3,200: both RMS medians are under
5%, and both search-quality gates pass. It still fails because health is not
unanimous at 3,200 and 6,400 and because the RMS upper confidence endpoints,
34.05% and 32.18%, greatly exceed 10%. With only five seeds, heterogeneous
per-seed changes prevent a plateau claim even though the medians are small.

`dalu` gives the opposite signal. Its final paired reductions are consistently
positive across all seeds: the medians are 17.55% and 20.52%, with lower
confidence endpoints of 11.66% and 10.03%. More training is still reducing
centered policy error at the cap.

## Endpoint health failures

At 6,400 trajectories all failures are normalization-bias failures. The common
limits are bias fraction at most `0.05` and standardized bias at most `0.25`.

| Circuit | Seed | Stratum | Bias fraction | Standardized bias | Failed gates |
|:--|--:|:--|--:|--:|:--|
| `bc0` | 4 | fixed-uniform | 0.086839 | 0.308378 | bias fraction, standardized bias |
| `dalu` | 1 | fresh-on-policy | 0.065401 | 0.264532 | bias fraction, standardized bias |
| `dalu` | 3 | fresh-on-policy | 0.050302 | 0.230143 | bias fraction |

The failure pattern is non-monotone. `dalu`, for example, is healthy in all ten
seed/stratum checks at 3,200 but loses two fresh-policy checks at 6,400. This
supports the Experiment 4 warning that `zcal` with rate `0.01` is a least-bad
oscillatory control rather than a stable normalization solution.

## Best-of-N search comparison

The table compares the mean nested search hypervolume at the first and last
training milestones. Each entry averages the same five seeds, with exactly 50
epsilon-zero search rollouts per seed and milestone.

| Circuit | Training budget | N=1 | N=2 | N=5 | N=10 | N=20 | N=50 | log2-N AUC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `bc0` | 200 | 0.174068 | 0.182692 | 0.200061 | 0.214828 | 0.215683 | 0.225498 | 0.202994 |
| `bc0` | 6,400 | 0.177225 | 0.182906 | 0.206658 | 0.218058 | 0.219100 | 0.225365 | 0.205935 |
| `dalu` | 200 | 0.000000 | 0.005602 | 0.005602 | 0.010303 | 0.019248 | 0.027825 | 0.011348 |
| `dalu` | 6,400 | 0.000000 | 0.004139 | 0.010978 | 0.010978 | 0.023295 | 0.029413 | 0.013291 |

Across the full 200-to-6,400 range, `bc0` archive hypervolume rises by 0.011906
(5.08%), but its N=50 fresh-search hypervolume is effectively unchanged
(-0.000133) and its AUC rises only 1.45%. `dalu` archive hypervolume rises by
0.010887 (33.42%), while N=50 hypervolume rises by 0.001588 (5.71%) and AUC by
17.12%. At the final doubling, however, both circuits' hypervolume and AUC
changes are already below the selection thresholds. This contrast indicates
that the accumulating archive continues to retain modest gains while a
50-rollout fresh search estimate is comparatively flat and noisy.

## Interpretation and recommendation

1. **Do not select 3,200 or 6,400 trajectories.** `bc0` at 3,200 is suggestive
   but not statistically confirmed, and `dalu` is still improving materially.
   The 6,400 checkpoint has no 12,800 confirmation and is ineligible by design.
2. **Do not interpret search saturation as optimizer convergence.** Archive HV
   and AUC pass their final-doubling thresholds, but centered RMS does not, and
   all-seed health is absent.
3. **Treat normalization stability as an unresolved confounder.** The endpoint
   failures are precisely the two bias gates associated with `logZ`, and they
   appear and disappear across milestones.
4. **For a direct budget answer, extend the cap to at least 12,800.** This gives
   6,400 a valid successor and tests whether `dalu`'s 18--21% RMS improvement
   persists. The same seeds, continuous-run semantics, epsilon clamp, and both
   validation strata must be retained.
5. **If Experiment 6 is chosen instead, resolve its health precondition
   explicitly.** A replay result cannot repair or waive failed `logZ` health
   gates. If replay passes its own screen, the entire Experiment 5 budget curve
   must be rerun and this experiment's candidate analysis becomes obsolete.

## Artifacts

Canonical Martin artifacts:

```text
/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-trajectory-budget-selection/tb-trajectory-budget-train-v1
/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-trajectory-budget-selection/tb-trajectory-budget-report-v1/report
```

The aggregate directory contains `decision_summary.json`,
`decision_report.md`, `seed_milestone_metrics.csv`, `health_gates.csv`,
`paired_budget_comparisons.csv`, `bootstrap.csv`, and `best_of_n.csv`. The
planned plot files were not present in the completed report artifact; all
numerical conclusions above were recomputed from the emitted CSV and JSON
tables.

The governing design and thresholds are documented in
[`experiment_5_trajectory_budget_selection.md`](experiment_5_trajectory_budget_selection.md)
and [`optimizer_experiment_protocol.md`](optimizer_experiment_protocol.md).
