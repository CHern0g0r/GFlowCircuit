# Experiment 3: calibrated logZ initialization

This repository contains an isolated runner and report pipeline for Experiment
3 in `optimizer_experiment_protocol.md`. It does not change the production TB
trainer or the completed Experiment 2 artifact schema.

## One logical run

```bash
python -m src.experiments.tb_logz_calibration run \
  --variant zcal \
  --config-name tb_zhuDOP \
  --circuit bc0 \
  --seed 0 \
  --output-dir /path/to/runs/zcal/bc0/seed_0 \
  --max-trajectories 800 \
  --schedule-trajectories 800 \
  --milestones 200 400 800 \
  --device cuda
```

Use `--variant z0` for the paired control. A run first collects 64 cached
calibration trajectories at epsilon 0.5, initializes logZ, processes the cache
in 16 collection-order minibatches, and then collects the remaining 736
training trajectories. The calibration cache, fixed validation cache, metrics,
tables, checkpoints, resolved configuration, RNG metadata, and summary are
stored below the run directory.

Resume only from a checkpoint belonging to the same variant, circuit, seed,
source tree, and scientific configuration:

```bash
python -m src.experiments.tb_logz_calibration run \
  --variant zcal --circuit bc0 --seed 0 \
  --output-dir /path/to/runs/zcal/bc0/seed_0 \
  --max-trajectories 800 --milestones 200 400 800 --device cuda \
  --resume-checkpoint /path/to/runs/zcal/bc0/seed_0/checkpoints/trajectory_400.pt
```

## Aggregate report

After all 12 runs are complete:

```bash
python -m src.experiments.tb_logz_calibration report \
  --runs-root /path/to/runs \
  --output-dir /path/to/report \
  --expected-variants z0 zcal \
  --expected-circuits bc0 dalu \
  --expected-seeds 0 1 2
```

The report validates pairing and artifact integrity before applying the 200-
trajectory improvement gates, 800-trajectory non-harm gates, oscillation rule,
health gates, bootstrap comparison, and simplicity tie-break. Its authoritative
decision is `decision_summary.json`; CSV tables, plots, the Markdown report,
and `phase_ledger.csv` provide supporting evidence.

## Experiment results and conclusions

### Execution and artifact validity

The full Martin array was job `16681`, and the aggregate report was job
`17068`. All 12 requested training runs completed: both `z0` and `zcal` on
`bc0` and `dalu`, with seeds 0--2. Every run contained exactly 800 unique
training trajectories and 200 optimizer updates. Of the 800 trajectories, 64
were calibration trajectories used in the first 16 collection-order
minibatches, and 736 were newly sampled after calibration. No run reported a
numerical failure.

The report accepted the complete run matrix and all artifact-integrity checks.
In particular, each `z0`/`zcal` pair had the same pre-calibration policy
checksum, fixed-validation trajectory-content checksum, calibration
trajectory-content checksum, paired configuration fingerprint, and source-tree
checksum. Each serialized cache also matched the checksum recorded by its own
run. Therefore the differences below can be attributed to the assigned initial
`logZ` within this paired design, rather than to different initial policies,
calibration samples, validation samples, or source code.

The 64-trajectory calibration estimates were stable in scale across seeds:

| Circuit | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| `bc0` | 41.5644 | 41.6943 | 41.3575 |
| `dalu` | 39.3235 | 39.3638 | 39.0529 |

These values were assigned exactly as the initial `logZ` for `zcal`; `z0`
started at zero. The roughly 39--42 unit discrepancy shows that zero was very
far from the scale implied by the initial trajectories.

### The 200-trajectory benefit gate passed decisively

The protocol required `zcal` to reduce both the median absolute validation
target gap and median bias fraction to at most 50% of `z0` for every circuit
and validation stratum. All four circuit/stratum comparisons passed:

| Circuit | Validation stratum | Median absolute gap, `z0` | Median absolute gap, `zcal` | `zcal / z0` | Median bias fraction, `z0` | Median bias fraction, `zcal` | `zcal / z0` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bc0` | fixed uniform | 41.5036 | 0.1014 | 0.00244 | 0.99948 | 0.07982 | 0.07986 |
| `bc0` | fresh on-policy | 40.6015 | 0.0760 | 0.00187 | 0.99954 | 0.04353 | 0.04355 |
| `dalu` | fixed uniform | 39.2142 | 0.0693 | 0.00177 | 0.99941 | 0.03233 | 0.03235 |
| `dalu` | fresh on-policy | 38.3358 | 0.0335 | 0.00087 | 0.99944 | 0.00842 | 0.00843 |

Thus calibration reduced the median absolute gap by more than 99.7% in every
comparison. It also reduced the bias fraction by about 92.0%--99.2%. This is
far beyond the required 50% reduction and strongly supports the narrow claim
that the Experiment 2 failure was dominated by a badly initialized global
offset. The result is consistent on both fixed-policy and fresh on-policy
validation, so it is not an artifact of only one validation distribution.

### The 800-trajectory non-harm gate also passed

At 800 trajectories, the protocol allowed `zcal` centered RMS to be at most
110% of `z0` and mean training-archive hypervolume to be at least 90% of `z0`.
All six comparisons passed:

| Metric | Circuit | Validation stratum | `z0` | `zcal` | `zcal / z0` |
| --- | --- | --- | ---: | ---: | ---: |
| Centered RMS | `bc0` | fixed uniform | 1.8061 | 0.2122 | 0.1175 |
| Centered RMS | `bc0` | fresh on-policy | 1.9992 | 0.2051 | 0.1026 |
| Centered RMS | `dalu` | fixed uniform | 1.6436 | 0.4035 | 0.2455 |
| Centered RMS | `dalu` | fresh on-policy | 1.8797 | 0.3560 | 0.1894 |
| Archive hypervolume | `bc0` | training archive | 0.24400 | 0.24352 | 0.9981 |
| Archive hypervolume | `dalu` | training archive | 0.04048 | 0.04027 | 0.9947 |

Calibration therefore did more than remove a constant residual offset. The
centered RMS, which removes the global offset before measuring policy error,
was 75.5%--89.7% lower than for `z0`. In other words, placing `logZ` near the
correct scale also allowed the policy parameters to learn a substantially
better *relative* trajectory model within the same trajectory and update
budget.

Search quality was essentially preserved, but it did not improve. Mean archive
hypervolume was 0.19% lower on `bc0` and 0.53% lower on `dalu`, both comfortably
inside the 10% non-harm allowance. This supports “no material search harm,” not
“better search.” It also warns against selecting an optimizer setting from TB
residual quality alone.

### Paired uncertainty analysis

The report bootstrapped the three paired seed differences 10,000 times. A
difference is `zcal - z0`, so a negative centered-RMS difference favors
`zcal`, while a positive hypervolume difference favors `zcal`.

| Metric | Circuit | Stratum | Mean paired difference | 95% paired bootstrap interval | Wins (`zcal`/tie/`z0`) |
| --- | --- | --- | ---: | ---: | ---: |
| Centered RMS | `bc0` | fixed uniform | -1.3987 | [-1.6401, -0.9620] | 3/0/0 |
| Centered RMS | `bc0` | fresh on-policy | -1.6393 | [-2.2578, -0.8350] | 3/0/0 |
| Centered RMS | `dalu` | fixed uniform | -1.1637 | [-1.5013, -0.8561] | 3/0/0 |
| Centered RMS | `dalu` | fresh on-policy | -1.2316 | [-1.5258, -0.6454] | 3/0/0 |
| Archive hypervolume | `bc0` | training archive | -0.000473 | [-0.001033, -0.000182] | 0/0/3 |
| Archive hypervolume | `dalu` | training archive | -0.000215 | [-0.000542, 0.000000] | 1/0/2 |

Every centered-RMS interval excludes zero and every seed favors `zcal`, so the
policy-error improvement is consistent across the observed seeds. The `bc0`
hypervolume interval also excludes zero in the unfavorable direction, although
the absolute loss is only about 0.00047 and remains well within the predefined
non-harm bound. The `dalu` hypervolume interval reaches zero. Because only three
paired seeds were run, these intervals should be read as evidence for this
experiment matrix, not as a precise estimate of behavior over a broad circuit
population.

### Health gates improved, but neither configuration was fully healthy

There are 12 health endpoints per milestone: two circuits, three seeds, and two
validation strata. The shared thresholds are absolute target gap at most 0.5,
bias fraction at most 0.05, and standardized bias at most 0.25, in addition to
finite-value, probability, gradient-tail, clipping, and policy-collapse gates.

| Variant | 200 trajectories | 400 trajectories | 800 trajectories |
| --- | ---: | ---: | ---: |
| `z0` | 0/12 passed | 0/12 passed | 0/12 passed |
| `zcal` | 8/12 passed | 9/12 passed | 11/12 passed |

Every `z0` endpoint failed all three offset-related gates: target gap, bias
fraction, and standardized bias. By 800 trajectories its median learned `logZ`
was only about 1.97, while validation still implied absolute gaps of about
35.96--41.31 and bias fractions of about 0.997--0.998. With this learning rate
and budget, optimization from zero cannot traverse the required 39--42 unit
distance. The absence of oscillation in `z0` is therefore not evidence of good
behavior; it is a consequence of remaining far from the target on the same
side throughout training.

For `zcal`, no health endpoint failed the absolute target-gap gate. Its failures
were limited to bias fraction and standardized bias. At 800 trajectories, 11
of 12 endpoints passed; the sole failure was `dalu`, seed 2, fixed-uniform
validation, with gap -0.0824, bias fraction 0.0657, and standardized bias
0.2651. The last two values narrowly exceed their thresholds of 0.05 and 0.25.
Median learned `logZ` at 800 was approximately 41.54 on `bc0` and 39.27 on
`dalu`, close to the initial calibration estimates. This confirms that
calibration placed the parameter in the correct region, although it did not
make every seed and stratum healthy.

### Persistent oscillation is the decisive rejection reason

The protocol marks oscillation as persistent when the per-update target gap
changes sign more than four times over updates 1--200 *and* the mean absolute
gap over updates 151--200 is larger than over updates 101--150. Exact zeros are
ignored. `zcal` crossed zero 83--101 times in every run. Five of its six runs
also worsened in the final window and therefore met the persistent-oscillation
definition:

| Circuit | Seed | Sign changes | Mean absolute gap, updates 101--150 | Mean absolute gap, updates 151--200 | Persistent |
| --- | ---: | ---: | ---: | ---: | --- |
| `bc0` | 0 | 90 | 0.0804 | 0.0821 | yes |
| `bc0` | 1 | 83 | 0.0788 | 0.0835 | yes |
| `bc0` | 2 | 101 | 0.0552 | 0.0671 | yes |
| `dalu` | 0 | 98 | 0.1655 | 0.1488 | no |
| `dalu` | 1 | 96 | 0.1421 | 0.1467 | yes |
| `dalu` | 2 | 101 | 0.1051 | 0.1577 | yes |

The `dalu` seed-0 run is not persistent under the exact rule because its final
mean absolute gap decreased, despite 98 sign changes. This distinction matters:
the protocol rejects repeated crossing only when late error is also worsening.
Nevertheless, five persistent cases are enough to fail the global oscillation
gate. The oscillation magnitude is small compared with the 36--41 unit `z0`
error, but the rejection rule is deterministic and was fixed before examining
the results.

### Final decision

The authoritative report records:

- `improvement_200_pass = true`;
- `nonharm_800_pass = true`;
- `numerical_pass = true`;
- `oscillation_pass = false`;
- `both_variants_healthy_200_800 = false`;
- decision `reject_calibrated_initialization`;
- selected variant `z0`.

Report job `17068` consequently exited with status 3. In this pipeline, status
3 means a completed report with a scientific rejection; it is not an execution
or artifact failure.

The selection of `z0` must be interpreted as retaining the control after the
candidate violated a rejection rule. It does **not** mean that `z0` is healthy
or empirically better: `z0` failed every health endpoint, had much worse
centered RMS, and learned only a small fraction of the required global offset.
Conversely, `zcal` should not be adopted unchanged even though it was much more
accurate, because persistent late oscillation occurred in five of six runs and
one final health endpoint still failed.

The combined conclusion is that calibrated initialization is mechanistically
useful but insufficient as a complete optimizer setting. It removes the large
initial offset, accelerates calibration, and improves centered policy learning
without material hypervolume harm. The remaining problem is the dynamics of
updating an already well-centered `logZ`, not estimation of its initial scale.
This result motivates Experiment 4: test a separate `logZ` learning rate while
retaining the paired calibrated initialization, and require the new setting to
preserve the observed residual improvements while eliminating persistent
oscillation and passing every shared health gate. Experiment 3 alone does not
justify a production trainer change.

The canonical report artifacts are stored at
`/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-logz-calibration/tb-logz-calibration-report-v1/report`.
The most important files are `decision_summary.json`, `seed_metrics.csv`,
`improvement_200.csv`, `nonharm_800.csv`, `bootstrap.csv`, `oscillation.csv`,
`health_gates.csv`, and the diagnostic plots.

## Martin packaging

The corresponding `myhpc` project is
`gflowcircuit-gfn-logz-calibration` in the external scripts repository. Its
preflight, 12-task training array, and CPU report SLURM files are canonical.
Review, commit, and push both repositories before using `myhpc sync` or
`myhpc run`. Repository preparation does not submit a job.
