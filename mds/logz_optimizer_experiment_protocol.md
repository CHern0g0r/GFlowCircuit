# `logZ` optimizer stabilization experiment protocol

## Objective

Determine whether the apparent `logZ` oscillation in Experiment 4 is primarily
minibatch-target noise, true scalar-parameter instability, or instability of
the GFlowNet policy. If a training change is needed, select the simplest
mechanism that makes `logZ` stable and passes TB health gates on both `bc0` and
`dalu` without harming circuit-search quality.

The current control is:

- calibrated initialization (`zcal`);
- policy learning rate `0.001`;
- constant `logZ` learning rate `0.01`;
- four trajectories per policy update;
- 800 total training trajectories and 200 optimizer updates.

Experiment 4 established that this is the best descriptive constant-rate
baseline, not a validated winner. It reaches the correct endpoint scale but
has persistent training-batch target-gap oscillation and a seed-specific
`dalu` bias failure. This protocol does not assume that frequent minibatch gap
sign changes prove policy oscillation.

## Mechanistic motivation

For a training minibatch $B$, define the TB residual

$$
\delta_i = z + \log P_F(\tau_i) - \log R_i - \log P_B(\tau_i),
$$

and loss

$$
L_B(z,\theta)=\frac{1}{|B|}\sum_{i\in B}\delta_i^2.
$$

With the policy fixed, the batch-optimal scalar is

$$
\hat z_B = \frac{1}{|B|}\sum_{i\in B}
\left(\log R_i+\log P_B(\tau_i)-\log P_F(\tau_i)\right),
$$

and

$$
\frac{\partial L_B}{\partial z}=2(z-\hat z_B).
$$

The active configuration estimates this global scalar from only four
trajectories while the policy and its on-policy sampling distribution are also
changing. Consequently, the signed training-batch gap can cross zero because
`logZ` is unstable, because the target estimate is noisy, or both. Adam's
momentum and adaptive scaling can add lag when the target moves.

## Shared experimental setup

Use the Experiment 4 scientific configuration unless an arm explicitly
changes the `logZ` update:

- circuits: development circuit `bc0` and stress-test circuit `dalu`;
- initialization: `zcal` only;
- policy learning rate: `0.001`, constant;
- trajectories per policy update: four;
- trajectory horizon: 20;
- total training budget: 800 trajectories;
- milestones: 200, 400, and 800 trajectories;
- calibration: 64 cached trajectories followed by the same 16 ordered
  minibatches of four used in Experiment 4;
- subsequent training: 736 unique on-policy trajectories;
- `fixed_uniform`: the same 256 cached legal action sequences per circuit and
  seed, rescored under the current policy;
- `fresh_on_policy`: 128 new rollouts at each milestone with external epsilon
  zero;
- search evaluation: 50 separate rollouts at each milestone, with nested
  best-of-`N` prefixes for `N = 1, 2, 5, 10, 20, 50`;
- checkpoint samples never enter the training archive.

Calibration trajectories count toward the 800-trajectory scientific budget.
Validation and search trajectories do not. Policy updates and `logZ` updates
must have separate counters because some arms update `logZ` less often than the
policy.

### Pairing and determinism

For every circuit and seed, create one immutable initial-policy state,
calibration cache, fixed-uniform cache, minibatch order, and set of named RNG
streams. Clone these artifacts into every arm rather than regenerating them
inside each run. Require exact matches for:

- pre-calibration parameter checksum;
- calibration trajectory/action-sequence checksum;
- calibration score and analytic-target checksum;
- assigned post-calibration parameter checksum;
- fixed-uniform sequence checksum;
- initial training, evaluation, and search RNG states;
- scientific configuration and source-tree fingerprints.

Compute the shared calibration scores deterministically, preferably on CPU in
float64, and store the scalar target in the immutable calibration artifact.
This prevents the GPU roundoff difference that invalidated the Experiment 4
`dalu` aggregate. A report must reject a pairing mismatch; it must not silently
replace exact checks with approximate equality.

### Seeds and staging

Use staged execution to avoid promoting a mechanism based on one circuit:

1. Diagnostic stage: control only, `bc0` and `dalu`, seeds 0--2.
2. Mechanism screen: all five core arms on `bc0`, seeds 0--2.
3. Cross-circuit confirmation: at most the best two non-control mechanisms
   plus the control on `dalu`, seeds 0--4.
4. Optional locked confirmation: the selected mechanism and control on `i10`,
   seeds 0--4, only after all rules are frozen.

Screening uses deterministic health and stability gates. Confirmation reports
paired seed differences and paired bootstrap confidence intervals with 10,000
resamples. Do not use the optional `i10` results to revise the selection rule.

## Enhanced oscillation diagnostics

The existing training-batch target gap remains logged at every update, but it
is not the sole oscillation measure. Add the following without advancing any
training RNG or changing model state:

- actual scalar value $z_t$ after every update;
- signed and absolute scalar update, $z_t-z_{t-1}$ and
  $|z_t-z_{t-1}|$;
- detached batch target \(\hat z_B\) and batch gap \(z_t-\hat z_B\);
- 10- and 20-update moving averages of the batch gap;
- fixed-uniform target gap every five policy updates, always using the same
  cached 256 trajectories;
- fixed-uniform bias fraction, standardized bias, and centered residual RMS at
  the same probe frequency;
- policy parameter update norm and policy-gradient norm;
- mean statewise KL divergence between consecutive policies on the cached
  fixed-uniform states;
- entropy, maximum action probability, normalization error, illegal-action
  probability, and collapse fraction.

Store raw measurements and derive smoothing only in the report. The probe must
restore model mode and global RNG state and must verify that policy parameters
are unchanged by evaluation.

### Oscillation classifications

Report three distinct classifications:

1. **Batch-target noise:** apply the existing rule to the 200 raw
   training-batch gaps: more than four sign changes and a larger mean absolute
   gap over updates 151--200 than over 101--150.
2. **Scalar instability:** apply the same worsening-window rule to the
   fixed-uniform probe gap, with windows mapped to the probes in updates
   101--150 and 151--200. Also report the number of reversals in
   \(z_t-z_{t-1}\), total scalar variation
   \(\sum_t |z_t-z_{t-1}|\), and final-50 scalar-update RMS.
3. **Policy instability:** flag a run when fixed-uniform centered residual RMS,
   fixed-state policy KL, or policy update norm worsens in the final window
   while fresh-on-policy health or search quality also deteriorates. Report
   the individual conditions rather than collapsing them into an unexplained
   label.

If the raw batch gap oscillates but the fixed-uniform gap, actual scalar,
policy KL, and endpoint health are stable, classify the phenomenon as
estimator noise rather than harmful training oscillation.

## Core experimental arms

All arms start from the exact same `zcal` value and paired policy state. The
policy continues to use the ordinary TB loss and Adam at learning rate
`0.001`. Only the scalar update differs.

### Arm A: constant-rate control

- `logZ` optimizer: the Experiment 4 Adam parameter group;
- rate: constant `0.01`;
- update frequency: every policy update.

This arm measures whether the enhanced fixed-distribution diagnostic confirms
the apparent instability. Rerun it rather than relying exclusively on
Experiment 4 because the required high-frequency fixed probes were not saved
there.

### Arm B: calibrated and frozen `logZ`

After assigning the immutable calibration target, set `requires_grad=False`
for `logZ` or exclude it from all optimizers. Keep it fixed during the 16
calibration-policy minibatches and all later policy updates.

This is the simplest stabilization mechanism and a causal test: if TB health
and search remain good, continued scalar optimization was unnecessary. Reject
the arm if the fixed or fresh target gap drifts above the health threshold as
the policy changes.

### Arm C: decaying `logZ` learning rate

Use Adam for `logZ`, starting at `0.01` and decaying to `0.0003`. Keep the
rate at `0.01` for the 16 calibration minibatches. For updates 17--200, use a
cosine schedule:

$$
\eta_z(t)=0.0003+\frac{0.01-0.0003}{2}
\left[1+\cos\left(\pi\frac{t-17}{200-17}\right)\right].
$$

Record the resolved rate at every update and the scheduler state in every
checkpoint. The policy learning rate remains constant, so this arm isolates
late scalar step size.

### Arm D: per-batch EMA target update

Remove `logZ` from Adam. At every policy update, compute the detached
pre-policy-update target \(\hat z_B\) and apply

$$
z_{t+1}=(1-\alpha)z_t+\alpha\hat z_B,
\qquad \alpha=0.03.
$$

Compute the policy gradient using the current $z_t$, update policy
parameters, and then apply the scalar EMA from the detached target calculated
on that same batch. Save the EMA coefficient, target, and update order in the
scientific fingerprint. This avoids Adam momentum and gives the scalar update
a directly interpretable tracking gain.

### Arm E: accumulated 32-trajectory EMA target

Remove `logZ` from Adam and keep it fixed between scalar updates. Accumulate
the detached implied targets from eight consecutive four-trajectory policy
batches, then compute their 32-trajectory mean. Update every eighth policy
update using

$$
z_{k+1}=(1-\alpha_{32})z_k+\alpha_{32}\hat z_{32},
$$

where

$$
\alpha_{32}=1-(1-0.03)^8\approx0.2163.
$$

This matches the approximate eight-batch tracking gain of Arm D while reducing
target-estimation variance. Flush only complete groups of eight; do not make a
short final update. Save the accumulation buffer and scalar-update counter in
checkpoints so resume is exact.

## Experiment 1: determine whether the oscillation is real

### Hypotheses

- H1a: frequent Experiment 4 sign changes are primarily noise in the
  four-trajectory target estimator.
- H1b: the actual scalar or policy is unstable on a fixed validation
  distribution.

### Procedure

Run Arm A on both circuits, seeds 0--2, with enhanced probes. Compare raw batch
gap, smoothed batch gap, fixed-uniform gap, actual `logZ`, scalar updates,
policy KL, centered residual RMS, and archive hypervolume.

### Interpretation

Support H1a when all seeds have stable fixed-uniform probes and healthy
endpoints even though the raw batch gap frequently changes sign. In that case,
do not modify training solely to reduce raw sign changes; revise the
oscillation gate to use the fixed probe or a preregistered smoothed statistic.

Support H1b when fixed-probe error worsens, scalar-update amplitude remains
large late in training, or policy KL/residual measures worsen together with a
health or search metric. Continue to the mechanism screen in either case, but
label the problem correctly.

## Experiment 2: core stabilization screen

### Hypothesis

At least one of freezing, scalar-rate decay, direct EMA tracking, or
larger-sample EMA tracking reduces genuine scalar instability without harming
policy learning or search quality.

### Procedure

Run Arms A--E on `bc0`, seeds 0--2, through 800 trajectories. Pair every arm by
the immutable artifacts and RNG streams. Do not terminate an arm merely because
an early diagnostic checkpoint fails; only numerical failure or corrupt
pairing permits early termination.

### Screening gates

A mechanism is screen-eligible only when:

- every seed passes every common health gate on both validation strata at 800;
- no seed has persistent fixed-probe scalar instability;
- no seed has numerical failure, policy collapse, corrupt probability
  normalization, or non-finite optimizer state;
- mean fixed and fresh centered residual RMS are no more than 10% worse than
  the paired control;
- mean archive hypervolume and best-of-`N` AUC are each no more than 10% worse
  than the paired control;
- final-50 fixed-probe mean absolute gap is no more than the control value.

Rank eligible mechanisms by:

1. fewest failed seed/stratum health gates across all checkpoints;
2. lowest final-50 fixed-probe mean absolute gap;
3. lowest final fixed/fresh mean bias fraction;
4. lowest total scalar variation;
5. highest archive hypervolume;
6. simplicity order: freeze, decay, per-batch EMA, accumulated EMA.

Advance at most two non-control mechanisms. If no mechanism is eligible, do
not select an unhealthy fallback.

## Experiment 3: `dalu` confirmation

### Hypothesis

The selected stabilization mechanism generalizes from `bc0` to `dalu` and is
not exploiting the development circuit's target scale or trajectory
distribution.

### Procedure

Run the control and at most two selected mechanisms on `dalu`, seeds 0--4,
using the same 800-trajectory budget and enhanced diagnostics. Do not reuse the
old Experiment 4 control because it lacks the enhanced probes and one old
factorial pair failed the exact post-calibration checksum.

### Confirmation rule

A mechanism is confirmed only when every seed passes both endpoint validation
strata, no seed has persistent fixed-probe instability, and the paired results
show:

- lower mean final-50 fixed-probe absolute gap than control;
- lower or equal mean endpoint bias fraction than control;
- no more than 10% degradation in centered residual RMS, archive
  hypervolume, or best-of-`N` AUC;
- a paired bootstrap 95% interval for final-50 fixed-probe gap that does not
  support a practically meaningful worsening of more than 10%.

If several mechanisms confirm and their primary stability metrics are within
one paired standard error, choose the simpler one. A complete result with no
confirmed mechanism is a scientific rejection, not an execution failure.

## Experiment 4: optional locked-circuit validation

After freezing all decisions, compare the confirmed mechanism and control on
`i10`, seeds 0--4. Require the same health and non-harm gates. This stage tests
transport to a circuit not used for mechanism selection. Do not tune the
method after observing `i10`; a failure motivates a new protocol.

## Contingent experiments

Run these only if Arms B--E reveal a specific remaining failure. They are not
part of the initial screen.

### Separate scalar optimizer and momentum ablation

If decay helps but late reversals remain, compare the existing Adam scalar
group with plain SGD and Adam with reduced or zero first-moment momentum. Keep
the effective initial scalar gain matched as closely as possible. This tests
whether optimizer memory, rather than step size alone, causes lag and
overshoot.

### Calibration freeze followed by controlled unfreezing

If permanent freezing is initially stable but later target gap drifts, freeze
`logZ` through a preregistered warm-up, such as update 100, then unfreeze with
EMA coefficient `0.01` or a scalar rate of `0.0003`. Do not choose the
unfreezing point by inspecting each seed separately.

### Calibration-anchor regularization

If `logZ` departs substantially from a reliable calibrated value, add

$$
L_{anchor}=\lambda(t)(z-z_{cal})^2
$$

with a preregistered decaying \(\lambda(t)\). Treat this as a biased estimator
and reject it if fresh-on-policy gap grows while fixed-uniform metrics improve.

### Robust target aggregation

If target distributions contain rare extreme values, compare the ordinary
mean with a trimmed mean, median-of-means, or Huber target estimator. Continue
to train the policy with the ordinary TB objective; apply robustness only to
the detached scalar estimator. Report the estimator-induced bias and do not
present a robust estimator as exact TB optimization.

### Larger scalar batches

If the 32-trajectory arm remains noisy but otherwise healthy, test 64- and
128-trajectory accumulations while matching the effective EMA gain. Keep
policy update frequency and total unique trajectory budget unchanged.

## Common health and non-harm gates

For each seed, both fixed-uniform and fresh-on-policy validation must satisfy:

- finite metrics, parameters, gradients, and optimizer state;
- maximum legal-probability normalization error at most `1e-6`;
- maximum illegal-action probability at most `1e-6`;
- absolute `logZ` target gap at most `0.5`;
- bias fraction at most `0.05`;
- standardized bias at most `0.25`;
- policy-gradient 99th-percentile/median ratio at most `20`;
- clipping on fewer than 5% of updates if clipping is enabled;
- collapse fraction at most `0.95` using maximum action probability `0.999`.

Use the implementation's recorded numerical floor consistently in every arm
and report. The same report code must evaluate all methods.

Health is decisive only at 800 trajectories; 200 and 400 are diagnostic.
Longitudinal fixed-probe instability and numerical failures remain decisive.
Hypervolume never overrides failed TB health.

## Severity decision

The report must conclude with one of the following classifications:

- **Benign estimator noise:** raw batch gaps oscillate, but fixed-probe gaps,
  actual `logZ`, policy KL, endpoint health, and search quality are stable.
  Retain the control and change the diagnostic rather than the optimizer.
- **Scalar-only instability:** fixed-probe or actual-`logZ` measures oscillate,
  but policy shape and search remain stable. Prefer the simplest confirmed
  scalar stabilization method and continue monitoring.
- **Policy-coupled instability:** scalar instability coincides with worsening
  centered residuals, policy KL/update norms, fresh-on-policy health, or search
  quality. Treat this as severe and do not retain the current optimizer.
- **Unresolved:** results are seed- or circuit-dependent and no mechanism
  passes all gates. Keep `zcal`/`0.01` only as an experimental control and do
  not claim a production-ready configuration.

## Artifact and report requirements

Each run must save:

- resolved scientific configuration and source provenance;
- immutable pairing-artifact checksums;
- raw update metrics and enhanced fixed-probe metrics;
- separate policy and scalar optimizer/scheduler state;
- scalar accumulation buffer and counters when applicable;
- RNG states, trajectory counters, milestones, checkpoints, and resume
  fingerprint;
- fixed/fresh validation tables and search tables at every milestone.

The aggregate report must produce:

- a machine-readable decision JSON;
- per-seed and per-update CSVs;
- endpoint health and failed-gate tables;
- batch-noise, scalar-instability, and policy-instability tables;
- paired candidate ranking and non-harm tables;
- plots of raw and smoothed batch gap, fixed-probe gap, actual `logZ`, scalar
  updates, policy KL, centered residual RMS, and search quality;
- a phase ledger and Markdown interpretation;
- an explicit no-selection outcome when nothing is healthy.

Resume must reproduce uninterrupted execution exactly and reject changed
mechanism, schedule, EMA coefficient, accumulation state, initialization,
source tree, calibration artifact, fixed cache, RNG streams, or scientific
configuration.

## Recommended execution order

1. Implement the enhanced diagnostics and exact shared calibration artifact.
2. Run Experiment 1 to determine whether the original signal is estimator
   noise or genuine instability.
3. Run the five-arm `bc0` screen.
4. Generate the screen report and select at most two healthy mechanisms.
5. Run the `dalu` confirmation with control and selected mechanisms.
6. Select the simplest confirmed method or report no healthy solution.
7. Optionally run the locked `i10` validation without further tuning.

The highest-priority candidate is the 32-trajectory EMA because it directly
reduces target-estimation variance while avoiding Adam momentum. Permanent
freezing is the simplest candidate and should be preferred if it remains
healthy across circuits. Learning-rate decay is useful as a comparator, but
Experiment 4 indicates that reducing step size alone may not eliminate a
noisy moving target.
