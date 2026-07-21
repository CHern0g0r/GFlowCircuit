# Trajectory Balance GFlowNet optimizer experiment protocol

## Objective

Identify the smallest trajectory budget and simplest optimizer configuration
that make the Trajectory Balance (TB) GFlowNet numerically healthy and improve
circuit-search quality. Search quality is the primary objective; correct
reward-proportional sampling is a required health condition.

The modeled terminal object is a complete action sequence. Different sequences
that produce the same circuit remain distinct objects. Every prefix therefore
has one parent, so `P_B = 1` and `log P_B = 0`.

## Shared experimental setup

- Model: current contract encoder and policy head.
- Environment: seven actions `[0, 1, 2, 3, 4, 5, 6]`, horizon 20 unless an
  exact test specifies horizon 4.
- Reward: diff-of-product with `reward_alpha = 4`, `reward_eps = 1e-8`, and
  improvement clip 2.
- Development circuit: `bc0`.
- Stress-test circuit: `dalu`.
- Locked holdout circuit: `i10`.
- Initial policy learning rate: `0.001`.
- Initial batch size: four newly collected trajectories per optimizer episode.
- Primary budget: complete training trajectories. Calibration and exploratory
  trajectories count; replay presentations and evaluation trajectories do not.
- Checkpoint budgets: 200, 400, 800, 1,600, 3,200, and 6,400 trajectories.
- Pilot seeds: 0--2. Budget-selection seeds: 0--4. Locked final seeds: 0--9.

Use separate, recorded random streams for parameter initialization, action
sampling, replay, and evaluation. Pair variants by initial parameter checksum
and seed. At every milestone evaluate:

- `fixed_uniform`: 256 cached legal action sequences;
- `fresh_on_policy`: 128 new rollouts sampled with external epsilon zero;
- search quality: 50 new rollouts, used as nested prefixes for best-of-
  `N = 1, 2, 5, 10, 20, 50`.

The primary search metric is mean per-seed archive hypervolume with reference
`[1, 1]`. Secondary metrics are best-of-`N`, mean terminal product improvement,
best size/depth reduction, distinct sequences, distinct `(size, depth)` points,
and nondominated endpoints. Checkpoint samples must never enter the final
archive.

For residual

\[
\delta = \log Z + \log P_F - \log R - \log P_B,
\]

log `mean(delta)^2`, `var(delta)`, RMS, quantiles, the analytic target

\[
\log Z_{target}=\operatorname{mean}(\log R+\log P_B-\log P_F),
\]

and the analytically recentered residual. Also log policy and `logZ` gradient
norms, parameter-update norms, learning rates, logits, entropy, maximum action
probability, clipping, NaN/Inf counts, trajectories, transitions, optimizer
updates, trajectory presentations, evaluation trajectories, and wall time.

### Common full-horizon health gates

A configuration is healthy only if both validation distributions satisfy:

- zero NaN/Inf values and correctly normalized legal-action probabilities;
- `abs(logZ - logZ_target) <= 0.5` nats;
- `mean(delta)^2 / mean(delta^2) <= 0.05`;
- `abs(mean(delta)) / (std(delta) + 1e-8) <= 0.25`;
- finite policy gradients, with the 99th percentile at most 20 times the
  median;
- gradient clipping on fewer than 5% of updates, if clipping is enabled;
- no near-deterministic collapse (`max probability > 0.999` on more than 95%
  of validation states) unless the exact target is equally concentrated.

Pilot experiments use these thresholds as deterministic screening rules, not
as significance tests. Final comparisons use paired bootstrap confidence
intervals over seeds with 10,000 resamples.

## Experiment 1: exact TB correctness

### Hypothesis

The implemented TB residual, legal-action masking, off-policy action scoring,
and `logZ` optimization can recover a known reward-proportional distribution.

### Setup

1. Build a depth-four synthetic prefix tree with nonuniform branching and
   strictly positive terminal rewards spanning at least a factor of 100.
2. Enumerate all terminal sequences, exact `Z`, and exact terminal
   probabilities.
3. Train a tabular policy under on-policy sampling, epsilon-0.5 behavior
   sampling, and uniform off-policy sampling. Always score actions using the
   learned policy probability, not the behavior-mixture probability.
4. Enumerate the real `bc0` environment at horizon four (`7^4 = 2,401`
   sequences if all actions stay legal). Train first a tabular prefix policy and
   then the contract neural policy.
5. Evaluate exact metrics every 100 updates, for at most 20,000 updates.

### Evaluation and rejection rule

Require the gates at three consecutive evaluations:

- absolute `logZ` error at most 0.05 nats;
- RMS residual at most 0.05 and maximum absolute residual at most 0.15;
- terminal-distribution total variation at most 0.02;
- maximum terminal-probability error at most 0.01.

Reject the implementation-correctness hypothesis if any tabular test misses a
gate at 20,000 updates. Stop all circuit optimizer experiments in that case. If
tabular policies pass but the contract policy fails, retain implementation
correctness but reject the hypothesis that the current observation/model can
represent the target; measure prefix-observation collisions before increasing
capacity.

## Experiment 2: active-baseline diagnosis

### Hypothesis

Most of the current TB loss is a global `logZ` offset, and the current
800-trajectory endpoint is undertrained.

### Setup

Run the active configuration unchanged on `bc0` and `dalu`, seeds 0--2, through
800 trajectories, with checkpoints at 200, 400, and 800. Save the fully
resolved `log_z_learning_rate`; historical runs are not assumed to match it.

Before training, collect 256 diagnostic trajectories with a separate
validation RNG. They do not influence parameters. Measure the initial
`log P_F`, `log R`, residual, and `logZ_target`. At every checkpoint compare the
learned-`logZ` loss with the analytically recentered loss.

### Evaluation and rejection rule

Support the global-offset hypothesis only when analytic recentering removes at
least 50% of validation MSE on both circuits and the median bias fraction is
above 0.5. Reject it otherwise and investigate policy error, representation, or
implementation instead of prioritizing `logZ` initialization.

Support the undertraining hypothesis when, from 400 to 800 trajectories,
centered residual RMS decreases by at least 5%, mean archive hypervolume rises
by at least 0.005, or best-of-`N` area under the curve rises by at least 5%.
Reject the claim that 800 is demonstrably insufficient if none of those changes
reaches its threshold and all health gates pass; retain 800 as a candidate
budget.

## Experiment 3: calibrated `logZ` initialization

### Hypothesis

Initializing `logZ` from the observed initial scale removes the harmful early
global residual and reaches a healthy sampler in fewer trajectories.

### Setup

For every paired run, collect the same 64 calibration trajectories from the
initial behavior policy. Count them toward the training budget and use them for
optimization in 16 minibatches of four, recomputing `log P_F` under the current
policy.

Compare:

- `Z0`: `logZ = 0`;
- `Zcal`: `logZ = mean(log R + log P_B - log P_F)` on the calibration set.

Keep all other fields fixed. Run `bc0` and `dalu`, seeds 0--2, through 800
trajectories.

### Evaluation and rejection rule

Reject the hypothesis if `Zcal` fails to reduce the median absolute validation
target gap and bias fraction by at least 50% at 200 trajectories, or if it
worsens analytically centered RMS or mean hypervolume by more than 10% at 800.
Also reject any variant with numerical failure or persistent `logZ` oscillation.
If both are healthy and statistically indistinguishable, prefer `Z0` because it
is simpler.

## Experiment 4: separate `logZ` learning rate

### Hypothesis

After calibrated initialization, a suitable separate `logZ` learning rate
tracks the changing target without oscillation and improves final calibration.

### Setup

With the winning initialization, screen rates `0.003`, `0.01`, `0.03`, and
`0.1` on `bc0`, seeds 0--1, through 800 trajectories. Confirm at most the best
two on `dalu`, seeds 0--2. Keep the policy learning rate at `0.001`.

### Evaluation and rejection rule

Reject a rate when it violates a health gate, has more than four target-gap
sign changes in the final 200 updates while its final-50-update mean absolute
gap exceeds the preceding 50-update mean, or has more than 10% worse
fixed-validation bias than `0.01`.
Reject the hypothesis that a rate change is needed if `0.01` passes all gates
and lies within one standard error of the best candidate. Among healthy rates,
select by target gap and bias; use hypervolume only to break a tie.

## Experiment 5: trajectory-budget selection

### Hypothesis

There is a finite common trajectory budget `B*` at which the healthy GFlowNet's
optimization and search curves have effectively plateaued.

### Setup

Using the selected `logZ` settings, run one continuous job for each of `bc0`
and `dalu`, seeds 0--4, saving all shared checkpoint budgets. Each checkpoint
uses exactly 50 search samples and the shared validation protocol. Do not start
a separate training run for each budget.

For a checkpoint `B`, compare it with its already completed `2B` checkpoint.
Define the per-circuit candidate as the smallest `B` where:

- health gates pass at both `B` and `2B`;
- median centered-RMS reduction from `B` to `2B` is below 5%, with the upper
  paired-bootstrap bound below 10%;
- absolute mean-hypervolume gain is below 0.005;
- best-of-`N` area-under-curve gain is below 5%.

Set `B*` to the larger candidate from `bc0` and `dalu`.

### Evaluation and rejection rule

Reject the finite-budget-within-cap hypothesis if either circuit has no
eligible checkpoint with a completed `2B` confirmation. The largest completed
checkpoint can never be selected without its successor. In that case extend
the cap or run Experiment 6, then repeat the entire budget curve.

## Experiment 6: uniform trajectory replay

### Hypothesis

Reusing trajectories reduces centered policy error and reaches equal or better
search quality with fewer environment trajectories.

### Setup

Run this experiment only if `logZ` is healthy but policy error or search is
still improving at the budget cap. Compare:

- `replay-1x`: one update on each new batch of four;
- `replay-4x`: one update on the four new trajectories plus three updates on
  uniformly sampled minibatches of four from a FIFO buffer of size 1,024.

Reconstruct observations and recompute `log P_F` for replayed sequences. Never
reuse stored computation graphs. Screen on both circuits, seeds 0--2, then
expand a surviving candidate to seeds 0--4. Test `replay-8x` only if `replay-4x`
is still improving and fits the per-job time limit.

### Evaluation and rejection rule

Reject the replay hypothesis if, at matched trajectory checkpoints, replay does
not reduce centered validation RMS by at least 5%, violates a health gate, or
decreases mean archive hypervolume by more than 0.005. If replay passes, repeat
Experiment 5; the original `B*` is no longer valid. Always report the extra
updates, presentations, wall time, and GPU hours.

## Experiment 7: policy learning rate and gradient clipping

### Hypothesis

Once `logZ` is centered, remaining slow or unstable learning is caused by the
policy optimizer rather than insufficient data or representation.

### Setup

Run only when Experiments 1--6 rule out implementation, representation, and
`logZ` problems. Screen policy learning rates `0.0003`, `0.001`, and `0.003` on
`bc0`, seeds 0--1; confirm at most two on `dalu`, seeds 0--2. Introduce gradient
clipping only if the unclipped run contains nonfinite updates or genuine norm
spikes. Set the first threshold to the observed 99th-percentile norm of the best
stable run.

### Evaluation and rejection rule

Reject a candidate if it violates a health gate or lowers mean hypervolume by
more than 0.005 relative to `0.001` at matched budgets. Reject the
policy-optimizer hypothesis if no rate improves centered RMS by at least 5%
without crossing that hypervolume limit. Reject a clipping threshold if it
activates on 5% or more updates. Any accepted change requires a new Experiment
5 budget curve.

## Experiment 8: external epsilon schedule

### Hypothesis

After optimizer health is established, shortening or removing external epsilon
improves low-epsilon consolidation and circuit-search quality.

### Setup

Freeze the reward, model, optimizer, replay, and `B*`. Compare on `bc0` and
`dalu`, seeds 0--4:

- the current epsilon schedule;
- epsilon zero throughout training;
- epsilon 0.1 decayed to zero during the first half, followed by a zero-epsilon
  consolidation half.

TB always uses the learned-policy `log P_F`, even when behavior actions come
from an epsilon mixture.

### Evaluation and rejection rule

Reject the hypothesis if neither alternative raises mean per-seed archive
hypervolume by at least 0.005 without violating health gates or reducing
sequence and terminal diversity by more than 10%. When alternatives are within
one standard error, retain the current schedule.

## Experiment 9: locked holdout and baseline comparison

### Hypotheses

1. The selected optimizer and `B*` generalize to the untouched `i10` circuit
   and improve search over the active GFlowNet baseline.
2. The resulting GFlowNet is competitive with the project baselines under an
   equal complete-trajectory budget.

### Setup

Before examining `i10`, lock the complete configuration, `B*`, code commit,
exact-test report, and development-circuit gate table. Run the locked GFlowNet
and active GFlowNet baseline on `i10`, seeds 0--9. Do not retune on `i10`.

Then run REINFORCE, DRiLLS-A2C, PPO, and PCN with the same `B*`:

| Algorithm | Budget translation |
| --- | --- |
| GFlowNet-TB | `B* / 4` four-trajectory collection episodes |
| REINFORCE | `B*` one-trajectory episodes |
| DRiLLS-A2C | `B* / 4` four-trajectory episodes |
| PPO | exactly `20 * B*` environment transitions |
| PCN | exactly `B*` complete collected trajectories |

Use 50 final samples per seed, identical evaluation-seed construction, and the
shared hypervolume and best-of-`N` protocol.

### Evaluation and rejection rule

Reject hypothesis 1 if optimizer-health gates fail on `i10`, or if the paired
95% bootstrap interval for optimized-minus-active GFlowNet mean hypervolume
includes zero. If health transfers but hypervolume does not improve, conclude
that optimizer health succeeded but search is reward/exploration limited; do
not retune on `i10`.

Reject hypothesis 2 if GFlowNet's mean per-seed hypervolume is lower than the
best baseline with a paired 95% interval excluding zero. Regardless of ranking,
report all seed-level values, best-of-`N`, diversity, wall time, GPU hours, and
optimizer updates. A search improvement with failed TB gates may be reported as
heuristic performance, not as a converged GFlowNet.

## Martin execution and artifacts

Run one Martin job per circuit, seed, and configuration. First schedule a
100-trajectory preflight and extrapolate training plus evaluation time. Target
at most 10.5 hours of predicted work inside a 12-hour job. Start with one whole
GPU, 32 GB RAM, eight CPUs, the `michalis` account, and
`guaranteed-michalis`; revise resources from measured utilization.

Use `myhpc` for Git validation, immutable synchronization, and later
submission. Save milestone checkpoints with policy, optimizer, replay buffer,
all RNG states, counters, resolved configuration, Git commit, circuit hashes,
and append-only metrics. Submit in waves compatible with Martin's limits of
four running and 20 queued-plus-running jobs. If a logical run needs more than
12 hours, resume exactly from a milestone in a second job without repeating or
omitting trajectories.
