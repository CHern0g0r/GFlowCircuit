# Optimizer-health check for the Trajectory Balance GFlowNet

## Purpose of this phase

The reward and exploration experiments should not start until the GFlowNet is
shown to train adequately at the selected budget. Otherwise, a comparison of
reward temperatures or exploration schedules mostly measures how those choices
interact with an optimizer that is still far from satisfying trajectory
balance.

This phase is not intended to maximize benchmark hypervolume. Its purpose is to
answer four narrower questions:

1. Is the trajectory-balance objective implemented consistently?
2. Are `logZ` and the forward policy learning on compatible numerical scales?
3. Does training approach a stable solution before its budget is exhausted?
4. Does better TB optimization translate into a better calibrated sampler and
   better circuit-search performance?

The current results make undertraining a strong hypothesis, but a high TB loss
alone is not proof. A persistent loss can also come from an implementation
error, an invalid backward-policy assumption, insufficient model capacity, or
an objective that cannot be fit well by the current policy. The health check
must distinguish these cases.

## The problem in the current runs

For one sampled trajectory, the implemented TB residual is

\[
\delta(\tau)
= \log Z + \log P_F(\tau) - \log R(x) - \log P_B(\tau),
\]

and the loss is

\[
L_{TB}=\mathbb{E}_{\tau}[\delta(\tau)^2].
\]

The current implementation sets `log P_B = 0`, initializes `logZ = 0`, and
uses 200 optimizer updates. When `log_z_learning_rate` is unset, its learning
rate is `10 * learning_rate = 0.01`.

For intuition, suppose a 20-step trajectory has seven legal actions at every
step and the initial policy is approximately uniform. Then

\[
\log P_F(\tau) \approx -20\log 7 \approx -38.9.
\]

With `reward_alpha = 4`, typical `log R` values are only on the order of a few
units. A residual near zero therefore requires `logZ` to be on the order of
approximately 40, subject to the actual legal-action counts and rewards. The
exact target must be measured rather than assumed, but it is clearly much
larger than zero.

In the July runs, `logZ` advances almost linearly:

```text
episode       1      50      100      150      200
logZ       ~0.01   ~0.50    ~1.00    ~1.49    ~1.98
```

This pattern is nearly identical across circuits and seeds. It looks like an
Adam parameter moving at approximately its configured step size throughout the
entire run, rather than a parameter approaching its optimum.

The reported final TB losses are generally around 1,350--1,560. Their square
roots are approximately 37--40, so the RMS TB residual remains comparable to
the entire 20-step forward-log-probability scale. Examples include:

- `max`, run 0: loss decreases from about 1,550 to 1,428 while `logZ` reaches
  only 1.98;
- `adder`, run 0: loss decreases from about 1,522 to 1,384 while `logZ` reaches
  only 1.98;
- `c1355`, run 0: loss decreases from about 1,632 to 1,470 while `logZ` reaches
  only 1.98.

The loss decreases somewhat, but it is still extremely large and still driven
by a global scale mismatch at the final checkpoint. This is the main
undertraining signal.

When the residual is large and mostly negative, the policy gradient from the TB
loss is also dominated by that shared offset. It can encourage increasing the
probability of whichever trajectories happened to be sampled, while the much
smaller differences in terminal reward provide relatively weak discrimination.
The policy can still discover good trajectories, but such discoveries do not
show that it has learned the intended reward-proportional distribution.

There are additional warning signs:

- only four trajectories contribute to each noisy optimizer update;
- every collected trajectory is used for only one update;
- epsilon starts at 0.5 and decays for almost the entire 200-update run, leaving
  little low-epsilon consolidation time;
- evaluation quality is often noisy or plateaus while the TB residual remains
  large;
- the current logs contain only the squared loss and `logZ`, so they cannot show
  whether the remaining error is a global `logZ` offset or trajectory-dependent
  policy error.

The poor mean sample quality and occasional strong rare endpoint are compatible
with an undertrained sampler, but they are not sufficient evidence by
themselves. They can also be a legitimate consequence of a
reward-proportional sampler, finite evaluation budget, or trajectory
multiplicity. Likewise, `adder`'s flat evaluation return is
not useful evidence of undertraining because it may be a reachability problem.

## Metrics that must be added

The scalar TB loss should be decomposed. For every training update and for a
fixed validation batch, log the following.

### TB components and residual distribution

- `logZ` before and after the update;
- mean, standard deviation, minimum, maximum, and quantiles of `log P_F`;
- the same statistics for `log R` and `log P_B`;
- mean, standard deviation, RMS, and quantiles of `delta`;
- `mean(delta)^2` and `var(delta)` separately.

The decomposition

\[
\mathbb{E}[\delta^2]
= \mathbb{E}[\delta]^2 + \operatorname{Var}(\delta)
\]

is especially important:

- a large `mean(delta)^2` with relatively small variance indicates primarily a
  `logZ` calibration problem;
- a mean near zero with large variance indicates that `logZ` is approximately
  centered but the policy does not assign the correct relative probabilities
  to different trajectories;
- both terms remaining large indicates that neither component is healthy.

For the current batch, also compute the value that minimizes squared residual
with the policy held fixed:

\[
\log Z_{\text{batch target}}
= \mathbb{E}[\log R + \log P_B - \log P_F].
\]

Log the gap between the learned `logZ` and this batch target. Because the batch
is sampled and possibly off-policy, this is a diagnostic target rather than an
exact estimate of the global partition function. It nevertheless immediately
shows whether `logZ` is tens of units behind the scale required by the current
policy.

### Optimization metrics

- gradient norm for `logZ`;
- policy gradient norm, separately from `logZ`;
- parameter-update norm for the policy and the absolute `logZ` update;
- configured and effective learning rates;
- action-logit mean, standard deviation, and maximum magnitude;
- policy entropy and maximum action probability by trajectory step;
- NaN/Inf counts and gradient-clipping frequency if clipping is introduced.

These metrics distinguish slow optimization from exploding gradients, policy
collapse, dead updates, and a `logZ` parameter that simply needs a different
initialization or learning rate.

### Reward and sampling metrics

- raw terminal improvement before clipping;
- fraction clipped at the lower and upper bounds;
- mean, standard deviation, and quantiles of `log R`;
- mean and best final product improvement;
- archive hypervolume at a fixed evaluation sample count;
- number of distinct action sequences and distinct terminal `(size, depth)`
  pairs;
- action entropy and terminal-outcome entropy;
- best-of-`N` curves using fixed evaluation seeds.

With the current diff-of-product utility, the upper improvement clip of 2
should never activate for positive size and depth because improvement is at
most 1. The lower clip activates only for very poor trajectories. The clipping
rate should still be measured so reward saturation can be ruled out.

### Training and validation curves

Report every metric against all of:

- optimizer updates;
- complete training trajectories;
- environment transitions;
- wall-clock time.

Training residuals should be measured on the sampled behavior trajectories.
Validation residuals should additionally be measured on fixed trajectory sets
or fresh on-policy trajectories with exploration epsilon set to zero. This
prevents a changing exploration schedule from being mistaken for optimizer
progress.

## Checks to run before changing the optimizer

### 1. Tiny exactly enumerable environment

First test TB outside the full circuit problem using a small tree for which all
complete trajectories and rewards can be enumerated. Compute the true

\[
Z=\sum_x R(x)
\]

and the target terminal probabilities exactly. Training should recover:

- a small TB residual on every trajectory, not just on average;
- `logZ` close to the enumerated `log Z`;
- terminal probability ratios close to reward ratios;
- consistent behavior with and without epsilon-based off-policy sampling.

Failure here is an implementation or optimization bug, not a circuit-search
difficulty. In particular, verify that actions may be sampled from the
epsilon-mixture while `log P_F` in the loss is evaluated under the learned
forward policy. Off-policy TB can do this, but the distinction should be
covered by a test.

### 2. Inspect the initial numerical scales

Before the first update, sample a few hundred trajectories and inspect
`log P_F`, `log R`, the residual, and the batch-target `logZ`. Repeat this for
`bc0` and `dalu`:

- `bc0` is a relatively successful, easier circuit;
- `dalu` is exploration-sensitive and has large between-seed variability.

If the target is around 35--45 while initialization is zero, the current
initialization is inappropriate. If the target is unexpectedly small, inspect
legal-action masking, trajectory length, and the calculation of `log P_F`.

### 3. Separate global offset error from policy error

Recompute validation residuals after analytically replacing learned `logZ` with
the batch-target value, without changing the policy. If the loss falls by
orders of magnitude, `logZ` lag is the immediate bottleneck. If substantial
variance remains, the policy also needs more or better updates.

This diagnostic is cheaper and more informative than immediately launching a
large learning-rate sweep.

### 4. Verify the modeled object and backward probability

The current `log P_B = 0` assumption is correct if states are treated as unique
action prefixes with one deterministic parent. If the intended terminal object
is a unique circuit and several action sequences can produce that circuit, the
learned endpoint probability is affected by trajectory multiplicity.

This issue cannot be fixed by longer training. Decide whether the modeled
object is a synthesis program or a deduplicated circuit before interpreting a
low TB loss as correct endpoint sampling.

### 5. Check learning beyond the current stopping point

Run a diagnostic learning curve substantially beyond 200 updates on one seed
per diagnostic circuit. Save checkpoints frequently. This run is not part of
the fair baseline comparison; it determines whether the current endpoint is
merely an early point on a continuing learning curve.

If TB residual, calibration, or search quality continues improving well after
the contractual 800 trajectories, then 800 trajectories is not a converged
budget for the current implementation. The final comparison must either raise
the common environment budget for every method or improve GFlowNet's use of
each trajectory through replay and additional optimization.

## Fixes to test, in order

Only change one class of issue at a time. Otherwise, a lower loss will not be
attributable to a particular fix.

### 1. Initialize `logZ` near the observed scale

At initialization, collect a calibration batch and set

\[
\log Z_0
= \operatorname{mean}(\log R + \log P_B - \log P_F).
\]

This does not claim to know the exact partition function. It removes the large
initial mean residual so early policy updates respond to relative trajectory
errors rather than spending hundreds or thousands of steps correcting a
global offset. A simpler alternative is an estimate based on the sum of log
legal-action counts along random trajectories plus the average `log R`, but the
empirical residual-centering estimate is easier to validate.

Compare zero initialization and calibrated initialization with all other
settings fixed.

### 2. Tune the separate `logZ` learning rate

After fixing initialization, sweep a small range of `logZ` learning rates while
holding the policy learning rate fixed. Select using validation residual
decomposition, not benchmark hypervolume alone. A high rate is useful only if
`logZ` tracks its target without oscillation.

The current `0.01` rate moves `logZ` by roughly 0.01 per update throughout the
run. Starting from zero, reaching a scale near 40 at that speed would require
thousands of updates. Initialization should therefore be fixed before relying
on a larger rate.

### 3. Increase optimization per collected trajectory

The current code discards each trajectory after one update. Under a fixed
environment-interaction budget, consider:

- a replay buffer containing recent and high-reward trajectories;
- several optimizer updates per collection step;
- mixing on-policy, exploratory, and replayed trajectories;
- SubTB or another objective that extracts learning signals from
  subtrajectories.

Record extra optimizer steps and wall time. This preserves the 800-trajectory
interaction contract but no longer constitutes a compute-matched comparison,
so both interaction-matched and wall-clock results should be reported.

Simply increasing batch size is not automatically a fix. With 800 total
trajectories, a larger batch produces fewer optimizer updates unless data are
reused.

### 4. Tune the policy optimizer after `logZ` is healthy

Once mean residual is controlled, examine residual variance and policy
gradients. Then tune:

- policy learning rate;
- gradient clipping if gradients spike;
- number of replay updates;
- batch composition and replay age.

Do not tune these while `logZ` is tens of units behind its target, because the
dominant global residual obscures whether relative reward learning works.

### 5. Revisit exploration last

After optimizer health is established, compare no external epsilon, low
epsilon, and the current schedule. The learned GFlowNet policy is already
stochastic, so `epsilon = 0` does not mean deterministic training.

Check whether the current schedule spends too much of a short run near uniform
behavior and ends immediately after reaching low epsilon. If so, shorten the
decay or add a fixed low-epsilon consolidation period. Exploration should be
selected by both coverage and reward calibration, not TB loss alone.

## Minimal optimizer-health experiment

Use fixed diff-of-product reward, `reward_alpha = 4`, all seven actions, horizon
20, and the contract model. Do not vary reward or exploration simultaneously
with the first optimizer checks.

1. Instrument the current configuration and rerun `bc0` and `dalu` with two
   paired seeds.
2. Measure the initial batch-target `logZ` and decompose TB loss into squared
   mean residual and residual variance.
3. Compare current zero initialization with calibrated `logZ` initialization.
4. With calibrated initialization, compare a small number of separate `logZ`
   learning rates.
5. Extend the best diagnostic setup beyond 200 updates for one seed and inspect
   checkpoints against trajectories, updates, and wall time.
6. If policy error remains after `logZ` centering, test trajectory replay or
   multiple updates per collection batch.
7. Repeat the chosen setup for at least five seeds before freezing optimizer
   settings for the reward/exploration screen.

This sequence avoids a full optimizer grid: each step is conditional on the
diagnosis from the preceding metrics.

## Exit criteria

There is no universal TB-loss threshold for this circuit environment. The
optimizer-health phase passes when all of the following are true:

- the tiny enumerable test recovers the known distribution and partition
  function;
- learned `logZ` is no longer moving monotonically at its maximum effective
  step size at the final checkpoint;
- the mean held-out residual is close to zero relative to residual standard
  deviation;
- held-out residual RMS and variance have plateaued rather than continuing a
  clear downward trend at the budget boundary;
- policy gradients and logits are finite and do not show collapse;
- reward clipping is rare and understood;
- fixed-budget sampling quality and diversity are stable across checkpoints
  and several seeds;
- additional training produces negligible improvement at the intended
  resolution, or the common experimental budget has been revised accordingly.

A low training TB loss alone is not sufficient. It must be accompanied by a
healthy validation residual and by evidence that sampled terminal probabilities
respond correctly to reward differences. Conversely, circuit hypervolume can
improve before TB is well fit; that is useful search behavior, but it should not
be presented as evidence that the GFlowNet objective has converged.
