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
rate is `10 * learning_rate = 0.01`. In this experiment the modeled terminal
object is explicitly an action sequence, so a state is its unique action
prefix and `log P_B = 0` is intentional. Different sequences that synthesize
the same circuit remain different terminal objects.

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

### 4. Verify action-prefix semantics and backward probability

The modeled terminal object for this experiment is an action sequence. States
must therefore be interpreted as unique action prefixes with one deterministic
parent, which makes `log P_B = 0` correct. Add a test that constructs two
different sequences producing the same circuit and verifies that they are
still treated as two terminal objects.

The policy observation does not encode the full prefix verbatim: it includes
the current graph, size/depth information, the step, and only a short recent
action history. Distinct prefixes can consequently be aliased by the policy.
This does not change the definition of `P_B`, but it can make the target
distribution unrepresentable by the current policy. The exact tests below must
distinguish an incorrect TB implementation from observation aliasing or
insufficient model capacity.

### 5. Check learning beyond the current stopping point

Run a diagnostic learning curve substantially beyond 200 updates on one seed
per diagnostic circuit. Save checkpoints frequently. This run is not part of
the fair baseline comparison; it determines whether the current endpoint is
merely an early point on a continuing learning curve.

If TB residual, calibration, or search quality continues improving well after
the currently configured 800 trajectories, then 800 trajectories is not a
converged budget for the current implementation. The 800-trajectory value is
provisional during this health check. The final comparison must select a new
common trajectory budget from learning curves, or improve GFlowNet's use of
each trajectory through replay and additional optimization, and then apply the
selected trajectory budget to every method.

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

Record extra optimizer steps and wall time. This preserves a fixed trajectory
interaction budget but no longer constitutes a compute-matched comparison, so
both interaction-matched results and the compute difference should be
reported.

Simply increasing batch size is not automatically a fix. At any fixed total
trajectory budget, a larger batch produces fewer optimizer updates unless data
are reused.

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

## Experimental decisions

The following decisions are fixed for this optimizer-health study.

- The GFlowNet models complete action sequences. Sequence prefixes have one
  parent, `P_B = 1`, and `log P_B = 0`. Sequence multiplicity is accepted:
  equivalent sequences leading to the same synthesized circuit are not merged.
- Circuit-search quality is the primary goal. Correct reward-proportional
  sampling is a required health check and the secondary goal.
- A configuration must pass the correctness and optimization gates before its
  search quality can be used for selection. Among healthy configurations,
  mean per-seed archive hypervolume is the primary selection metric.
- The common comparison resource is the number of complete training
  trajectories. A trajectory has 20 environment transitions. Optimizer steps,
  trajectory presentations through replay, evaluation trajectories, wall time,
  GPU hours, and peak memory are separate counters.
- The trajectory budget is not frozen at 800. It will be selected from the
  GFlowNet learning curve and then translated to REINFORCE, DRiLLS-A2C, PPO,
  and PCN. The selected value must replace, rather than silently contradict,
  the frozen value in a later version of `cfg/CONTRACT.yaml`.
- Extra optimizer updates and uniform replay are allowed under the fixed
  trajectory budget. This is interaction-matched but not compute-matched, so
  compute usage must be reported explicitly.
- Each Martin training job should be planned to finish within 12 hours. This is
  a per-job target, not a limit on the complete campaign.

Historical July runs are evidence for the undertraining hypothesis, not the
baseline dataset for model selection. They used the same algorithm but are
reported to have used a substantially lower `logZ` learning rate. Their exact
configuration is therefore uncertain relative to the active fallback of
`0.01`. The first circuit phase must rerun the active baseline with complete
instrumentation and save its resolved configuration.

## Assessment of the hypothesis and proposed fixes

The current hypothesis is strong for three independent reasons:

1. The observed `logZ` trajectory has the characteristic Adam step-size-limited
   shape expected from a scalar with a large, same-sign gradient.
2. The RMS residual is approximately the full negative log-probability of a
   20-step near-uniform trajectory.
3. The run ends after only 200 policy updates, using four new trajectories once
   per update, while the external epsilon schedule is still changing.

The code inspection also confirms that the loss has the intended sign, that
`logZ` is in a separate Adam parameter group, and that actions sampled from the
epsilon mixture are scored under the learned forward policy. Off-policy TB can
use this behavior distribution because its ideal residual is zero pointwise,
although the behavior distribution changes how approximation error is
weighted.

The proposed fixes nevertheless have important limits:

- Calibrated `logZ` initialization removes an initial mean residual; it is not
  an exact estimate of the global partition function. The batch mean of
  `log R - log P_F` is the least-squares centering value under that batch's
  sampling distribution, whereas the true `log Z` contains a log-sum.
- A larger `logZ` learning rate can shorten the transient but can oscillate or
  track batch noise. It must be judged by a fixed validation set, not just the
  training batch.
- Replay increases trajectory reuse and changes the weighting of residuals.
  Uniform replay is the first test; reward-prioritized replay is deferred
  because it can make approximation bias harder to interpret.
- More training cannot repair a non-Markov or aliased observation. If a tabular
  prefix policy succeeds and the contract encoder fails on an enumerable task,
  representation is the bottleneck.
- Epsilon changes data coverage, not the definition of the learned forward
  policy. It should be tuned only after mean residual and policy optimization
  are healthy.
- Search quality and TB calibration can disagree. Healthy calibration without
  better search passes the optimizer phase and sends the project to the reward
  and exploration phase. Better search with unhealthy calibration may be
  reported as heuristic search behavior, but not as a converged GFlowNet.

## Diagnostic hypotheses and decisions

| Hypothesis | Decisive observation | Next action |
| --- | --- | --- |
| TB implementation is wrong | A tabular policy fails on the synthetic enumerated tree | Stop circuit experiments and fix the implementation |
| `logZ` lag dominates | Analytic recentering removes most validation MSE | Calibrate initialization, then tune only the `logZ` rate |
| Policy is underoptimized | Mean residual is healthy, but centered residual RMS keeps falling with more updates | Add uniform replay or extra updates per collection |
| Policy observation/capacity is inadequate | Tabular prefix policy passes, contract encoder plateaus with large centered error on an enumerable circuit tree | Inspect observation aliasing and encoder/head capacity |
| External epsilon is harmful | Optimizer gates pass, but low-epsilon consolidation improves held-out search and calibration | Shorten or remove the epsilon schedule |
| Reward, rather than optimizer, limits search | TB gates and probability calibration pass, but hypervolume does not improve | End optimizer tuning and begin the reward/exploration study |

## Accounting rules

Let `B` be the common number of complete 20-action training trajectories per
circuit and seed. The following rules prevent hidden budget changes.

- Every trajectory that influences parameters or `logZ` counts toward `B`,
  including exploratory and initialization-calibration trajectories.
- Calibration trajectories are retained and used for optimization; they are
  not consumed only to estimate `logZ` and then discarded.
- Replaying an existing trajectory does not increment `B`. It increments
  `trajectory_presentations` and normally `optimizer_updates`.
- Checkpoint validation, fixed health-validation sets, and final search samples
  do not count toward `B`. They must be counted separately as evaluation
  trajectories and environment transitions.
- A stopped or truncated trajectory still counts if it caused environment
  transitions; no such truncation is expected at the fixed horizon.
- Report `B`, environment transitions, optimizer updates, trajectory
  presentations, evaluation trajectories, wall time, GPU hours, and peak CPU,
  RAM, and VRAM for every run.
- Choose `B` to be divisible by four so translations do not require fractional
  batches or iterations.

For the final comparison, translate `B` as follows while retaining each
algorithm's established update semantics.

| Algorithm | Translation of the common trajectory budget |
| --- | --- |
| GFlowNet-TB | `episodes = B / 4`, with four new trajectories per collection episode |
| REINFORCE | `episodes = B`, with one complete trajectory per episode |
| DRiLLS-A2C | `episodes = B / 4`, with four complete trajectories per episode |
| PPO | collect exactly `20 * B` transitions; with 80 rollout transitions per iteration this is `B / 4` iterations |
| PCN | stop after exactly `B` collected complete trajectories |

This is an interaction-matched comparison. Do not force equal optimizer steps
or equal wall time. Instead, present a second compute table so readers can see
the cost of GFlowNet replay or extra updates. Checkpoint and final evaluation
budgets must be identical across algorithms.

## Circuits, seeds, and separation of roles

Use only three real circuits during this study.

- `bc0` is the development circuit. It is relatively easy and is also used for
  the short-horizon exact circuit integration test.
- `dalu` is the stress circuit. It is exploration-sensitive and exposes
  between-seed instability. It confirms that a setting chosen on `bc0` is not
  healthy only on the easy case.
- `i10` is the locked holdout circuit aligned with the current comparison
  manifest. It is not used to tune initialization, learning rates, replay, or
  the selected budget.

Use paired seeds and separate random streams for model initialization, circuit
selection, action sampling, replay sampling, and evaluation. Store all five
resolved seeds. A practical successive-fidelity schedule is:

- deterministic seeds for unit and exact tests;
- seeds 0--2 for intervention screens on `bc0` and `dalu`;
- seeds 0--4 for the budget curve and any surviving replay intervention;
- seeds 0--9 only after the configuration and `B` are locked, for the final
  `i10` GFlowNet and baseline comparison.

Using the same integer seed across variants is necessary but not sufficient:
record the initial parameter checksum and the first calibration action
sequences to verify pairing.

## Instrumentation and artifacts to implement first

### Per-update metrics

At every update, write machine-readable metrics in addition to TensorBoard:

- all TB component and residual statistics listed earlier;
- `tb/residual_bias_sq`, `tb/residual_variance`, and their fractions of MSE;
- `tb/logz_batch_target` and `tb/logz_target_gap`;
- `optim/logz_grad`, `optim/logz_update`, policy gradient and update norms;
- learning rates, gradient clipping threshold and occurrence;
- logit magnitude, entropy, maximum probability, and legal-action count by
  trajectory step;
- training trajectories, transitions, trajectory presentations, optimizer
  updates, evaluation trajectories, and elapsed seconds.

Compute gradient norms after `backward()` and before clipping or
`optimizer.step()`. Compute parameter-update norms from parameter snapshots
immediately before and after the step. Log `logZ` on both sides of the update;
the current logger records only the value after it.

### Validation strata

Residuals must be evaluated on three distributions whose names appear in every
artifact.

1. `train_behavior`: the trajectories used for the current update.
2. `fixed_uniform`: 256 legal action sequences per circuit, produced once from
   a fixed seed and cached with their per-step observations and terminal
   rewards. Recompute their learned `log P_F` at every milestone.
3. `fresh_on_policy`: 128 new trajectories at every milestone, sampled from
   the learned policy with external epsilon equal to zero.

The fixed set supports checkpoint-to-checkpoint comparison. The fresh set
measures the part of the tree on which the current policy places mass. For each
stratum report both learned-`logZ` residuals and analytically recentered
residuals. Never optimize on either validation set.

### Checkpoints and reproducibility

The current checkpoint contains the policy but is not sufficient to resume an
adaptive experiment. Each milestone checkpoint must contain:

- policy and `logZ` parameters;
- optimizer state and learning-rate scheduler state;
- replay-buffer contents and insertion order, if enabled;
- Python, NumPy, Torch, action-sampling, and replay RNG states;
- completed-trajectory, transition, presentation, update, and evaluation
  counters;
- the fully resolved Hydra configuration, Git commit, environment versions,
  circuit hashes, and initial parameter checksum.

Write one append-only metrics file per run and a summary JSON. The analysis
script must regenerate every table and plot from these files without parsing
console output.

### Required tests

Add tests for:

- the residual decomposition identity and analytic batch target;
- correct exclusion of `logZ` from the policy parameter group;
- learned-policy scoring under epsilon-mixture action sampling;
- counting calibration and exploratory trajectories exactly once;
- recomputing, rather than reusing, stored `log P_F` during replay;
- fixed-validation reproducibility across checkpoints;
- complete checkpoint/resume equivalence for a short deterministic run;
- two equivalent circuits reached by different action sequences remaining two
  distinct terminal sequence objects with `log P_B = 0`.

## Phase 1: exact correctness tests

### 1A. Synthetic prefix tree

Create a depth-four tree with nonuniform branching and strictly positive
terminal rewards spanning at least a factor of 100. Enumerate every complete
sequence, its reward, the exact `Z`, and its target probability. Train a
tabular forward policy and scalar `logZ` under:

- on-policy sampling with no external epsilon;
- behavior sampling with epsilon 0.5 while scoring `log P_F` under the learned
  policy;
- uniform off-policy sampling.

All three cases must pass independently. This test isolates TB algebra,
masking, off-policy scoring, and `logZ` optimization from circuit synthesis and
neural representation. Evaluate exact metrics every 100 updates and require
the gates at three consecutive evaluations. Cap this test at 20,000 updates;
failure at the cap is a failed test, not permission to weaken the thresholds.

### 1B. Enumerable circuit tree

Run `bc0` at horizon four. With seven legal actions at every step there are
only `7^4 = 2,401` action sequences; if legal actions vary, enumerate the
actual legal prefix tree instead. Cache every sequence, reward, and observation
and compute exact `Z` and exact target probabilities.

First train a tabular prefix policy. Then train the contract encoder and head.
Compute learned terminal probabilities directly by enumerating the policy tree,
not by Monte Carlo sampling. If tabular succeeds and the contract model fails,
measure observation collisions between different prefixes before adding model
capacity. Use the same 100-update evaluation interval, three-evaluation
persistence rule, and 20,000-update cap as the synthetic test.

### Exact-test gates

Both the synthetic and tabular circuit tests must satisfy all of:

- absolute `logZ` error at most 0.05 nats;
- RMS TB residual at most 0.05 and maximum absolute residual at most 0.15;
- total-variation distance between exact and learned terminal distributions at
  most 0.02;
- maximum absolute terminal-probability error at most 0.01;
- zero NaN or Inf values and normalized legal-action probabilities within
  `1e-6`.

The contract encoder on the enumerable circuit must meet the same `logZ` and
total-variation gates before it is considered able to represent the intended
distribution. If it misses them while the tabular policy passes, do not tune
full-horizon optimizer budgets yet.

## Phase 2: instrumented active-baseline reproduction

Run the unmodified active optimizer settings on `bc0` and `dalu`, seeds 0--2,
for at least the current 800 trajectories. Resolve and store the actual
`log_z_learning_rate`; do not infer it from historical logs. Save milestones at
200, 400, and 800 trajectories.

Before the first update, collect a diagnostic batch of 256 trajectories only
for scale inspection. This particular batch is validation-only and must not
affect parameters, so it is counted as evaluation rather than training. Record
the distributions of `log P_F`, `log R`, residual, and batch-target `logZ`.
Generate it with the dedicated validation RNG; collecting it must not advance
or reset any training RNG state.

The reproduction answers three questions before any intervention:

1. What fraction of TB MSE is squared mean residual?
2. How far is learned `logZ` from the fixed and fresh-on-policy targets?
3. Does analytic recentering expose a policy-error variance that is already
   learning, flat, or diverging?

If squared bias is not the dominant component, skip directly to the relevant
policy, representation, or implementation branch rather than assuming that
initialization is the fix.

## Phase 3: isolate `logZ` initialization and learning rate

### Matched calibration protocol

Use a 64-trajectory calibration prefix for every initialization ablation. Draw
it from the initial training behavior, count all 64 trajectories toward `B`,
and retain it for optimization. Use the same sequences and minibatch order for
the paired zero- and calibrated-initialization variants.

The two causal controls are:

- `Z0`: initialize `logZ = 0`;
- `Zcal`: initialize
  `logZ = mean(log R + log P_B - log P_F)` on the 64 trajectories.

After initialization, consume the calibration trajectories exactly once in 16
minibatches of four before collecting more trajectories. This protocol avoids
giving `Zcal` free environment interactions and prevents the calibration data
from being discarded. Reconstruct their observations and recompute `log P_F`
under the current policy when each minibatch is used. Retain the original
active run from Phase 2 as a compatibility reference; compare `Z0` with `Zcal`
for the causal conclusion.

Run `Z0` and `Zcal` with policy learning rate `0.001`, the active `logZ` rate,
the current reward, and the current epsilon schedule on `bc0` and `dalu`, seeds
0--2, through 800 trajectories.

### Conditional `logZ`-rate screen

With calibrated initialization, screen `log_z_learning_rate` values
`0.003`, `0.01`, `0.03`, and `0.1`. Run all four only on `bc0`, seeds 0--1,
through 800 trajectories. Eliminate any rate with nonfinite values,
oscillation across the target, or worse fixed-validation bias than `0.01`.
Confirm at most the best two rates on `dalu`, seeds 0--2.

Choose the rate using residual bias and target tracking. Search hypervolume
breaks ties only after the health gates pass. If calibrated initialization with
`0.01` already passes and the other rates have no meaningful advantage, keep
`0.01` rather than adding a new hyperparameter.

## Phase 4: select the trajectory budget

Run one continuous job per circuit and seed with the best healthy `logZ`
setting. Save milestones at

```text
B = 200, 400, 800, 1,600, 3,200, 6,400 trajectories.
```

Use `bc0` and `dalu`, seeds 0--4. A milestone is a checkpoint, not a separate
training run. This avoids retraining the prefixes and makes budget comparisons
paired. If preflight timing predicts that 6,400 trajectories plus milestone
evaluation will exceed 10.5 hours, reduce the maximum before submission and
record the timing calculation. The remaining 1.5 hours is a safety margin for
checkpoint evaluation and artifact writes.

For each milestone, evaluate the fixed and fresh validation strata and draw 50
search samples using a fixed evaluation seed. Construct nested best-of-`N`
curves from the first `N` samples for `N = 1, 2, 5, 10, 20, 50` so larger
budgets do not receive more favorable evaluation randomness.

For each circuit, define its candidate budget as the smallest `B` for which:

- all full-horizon optimizer-health gates below pass at `B` and `2B`;
- the median reduction in analytically centered validation RMS from `B` to
  `2B` is below 5%, with the upper 95% paired-bootstrap bound below 10%;
- the absolute increase in mean per-seed archive hypervolume from `B` to `2B`
  is below 0.005, and best-of-`N` area under the curve improves by less than 5%;
- no seed shows a late nonfinite value, policy collapse, or renewed monotonic
  `logZ` drift.

The common `B*` is the larger of the two circuit candidate budgets. Because the
rule needs the next checkpoint, a budget cannot be selected at the largest
completed checkpoint. If either circuit has not plateaued by the cap, do not
declare the cap adequate; proceed to Phase 5 or extend the job limit.

## Full-horizon optimizer-health gates

Apply these gates to both `fixed_uniform` and `fresh_on_policy` at the selected
budget and at the next checkpoint.

### Correctness and numerical health

- zero NaN or Inf values in rewards, logits, probabilities, residuals,
  gradients, parameters, optimizer state, and metrics;
- legal-action probabilities sum to one within `1e-6`;
- no illegal action receives positive probability;
- upper reward-clipping rate is exactly zero for diff-of-product reward;
- lower reward-clipping above 5% triggers a reward-range investigation and
  blocks optimizer selection until understood;
- if gradient clipping is enabled, fewer than 5% of updates clip; otherwise
  the learning rate or representation must be investigated.

### `logZ` centering

Define

\[
q_{bias} = \frac{\operatorname{mean}(\delta)^2}
{\mathbb{E}[\delta^2] + 10^{-12}},
\qquad
s_{bias} = \frac{|\operatorname{mean}(\delta)|}
{\operatorname{std}(\delta) + 10^{-8}}.
\]

Require all of:

- absolute learned-`logZ` target gap at most 0.5 nats;
- `q_bias <= 0.05`;
- `s_bias <= 0.25`;
- the target gap does not keep the same sign while changing by approximately
  one full configured Adam step throughout the final 10% of updates.

These are scale-free bias gates plus an interpretable absolute tolerance. If
the residual standard deviation is almost zero, the absolute gap is the
decisive gate.

### Policy error and behavior

- analytically centered RMS is stable according to the plateau rule in
  Phase 4;
- policy gradient and parameter-update norms are finite in every update and
  their 99th percentiles are no more than 20 times their medians;
- a maximum action probability above 0.999 on more than 95% of validation
  states is treated as collapse unless the exact short-horizon target shows
  equivalent concentration;
- distinct action-sequence and terminal `(size, depth)` counts do not decrease
  for two consecutive budget doublings while hypervolume is also flat;
- on fixed trajectories, the regression of `log P_F` on `log R + log P_B`
  has a bootstrap confidence interval containing slope one. Report the slope
  and centered RMSE even when the reward range is too narrow for a stable hard
  test.

## Phase 5: improve policy optimization conditionally

Enter this phase only if `logZ` gates pass but centered residual variance or
search quality is still improving at the Phase 4 budget cap.

### Uniform replay test

Use a FIFO buffer of the latest 1,024 complete trajectories. After collecting
four fresh trajectories:

1. perform one update on those four trajectories;
2. perform three additional updates on uniformly sampled replay minibatches of
   four trajectories;
3. recompute every replayed trajectory's `log P_F` under the current policy;
4. reuse stored action sequences and terminal rewards, with `log P_B = 0`.

This is `replay-4x`. Compare it with `replay-1x`, the no-extra-update control,
at identical trajectory checkpoints. Use `bc0` and `dalu`, seeds 0--2 first;
expand the survivor to seeds 0--4. Test `replay-8x` only if `replay-4x` is still
clearly improving at the job boundary and remains within the 12-hour target.

Do not introduce reward-prioritized replay in this phase. If uniform replay
passes health gates but hurts hypervolume, keep the no-replay optimizer and
select a larger `B` rather than optimizing the residual at the expense of the
primary search objective.

### Policy learning rate and clipping

Tune the policy learning rate only after the replay decision. If gradients are
finite but learning is slow, screen `0.0003`, `0.001`, and `0.003` on `bc0`,
seeds 0--1, and confirm at most two values on `dalu`, seeds 0--2. Add gradient
clipping only if the unclipped norm distribution has genuine spikes or
nonfinite updates; set the first threshold to the 99th percentile of gradient
norms from the best stable run rather than choosing it from hypervolume.

After any replay or policy-rate change, rerun the Phase 4 budget curve. Extra
updates can move the plateau earlier, so the old `B*` cannot be copied without
verification.

If centered error plateaus high despite tabular exact-test success, inspect
prefix observation collisions and increase representation capacity before
considering SubTB. Treat SubTB as a new objective study, not as a minor
optimizer ablation.

## Phase 6: exploration check after optimizer health

Freeze initialization, both learning rates, replay, reward, and `B*`. Compare
only:

- the current epsilon schedule;
- no external epsilon;
- epsilon 0.1 decayed to zero during the first half of training, followed by a
  zero-epsilon consolidation half.

Run `bc0` and `dalu`, seeds 0--4. All action log-probabilities in TB remain
those of the learned policy. Select exploration by mean per-seed archive
hypervolume, subject to retaining the optimizer-health gates and acceptable
sequence diversity. This phase marks the boundary between optimizer health and
the later reward/exploration study; do not vary reward temperature here.

## Phase 7: lock, hold out, and compare baselines

Before running `i10`, write a locked configuration containing:

- selected `B*` and the reason it passed the plateau rule;
- `logZ` initialization and learning rate;
- policy learning rate, replay ratio and buffer policy;
- epsilon schedule;
- exact-test results and full-horizon health-gate table;
- code commit and resolved environment.

Run the locked GFlowNet on `i10`, seeds 0--9. Do not retune after seeing these
results. A successful holdout confirmation requires:

- all exact and full-horizon optimizer gates to remain satisfied;
- a positive paired difference in mean per-seed archive hypervolume relative
  to the instrumented active GFlowNet baseline at the same `B*`, with a 95%
  bootstrap confidence interval excluding zero;
- no regression in best-of-50 mean product improvement or terminal diversity;
- probability-calibration diagnostics consistent with `bc0` and `dalu`.

If health transfers but the hypervolume improvement does not, record optimizer
health as successful and move to the reward/exploration phase. Do not continue
optimizer tuning on `i10`.

Then run REINFORCE, DRiLLS-A2C, PPO, and PCN on `i10`, seeds 0--9, using the
budget translations above. Use the same 50 final samples, evaluation seed
scheme, hypervolume reference `[1, 1]`, and nested best-of-`N` calculation.
Report:

- mean per-seed archive hypervolume with paired bootstrap intervals;
- pooled archive hypervolume, clearly labeled secondary;
- best-of-`N` curves and their area under the curve;
- mean terminal product improvement, best size and depth reduction;
- distinct sequences, distinct terminal points, and nondominated endpoints;
- trajectories, transitions, updates, presentations, wall time, and GPU hours.

If healthy configurations are statistically tied in primary hypervolume, use
the one-standard-error rule: choose the simpler and cheaper configuration whose
mean lies within one standard error of the best.

## Martin execution plan

Use one job per circuit, seed, and configuration. This isolates failures,
supports paired analysis, and keeps each logical run below the 12-hour target.
Submit in successive-fidelity waves so rejected configurations do not consume
the expensive seed budget.

Before the first long wave, run a one-hour scheduled preflight containing 100
training trajectories and one validation milestone. Measure trajectories per
second, evaluation cost, CPU utilization, RAM, and peak VRAM. Extrapolate the
largest planned milestone and require a predicted runtime below 10.5 hours.

The initial resource request should be reviewed after preflight, but a
conservative training template is:

```text
--gres=gpu:1
--mem=32G
--cpus-per-task=8
--time=12:00:00
--account=michalis
--qos=guaranteed-michalis
--exclude=mbz-titan-3
```

Use a whole GPU for training and the guaranteed group allocation. Do not use
the preemptable QoS unless checkpoint/requeue support has been verified and a
separate decision authorizes it. Respect Martin's limit of four running and 20
total queued-plus-running jobs by submitting waves of at most 20. If a selected
logical run cannot fit within 12 hours, resume it from an exact milestone in a
second job rather than silently reducing `B*`; the combined segments still
constitute one run and must not repeat or omit trajectories.

Use `myhpc` for Git validation, immutable synchronization, preflight, and any
later submission. The SLURM script is the canonical entrypoint. It must write
all metrics, checkpoints, resolved configs, and summaries under
`/shared/home/fedor.chernogorskii/agent/art/<project>/<job-name>` and must never
run GPU work on the login node. No job is submitted as part of writing this
plan.

## Analysis and reporting protocol

Generate the following figures for every surviving configuration:

1. TB MSE split into squared bias and variance versus training trajectories.
2. learned `logZ`, fixed-batch target, fresh-on-policy target, and target gap.
3. learned and analytically centered residual RMS versus trajectories and
   optimizer updates.
4. policy and `logZ` gradient and update norms, logits, entropy, and maximum
   probability.
5. archive hypervolume and distinct nondominated endpoints versus trajectories.
6. nested best-of-`N` curves at every budget milestone.
7. wall time and trajectory presentations versus search and calibration
   metrics.

Use paired bootstrap resampling over seeds with 10,000 resamples for confidence
intervals. Report every seed as a point; do not show only aggregate error bars.
For zero-inflated hypervolume, also report the number of paired seed wins,
ties, and losses. Do not use checkpoint evaluation samples in the final archive
and do not pool samples across seeds for the primary metric.

Maintain a phase ledger with one row per attempted configuration containing its
hypothesis, changed field, circuits, seeds, maximum `B`, status, rejection
reason, health gates, primary metric, runtime, and artifact path. A phase may
advance only when its stated gate passes. This makes the conditional experiment
auditable and prevents a nominally small screen from becoming an unreported
optimizer grid.

## Final stop conditions

The optimizer-health campaign ends in exactly one of these states.

1. **Healthy and search-improving:** exact tests pass, full-horizon gates and
   plateau rules pass, and holdout hypervolume improves. Freeze the optimizer
   and common budget for baseline comparison.
2. **Healthy but search-neutral:** calibration and convergence pass but holdout
   search does not improve. Freeze the optimizer-health conclusion and start a
   separate reward/exploration investigation.
3. **Implementation failure:** the tabular exact test fails. Stop all circuit
   sweeps until the code is corrected.
4. **Representation failure:** tabular tests pass but the contract policy fails
   the enumerable circuit test. Improve observation sufficiency or capacity
   before allocating a longer full-horizon budget.
5. **Budget unresolved:** health or search curves still improve at the largest
   checkpoint. Do not declare convergence or copy that budget to baselines;
   extend the cap or add trajectory reuse, then repeat the budget curve.
