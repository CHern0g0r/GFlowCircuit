# GFlowNet circuit-optimization comparison protocol

## Objective

Evaluate whether a Trajectory Balance (TB) GFlowNet discovers a higher-quality
and more diverse size/depth Pareto front than established circuit-optimization
and reinforcement-learning methods.

The comparison has two distinct questions:

1. **End-to-end comparison:** does the complete GFlowNet system outperform
   acknowledged methods using each method's native representation and training
   procedure?
2. **Controlled algorithm comparison:** with the same policy backbone,
   environment, reward information, training-evaluation budget, and final
   sampling budget, does the GFlowNet objective outperform conventional RL?

These questions require separate result tables. A native-paper baseline should
not be modified and then described as a reproduction. Conversely, a baseline
should not be forced to use a representation known to be inadequate merely to
preserve a paper-scale architecture.

## Selected baselines

### Zhu-style REINFORCE

- **Original paper:** [Exploring Logic Optimizations with Reinforcement
  Learning and Graph Convolutional
  Network](https://doi.org/10.1145/3380446.3430622)
- **Repository configuration:** `cfg/zhuDOP.yaml`
- **Status:** implemented.

#### Original method and backbone

Zhu et al. formulate logic-synthesis operator selection as an MDP and train a
REINFORCE policy with a learned state-value baseline. The state contains an AIG
graph branch and a compact vector branch. The graph uses six node classes,
four graph-convolution layers with dimensions `6 -> 12 -> 12 -> 12 -> 4`, and
mean pooling over nodes. The compact vector contains normalized circuit
statistics, recent-action information, and progress through the synthesis
sequence. The policy combines the graph and vector representations and applies
fully connected layers before the action softmax. The value baseline uses only
the vector state.

#### Repository adaptation

The repository uses the shared `hybrid_zhu` encoder:

- Zhu GCN graph branch with hidden dimension 12, output dimension 4, four
  layers, and mean pooling;
- a Zhu-style vector branch projected to 28 dimensions;
- concatenation into a 32-dimensional policy representation;
- an MLP policy head with one 32-unit hidden layer;
- a vector-only MLP value baseline with one 32-unit hidden layer.

The repository action set has seven actions, so its dynamic Zhu vector is not
literally the paper's fixed ten-dimensional input. This is a paper-style
adaptation, not a bit-for-bit reproduction.

Use this method in two roles:

1. `REINFORCE-native`, using the published-scale Zhu-style backbone;
2. `REINFORCE-shared`, using the capacity-qualified shared backbone selected
   for the controlled comparison.

### Proximal Policy Optimization

- **Original paper:** [Proximal Policy Optimization
  Algorithms](https://arxiv.org/abs/1707.06347)
- **Repository configuration:** `cfg/ppo_zhuDOP.yaml`
- **Status:** implemented.

#### Original method and backbone

PPO is a general policy-optimization algorithm, not a circuit-synthesis
architecture. The original algorithm alternates between collecting on-policy
experience and performing multiple minibatch epochs on a clipped surrogate
objective. It does not prescribe an AIG encoder or a circuit-specific policy
backbone.

#### Repository adaptation

The repository uses the same `hybrid_zhu` policy encoder and 32-unit policy
head as the Zhu-style REINFORCE baseline. Its critic is a vector-only MLP with
one 32-unit hidden layer. Training uses clipped PPO with generalized advantage
estimation, normalized advantages, 80 rollout transitions per iteration,
20 PPO epochs, minibatches of 64, clipping parameter 0.2, and GAE parameter
0.95 under the current default configuration.

Because PPO has no native circuit backbone, the primary PPO result should use
the shared capacity-qualified backbone. The published-scale Zhu backbone may
be retained as a backbone ablation.

### DRiLLS

- **Original paper:** [DRiLLS: Deep Reinforcement Learning for Logic
  Synthesis](https://arxiv.org/abs/1911.04021)
- **Repository configurations:** `cfg/drills_a2c_zhu.yaml` and
  `cfg/drillsDOP.yaml`
- **Status:** implemented.

#### Original method and backbone

DRiLLS uses an Advantage Actor-Critic agent to minimize circuit area subject
to a delay constraint. Its original state is a handcrafted vector of design
metrics extracted from the synthesis tool rather than a graph-neural
representation. The actor and critic are feed-forward networks over this state:
the actor outputs a distribution over synthesis transformations and the critic
estimates state value. The original reward is a discrete size/delay reward
table encoding area improvement and timing-constraint satisfaction.

#### Repository adaptations

The repository's DRiLLS-style trainer uses one-step TD targets and a joint A2C
actor/critic loss.

- `drills_a2c_zhu.yaml` retains the DRiLLS-style constrained size/depth reward
  and A2C update, while using repository-compatible Zhu observations and
  models.
- `drillsDOP.yaml` uses the shared `hybrid_zhu` actor backbone and the same
  diff-of-product reward used by the current controlled GFlowNet comparison.

Report the first as a native-style DRiLLS adaptation and the second as the
controlled same-reward/same-backbone baseline. Do not label either as an exact
reproduction unless the original feature vector, reward table, action set,
technology mapping, and training procedure have all been matched and
validated.

### BOiLS

- **Original paper:** [BOiLS: Bayesian Optimisation for Logic
  Synthesis](https://arxiv.org/abs/2111.06178)
- **Repository configuration:** not yet implemented.
- **Status:** selected baseline to add.

#### Original method and backbone

BOiLS is not an RL algorithm and has no neural policy backbone. It treats the
quality of a complete synthesis sequence as a black-box function. A Gaussian
process surrogate represents synthesis sequences with a subsequence string
kernel. Expected improvement proposes new sequences, and acquisition
optimization is constrained to an adaptive Hamming-distance trust region
around the best sequence observed so far.

BOiLS is an important domain baseline because it directly addresses
sample-efficient synthesis-sequence optimization. It must retain its own
surrogate and acquisition machinery; forcing it to use the neural shared
backbone would define a different method.

For integration, adapt only the action alphabet, fixed sequence horizon,
objective calculation, circuit interface, random-seed handling, and artifact
format. Count every initial-design and acquisition-proposed sequence evaluated
by ABC against the shared training-evaluation budget.

## Candidate multi-objective RL comparisons

### Pareto Conditioned Networks

- **Original paper:** [Pareto Conditioned
  Networks](https://arxiv.org/abs/2204.05036)
- **Repository components:** `src/algorithms/pcn/` and
  `cfg/algorithm/pcn.yaml`
- **Status:** implemented in the repository, but it must pass method-specific
  validation before inclusion in the final comparison.

PCN is designed to represent multiple nondominated policies in one network.
It stores transitions with the return achieved by their episode and trains a
policy by supervised action prediction conditioned on a desired return and
remaining horizon. At inference time, selecting different desired size/depth
returns should produce different regions of the Pareto front.

The repository PCN combines the shared state encoder with embeddings of desired
two-objective return and desired horizon. State and condition embeddings are
fused multiplicatively before the policy head. Its initial random episodes,
subsequent collected episodes, and every terminal sequence added to the PCN
archive count against the training-evaluation budget. Replay presentations and
supervised gradient updates do not count as new circuit evaluations, but must
be reported separately.

Use the capacity-qualified shared backbone for the controlled result. Validate
that conditioning changes the sampled objective distribution and that
checkpoints reproduce their recorded archive and evaluation metrics.

### Scalarized RL sweep

A scalarized RL sweep is a simple and strong multi-objective control. Choose a
fixed, preregistered set of nonnegative weights

\[
W = \{(w_i, 1-w_i)\},
\]

including both endpoints and evenly spaced interior preferences. Train an
independent REINFORCE or PPO policy for each size/depth scalarization, then take
the nondominated union of all returned sequences.

The total budget belongs to the sweep, not to each scalarization. If the shared
budget is 6,400 complete trajectories and eight weights are used, each weight
receives 800 trajectories. Likewise, a final allowance of 400 fresh samples
means 50 samples per weight. Giving every weight the entire budget would make
the sweep eight times more expensive than a single GFlowNet or PCN.

Use the shared backbone for every weight. Pair initialization and sampling
seeds where meaningful, and report per-weight results in addition to the union.
The weight grid and budget allocation must be selected on development circuits
before examining the locked holdout.

## Comparison contract

### 1. Select baselines before final evaluation

The minimum final comparison should contain:

- TB GFlowNet;
- Zhu-style REINFORCE;
- PPO;
- DRiLLS;
- BOiLS;
- PCN;
- one scalarized RL sweep;
- Resyn2 and uniform random sequence search as non-learning controls.

Use two result tracks:

1. **Native/end-to-end track:** each acknowledged method uses its native or
   closest validated paper-style representation, reward semantics, and
   optimizer. This measures complete-system performance.
2. **Shared-backbone track:** TB GFlowNet, REINFORCE, PPO, DRiLLS-A2C, and PCN
   use the same capacity-qualified policy encoder and head. Actor-critic methods
   may add a value head, and PCN may add its required return/horizon condition.
   This isolates the learning objective as far as practical.

BOiLS belongs only to the end-to-end track because it has no neural policy
backbone.

Select the shared backbone using only representability tests and development
circuits. Choose the smallest candidate that passes the preregistered
representability gates. Do not select it using downstream holdout hypervolume
or by choosing the model on which GFlowNet has the largest advantage. Freeze
the encoder, policy-head dimensions, observation definition, and parameter
count before the final method comparison.

### 2. Fix the task definition

All methods must use:

- the same ABC build and circuit files;
- the same seven-action alphabet and action semantics;
- the same fixed horizon;
- the same initial circuit state;
- identical size/depth extraction and normalization;
- identical legality handling;
- the same development, stress-test, and locked-holdout split;
- the same definition of a unique terminal object;
- the same random-seed list;
- immutable resolved configurations and circuit/code hashes.

Native methods may keep their published training rewards, but evaluation must
always use the common raw size/depth objectives. The controlled track must use
the same reward information wherever the algorithms permit it. Any unavoidable
difference, such as PCN's vector return or DRiLLS's timing constraint, must be
declared rather than hidden inside a common method name.

### 3. Separate tuning, training, and final evaluation

Count three budgets independently:

1. **Tuning budget:** all circuit evaluations used for hyperparameter and
   architecture selection. Give every method the same maximum tuning budget,
   or publish the actual unequal tuning cost.
2. **Training-evaluation budget:** new complete synthesis sequences evaluated
   while fitting or searching. This is the primary sample-complexity budget.
3. **Final sampling budget:** fresh sequences drawn from a frozen method after
   training. These samples must not update models, surrogates, replay buffers,
   or training archives.

Replay presentations, PPO epochs, gradient steps, and GFlowNet reuse of sampled
trajectories are not new circuit evaluations. Record them as optimizer updates
and trajectory presentations. Also report environment transitions, ABC
transformation executions, wall time, CPU/GPU hours, and peak memory, because
two methods with the same number of terminal sequences can have very different
internal costs.

### 4. Use common budget checkpoints

Do not compare every baseline only at a trajectory count chosen because it is
where GFlowNet fits best. That answers the narrow question "what do other
methods do at GFlowNet's preferred budget?" and can favor GFlowNet when methods
have different learning curves.

Instead, run every method continuously to the same preregistered maximum and
save results at common cumulative training-evaluation budgets:

\[
200,\ 400,\ 800,\ 1{,}600,\ 3{,}200,\ 6{,}400
\]

complete trajectories per circuit and seed, or another grid frozen before the
comparison. A scalarized sweep divides each checkpoint across its weights;
PCN's random initialization episodes and BOiLS's initial design count toward
the same checkpoints.

Use three complementary budget analyses:

1. **Primary matched-budget endpoint:** final hypervolume at the common maximum
   budget. The maximum must be chosen from practical resource limits or
   development-only convergence evidence, not from holdout results for one
   method.
2. **Sample-efficiency result:** area under the hypervolume-versus-trajectory
   curve through the common maximum. A method receives credit for reaching a
   good front earlier.
3. **Method-specific stopping result:** on development circuits only, choose
   the smallest checkpoint at which each method satisfies its own
   preregistered plateau rule. Freeze those budgets and evaluate them on the
   holdout. Report both quality and the different compute costs; do not call
   this an equal-budget comparison.

It is acceptable to use the selected GFlowNet budget as an additional vertical
slice, provided it is labeled `GFlowNet-selected budget` and the full common
learning curves and common-maximum result remain primary.

### 5. Define archives and fresh samples consistently

Maintain two outputs per run:

- **Search archive:** the nondominated set among all terminal sequences whose
  circuit objectives were evaluated within the training-evaluation budget.
  Every method, including random search and BOiLS, receives the same right to
  retain evaluated solutions.
- **Frozen-policy sample:** a fresh, fixed-size set generated after training.
  It measures what the learned sampler produces rather than what it happened
  to discover during training.

Checkpoint evaluation samples must not enter the search archive or influence
training. Deduplicate exact action sequences before sequence-count metrics and
deduplicate identical objective pairs before Pareto-front geometry metrics.

Evaluate frozen methods at shared nested sample counts such as

\[
N \in \{1, 2, 5, 10, 20, 50\}.
\]

For PCN, distribute the allowance across preregistered target returns. For a
scalarized sweep, distribute it across weights. For GFlowNet and ordinary RL,
sample from the frozen policy with external exploration disabled unless the
evaluation protocol explicitly defines otherwise.

### 6. Metrics

#### Primary metrics

1. **Normalized two-dimensional hypervolume** of the per-run nondominated
   archive at each common budget, with a reference point fixed before final
   experiments.
2. **Hypervolume AUC versus complete trajectories**, normalized by the maximum
   budget.
3. **Frozen-policy best-of-\(N\) hypervolume curve** and its AUC over the shared
   nested sample counts.

Hypervolume must be computed per seed and circuit before aggregation. Never
pool all seeds into one front for the primary result; one lucky seed could then
hide unreliable training.

#### Pareto quality and coverage

- best normalized size improvement;
- best normalized depth improvement;
- number of unique nondominated objective pairs;
- additive epsilon indicator against a fixed reference front;
- IGD+ against a fixed reference front, when such a front can be constructed
  without using holdout tuning;
- extent of the front along both objectives;
- spacing or gap statistics along the nondominated front;
- pairwise coverage \(C(A,B)\) as a secondary directional comparison.

If the true front is unavailable, construct a descriptive reference front from
the nondominated union of all methods and seeds after the experiment is frozen.
Mark IGD+ and epsilon results against this empirical front as descriptive,
because their reference depends on the compared method set.

#### Diversity and reliability

- number and fraction of unique action sequences;
- duplicate rate;
- mean pairwise normalized Hamming distance among sampled sequences;
- number of unique terminal objective pairs;
- fraction of seeds passing method-specific numerical or sampler-health gates;
- median and worst-seed hypervolume;
- wall time and compute cost to reach each checkpoint.

Raw nondominated-point count is not sufficient evidence of useful diversity:
many points can be tightly clustered or dominated after exact deduplication.

### 7. Statistical comparison

Use paired seeds and common circuit instances. For final claims, use at least
ten seeds per method. Report each circuit separately and a macro-average across
circuits.

For each primary metric, report:

- per-seed values;
- mean, median, and standard deviation;
- paired method-minus-GFlowNet differences;
- paired 95% bootstrap confidence intervals with 10,000 resamples;
- the number of circuits and seeds on which each method wins.

Choose one primary hypothesis before final evaluation, for example:

> At the common maximum trajectory budget, TB GFlowNet has higher mean
> per-seed normalized archive hypervolume than the strongest shared-backbone
> RL baseline.

Treat remaining pairwise tests and diversity metrics as secondary, and disclose
the number of comparisons. A search-quality win with failed GFlowNet sampler
health may be reported as heuristic search performance, not as successful
reward-proportional GFlowNet learning.

### 8. Required reporting tables

Publish at least:

1. **Native/end-to-end table:** GFlowNet, native-style REINFORCE, native-style
   DRiLLS, BOiLS, Resyn2, and random search.
2. **Shared-backbone table:** GFlowNet, REINFORCE, PPO, DRiLLS-A2C, PCN, and
   scalarized RL with the frozen shared backbone.
3. **Budget-curve table or figure:** hypervolume and best-of-\(N\) metrics at
   every common checkpoint.
4. **Backbone ablation:** published-scale backbone versus the selected shared
   backbone for GFlowNet and at least REINFORCE.
5. **Compute table:** circuit evaluations, transitions, optimizer updates,
   trajectory presentations, wall time, and CPU/GPU hours.

All exclusions, crashes, resumed runs, and numerical failures remain in the
seed-level report. Do not silently replace failed seeds.
