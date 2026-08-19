# Exploration tuning protocol

## Goal and scope

This experiment selects an exploration configuration **separately for each
trainable algorithm** before the final algorithm comparison. It is not a joint
search over learning rates, rewards, model capacity, action spaces, or training
budgets. Those values must remain fixed so that an apparent exploration gain is
not caused by a different optimization problem.

The algorithms currently exposed by `src.run` are:

| Algorithm | Exploration mechanism to tune | No-explicit-exploration control |
| --- | --- | --- |
| GFlowNet-TB | epsilon mixture with the uniform distribution over legal actions | epsilon disabled; the learned categorical policy is still stochastic |
| REINFORCE | policy-entropy coefficient `entropy_beta` | `entropy_beta=0` |
| PPO | policy-entropy coefficient `algorithm.ppo.entropy_beta` | coefficient 0 |
| DRiLLS-A2C | policy-entropy coefficient `algorithm.drills.entropy_beta` | coefficient 0 |
| PCN | random archive seeding and fallback desired-return perturbation | 32 random seed episodes and no fallback perturbation |

`resyn2` is deterministic in this repository and has no exploration parameter,
so it is a reference baseline rather than a member of this tuning experiment.

## Repository version and implementation audit

This protocol was prepared against commit
`3c7cd6bdd09750f392bdcdb7dddeb7ca3766fcd2` (`3c7cd6b`, 2026-07-20,
“exploration_tuning exp configs”). At the time of inspection,
`mds/exploration_tuning.md` was the only untracked path and was empty.

The commit already provides a coarse set of Hydra fragments under
`cfg/exp/exploration_tuning/`:

- GFlowNet-TB: `epsilon_none`, `epsilon_low`, `epsilon_current`;
- REINFORCE, PPO, and DRiLLS-A2C: `entropy_none`, `entropy_low`, `entropy_high`;
- PCN: `no_fallback_noise`, `balanced`, `more_random_seeding`.

These are useful as a first screen, but they are not a complete tuning search.
In particular, the two nonzero GFlowNet presets change the initial epsilon and
warmup simultaneously, and both leave `exploration_decay_episodes=null`. With
50 pilot episodes, `null` means that epsilon reaches its final value only at the
last episode, leaving no low-epsilon consolidation period.

The implementation has the following semantics that must be preserved when
interpreting results:

- GFlowNet sampling uses
  `(1 - epsilon) * learned_policy + epsilon * uniform_legal_policy`. The TB loss
  records log-probabilities under the learned forward policy, not under the
  epsilon behavior mixture. Evaluation through `src.sample_exp` uses the
  learned categorical policy without the external epsilon mixture.
- REINFORCE applies an entropy bonus at every step; PPO and DRiLLS-A2C use the
  mean legal-action entropy in their batch losses. The same numeric coefficient
  therefore need not be optimal across the three algorithms.
- PCN always explores stochastically while collecting trajectories. Its target
  perturbation normally uses the observed nondominated-archive standard
  deviation. `target_noise_scale` is only a fallback perturbation range when
  that standard deviation is at most `target_min_sigma`; it is not a global
  noise multiplier. Setting both fields to zero removes only the fallback.
- `discovery_metrics` records terminal states encountered during training. It
  is diagnostic data and must not be mixed with post-training evaluation
  samples used for model selection.

The protocol is implemented by `python -m src.exploration_tuning`, with the
machine-readable matrix in `cfg/exp/exploration_tuning/protocol.yaml` and
stage-specific Martin wrappers in the configured `gflowcircuit` scripts
project. The driver keeps one seed per resumable task, validates Hydra
composition before training, samples final checkpoints, and produces the CSV,
JSON, and Markdown selection artifacts described below. The reviewed project
commit remains an explicit pre-submission gate.

## Experimental invariants

Use a two-stage design: a cheap screen followed by a full-budget confirmation.
All settings within a stage use common training seeds and identical evaluation
seeds.

| Field | Screening stage | Confirmation stage |
| --- | ---: | ---: |
| Tuning circuits | `C1355`, `dalu` | `C1355`, `dalu` |
| Training seeds | 0, 1, 2 | 0 through 9 |
| Training trajectories per seed and circuit | 200 | 800 |
| Horizon | 20 actions | 20 actions |
| Available actions | `[0,1,2,3,4,5,6]` | same |
| Final evaluation samples | 20 per seed and circuit | 50 per seed and circuit |
| Evaluation seed | 42 | 42 |
| Hypervolume reference | normalized `(1,1)` | same |

`C1355` and `dalu` are the predeclared tuning set. `dalu` is deliberately
included because prior repository analysis identifies it as
exploration-sensitive and high-variance. Do not add or remove circuits after
viewing tuning results. Generalization claims require a later, untouched
benchmark evaluation; the two tuning circuits alone are only sufficient to
select settings for the controlled comparison.

Freeze all remaining fields to the algorithm's named base configuration:

- GFlowNet-TB: `tb_zhuDOP`;
- REINFORCE: `zhuDOP`;
- PPO: `ppo_zhuDOP`;
- DRiLLS-A2C: `drillsDOP`;
- PCN: `pcn_zhu`.

This includes reward, encoder, head, optimizer, learning rates, discount,
baseline, PPO epochs and clipping, PCN archive/training parameters, and the
GFlowNet reward temperature. Do not tune reward temperature and exploration in
the same sweep. For this campaign, the manual optimizer-health approval is
waived and the `gfn_health` branch's descriptive control is pinned explicitly:
Adam, policy learning rate `0.001`, separate `logZ` learning rate `0.01`, and
four trajectories per update (source commit
`6f9d3fb48e244c671d8d7838aac45c7bd2b6bd9a`). That branch did not establish a
fully healthy optimizer winner, so any GFlowNet exploration conclusion must be
reported with this unresolved-optimizer caveat.

Translate the trajectory budget as follows:

| Algorithm | Screen | Confirmation |
| --- | --- | --- |
| GFlowNet-TB | 50 episodes x 4 trajectories | 200 x 4 |
| REINFORCE | 200 episodes x 1 trajectory | 800 x 1 |
| PPO | 50 iterations x 80 transitions / 20 steps | 200 x 80 / 20 |
| DRiLLS-A2C | 50 episodes x 4 trajectories | 200 x 4 |
| PCN | 200 collected episodes | 800 collected episodes |

Random PCN archive episodes count against this budget. Evaluation rollouts do
not. Record optimizer updates and wall-clock time as secondary resource
measurements because equal environment budgets do not imply equal compute.

Set `eval_every` to the final training episode/iteration and
`paper_mode.infer_rollouts=1` during tuning. This prevents checkpoint evaluation
from becoming a large, algorithm-dependent extra environment budget. The
selection samples are generated after training from saved checkpoints.

## Search procedure

Every algorithm is tuned independently. Complete the coarse screen, retain the
best two settings, refine only those settings, and then run confirmation. Do
not select a global value shared by all algorithms.

### 1. Coarse screen using the existing fragments

Run the three checked-in settings for each algorithm:

| Algorithm | Settings |
| --- | --- |
| GFlowNet-TB | disabled; 0.10 to 0.01 with 5-episode warmup; 0.50 to 0.01 with 20-episode warmup |
| REINFORCE | entropy beta 0, `1e-3`, `1e-2` |
| PPO | entropy beta 0, `1e-3`, `1e-2` |
| DRiLLS-A2C | entropy beta 0, `1e-3`, `1e-2` |
| PCN | 32 seeds/no fallback; 32 seeds with `(scale=0.05,min_sigma=0.01)`; 64 seeds with that fallback |

This is a screening comparison of bundled strategies. It must not be used to
claim that one individual GFlowNet or PCN parameter caused the result.

### 2. Algorithm-specific refinement

Use the same 200-trajectory, three-seed screen budget. Evaluate only settings
not already measured in the coarse screen.

For GFlowNet-TB, tune one schedule component at a time:

1. Fix `epsilon_end=0.01`, warmup 5, and decay 25. Compare epsilon disabled
   against `epsilon_start` in `{0.05, 0.10, 0.25, 0.50}`. This leaves 20 of the
   50 episodes at the final epsilon.
2. With the winning nonzero start value, compare `epsilon_end` in
   `{0, 0.01, 0.05}`.
3. With the winning start/end pair, compare `(warmup, decay)` in
   `{(0,25), (5,15), (5,25), (5,40), (20,20)}`. These schedules separate early
   random exploration from low-epsilon consolidation.
4. Carry the disabled control forward even if it is not in the top two. It is
   essential for determining whether explicit epsilon exploration helps a
   policy that is already stochastic.

For each entropy-regularized baseline, independently compare beta in
`{0, 1e-4, 1e-3, 1e-2}`. Around the best nonzero value, evaluate one lower and
one higher neighbor separated by a factor of about three (for example,
`3e-4` and `3e-3` around `1e-3`). If beta 0 wins, refine only with `1e-5` and
`3e-5`; do not force a nonzero winner. Do not transfer the REINFORCE winner to
PPO or DRiLLS-A2C.

For PCN, use a coordinate screen followed by a small interaction check:

1. With fallback noise off, compare `random_seed_episodes` in `{16,32,64}`.
2. At the best seed count, compare `(target_noise_scale,target_min_sigma)` in
   `{(0,0), (0.01,0.01), (0.05,0.01), (0.10,0.01)}`.
3. Evaluate the four combinations of the best two seed counts and best two
   fallback settings. This is necessary because more random seeding leaves
   fewer learned collection episodes within the fixed trajectory budget.

If a selected value lies at the edge of a grid, add one adjacent value in that
direction before confirmation. Stop refinement once adjacent settings are
within the practical tie threshold defined below; three pilot seeds cannot
support fine distinctions.

### 3. Full-budget confirmation

For each algorithm, run exactly these two configurations with 800 training
trajectories and ten seeds:

1. the selected exploration configuration;
2. the no-explicit-exploration control.

If the selected configuration is the control, confirm it against the best
nonzero runner-up instead. This final paired comparison determines the setting
used in the later algorithm benchmark. Pilot results alone do not.

## Running the experiments

First perform a one-seed smoke run for every base config and inspect the saved
`.hydra/config.yaml`. Confirm the algorithm name, circuit manifest, exploration
fields, 20-step horizon, seven-action set, seed count, and budget translation.
Abort the sweep if any resolved value differs from the protocol.

The base command pattern is:

```bash
python -m src.run --config-name BASE_CONFIG \
  "+exp/exploration_tuning=FRAGMENT" \
  dataset_cfg=cfg/data/zhu2020/CIRCUIT.yaml \
  num_steps=20 available_actions='[0,1,2,3,4,5,6]' \
  episodes=EPISODES eval_every=EPISODES \
  paper_mode.num_runs=3 paper_mode.infer_rollouts=1 \
  discovery_metrics.enabled=true \
  discovery_metrics.emit_every_trajectories=50 \
  run_name=explore_ALGORITHM_SETTING_CIRCUIT
```

Use `dataset_cfg=...`; the current base configs do not declare a Hydra `data`
defaults group, and `src.run` reads `cfg.dataset_cfg` directly.

Concrete coarse-screen mappings are:

| Algorithm | `BASE_CONFIG` | `FRAGMENT` prefix | screen `EPISODES` |
| --- | --- | --- | ---: |
| GFlowNet-TB | `tb_zhuDOP` | `gflownet/` | 50 |
| REINFORCE | `zhuDOP` | `reinforce/` | 200 |
| PPO | `ppo_zhuDOP` | `ppo/` | 50 |
| DRiLLS-A2C | `drillsDOP` | `drills/` | 50 |
| PCN | `pcn_zhu` | `pcn/` | 200 |

For example:

```bash
python -m src.run --config-name tb_zhuDOP \
  "+exp/exploration_tuning=gflownet/epsilon_low" \
  dataset_cfg=cfg/data/zhu2020/C1355.yaml \
  num_steps=20 available_actions='[0,1,2,3,4,5,6]' \
  episodes=50 eval_every=50 \
  paper_mode.num_runs=3 paper_mode.infer_rollouts=1 \
  discovery_metrics.enabled=true \
  discovery_metrics.emit_every_trajectories=50 \
  run_name=explore_gfn_epsilon_low_C1355
```

Pass refinement values as explicit Hydra overrides in the same command. For
example, the GFlowNet schedule uses top-level fields such as
`tb.exploration_epsilon_start=0.25`, whereas PPO uses
`algorithm.ppo.entropy_beta=0.001`. Keep a manifest with one row per job:
commit, algorithm, setting ID, full resolved exploration fields, circuit,
training seeds, trajectory budget, evaluation seed, output directory, status,
and wall-clock time.

After training, create fresh samples for each output directory and its single
tuning circuit:

```bash
python -m src.sample_exp \
  --experiment /absolute/path/to/output_dir \
  --circuit /absolute/path/to/C1355.blif \
  --num-samples 20 --seed 42
```

Use 50 samples in confirmation. For PCN, keep the default `target` sampling
mode and record it in the manifest; this evaluates conditioned target commands
rather than pretending PCN has the same unconditional sampler as the policy
gradient methods. Never combine PCN modes within one selection table.

## Metrics and selection rule

For each seed, deduplicate raw terminal `(size,depth)` endpoints, normalize both
coordinates by the original circuit, retain the weak Pareto-minimal front, and
compute strict two-dimensional hypervolume with reference `(1,1)`. The original
circuit row in `points.csv` is a reference point, not a model sample.

Report, per algorithm, setting, and circuit:

- mean, standard deviation, and individual values of per-seed hypervolume;
- pooled hypervolume across all seeds, clearly labeled as a discovery metric;
- distinct sampled endpoints and distinct nondominated endpoints;
- best size reduction and best depth reduction;
- mean terminal product/QoR improvement;
- fraction of samples that belong to the setting's pooled nondominated front;
- training discovery hypervolume versus completed trajectories;
- wall-clock time and optimizer-update count.

Mean per-seed post-training hypervolume is primary. Pooled hypervolume is not
primary because one lucky seed can make an unreliable setting look best. A
zero hypervolume can still contain a useful one-objective improvement at the
`(1,1)` boundary, so always report the separate size and depth reductions.

For setting `s`, circuit `c`, and algorithm `a`, calculate

```text
relative_score(a,s,c) = mean_HV(a,s,c) / max_s mean_HV(a,s,c)
selection_score(a,s)  = mean over the two circuits of relative_score(a,s,c)
```

If all settings have zero mean hypervolume on a circuit, assign every setting a
relative score of 1 for that circuit and use the declared secondary metrics;
never divide by zero. Select the largest `selection_score`. Treat settings
within 0.02 as practically tied, then prefer in order:

1. the larger minimum relative score across the two circuits;
2. the larger mean per-seed terminal product improvement;
3. the simpler/lower-exploration setting (lower epsilon or entropy, fewer PCN
   random seed episodes, then no fallback noise).

Pooled hypervolume is reported but is not a tie-breaker for reliability-based
selection. In confirmation, show paired seed differences and a bootstrap
confidence interval, but apply the same predeclared practical threshold rather
than selecting on a p-value. Do not choose separately per circuit; the output
is one exploration setting per algorithm.

## Validity and stopping checks

A job is invalid and must be rerun if it has the wrong resolved config, fewer
than the required training trajectories, missing checkpoints, non-finite loss,
failed circuit synthesis, or fewer than the required final samples. Do not
silently replace a failed seed or discard a poor seed. Record the failure and
rerun the same seed after fixing the operational cause.

Before accepting a winner, verify that:

- all compared settings use the same commit, circuits, actions, horizon,
  architecture, reward, optimizer settings, and environment budget;
- training and evaluation samples are stored separately;
- GFlowNet evaluation does not apply training epsilon;
- the selected result is not driven only by pooled rare discoveries;
- PCN target sampling mode and number of evaluated commands are identical
  across its settings;
- the selected configuration survives the 800-trajectory, ten-seed paired
  confirmation.

Only after these checks should the per-algorithm winners be frozen and used in
the final GFlowNet-versus-baselines experiment on untouched circuits.

## Results

### Campaign status and interpretation

Results were collected from the Martin artifacts on 2026-08-19. All stages
from smoke validation through `pcn_interaction` completed on project commit
`da80471c4ad292b84053a3359fe70cd7ce56e3c6` with protocol hash
`25ce67a9122520341e7e59b7eed4b64fc92e16f095bff9d82590257add43fe93`.
The acceptance audit covered 431 stage-task records representing 239 actual
training tasks after duplicate reuse. Every new task had one validated
checkpoint, the exact trajectory and sample counts, successful training and
sampling exit codes, and a single `attempt_001`; no retry was required.

The ten-seed, 800-trajectory `confirmation` stage has not yet been run.
Consequently, the decisions below are the best **screening selections** and
the configurations to carry forward, not yet confirmed settings for the final
algorithm benchmark. Canonical machine-readable results remain under
`/shared/home/fedor.chernogorskii/agent/art/gflowcircuit/gfc-explore-*-v1`.

| Stage | New tasks | Reused tasks | Selection |
| --- | ---: | ---: | --- |
| `smoke` | 5 | 0 | All five base configurations validated |
| `coarse` | 90 | 0 | GFlowNet `epsilon_current`; entropy `1e-2` for all three policy-gradient baselines; PCN `balanced` |
| `gfn_start` | 24 | 6 | epsilon start `0.10` |
| `gfn_floor` | 12 | 12 | epsilon end `0` |
| `gfn_schedule` | 24 | 12 | warmup 5, decay 25 |
| `entropy_grid` | 18 | 54 | DRiLLS-A2C `1e-4`; PPO `1e-2`; REINFORCE `1e-2` |
| `entropy_neighbors` | 36 | 72 | DRiLLS-A2C `3e-4`; PPO `3e-2`; REINFORCE `1e-2` |
| `pcn_seeds` | 12 | 6 | 64 random seed episodes |
| `pcn_noise` | 12 | 12 | scale `0.10`, minimum sigma `0.01` |
| `pcn_interaction` | 6 | 18 | 64 seeds with scale `0.10`, minimum sigma `0.01` |

In the comparison tables, `HV` is mean per-seed strict normalized
hypervolume at reference `(1,1)`, shown as mean +/- standard deviation over
three seeds. `Score` is the declared mean two-circuit relative score, `Min` is
the smaller circuit-relative score, and `Product` is the mean terminal product
improvement over both circuits. Tables are ordered by the recorded selection
ranking. Pooled hypervolume was checked in the artifacts but was not used for
selection.

| Method | Best observed exploration configuration | Winner score | Control score | Score gain |
| --- | --- | ---: | ---: | ---: |
| GFlowNet-TB | epsilon `0.10 -> 0`, warmup 5, decay 25 | 0.981 | 0.705 | +0.276 |
| REINFORCE | entropy beta `0.01` | 1.000 | 0.814 | +0.186 |
| PPO | entropy beta `0.03` | 0.985 | 0.861 | +0.123 |
| DRiLLS-A2C | entropy beta `0.0003` | 0.972 | 0.690 | +0.281 |
| PCN | 64 seed episodes, noise scale `0.10`, minimum sigma `0.01` | 0.901 | 0.500 | +0.401 |

### GFlowNet-TB

#### Results

The start sweep fixed epsilon end at `0.01`, warmup at 5, and decay at 25.

| Epsilon start | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.10** | 0.07504 +/- 0.00849 | 0.01989 +/- 0.01427 | **0.963** | **0.927** | 25.33% |
| 0.05 | 0.08069 +/- 0.00040 | 0.01465 +/- 0.01622 | 0.866 | 0.737 | 24.76% |
| disabled | 0.08099 +/- 0.00007 | 0.00886 +/- 0.00836 | 0.723 | 0.446 | 22.96% |
| 0.50 | 0.07489 +/- 0.00425 | 0.00672 +/- 0.00553 | 0.631 | 0.338 | 24.03% |
| 0.25 | 0.07489 +/- 0.00419 | 0.00457 +/- 0.00646 | 0.577 | 0.230 | 23.89% |

The floor sweep then fixed epsilon start at `0.10`, warmup at 5, and decay at
25.

| Epsilon end | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0** | 0.07792 +/- 0.00416 | 0.02162 +/- 0.00845 | **0.981** | **0.962** | 23.12% |
| 0.01 | 0.07504 +/- 0.00849 | 0.01989 +/- 0.01427 | 0.923 | 0.920 | 25.33% |
| disabled | 0.08099 +/- 0.00007 | 0.00886 +/- 0.00836 | 0.705 | 0.410 | 22.96% |
| 0.05 | 0.06891 +/- 0.00421 | 0.00225 +/- 0.00318 | 0.477 | 0.104 | 23.21% |

The schedule sweep fixed epsilon at `0.10 -> 0`.

| Warmup, decay | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **5, 25** | 0.07792 +/- 0.00416 | 0.02162 +/- 0.00845 | **0.981** | **0.962** | 23.12% |
| 0, 25 | 0.06899 +/- 0.00842 | 0.02109 +/- 0.00349 | 0.914 | 0.852 | 25.08% |
| 5, 40 | 0.06601 +/- 0.00426 | 0.02000 +/- 0.01446 | 0.870 | 0.815 | 24.19% |
| 5, 15 | 0.07199 +/- 0.00735 | 0.01441 +/- 0.01220 | 0.778 | 0.666 | 24.35% |
| disabled | 0.08099 +/- 0.00007 | 0.00886 +/- 0.00836 | 0.705 | 0.410 | 22.96% |
| 20, 20 | 0.07199 +/- 0.00735 | 0.00909 +/- 0.00327 | 0.655 | 0.420 | 24.45% |

#### Analysis

An initial epsilon of `0.10` was the best compromise. The disabled policy and
the `0.05` start retained slightly more C1355 hypervolume, but `0.10` was much
stronger on dalu. Starts of `0.25` and `0.50` were particularly poor on dalu,
showing that aggressive early randomization is not beneficial at this budget.

Decaying fully to zero was clearly better than retaining an epsilon floor. A
floor of `0.05` reduced the minimum circuit-relative score to 0.104. With the
winning start and floor, warmup 5 and decay 25 beat the next schedule by 0.067
score, well outside the 0.02 tie threshold. Relative to epsilon disabled, the
winner lost 0.00308 mean HV on C1355 but gained 0.01276 on dalu; its strength is
cross-circuit robustness rather than universal dominance. Its best observed
size/depth reductions were 23.41%/34.62% on C1355 and 32.90%/14.29% on dalu.

The optimizer-health caveat remains: this result tunes exploration for the
pinned descriptive Adam control and does not establish that the underlying
GFlowNet optimizer is itself optimal.

#### Final decision

Carry `tb.exploration_epsilon_enabled=true`,
`tb.exploration_epsilon_start=0.10`, `tb.exploration_epsilon_end=0`,
`tb.exploration_warmup_episodes=5`, and
`tb.exploration_decay_episodes=25` into confirmation against epsilon disabled.
Zero is the natural lower boundary for the epsilon floor, so no lower edge
extension is possible.

### REINFORCE

#### Results

| Entropy beta | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.01** | 0.08033 +/- 0.00014 | 0.03442 +/- 0.00027 | **1.000** | **1.000** | 33.92% |
| 0.003333 | 0.08003 +/- 0.00050 | 0.03069 +/- 0.00247 | 0.944 | 0.892 | 36.19% |
| 0.0001 | 0.07713 +/- 0.00449 | 0.02764 +/- 0.00098 | 0.882 | 0.803 | 35.88% |
| 0.001 | 0.07158 +/- 0.00680 | 0.02905 +/- 0.00205 | 0.868 | 0.844 | 34.55% |
| 0 | 0.07702 +/- 0.00440 | 0.02300 +/- 0.00256 | 0.814 | 0.668 | 36.63% |
| 0.03 | 0.06583 +/- 0.00835 | 0.02313 +/- 0.01173 | 0.746 | 0.672 | 27.36% |

#### Analysis

Beta `0.01` achieved the highest mean hypervolume on both circuits, so its
score and minimum relative score are both 1. It improved mean HV over the
zero-entropy control by 0.00331 on C1355 and 0.01142 on dalu. The larger `0.03`
neighbor degraded both reliability and product improvement, while the smaller
`0.003333` coefficient remained competitive but was 0.056 score behind the
winner. The winner's best size/depth reductions were 23.41%/34.62% on C1355
and 26.55%/14.29% on dalu.

#### Final decision

Carry `entropy_beta=0.01` into confirmation against `entropy_beta=0`. The
winner is bracketed by tested lower and higher values and is not an unresolved
grid-edge choice.

### PPO

#### Results

| Entropy beta | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.03** | 0.07764 +/- 0.00482 | 0.03517 +/- 0.00314 | **0.985** | **0.969** | 37.40% |
| 0.003333 | 0.08010 +/- 0.00067 | 0.03202 +/- 0.00019 | 0.955 | 0.910 | 38.01% |
| 0.01 | 0.07705 +/- 0.00443 | 0.03059 +/- 0.00654 | 0.916 | 0.870 | 37.18% |
| 0 | 0.07962 +/- 0.00004 | 0.02563 +/- 0.00527 | 0.861 | 0.729 | 36.90% |
| 0.001 | 0.07982 +/- 0.00027 | 0.02449 +/- 0.00346 | 0.846 | 0.696 | 37.50% |
| 0.0001 | 0.07959 +/- 0.00000 | 0.02394 +/- 0.00280 | 0.837 | 0.681 | 37.33% |

#### Analysis

Beta `0.03` won by improving dalu substantially. Relative to beta zero it lost
0.00198 mean HV on C1355 but gained 0.00954 on dalu, increasing the normalized
two-circuit score by 0.123. Beta `0.003333` had the highest C1355 HV and product
improvement, but its score was 0.029 below `0.03`, just outside the 0.02
practical tie band. The winner's best size/depth reductions were
23.41%/34.62% on C1355 and 25.60%/17.14% on dalu.

Beta `0.03` is the largest tested value. The protocol requires one additional
neighbor in the winning direction when a selected value is at a grid edge, so
the present sweep does not yet bracket the PPO optimum.

#### Final decision

The best observed setting is `algorithm.ppo.entropy_beta=0.03`. Do not freeze
it for the final benchmark yet: first add one declared higher-beta screening
point, rerun selection, and then confirm the resulting choice against beta
zero. Until that edge check is complete, `0.03` is the provisional
confirmation candidate.

### DRiLLS-A2C

#### Results

| Entropy beta | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.0003** | 0.07794 +/- 0.00428 | 0.02422 +/- 0.00527 | **0.972** | **0.943** | 27.44% |
| 0.0001 | 0.07479 +/- 0.00831 | 0.02206 +/- 0.00433 | 0.909 | 0.859 | 26.03% |
| 0.00003333 | 0.06298 +/- 0.00729 | 0.02568 +/- 0.00697 | 0.904 | 0.808 | 27.82% |
| 0.01 | 0.06596 +/- 0.01117 | 0.02164 +/- 0.00398 | 0.844 | 0.843 | 25.12% |
| 0.001 | 0.06303 +/- 0.00735 | 0.01858 +/- 0.01421 | 0.766 | 0.724 | 27.05% |
| 0 | 0.07474 +/- 0.00414 | 0.01084 +/- 0.00791 | 0.690 | 0.422 | 26.35% |

#### Analysis

The coarse `1e-4` winner improved further at its upper neighbor `3e-4`.
Although `3.333e-5` had slightly higher dalu HV and product improvement,
`3e-4` was much stronger on C1355 and had the best balanced score. Its 0.062
lead over `1e-4` exceeds the practical tie threshold. Against beta zero it
gained 0.00321 mean HV on C1355 and 0.01339 on dalu. Its best size/depth
reductions were 23.41%/34.62% on C1355 and 30.93%/14.29% on dalu.

#### Final decision

Carry `algorithm.drills.entropy_beta=0.0003` into confirmation against beta
zero. The winner lies between tested `1e-4` and `1e-3` values and therefore is
adequately bracketed.

### PCN

#### Results

With fallback noise disabled, the seed-count comparison was:

| Random seed episodes | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **64** | 0.02406 +/- 0.01576 | 0.00245 +/- 0.00346 | **0.901** | **0.802** | 10.50% |
| 32 | 0.03002 +/- 0.02122 | 0.00000 +/- 0.00000 | 0.500 | 0.000 | 20.58% |
| 16 | 0.00682 +/- 0.00943 | 0.00000 +/- 0.00000 | 0.114 | 0.000 | 10.87% |

At 64 seed episodes, the fallback-noise comparison was:

| Noise scale, minimum sigma | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.10, 0.01** | 0.02406 +/- 0.01576 | 0.00245 +/- 0.00346 | **1.000** | **1.000** | 11.79% |
| 0, 0 | 0.02406 +/- 0.01576 | 0.00245 +/- 0.00346 | 1.000 | 1.000 | 10.50% |
| 0.01, 0.01 | 0.02106 +/- 0.01186 | 0.00245 +/- 0.00346 | 0.938 | 0.875 | 15.38% |
| 0.05, 0.01 | 0.02096 +/- 0.01173 | 0.00000 +/- 0.00000 | 0.436 | 0.000 | 7.18% |

The top-two-by-top-two interaction check produced:

| Seed episodes; noise scale, sigma | C1355 HV | dalu HV | Score | Min | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| **64; 0.10, 0.01** | 0.02406 +/- 0.01576 | 0.00245 +/- 0.00346 | **0.901** | **0.802** | 11.79% |
| 64; 0, 0 | 0.02406 +/- 0.01576 | 0.00245 +/- 0.00346 | 0.901 | 0.802 | 10.50% |
| 32; 0, 0 | 0.03002 +/- 0.02122 | 0.00000 +/- 0.00000 | 0.500 | 0.000 | 20.58% |
| 32; 0.10, 0.01 | 0.01938 +/- 0.01813 | 0.00000 +/- 0.00000 | 0.323 | 0.000 | 15.24% |

#### Analysis

Sixty-four seed episodes were selected because they were the only seed-count
setting to achieve positive mean hypervolume on dalu. The 32-seed control had
higher C1355 HV and product improvement but a zero minimum circuit score.

At 64 seeds, scale `0.10` and no fallback noise had identical per-circuit HV,
selection score, and minimum score. The declared tie-break therefore moved to
mean product improvement, where `0.10` scored 11.79% versus 10.50%; this is the
sole basis for choosing fallback noise. The interaction check also shows that
scale `0.10` is harmful at 32 seeds, so the noise decision must not be detached
from the selected seed count. The selected configuration's best size/depth
reductions were 23.41%/19.23% on C1355 and 20.20%/5.71% on dalu.

Both 64 seed episodes and noise scale `0.10` are upper boundaries of their
tested grids. Under the protocol's edge rule, the PCN optimum is not yet
bracketed and the exact-HV tie makes the evidence for fallback noise weak.

#### Final decision

The best observed joint setting is `pcn.random_seed_episodes=64`,
`pcn.target_noise_scale=0.10`, and `pcn.target_min_sigma=0.01`. Treat it as
provisional: declare and test one adjacent higher seed count and one adjacent
higher noise scale before confirmation. After that edge check, compare the
resulting selection against the explicit control of 32 seed episodes with
scale and minimum sigma both zero, using PCN `target` sampling mode throughout.

### Confirmation gate

The completed screens nominate GFlowNet `0.10 -> 0` with schedule `(5,25)`,
REINFORCE beta `0.01`, PPO beta `0.03`, DRiLLS-A2C beta `0.0003`, and PCN
`(64,0.10,0.01)`. GFlowNet, REINFORCE, and DRiLLS-A2C are ready for the
predeclared ten-seed confirmation. PPO and PCN require their grid-edge checks
first. None of these settings should be described as a confirmed final winner
until its 800-trajectory paired comparison and deterministic bootstrap report
have completed.
