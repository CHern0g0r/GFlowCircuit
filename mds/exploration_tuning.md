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

The repository currently contains no exploration-specific launcher, SLURM
script, or final-sample aggregation script. `cfg/exp/exploration_tuning/README.md`
describes such scripts, but they are not present in this version. The commands
and analysis below therefore form the executable contract. Hydra could not be
composition-tested in the inspected shell because `hydra-core` was not
installed there; it is declared in `requirements.txt` and must be installed in
the experiment environment before the smoke test.

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
the same sweep. The optimizer-health gate in `mds/undertrained_gfn.md` must be
satisfied before using this experiment to draw conclusions about GFlowNet
exploration.

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
