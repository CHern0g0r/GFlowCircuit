# Exploration-tuning pilot

This directory contains exploration-only Hydra fragments for a fast screening
experiment. The training protocol, circuit selection, and output paths are
specified explicitly by the accompanying SLURM scripts rather than by these
fragments.

## Experiment matrix

| Algorithm | Fragment | Exploration setting |
| --- | --- | --- |
| GFlowNet-TB | `gflownet/epsilon_none` | categorical policy sampling, epsilon disabled |
| GFlowNet-TB | `gflownet/epsilon_low` | epsilon 0.10 to 0.01, 5-episode warmup |
| GFlowNet-TB | `gflownet/epsilon_current` | epsilon 0.50 to 0.01, 20-episode warmup |
| REINFORCE | `reinforce/entropy_none` | entropy beta 0 |
| REINFORCE | `reinforce/entropy_low` | entropy beta 1e-3 |
| REINFORCE | `reinforce/entropy_high` | entropy beta 1e-2 |
| PPO | `ppo/entropy_none` | entropy beta 0 |
| PPO | `ppo/entropy_low` | entropy beta 1e-3 |
| PPO | `ppo/entropy_high` | entropy beta 1e-2 |
| DRiLLS-A2C | `drills/entropy_none` | entropy beta 0 |
| DRiLLS-A2C | `drills/entropy_low` | entropy beta 1e-3 |
| DRiLLS-A2C | `drills/entropy_high` | entropy beta 1e-2 |
| PCN | `pcn/no_fallback_noise` | 32 random seeds, zero fallback noise and minimum sigma |
| PCN | `pcn/balanced` | 32 random seeds, 0.05 fallback noise, 0.01 minimum sigma |
| PCN | `pcn/more_random_seeding` | 64 random seeds, 0.05 fallback noise, 0.01 minimum sigma |

For PCN, zero fallback noise does not suppress perturbations derived from
observed archive variance. It only removes the configured fallback when that
variance is very small.

## Pilot protocol and budgets

Each setting is evaluated on `C1355` and `dalu`, with three training seeds per
circuit. Every seed receives exactly 200 terminal training trajectories, uses a
20-action horizon and actions `[0, 1, 2, 3, 4, 5, 6]`, and is sampled 20 times
after training. Evaluation rollouts are not included in the training budget.

| Algorithm | Training calculation per seed |
| --- | --- |
| GFlowNet-TB | 50 episodes x 4 trajectories = 200 |
| REINFORCE | 200 episodes x 1 trajectory = 200 |
| PPO | 50 iterations x 80 transitions / 20 steps = 200 |
| DRiLLS-A2C | 50 episodes x 4 trajectories = 200 |
| PCN | 200 collected episodes = 200 |

Consequently, one setting script consumes `2 circuits x 3 seeds x 200 = 1,200`
training trajectories. `eval_every` is set to the final training iteration or
episode and `paper_mode.infer_rollouts=1` limits in-training evaluation cost.
The post-training sampling stage uses seed 42.

## Hydra usage

Append a fragment to the normal base configuration with a Hydra defaults-list
override. For example:

```bash
python -m src.run --config-name tb_zhuDOP \
  "+exp/exploration_tuning=gflownet/epsilon_low" \
  data=zhu2020/C1355
```

The SLURM scripts also pass all budget-sensitive values explicitly, including
the action set, horizon, seed count, evaluation cadence, and algorithm-specific
batch size. This avoids relying on mutable defaults.

## Evaluation and selection

For each algorithm and circuit, report the mean and standard deviation of
per-seed hypervolume, pooled hypervolume, nondominated-point count, distinct
endpoints per seed, mean product/QoR improvement, and the probability that a
sampled endpoint belongs to the pooled nondominated front.

Mean per-seed hypervolume is the primary criterion. Normalize each setting's
mean hypervolume by the best mean hypervolume for that algorithm on the same
circuit, then average the two normalized circuit scores. If two settings are
within 0.02, prefer the setting with the higher minimum normalized circuit
score. If still tied, use pooled hypervolume and then expected QoR.

This suite is a screening experiment, not a final algorithm comparison. Confirm
the selected setting and the no-explicit-exploration control later with the full
800-trajectory budget and at least 10 seeds.
