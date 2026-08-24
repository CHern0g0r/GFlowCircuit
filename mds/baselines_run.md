# Baseline experiment matrix

## Scope

This campaign compares three policy-gradient baselines on nine circuits:
REINFORCE, DRiLLS-A2C, and PPO. It will run on the Martin SLURM cluster as 27
jobs, with one job for each method/circuit pair. Each job trains ten models
using training seeds 0 through 9.

All methods use 800 training trajectories per trained seed. PPO deliberately
uses 800 trajectories for this campaign, although the hyperparameter search
selected 400 as its formal plateau.

## Shared experiment definition

The three methods share the following environment, model, reward, and
evaluation contract:

| Component | Setting |
| --- | --- |
| Synthesis horizon | 20 steps |
| Available actions | `[0, 1, 2, 3, 4, 5, 6]` |
| Encoder | Hybrid Zhu encoder with concatenated graph and vector branches |
| Graph branch | Zhu GCN; 4 layers; hidden dimension 12; output dimension 4; mean pooling; no dropout |
| Vector branch | Zhu10 features; output dimension 28; no hidden layers |
| Policy head | MLP with one hidden layer of width 32 |
| Value model | Zhu10 input and one hidden layer of width 32 |
| Reward | Dense difference of normalized size-depth product, with `c_size = 1` and `c_depth = 1` |
| Reference baseline | `zhu_resyn2`, scale 1.0 |
| Training seeds | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` |
| Evaluation seeds per trained seed | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` |
| Evaluation sample budgets | `[10, 50, 100, 200]`, as nested prefixes of one ordered 200-sample sequence |
| Unique samples per evaluation seed | 200 |
| Unique samples per trained model | 2,000 |
| Trained models per method/circuit | 10 |
| Unique evaluation samples per method/circuit | 20,000 |

Across the complete campaign, the 27 jobs train 270 models and generate
540,000 unique post-training evaluation samples. The smaller best-of-N budgets
reuse prefixes of the same samples and do not require additional rollouts.

## Methods

### REINFORCE

Configuration: `cfg/baseline_configs/reinforce.yaml`

| Setting | Value |
| --- | --- |
| Selected profile | `policy_lr_high` |
| Training budget | 800 episodes × 1 trajectory = 800 trajectories |
| Policy learning rate | `2e-3` |
| Value learning rate | `3e-3` |
| Discount factor | `0.9` |
| Entropy coefficient | `0.01` |
| Return normalization | Disabled |
| Policy gradient clipping | Disabled |
| Value gradient clipping | Disabled |
| Terminal-only reward | Disabled |

### DRiLLS-A2C

Configuration: `cfg/baseline_configs/drills.yaml`

| Setting | Value |
| --- | --- |
| Selected profile | `lr_low_value_high` |
| Training budget | 200 episodes × 4 trajectories = 800 trajectories |
| Learning rate | `3e-4` |
| Discount factor | `0.9` |
| Value-loss coefficient | `1.0` |
| Entropy coefficient | `0.0003` |
| Advantage normalization | Disabled |
| Gradient clipping | Disabled |
| Depth constraint ratio | `1.0` |

### PPO

Configuration: `cfg/baseline_configs/ppo.yaml`

| Setting | Value |
| --- | --- |
| Selected profile | `epochs_high` |
| Training budget | 200 iterations × 80 transitions ÷ 20 steps = 800 trajectories |
| Learning rate | `1e-3` |
| Discount factor | `0.9` |
| PPO epochs per iteration | 40 |
| Rollout length | 80 transitions (4 complete trajectories) |
| Minibatch size | 64 |
| Clipping epsilon | `0.2` |
| Value-loss coefficient | `0.5` |
| Entropy coefficient | `0.03` |
| Advantage normalization | Enabled |
| GAE lambda | `0.95` |
| Gradient clipping | Disabled |

## Circuits

| Circuit | Family | Format | Dataset configuration |
| --- | --- | --- | --- |
| `C1355` | MCNC combinational | BLIF | `cfg/data/zhu2020/C1355.yaml` |
| `C5315` | MCNC combinational | BLIF | `cfg/data/zhu2020/C5315.yaml` |
| `adder` | EPFL arithmetic | AIG | `cfg/data/epfl_arithmetic/adder.yaml` |
| `apex1` | MCNC combinational | BLIF | `cfg/data/zhu2020/apex1.yaml` |
| `bc0` | MCNC combinational | BLIF | `cfg/data/zhu2020/bc0.yaml` |
| `dalu` | MCNC combinational | BLIF | `cfg/data/zhu2020/dalu.yaml` |
| `k2` | MCNC combinational | BLIF | `cfg/data/zhu2020/k2.yaml` |
| `max` | EPFL arithmetic | AIG | `cfg/data/epfl_arithmetic/max.yaml` |
| `multiplier` | EPFL arithmetic | AIG | `cfg/data/epfl_arithmetic/multiplier.yaml` |

## Martin SLURM job matrix

Each cell below is one SLURM job. The method configuration is combined with
the circuit's dataset configuration at launch time. Every cell contains ten
training seeds and the evaluation contract defined above.

| Circuit | REINFORCE | DRiLLS-A2C | PPO |
| --- | --- | --- | --- |
| `C1355` | `reinforce_C1355` | `drills_C1355` | `ppo_C1355` |
| `C5315` | `reinforce_C5315` | `drills_C5315` | `ppo_C5315` |
| `adder` | `reinforce_adder` | `drills_adder` | `ppo_adder` |
| `apex1` | `reinforce_apex1` | `drills_apex1` | `ppo_apex1` |
| `bc0` | `reinforce_bc0` | `drills_bc0` | `ppo_bc0` |
| `dalu` | `reinforce_dalu` | `drills_dalu` | `ppo_dalu` |
| `k2` | `reinforce_k2` | `drills_k2` | `ppo_k2` |
| `max` | `reinforce_max` | `drills_max` | `ppo_max` |
| `multiplier` | `reinforce_multiplier` | `drills_multiplier` | `ppo_multiplier` |
