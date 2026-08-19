# Baseline training hyperparameter and trajectory-budget protocol

## Objective and scope

Select training hyperparameters and a sample-efficient trajectory budget for
REINFORCE, DRiLLS-A2C, and PPO. The campaign does not tune GFlowNet and does not
borrow the unresolved GFlowNet budget from `gfn_health`.

The scientific questions are:

1. Which compact non-exploration training profile is strongest for each
   baseline at 800 trajectories?
2. For each baseline, what is the smallest trajectory budget that is within 2%
   of its best observed 50-sample hypervolume and has no established gain at
   the next doubling?
3. Do the chosen profile and plateau survive ten paired seeds?
4. How do all three baselines compare at the median confirmed budget?

An algorithm that reaches stronger hypervolume with fewer circuit trajectories
is more **sample efficient**. It is not necessarily faster in wall-clock or
compute terms. Every report therefore includes optimizer updates, wall time,
and GPU-hours.

## Fixed experimental contract

| Field | Value |
| --- | --- |
| Circuits | `C1355`, `dalu` |
| Structural roles | smaller structured logic; medium ALU/datapath |
| Horizon | 20 synthesis actions |
| Actions | 0 through 6 |
| Objective | normalized size/depth minimization |
| HV reference | `(1,1)` |
| Final samples | exactly 50 per trained seed |
| Evaluation seed | 42, paired across configurations |
| Screen seeds | 0, 1, 2 |
| Confirmation seeds | 0 through 9 |
| Model, encoder, reward | existing Zhu diff-of-product baseline contract |

Exploration is frozen from `mds/exploration_tuning.md` at REINFORCE beta
`0.01`, PPO beta `0.03`, and DRiLLS-A2C beta `0.0003`. The PPO value was a
provisional grid-edge winner; it is frozen here by explicit user decision and
is not retuned in this campaign.

## Compact profile screens

Every profile is an explicit complete override of the screened fields. This
prevents mutable defaults from silently changing a result.

### REINFORCE

The control uses policy/value learning rates `8e-4/3e-3`, gamma `0.9`, and no
return normalization or clipping. One-factor profiles test policy rates
`3e-4` and `2e-3`, value rates `1e-3` and `1e-2`, gamma `0.99`, and normalized
returns.

### DRiLLS-A2C

The control uses learning rate `1e-3`, gamma `0.9`, value-loss coefficient
`0.5`, and raw advantages. Profiles test rates `3e-4` and `3e-3`, value-loss
coefficients `0.25` and `1.0`, gamma `0.99`, and advantage normalization.

### PPO

The control uses learning rate `1e-3`, gamma `0.9`, 20 PPO epochs, minibatches
of 64, clipping `0.2`, value coefficient `0.5`, normalized advantages, and GAE
lambda `0.95`. Profiles test rates `3e-4` and `3e-3`, 10 and 40 epochs,
clipping `0.1` and `0.3`, and gamma `0.99`.

This is a compact one-factor screen, not a claim that the full interaction
surface was exhaustively optimized. The top two complete profiles advance so
their budget curves reveal major budget-dependent reversals.

## Selection rules

### Profile screen

For each circuit, divide a profile's mean per-seed hypervolume by the best
profile mean on that circuit. Average the two ratios. Within a 0.02 score band,
prefer the higher minimum circuit-relative score, then product improvement.
Advance two profiles per algorithm.

### Budget curve

Train both finalists independently at 200, 400, 800, 1,600, and 3,200
trajectories. A budget `B` is eligible only when:

- its two-circuit normalized score is at least 98% of the global best score;
- a paired bootstrap over circuit/training-seed values does not establish that
  `2B` is better (`CI95_low <= 0`); and
- `2B` was actually run.

If no budget is eligible, run both finalists at 6,400. If no budget through
3,200 is then eligible, report `budget_unresolved_at_6400`; do not choose 6,400
without a 12,800 successor.

### Ten-seed confirmation

For each algorithm, run or reuse three arms on both circuits and seeds 0--9:

1. selected profile at selected `B`;
2. runner-up profile at selected `B`;
3. selected profile at `2B`.

The profile gate passes when the selected score is no more than 0.02 below the
runner-up and is not significantly worse. The budget gate repeats the 98% and
no-significant-successor-gain rules. Both gates must pass.

### Common budget comparison

Take the median of the three confirmed budgets and round upward to the tested
grid. Run every selected baseline profile at that exact trajectory count with
ten paired training seeds and 50 post-training samples. Reuse matching
confirmation tasks by immutable task identity.

## Outputs and stopping rules

Each stage produces `stage_manifest.json`, `selection.json`, `report.md`,
`per_seed_metrics.csv`, `setting_summary.csv`, `pooled_front.csv`, and
`discovery_curves.csv`. The smoke stage also produces
`runtime_projection.json`.

Stop when any of these occurs:

- a task fails resolved-config, checkpoint, report, or sample-count validation;
- a dependency comes from another protocol hash or project commit;
- a baseline remains budget-unresolved after 6,400;
- a ten-seed profile or budget gate fails; or
- smoke projections show that a production wrapper must be split to stay
  within Martin's 72-hour maximum.

No stage submits its successor, and preparation does not synchronize or submit
any Martin job.
