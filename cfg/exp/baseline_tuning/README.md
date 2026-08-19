# Baseline training hyperparameter tuning

This campaign tunes only REINFORCE, DRiLLS-A2C, and PPO. GFlowNet training
hyperparameters and its unresolved trajectory budget are explicitly out of
scope.

The canonical specification is `protocol.yaml`; the executable interface is
`python -m src.baseline_tuning`. Every non-smoke trained model is sampled
exactly 50 times with evaluation seed 42. Training seed is the statistical
unit, and the same circuit/seed pairs are used for every comparison.

## Locked decisions

- Circuits: `C1355` (smaller structured logic) and `dalu` (medium ALU/datapath).
- Horizon/actions: 20 steps and actions `[0,1,2,3,4,5,6]`.
- Primary metric: mean per-training-seed strict normalized hypervolume with
  reference `(1,1)`.
- Secondary metrics: pooled hypervolume, training-discovery hypervolume,
  product improvement, size/depth reductions, diversity, wall time, GPU-hours,
  and optimizer updates.
- Exploration is frozen to the accepted screening values: REINFORCE `0.01`,
  PPO `0.03`, and DRiLLS-A2C `0.0003` entropy coefficients.
- Budget grid: 200, 400, 800, 1,600, and 3,200 trajectories. The 6,400 point is
  generated only when the first grid has no eligible plateau.
- A budget is eligible only when its two-circuit score is within 2% of the best
  observed score and the paired bootstrap does not establish an improvement at
  the next doubling. Therefore 6,400 cannot itself be selected without a
  successor; continued improvement at 6,400 is reported as unresolved.
- The final common budget is the median of the three confirmed budgets, rounded
  upward to the tested grid.

Equal trajectories measure sample efficiency. They do not imply equal
optimizer work: REINFORCE updates at every trajectory step, DRiLLS-A2C updates
once per four trajectories, and PPO performs repeated minibatch epochs. The
reports retain wall time, GPU-hours, and update counts to avoid calling sample
efficiency general training speed.

## Stage order

1. `smoke`
2. `screen_reinforce`, `screen_drills`, `screen_ppo`
3. `budget_reinforce`, `budget_drills`, `budget_ppo`
4. `extend_reinforce`, `extend_drills`, `extend_ppo`
5. `confirm_reinforce`, `confirm_drills`, `confirm_ppo`
6. `common_budget`

The three algorithm-specific stages at each numbered step are independent once
their own dependencies complete. Extension stages must always be invoked: they
become a zero-training forwarding stage when the 200--3,200 curve already
selects a budget.

## Commands

Protocol-only validation works without the training environment:

```bash
python -m src.baseline_tuning validate --stage smoke --skip-compose
```

The Martin jobs use the full validation path and abort if Hydra composition,
resolved overrides, checkpoints, reports, or exact sample counts differ:

```bash
python -m src.baseline_tuning run-stage \
  --stage smoke \
  --artifact-root /shared/home/fedor.chernogorskii/agent/art/gflowcircuit/gfc-base-smoke-v1 \
  --artifact-base /shared/home/fedor.chernogorskii/agent/art/gflowcircuit \
  --workers 4 \
  --project-commit PROJECT_COMMIT
```

Each task writes into an immutable `attempt_NNN` directory. Completed tasks are
validated and reused; failed or invalid attempts are never overwritten.
`selection.json` is the machine-readable gate for the next stage.

