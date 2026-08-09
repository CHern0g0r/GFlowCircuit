# Experiment 5: trajectory-budget selection

## Scientific status and fixed settings

Experiment 5 tests whether a common finite training-trajectory budget is enough
for both `bc0` and `dalu`. It uses the best practical cross-circuit control from
Experiment 4:

- calibrated initialization (`zcal`);
- `logZ` learning rate `0.01`;
- policy learning rate `0.001`.

This is a least-bad baseline, not a validated Experiment 4 winner. It was
persistently oscillatory and had an isolated `dalu` health failure. Experiment
5 may select a budget only if its own health gates pass; the control's name or
endpoint performance cannot waive a failed gate.

Each circuit uses seeds 0--4. A seed is one uninterrupted logical training run
through 6,400 unique training trajectories, with checkpoints at 200, 400, 800,
1,600, 3,200, and 6,400. Separate runs at individual budgets are invalid.

The first 64 trajectories are cached calibration trajectories. They count
toward the environment budget and are optimized once in 16 collection-order
minibatches of four. The runner then collects 6,336 on-policy trajectories,
for 6,400 trajectory presentations and 1,600 optimizer updates in total.

The Experiment 4 epsilon schedule is retained: epsilon starts at `0.5`, remains
there for the 20-update warm-up, decays to `0.01` over the original
800-trajectory schedule, and stays clamped at `0.01` thereafter. Stretching the
decay across 6,400 trajectories would change the Experiment 4 control and is
not allowed.

## Run interface and artifacts

Run one cell with:

```bash
python -m src.experiments.tb_trajectory_budget run \
  --circuit bc0 \
  --seed 0 \
  --output-dir /path/to/runs/bc0/seed_0 \
  --max-trajectories 6400 \
  --schedule-trajectories 800 \
  --milestones 200 400 800 1600 3200 6400 \
  --device cuda
```

The public runner accepts only circuits `bc0` and `dalu`, seeds 0--4, and the
fixed scientific budget above. Its schema and scientific fingerprint record
the experiment identity, initialization, both learning rates, maximum budget,
epsilon schedule, complete milestone list, model/environment/reward settings,
calibration protocol, and evaluation sizes.

Resume by supplying the same command plus `--resume-checkpoint`. Resume rejects
a different source tree, fingerprint, maximum budget, schedule, milestone list,
fixed-validation cache, or calibration cache. Checkpoints store policy and
optimizer state, archive state, every RNG stream, counters, completed
milestones, and provenance.

At every milestone the runner:

1. rescores the same 256 cached legal fixed-uniform sequences;
2. samples 128 external-epsilon-zero fresh-policy trajectories;
3. samples 50 external-epsilon-zero search trajectories and evaluates nested
   prefixes at `N = 1, 2, 5, 10, 20, 50`;
4. writes validation and best-of-N tables and a resumable checkpoint.

Initialization, training actions, circuit selection, replay, fixed validation,
fresh validation, and search use separately derived RNG streams. Evaluation
must neither mutate model parameters nor advance the training-action stream.
Fixed, fresh, and search samples never enter the training archive or training
trajectory counter.

## Aggregate validation and decision

After all ten cells finish, run:

```bash
python -m src.experiments.tb_trajectory_budget report \
  --runs-root /path/to/runs \
  --output-dir /path/to/report
```

Before analysis, the report requires the exact two-circuit, five-seed matrix.
It validates identities, source and scientific fingerprints, seed-paired
initial policies, caches and checksums, all milestone tables and checkpoints,
1,600 ordered updates, the `0.01` terminal epsilon, counters, and trajectory
source counts. Exit codes are:

- `0`: a common budget was selected;
- `1`: a corrupt, incompatible, or otherwise invalid artifact;
- `2`: an incomplete run matrix;
- `3`: a complete scientific rejection with no selectable budget.

For each circuit, consider `B` in `200, 400, 800, 1,600, 3,200` and compare it
with the already completed `2B` milestone. For each validation stratum and seed,
define centered-RMS reduction as

```text
(RMS_B - RMS_2B) / max(abs(RMS_B), 1e-12).
```

A candidate is eligible only when all of the following strict inequalities
hold:

- every seed passes every health gate on fixed-uniform and fresh-on-policy at
  both `B` and `2B`;
- separately for each stratum, the median RMS reduction is below `0.05` and the
  upper endpoint of its deterministic 95% paired-bootstrap interval is below
  `0.10` (10,000 seed resamples);
- `abs(mean(HV_2B) - mean(HV_B)) < 0.005`, using per-seed training-archive
  hypervolume;
- `(mean(AUC_2B) - mean(AUC_B)) / max(abs(mean(AUC_B)), 1e-12) < 0.05`, using
  nested best-of-N log2-budget AUC.

The per-circuit candidate is the smallest eligible `B`. The common `B*` is the
larger circuit candidate. A 6,400-trajectory checkpoint is confirmation-only:
it cannot be selected without a completed 12,800-trajectory successor. If
either circuit has no candidate, reject the finite-budget-within-cap hypothesis
and extend the cap or run Experiment 6 before repeating the entire curve.

The report writes `decision_summary.json`, `decision_report.md`, seed/milestone,
health, paired-budget, bootstrap, and best-of-N CSVs, plus plots for health,
centered RMS, training-archive hypervolume, and best-of-N AUC.

## Martin execution

The immutable `myhpc` project is
`gflowcircuit-gfn-trajectory-budget-selection`. Its canonical remote roots are:

```text
/shared/home/fedor.chernogorskii/agent/code/gflowcircuit-gfn-trajectory-budget-selection
/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-trajectory-budget-selection
```

Execute three stages in order:

1. `tb_trajectory_budget_preflight_v1.slurm` runs an 800-trajectory timing and
   infrastructure preflight for both circuits. Require successful CUDA use,
   projected 6,400-trajectory runtime below 66 hours, and host RSS below
   32 GiB before continuing.
2. `tb_trajectory_budget_train_v1.slurm` runs the ten-cell array with at most
   four concurrent tasks. Each task requests one whole GPU, 32 GiB RAM, eight
   CPUs, and 72 hours.
3. `tb_trajectory_budget_report_v1.slurm` is CPU-only and first verifies all ten
   successful run summaries before invoking the aggregate report.

All jobs use the guaranteed Michalis allocation, exclude `mbz-titan-3`, activate
the cluster `ospiel` environment, and write only below their job artifact root.
The preflight is infrastructure-only and never enters the scientific matrix.

After reviewing, committing, and pushing both repositories, run each stage via
`myhpc` and wait for it to complete before starting the next:

```bash
MYHPC=/Users/fedor.chernogorskii/.local/bin/myhpc
SCRIPTS=/Users/fedor.chernogorskii/workspace/local/scripts
PROJECT=gflowcircuit-gfn-trajectory-budget-selection

"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_trajectory_budget_preflight_v1.slurm"

"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_trajectory_budget_train_v1.slurm"

"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_trajectory_budget_report_v1.slurm"
```

Do not submit the full array when the preflight limit fails. Do not synchronize
or submit from dirty, unpushed, detached, or upstream-divergent repositories,
and do not overwrite an existing immutable remote project tree.
