# Experiment 4: separate `logZ` learning rate

This experiment is an isolated factorial study of `logZ` initialization and
learning rate. It does not change the production TB trainer or the completed
Experiment 3 artifact schema.

## Design

The `bc0` screen runs both `z0` and `zcal` at rates `0.003`, `0.01`, `0.03`,
and `0.1`, seeds 0--1, through 800 training trajectories. The policy learning
rate remains `0.001`. Every run uses the Experiment 3 calibration mechanics:
64 cached calibration trajectories, 16 collection-order minibatches, and 736
subsequent on-policy trajectories. Milestones are 200, 400, and 800.

Only the 800-trajectory health gates reject a cell. Earlier gates are recorded
as diagnostics. A cell is also rejected for persistent target-gap oscillation
or a median fixed-uniform bias fraction more than 1.1 times its
same-initialization rate-`0.01` control. At most two cells advance to `dalu`,
seeds 0--2.

## Run one cell

```bash
python -m src.experiments.tb_logz_learning_rate run \
  --initialization zcal \
  --log-z-learning-rate 0.003 \
  --config-name tb_zhuDOP \
  --circuit bc0 \
  --seed 0 \
  --output-dir /path/to/runs/zcal/rate_0p003/bc0/seed_0 \
  --max-trajectories 800 \
  --schedule-trajectories 800 \
  --milestones 200 400 800 \
  --device cuda
```

Resume with the same arguments plus `--resume-checkpoint`. Resume rejects a
different initialization, rate, scientific configuration, source tree, or
cache checksum.

The stable rate directory names are `rate_0p003`, `rate_0p01`, `rate_0p03`,
and `rate_0p1`.

## Screen report

After all 16 `bc0` runs complete:

```bash
python -m src.experiments.tb_logz_learning_rate screen-report \
  --runs-root /path/to/screen/runs \
  --output-dir /path/to/screen/report
```

The authoritative output is `decision_summary.json`.
`confirmation_candidates.json` is the machine-readable input to the `dalu`
stage. The report first validates the complete matrix, scientific fingerprints,
source checksum, initial-policy pairing, fixed-validation sequences,
calibration sequences, milestones, checkpoints, counters, and trajectory
sources.

Within each initialization, rate `0.01` is preferred when it is healthy and
both its mean absolute target gap and mean bias fraction lie within one paired
standard error of that initialization's best healthy rate. Remaining cells are
ranked by target gap, bias, and then archive hypervolume.

## Final report and Experiment 3 controls

Run non-`0.01` candidates on `dalu` using the same directory convention. A
selected `0.01` cell is not rerun: its Experiment 3 artifact is its
confirmation. Then run:

```bash
python -m src.experiments.tb_logz_learning_rate final-report \
  --screen-runs-root /path/to/screen/runs \
  --confirmation-runs-root /path/to/confirmation/runs \
  --experiment3-runs-root /path/to/experiment3/runs \
  --candidates-manifest /path/to/screen/report/confirmation_candidates.json \
  --output-dir /path/to/final/report
```

Experiment 3 `dalu` artifacts are accepted only as rate-`0.01` controls. Their
source checksum may differ, but the report requires compatible scientific
settings. For a new non-`0.01` `dalu` run it also requires the exact same
pre-calibration policy, fixed-sequence, and calibration-sequence checksums as
the paired Experiment 3 control.

Both report stages distinguish an incomplete matrix (exit 2), a corrupt or
incompatible artifact (exit 1), and a completed scientific rejection (exit 3).
They write decision JSON, Markdown, seed and update tables, health and
oscillation tables, rankings, plots, and a phase ledger. No unhealthy fallback
is selected.

## Martin execution order

The `myhpc` project is `gflowcircuit-gfn-logz-learning-rate`. Execute its jobs
in order: preflight, `bc0` screen, screen report, `dalu` confirmation, and final
report. The confirmation array reads the candidate manifest, so it must not be
submitted before the screen report succeeds. Review, commit, and push both the
project and scripts repositories before synchronization or submission.
