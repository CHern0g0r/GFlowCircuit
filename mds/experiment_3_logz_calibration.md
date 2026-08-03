# Experiment 3: calibrated logZ initialization

This repository contains an isolated runner and report pipeline for Experiment
3 in `optimizer_experiment_protocol.md`. It does not change the production TB
trainer or the completed Experiment 2 artifact schema.

## One logical run

```bash
python -m src.experiments.tb_logz_calibration run \
  --variant zcal \
  --config-name tb_zhuDOP \
  --circuit bc0 \
  --seed 0 \
  --output-dir /path/to/runs/zcal/bc0/seed_0 \
  --max-trajectories 800 \
  --schedule-trajectories 800 \
  --milestones 200 400 800 \
  --device cuda
```

Use `--variant z0` for the paired control. A run first collects 64 cached
calibration trajectories at epsilon 0.5, initializes logZ, processes the cache
in 16 collection-order minibatches, and then collects the remaining 736
training trajectories. The calibration cache, fixed validation cache, metrics,
tables, checkpoints, resolved configuration, RNG metadata, and summary are
stored below the run directory.

Resume only from a checkpoint belonging to the same variant, circuit, seed,
source tree, and scientific configuration:

```bash
python -m src.experiments.tb_logz_calibration run \
  --variant zcal --circuit bc0 --seed 0 \
  --output-dir /path/to/runs/zcal/bc0/seed_0 \
  --max-trajectories 800 --milestones 200 400 800 --device cuda \
  --resume-checkpoint /path/to/runs/zcal/bc0/seed_0/checkpoints/trajectory_400.pt
```

## Aggregate report

After all 12 runs are complete:

```bash
python -m src.experiments.tb_logz_calibration report \
  --runs-root /path/to/runs \
  --output-dir /path/to/report \
  --expected-variants z0 zcal \
  --expected-circuits bc0 dalu \
  --expected-seeds 0 1 2
```

The report validates pairing and artifact integrity before applying the 200-
trajectory improvement gates, 800-trajectory non-harm gates, oscillation rule,
health gates, bootstrap comparison, and simplicity tie-break. Its authoritative
decision is `decision_summary.json`; CSV tables, plots, the Markdown report,
and `phase_ledger.csv` provide supporting evidence.

## Martin packaging

The corresponding `myhpc` project is
`gflowcircuit-gfn-logz-calibration` in the external scripts repository. Its
preflight, 12-task training array, and CPU report SLURM files are canonical.
Review, commit, and push both repositories before using `myhpc sync` or
`myhpc run`. Repository preparation does not submit a job.
