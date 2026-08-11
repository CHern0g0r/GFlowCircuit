# Experiment 5.5: batch-size influence

## Scientific configuration

Experiment 5.5 compares optimizer batch sizes `1, 4, 8, 16, 32` at a fixed
budget of 1,600 complete training trajectories. It runs `bc0` and `dalu`, seeds
0--4, for 50 cells in total. The optimizer control is calibrated initialization
(`zcal`), policy learning rate `0.001`, and `logZ` learning rate `0.01`, with no
replay. This is the unresolved descriptive control used by Experiment 5, not a
healthy optimizer selected by that experiment.

The first 64 trajectories are the paired calibration set and are optimized
once in collection order. Checkpoints are written at 256, 512, 1,024, and 1,600
trajectories. These budgets and the calibration size are divisible by every
candidate. The variants perform 1,600, 400, 200, 100, and 50 optimizer updates,
respectively, without extra epochs or presentations.

The epsilon schedule is paired by trajectory rather than optimizer update. It
preserves the original batch-four behavior exactly: each consecutive group of
four trajectories shares epsilon, trajectories 1--80 use `0.5`, epsilon decays
through trajectory 800, and later trajectories use `0.01`. A large optimizer
batch can therefore contain several per-trajectory epsilon values. TB always
uses learned-policy `log P_F`, never behavior-mixture probabilities.

## Interfaces and artifacts

Run one scientific cell with:

```bash
python -m src.experiments.tb_batch_size_influence run \
  --batch-size 8 --circuit bc0 --seed 0 \
  --output-dir /path/to/runs/batch_8/bc0/seed_0 \
  --device cuda
```

Resume by adding `--resume-checkpoint` with a checkpoint from the same batch,
circuit, seed, source tree, scientific fingerprint, caches, and maximum budget.
The runner rejects any mismatch.

After all waves complete, aggregate repeatable run roots with:

```bash
python -m src.experiments.tb_batch_size_influence report \
  --runs-root /path/to/train-a/runs \
  --runs-root /path/to/train-b/runs \
  --runs-root /path/to/train-c/runs \
  --output-dir /path/to/report
```

The report requires exactly one copy of every batch/circuit/seed cell. It
validates source and configuration fingerprints, initial parameter pairing,
fixed and calibration sequence pairing, caches, checkpoints, counters, exact
epsilon vectors, evaluation isolation, resource records, and trajectory-source
counts. Exit codes are `0` for a selected alternative, `1` for invalid
artifacts or execution failure, `2` for an incomplete matrix, and `3` for a
complete scientific rejection that retains batch size four.

At 1,600 trajectories, each alternative is compared with batch four using
10,000 deterministic paired-mean bootstrap resamples. A candidate must pass all
health gates on both circuits, improve fixed-uniform centered RMS by at least
5% or archive hypervolume by at least `0.005` on at least one circuit, and be
non-inferior on both metrics and both circuits. Among qualifiers, select the
smallest batch within one paired standard error of the best equal-weight
cross-circuit hypervolume. Any selected change invalidates the prior `B*` and
requires a new Experiment 5 curve.

## Martin execution

The immutable `myhpc` project is
`gflowcircuit-gfn-batch-size-influence`. Its remote roots are:

```text
/shared/home/fedor.chernogorskii/agent/code/gflowcircuit-gfn-batch-size-influence
/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-batch-size-influence
```

After reviewing, committing, and pushing both the project and scripts
repositories, execute the five stages in order. Wait for each stage to finish
before starting the next so the 20-job queue limit is never exceeded.

```bash
MYHPC=/Users/fedor.chernogorskii/.local/bin/myhpc
SCRIPTS=/Users/fedor.chernogorskii/workspace/local/scripts
PROJECT=gflowcircuit-gfn-batch-size-influence

"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_batch_size_preflight_v1.slurm"
"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_batch_size_train_a_v1.slurm"
"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_batch_size_train_b_v1.slurm"
"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_batch_size_train_c_v1.slurm"
"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_batch_size_report_v1.slurm"
```

The 10-cell preflight covers every batch/circuit combination at seed zero and
128 trajectories. Each training wave refuses to start unless all preflights
completed on CUDA, project below 10.5 hours for 1,600 trajectories, and remain
below 32 GiB peak host RSS. Training waves contain 20, 20, and 10 cells and cap
concurrency at four. Canonical outputs remain below each job's Martin artifact
root.
