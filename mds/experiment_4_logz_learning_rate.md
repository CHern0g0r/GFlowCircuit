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

## Results

### Execution and validity

The `bc0` screen completed all 16 planned runs (two initializations, four
learning rates, and seeds 0--1). Every run reached 800 unique training
trajectories and 200 optimizer updates without a numerical failure. The
accounting checks also passed: each run used 64 calibration trajectories, 16
ordered calibration minibatches, and 736 later on-policy trajectories. The
report validated the scientific fingerprints, paired initial-policy checksum,
fixed-validation sequence, calibration cache and sequence, milestones, and
source tree before comparing cells.

The recorded code commit is
`e81649f752e93ae8cbb36bb2ad36213ea830b473`, the scientific source-tree hash is
`b7af20532f71811dbe97f240773f5e49a69923cfeebc9b10811baa4239b04ded`, and the
paired-configuration fingerprint is
`bdc97b03eab175dd3a82cd178ac422e1acc80b1acdc46e20440750006d8941c3`.

One screen task (`z0`, rate `0.03`, seed 0) initially failed before training
because its allocated shard did not expose CUDA. It was rerun successfully as
SLURM task `17411_4`; the resulting run passed the same pairing and artifact
checks as the other 15 cells. The screen-report job was `17442`. It exited with
code 3, the documented code for a completed scientific rejection, and produced
the decision `reject_no_healthy_screen_candidate`.

The authoritative remote artifacts are:

- screen runs:
  `/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-logz-learning-rate/tb-logz-learning-rate-screen-v1/runs`;
- screen report:
  `/shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-logz-learning-rate/tb-logz-learning-rate-screen-report-v1/report`;
- decision JSON: `decision_summary.json`;
- candidate manifest: `confirmation_candidates.json`.

The manifest is complete but contains no candidates. Consequently, no `dalu`
confirmation or final aggregation job was run: those stages require one or two
eligible screen cells and deliberately reject an empty manifest.

### Reading the checkpoint tables

The tables below report means over seeds 0 and 1. `Fixed gap` and `fresh gap`
are the absolute `logZ` target gaps on the fixed-uniform and fresh-on-policy
validation strata. `Fixed bias` and `fresh bias` are the fractions of TB MSE
explained by the squared global-offset bias. Lower is better for all four
columns. `F/U pass` gives the number of seeds passing every health gate in the
fixed/uniform and fresh/on-policy strata, respectively; each count is out of
two.

The relevant health limits are absolute target gap at most 0.5, bias fraction
at most 0.05, and standardized bias at most 0.25. The gates also check finite
values, probability normalization, illegal-action probability, gradient-tail
ratio, clipping, and policy collapse. None of the latter checks caused a
failure in this screen. Only the 800-trajectory rows are decisive; the 200 and
400 rows diagnose the learning trajectory.

#### `z0` initialization

| rate | trajectories | fixed gap | fixed bias | fresh gap | fresh bias | learned `logZ` | archive HV | F/U pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.003 | 200 | 41.9484 | 0.999365 | 40.8611 | 0.999482 | 0.1496 | 0.242422 | 0/0 |
| 0.003 | 400 | 42.0927 | 0.999083 | 40.3869 | 0.998890 | 0.2988 | 0.243830 | 0/0 |
| 0.003 | 800 | 42.7098 | 0.998138 | 38.9316 | 0.996771 | 0.5948 | 0.244134 | 0/0 |
| 0.01 | 200 | 41.6010 | 0.999353 | 40.5121 | 0.999471 | 0.4983 | 0.242422 | 0/0 |
| 0.01 | 400 | 41.3976 | 0.999053 | 39.6923 | 0.998852 | 0.9931 | 0.243830 | 0/0 |
| 0.01 | 800 | 41.3224 | 0.998070 | 37.5448 | 0.996447 | 1.9686 | 0.244134 | 0/0 |
| 0.03 | 200 | 40.6123 | 0.999316 | 39.5192 | 0.999441 | 1.4903 | 0.242422 | 0/0 |
| 0.03 | 400 | 39.3782 | 0.999028 | 37.7901 | 0.998732 | 2.9544 | 0.243830 | 0/0 |
| 0.03 | 800 | 37.4551 | 0.997767 | 33.7763 | 0.995878 | 5.7834 | 0.244002 | 0/0 |
| 0.1 | 200 | 37.1793 | 0.999191 | 36.1146 | 0.999354 | 4.9137 | 0.242442 | 0/0 |
| 0.1 | 400 | 32.7652 | 0.998653 | 31.2301 | 0.998195 | 9.5329 | 0.243708 | 0/0 |
| 0.1 | 800 | 25.2500 | 0.995816 | 22.2216 | 0.992227 | 17.6894 | 0.243881 | 0/0 |

The `z0` runs show the expected rate ordering in the learned scalar. At 200,
400, and 800 trajectories, rate `0.003` reaches approximately 0.15, 0.30, and
0.59; rate `0.1` reaches 4.91, 9.53, and 17.69. A larger learning rate therefore
moves `logZ` toward its target substantially faster. It does not, however,
bring the scalar close enough within the fixed 800-trajectory budget. Even the
best `z0` cell, rate `0.1`, retains mean final gaps of 25.25 and 22.22, roughly
45--50 times the allowed 0.5 gap.

All `z0` bias fractions remain above 0.99. Thus almost the entire TB error is a
global normalization-offset error rather than centered trajectory-to-trajectory
variation. Every `z0` seed and stratum fails target gap, bias fraction, and
standardized bias at every checkpoint. The absence of sign oscillation in
these runs is not evidence of health: it reflects slow one-directional motion
toward a distant target.

#### `zcal` initialization

| rate | trajectories | fixed gap | fixed bias | fresh gap | fresh bias | learned `logZ` | archive HV | F/U pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.003 | 200 | 0.0712 | 0.077565 | 0.1149 | 0.154952 | 41.6305 | 0.241399 | 1/1 |
| 0.003 | 400 | 0.0412 | 0.089499 | 0.0623 | 0.135366 | 41.5943 | 0.241976 | 1/1 |
| 0.003 | 800 | 0.0054 | 0.000723 | 0.0120 | 0.003906 | 41.5441 | 0.243698 | 2/2 |
| 0.01 | 200 | 0.0532 | 0.040069 | 0.1043 | 0.126002 | 41.6164 | 0.241399 | 1/1 |
| 0.01 | 400 | 0.0103 | 0.002699 | 0.0269 | 0.021372 | 41.5463 | 0.241976 | 2/2 |
| 0.01 | 800 | 0.0104 | 0.004726 | 0.0120 | 0.004955 | 41.5311 | 0.243526 | 2/2 |
| 0.03 | 200 | 0.0446 | 0.019259 | 0.0265 | 0.007349 | 41.5213 | 0.241399 | 2/2 |
| 0.03 | 400 | 0.0085 | 0.002870 | 0.0349 | 0.039008 | 41.5609 | 0.241976 | 2/1 |
| 0.03 | 800 | 0.0246 | 0.014903 | 0.0208 | 0.011952 | 41.5144 | 0.243526 | 2/2 |
| 0.1 | 200 | 0.1469 | 0.183494 | 0.0671 | 0.065464 | 41.4239 | 0.241611 | 1/1 |
| 0.1 | 400 | 0.0268 | 0.016915 | 0.0686 | 0.094304 | 41.5781 | 0.242118 | 2/0 |
| 0.1 | 800 | 0.0146 | 0.006829 | 0.0078 | 0.001413 | 41.5301 | 0.243617 | 2/2 |

Calibration immediately places `logZ` in the correct numerical region. At 200
trajectories, every `zcal` rate already has a mean absolute gap below 0.15 in
both strata. Early health failures are instead caused by bias fraction and
standardized bias. Rate `0.03` is the only cell whose four seed/stratum gates
all pass at 200, but this early result is diagnostic and does not override the
later stability checks.

At 400 trajectories, rate `0.01` is healthy in all four seed/stratum checks.
Rate `0.003` still fails the bias gates for one seed in each stratum, rate
`0.03` fails them for one fresh-on-policy seed, and rate `0.1` fails both fresh
seeds. Again, these are diagnostic snapshots. By 800 trajectories all four
rates pass all four endpoint health gates. Their final mean gaps are at most
0.0246 and their mean bias fractions are at most 0.014903, comfortably inside
the absolute limits.

The endpoint alone is therefore misleading. `zcal` makes every tested rate
look healthy at trajectory 800, but the update-by-update gaps cross zero very
frequently and sometimes worsen in the final training window.

### Oscillation and relative-bias rejection

Persistent oscillation requires more than four gap-sign changes over updates
1--200 and a larger mean absolute gap over updates 151--200 than over updates
101--150. A cell is rejected if either seed is persistent.

| initialization | rate | seed 0 changes | seed 0 windows 101--150 -> 151--200 | seed 0 persistent | seed 1 changes | seed 1 windows 101--150 -> 151--200 | seed 1 persistent |
|---|---:|---:|:---:|:---:|---:|:---:|:---:|
| `z0` | 0.003 | 0 | 40.7227 -> 40.5159 | no | 0 | 40.7506 -> 39.5748 | no |
| `z0` | 0.01 | 0 | 39.8531 -> 39.2999 | no | 0 | 39.8727 -> 38.3574 | no |
| `z0` | 0.03 | 0 | 37.4839 -> 35.9681 | no | 0 | 37.4714 -> 35.0373 | no |
| `z0` | 0.1 | 0 | 29.4740 -> 25.3474 | no | 0 | 29.5363 -> 24.4693 | no |
| `zcal` | 0.003 | 94 | 0.0916 -> 0.1126 | **yes** | 63 | 0.1179 -> 0.0946 | no |
| `zcal` | 0.01 | 90 | 0.0804 -> 0.0821 | **yes** | 83 | 0.0788 -> 0.0835 | **yes** |
| `zcal` | 0.03 | 82 | 0.0790 -> 0.0820 | **yes** | 85 | 0.0760 -> 0.0778 | **yes** |
| `zcal` | 0.1 | 82 | 0.0870 -> 0.1029 | **yes** | 93 | 0.0849 -> 0.0761 | no |

Reducing the rate from `0.01` to `0.003` does not remove oscillation: seed 0
still crosses the target 94 times and its final-window error worsens by about
23%. Increasing the rate also does not solve the problem. Rates `0.03` and
`0.1` retain 82--93 crossings per seed, with at least one worsening seed.

The separate fixed-uniform bias-control rule compares each cell's median final
bias with the same-initialization rate-`0.01` control:

| initialization | rate | median fixed bias | ratio to `0.01` | bias rule | final health | oscillation | eligible |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|
| `z0` | 0.003 | 0.998138 | 1.0001 | pass | fail | pass | no |
| `z0` | 0.01 | 0.998070 | 1.0000 | pass | fail | pass | no |
| `z0` | 0.03 | 0.997767 | 0.9997 | pass | fail | pass | no |
| `z0` | 0.1 | 0.995816 | 0.9977 | pass | fail | pass | no |
| `zcal` | 0.003 | 0.000723 | 0.1530 | pass | pass | **fail** | no |
| `zcal` | 0.01 | 0.004726 | 1.0000 | pass | pass | **fail** | no |
| `zcal` | 0.03 | 0.014903 | 3.1535 | **fail** | pass | **fail** | no |
| `zcal` | 0.1 | 0.006829 | 1.4451 | **fail** | pass | **fail** | no |

The bias ratio is only a relative safeguard. The `z0` ratios pass because all
four rates are similarly poor in absolute terms, not because they are healthy.
For `zcal`, rate `0.003` has the best final fixed-uniform bias, but its seed-0
oscillation is independently disqualifying. Rates `0.03` and `0.1` fail both
the oscillation rule and the 1.1 relative-bias limit. Since no cell reaches the
eligible set, the one-standard-error preference for rate `0.01` and the
cross-cell ranking tie-breakers are never invoked.

### Search behavior

Archive hypervolume increases slightly with trajectory budget for every cell,
from approximately 0.241--0.242 at 200 trajectories to 0.244 at 800. At the
final checkpoint the full spread across cells is only 0.243526--0.244134,
about 0.25% of the lower value. There is therefore no evidence that changing
only the `logZ` learning rate materially improves search quality in this
screen. This is consistent with `logZ` acting primarily as a global TB offset;
it does not rescue an unhealthy normalization estimate, and hypervolume cannot
override a failed TB-health criterion.

### Conclusions

1. **The Experiment 4 hypothesis is rejected under the prespecified screen.**
   None of the eight initialization/rate cells is eligible, so advancing a
   cell to `dalu` would be an unplanned unhealthy fallback.

2. **A larger rate accelerates `z0`, but the tested rates do not make it viable
   within 800 trajectories.** Rate `0.1` is clearly faster than `0.003`, yet its
   final target gap remains above 22 in both validation strata and its TB loss
   remains almost entirely offset bias. This result does not prove that `z0`
   can never converge; it shows that it cannot do so within this trajectory
   budget and rate range.

3. **Calibration fixes the initial scale but a constant scalar learning-rate
   change does not fix `zcal` stability.** All `zcal` cells have excellent
   endpoint gaps at 800, including rate `0.003`, but every rate has persistent
   oscillation in at least one seed. The failure is longitudinal and would be
   missed by selecting on the final checkpoint alone.

4. **The instability is not a simple “rate too high” phenomenon.** The smallest
   tested rate still oscillates persistently, while larger rates add no stable
   advantage and can worsen relative bias. A follow-up should therefore test a
   mechanism that changes dynamics over time—such as a decaying `logZ` rate,
   damping/averaging, or a calibration freeze followed by controlled
   unfreezing—rather than merely another nearby constant rate.

5. **Policy/search outcomes are essentially insensitive to this factorial
   change over the tested horizon.** Final archive hypervolumes are nearly
   identical. The study isolates a normalization-health problem rather than a
   search-quality improvement.

These conclusions are intentionally limited to `bc0`, two screen seeds, the
fixed policy rate `0.001`, constant `logZ` rates `0.003`--`0.1`, and the
800-trajectory horizon. The lack of `dalu` confirmation is a consequence of
the prespecified rejection rule, not missing execution.

## Forced `dalu` factorial replication

The rejected `bc0` screen leaves open a narrower descriptive question: does
the observed `z0` under-normalization and `zcal` oscillation also occur on
`dalu`? This follow-up repeats the complete Experiment 4 factorial on `dalu`
despite the absence of confirmation candidates. It is explicitly a replication
study, not a post-hoc confirmation and not an override of the original
decision.

The replication runs all eight initialization/rate cells with seeds 0--2,
giving 24 runs. Seeds 0--1 are paired with the completed `bc0` screen for
direct circuit comparisons. Seed 2 is an additional `dalu` replication and is
not treated as paired because Experiment 4 did not run `bc0` seed 2. Every run
uses the same policy rate, calibration mechanics, trajectory accounting,
milestones, validation protocol, and 800-trajectory horizon as the original
screen.

The separate Martin project is
`gflowcircuit-gfn-logz-learning-rate-dalu`. A new project name is required
because the completed Experiment 4 remote code tree is immutable. The report
accepts this source-provenance difference only after validating compatible
scientific settings, the pairing-configuration fingerprint, and the exact
pre-calibration policy checksum for every seed-0/1 circuit pair. Fixed and
calibration sequence checksums are expected to differ between circuits.

### Prepared stages

Run the stages strictly in this order:

1. `tb_logz_lr_dalu_preflight_v1.slurm` runs `z0` and `zcal` at the two extreme
   rates for 100 trajectories. Wait for all four tasks to complete. Before
   continuing, require scientific exit code 0, predicted full runtime below
   10.5 hours, host RSS below 32 GiB, and no CUDA error.
2. `tb_logz_lr_dalu_factorial_seeds01_v1.slurm` runs the 16 paired seed-0/1
   cells. It validates all four preflight summaries before training. Wait for
   every task and validate 16 complete summaries, each with 800 trajectories,
   200 updates, and no numerical failure.
3. `tb_logz_lr_dalu_factorial_seed2_v1.slurm` runs the eight seed-2 cells. It
   refuses to train until all 16 seed-0/1 prerequisites are complete. Wait for
   all eight tasks and validate the resulting complete 24-run matrix.
4. `tb_logz_lr_dalu_report_v1.slurm` runs the CPU report. It validates both
   factorial matrices, applies the original health, oscillation, relative-bias,
   simplicity, and ranking rules descriptively to `dalu`, and writes paired
   `dalu - bc0` metrics for seeds 0--1 at every checkpoint and validation
   stratum.

The two training-array scripts intentionally share the SLURM job name
`tb-logz-lr-dalu-factorial-v1` and write disjoint seed directories below one
canonical artifact root. They are separate submissions because a 24-task
array would exceed Martin's limit of 20 queued-plus-running jobs. Do not submit
the seed-2 array until the seed-0/1 array has fully finished.

Training tasks request one GPU shard, 32 GiB host RAM, eight CPUs, and 12 hours,
with at most four concurrent tasks. The shard request is retained because the
completed `bc0` recovery used about 2 GiB host RSS and the model is small; the
new `dalu` preflight remains decisive for CUDA visibility, runtime, and memory
before launching the matrix. The report requests no GPU, 8 GiB RAM, two CPUs,
and 30 minutes.

### Commands after review and commit

After committing and pushing both repositories, run:

```bash
MYHPC=/Users/fedor.chernogorskii/.local/bin/myhpc
SCRIPTS=/Users/fedor.chernogorskii/workspace/local/scripts
PROJECT=gflowcircuit-gfn-logz-learning-rate-dalu

"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_logz_lr_dalu_preflight_v1.slurm"

# Wait for and validate all four preflight tasks.
"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_logz_lr_dalu_factorial_seeds01_v1.slurm"

# Wait for and validate all 16 seed-0/1 tasks.
"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_logz_lr_dalu_factorial_seed2_v1.slurm"

# Wait for and validate all eight seed-2 tasks.
"${MYHPC}" run nils "${PROJECT}" \
  "${SCRIPTS}/projects/${PROJECT}/scr/tb_logz_lr_dalu_report_v1.slurm"
```

The report writes `decision_summary.json`, `decision_report.md`, seed/update,
health, oscillation, ranking, simplicity, best-of-N and phase-ledger CSVs,
three `dalu` plots, and `cross_circuit_paired_metrics.csv`. Any cells described
as eligible in this report are descriptive results only; they do not
retroactively satisfy the original `bc0` confirmation gate.
