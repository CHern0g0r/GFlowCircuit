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

### `dalu` execution and artifact validity

The forced replication completed all 24 planned `dalu` runs: two
initializations, four constant `logZ` learning rates, and seeds 0--2. All 24
run summaries are complete and report no numerical failure. Each run has the
required 800 training trajectories, 200 optimizer updates, 64 cached
calibration trajectories, 64 calibration presentations, and 736 new
on-policy trajectories. The per-run artifact validator also accepted the
milestone files, trajectory-source counts, checkpoints, fixed-validation
cache, calibration cache, and ordered calibration minibatches before the
aggregate pairing check was reached.

The `dalu` runs use code commit
`da22485780625267772b49f7f24093cd943bba88` and source-tree hash
`ef590bb8bd06eb6c27de4ba2d68b0f061a7f75a6f73f12cca7f7db2c7fb32510`.
The corresponding `bc0` screen uses commit
`e81649f752e93ae8cbb36bb2ad36213ea830b473` and source-tree hash
`b7af20532f71811dbe97f240773f5e49a69923cfeebc9b10811baa4239b04ded`.
This source-provenance difference was expected and allowed by the replication
design after scientific-configuration validation.

Several SLURM tasks had to be retried because an allocated GPU shard was busy
or not visible at CUDA initialization. These failures occurred before
training. The completed artifacts have the required counters and contain no
numerical failure, so the infrastructure retries do not change the scientific
sample.

The CPU aggregate was job `17790`. It used 2.95 GiB peak RSS and exited with
code 1 after 1 minute 41 seconds. Its authoritative decision is:

```json
{
  "complete": false,
  "failure_type": "artifact_or_execution_failure",
  "message": "dalu post-initialization pairing mismatch for zcal, seed 1"
}
```

This is an exact reproducibility failure in one run, not a missing or
incomplete training result. For `zcal`, seed 1, rates `0.003`, `0.01`, and
`0.03` all assigned the calibration target `39.36384963989258` and have the
same post-initialization checksum. Rate `0.1` assigned
`39.36384582519531`, a difference of about `3.8e-6`, and therefore has a
different bitwise checksum. The pre-calibration policy checksum and ordered
calibration-sequence checksum are identical across all four rates. The cached
policy log-probabilities differ at floating-point roundoff scale, apparently
because the calibration evaluation was not bitwise deterministic across GPU
shards.

The strict check behaved correctly: the official report must not claim a
paired factorial result when exact pairing is absent. The descriptive tables
below are reconstructed directly from the immutable, individually valid run
summaries and update streams. They are sufficient to assess the large
training effects, but they are not a substitute for a successful official
aggregate. In particular, the `zcal`, rate-`0.1`, seed-1 comparison should be
treated as approximately rather than exactly paired.

## Dalu circuit results

The following values are means over seeds 0--2. `F/U pass` is the number of
seeds passing every health gate on the fixed-uniform and fresh-on-policy
strata, respectively. Counts are out of three. As in the original screen,
only the 800-trajectory rows are decisive.

### Dalu `z0` checkpoint table

| initialization | rate | trajectories | fixed gap | fixed bias | fresh gap | fresh bias | learned `logZ` | archive HV | F/U pass |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `z0` | 0.003 | 200 | 39.629025 | 0.999310 | 38.650311 | 0.999403 | 0.149557 | 0.028356 | 0/3 / 0/3 |
| `z0` | 0.003 | 400 | 39.662939 | 0.999064 | 38.278722 | 0.998994 | 0.298659 | 0.035560 | 0/3 / 0/3 |
| `z0` | 0.003 | 800 | 39.944182 | 0.998314 | 37.247344 | 0.997755 | 0.594858 | 0.040290 | 0/3 / 0/3 |
| `z0` | 0.01 | 200 | 39.282015 | 0.999296 | 38.301335 | 0.999391 | 0.497982 | 0.028356 | 0/3 / 0/3 |
| `z0` | 0.01 | 400 | 38.965989 | 0.999032 | 37.586333 | 0.998950 | 0.992452 | 0.035560 | 0/3 / 0/3 |
| `z0` | 0.01 | 800 | 38.497039 | 0.998291 | 35.977199 | 0.997786 | 1.968086 | 0.040485 | 0/3 / 0/3 |
| `z0` | 0.03 | 200 | 38.297131 | 0.999250 | 37.304174 | 0.999351 | 1.489231 | 0.028356 | 0/3 / 0/3 |
| `z0` | 0.03 | 400 | 36.984039 | 0.998969 | 35.629713 | 0.998877 | 2.950421 | 0.035567 | 0/3 / 0/3 |
| `z0` | 0.03 | 800 | 34.612243 | 0.997971 | 32.224291 | 0.997492 | 5.773367 | 0.040464 | 0/3 / 0/3 |
| `z0` | 0.1 | 200 | 34.887298 | 0.999079 | 33.874350 | 0.999218 | 4.906503 | 0.028356 | 0/3 / 0/3 |
| `z0` | 0.1 | 400 | 30.434962 | 0.998479 | 29.114852 | 0.998303 | 9.500083 | 0.035560 | 0/3 / 0/3 |
| `z0` | 0.1 | 800 | 22.595200 | 0.996224 | 20.759305 | 0.995679 | 17.562441 | 0.040339 | 0/3 / 0/3 |

### Dalu `zcal` checkpoint table

| initialization | rate | trajectories | fixed gap | fixed bias | fresh gap | fresh bias | learned `logZ` | archive HV | F/U pass |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `zcal` | 0.003 | 200 | 0.095698 | 0.065564 | 0.100647 | 0.065403 | 39.269733 | 0.030676 | 2/3 / 2/3 |
| `zcal` | 0.003 | 400 | 0.099412 | 0.076462 | 0.064368 | 0.053047 | 39.267944 | 0.037678 | 1/3 / 2/3 |
| `zcal` | 0.003 | 800 | 0.075975 | 0.041319 | 0.064231 | 0.031170 | 39.269981 | 0.040270 | 2/3 / 2/3 |
| `zcal` | 0.01 | 200 | 0.056972 | 0.027095 | 0.086894 | 0.050310 | 39.301076 | 0.030676 | 3/3 / 2/3 |
| `zcal` | 0.01 | 400 | 0.079869 | 0.044852 | 0.040123 | 0.023686 | 39.277121 | 0.037678 | 2/3 / 2/3 |
| `zcal` | 0.01 | 800 | 0.066698 | 0.036223 | 0.047209 | 0.017653 | 39.269758 | 0.040270 | 2/3 / 3/3 |
| `zcal` | 0.03 | 200 | 0.068764 | 0.043269 | 0.074574 | 0.032696 | 39.284804 | 0.030676 | 2/3 / 2/3 |
| `zcal` | 0.03 | 400 | 0.064279 | 0.034815 | 0.056055 | 0.033082 | 39.284501 | 0.037692 | 2/3 / 2/3 |
| `zcal` | 0.03 | 800 | 0.056287 | 0.039019 | 0.079359 | 0.043525 | 39.286261 | 0.040283 | 2/3 / 2/3 |
| `zcal` | 0.1 | 200 | 0.167113 | 0.136812 | 0.044348 | 0.013192 | 39.200462 | 0.028488 | 0/3 / 3/3 |
| `zcal` | 0.1 | 400 | 0.049264 | 0.019308 | 0.090330 | 0.051099 | 39.320684 | 0.037699 | 3/3 / 1/3 |
| `zcal` | 0.1 | 800 | 0.041347 | 0.011595 | 0.099686 | 0.081044 | 39.355451 | 0.040290 | 3/3 / 1/3 |

The `z0` rate ordering is monotonic and almost identical to `bc0`: increasing
the rate moves the scalar farther in the available 200 updates. It still does
not move nearly far enough. At rate `0.1`, the mean learned value is only
`17.56`, while the seed-specific calibrated targets are approximately
`39.05`--`39.36`. The final fixed and fresh gaps remain `22.60` and `20.76`,
and more than 99.5% of TB MSE remains explained by global-offset bias. Every
`z0` seed fails target-gap, bias-fraction, and standardized-bias gates in both
strata at every checkpoint.

Calibration again fixes the scale immediately. All final `zcal` mean target
gaps are below `0.10`, and all mean bias fractions are below `0.082`.
Nevertheless, no `zcal` cell passes every seed/stratum gate at trajectory 800.
Mean values hide isolated but decisive failures. The next table is the
dedicated Dalu `zcal` seed-level endpoint table:

| rate | seed | fixed gap | fixed bias | fixed standardized bias | fresh gap | fresh bias | fresh standardized bias | learned `logZ` | archive HV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.003 | 0 | 0.067714 | 0.015019 | 0.123482 | 0.049818 | 0.008374 | 0.091897 | 39.289948 | 0.035886 |
| 0.003 | 1 | 0.055603 | 0.016093 | 0.127890 | 0.108624 | 0.075807 | **0.286401** | 39.306656 | 0.042659 |
| 0.003 | 2 | 0.104607 | **0.092846** | **0.319921** | 0.034250 | 0.009328 | 0.097036 | 39.213341 | 0.042263 |
| 0.01 | 0 | 0.038195 | 0.005579 | 0.074905 | 0.042793 | 0.007653 | 0.087818 | 39.304916 | 0.035886 |
| 0.01 | 1 | 0.079544 | 0.037413 | 0.197147 | 0.069358 | 0.036576 | 0.194846 | 39.274937 | 0.042659 |
| 0.01 | 2 | 0.082354 | **0.065677** | **0.265129** | 0.029477 | 0.008729 | 0.093837 | 39.229420 | 0.042263 |
| 0.03 | 0 | 0.006766 | 0.000206 | 0.014343 | 0.062633 | 0.018247 | 0.136332 | 39.327526 | 0.035886 |
| 0.03 | 1 | 0.056631 | 0.015272 | 0.124536 | 0.125060 | **0.089864** | **0.314224** | 39.315166 | 0.042701 |
| 0.03 | 2 | 0.105463 | **0.101579** | **0.336250** | 0.050386 | 0.022463 | 0.151590 | 39.216091 | 0.042263 |
| 0.1 | 0 | 0.053429 | 0.012842 | 0.114056 | 0.114606 | **0.061817** | **0.256691** | 39.390793 | 0.035928 |
| 0.1 | 1 | 0.043789 | 0.014278 | 0.120351 | 0.150060 | **0.169941** | **0.452475** | 39.380280 | 0.042680 |
| 0.1 | 2 | 0.026824 | 0.007664 | 0.087883 | 0.034393 | 0.011374 | 0.107259 | 39.295280 | 0.042263 |

Bold values exceed the applicable bias-fraction limit of `0.05` or
standardized-bias limit of `0.25`. All target gaps remain below the `0.5`
limit. Rate `0.01` is closest to endpoint health, failing only the fixed
stratum for seed 2. Rate `0.1` passes all fixed-stratum checks but fails the
fresh stratum for seeds 0 and 1. This reversal is evidence of distribution
sensitivity rather than a clean rate effect.

### Dalu oscillation table

The persistent-oscillation rule is unchanged: more than four sign changes in
200 updates and a larger mean absolute gap in updates 151--200 than in
101--150. The table gives `sign changes; preceding window -> final window`.

| initialization | rate | seed 0 | seed 1 | seed 2 | cell oscillation pass |
|:---|---:|:---|:---|:---|:---:|
| `z0` | 0.003 | 0; 38.4948 -> 38.3708 | 0; 38.6090 -> 37.3405 | 0; 38.2435 -> 38.2796 | yes |
| `z0` | 0.01 | 0; 37.6338 -> 37.1910 | 0; 37.7305 -> 36.1202 | 0; 37.4333 -> 37.0910 | yes |
| `z0` | 0.03 | 0; 35.2303 -> 33.9177 | 0; 35.3222 -> 32.7308 | 0; 35.0072 -> 33.7796 | yes |
| `z0` | 0.1 | 0; 27.3508 -> 23.3716 | 0; 27.4611 -> 22.4566 | 0; 27.0744 -> 23.1866 | yes |
| `zcal` | 0.003 | 92; 0.1761 -> 0.1597 | 92; 0.1424 -> 0.1320 | **87; 0.1112 -> 0.1471** | **no** |
| `zcal` | 0.01 | 98; 0.1655 -> 0.1488 | **96; 0.1421 -> 0.1467** | **101; 0.1051 -> 0.1577** | **no** |
| `zcal` | 0.03 | **92; 0.1545 -> 0.1686** | 94; 0.1464 -> 0.1449 | **101; 0.1002 -> 0.1637** | **no** |
| `zcal` | 0.1 | 86; 0.1810 -> 0.1707 | 98; 0.1424 -> 0.1310 | **97; 0.1120 -> 0.1664** | **no** |

All `zcal` cells cross the target 86--101 times per seed. Each rate has at
least one seed whose final window worsens, so every calibrated cell fails the
persistent-oscillation criterion. Seed 2 is persistent at all four rates and
is the main reason that the three-seed replication is more clearly negative
than the two-seed `bc0` screen. Lowering the rate to `0.003` reduces neither
the number of crossings nor the cross-seed fragility.

### Dalu final cell-selection table

The endpoint cell metrics average both validation strata and all three
`dalu` seeds. The relative-bias ratio uses the median fixed-uniform bias and
the same-initialization rate-`0.01` denominator.

| initialization | rate | mean absolute gap | mean bias | median fixed bias | ratio to `0.01` | mean archive HV | final health | oscillation | eligible |
|:---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| `z0` | 0.003 | 38.595763 | 0.998035 | 0.998206 | 1.000015 | 0.040290 | fail | pass | no |
| `z0` | 0.01 | 37.237119 | 0.998039 | 0.998192 | 1.000000 | 0.040485 | fail | pass | no |
| `z0` | 0.03 | 33.418267 | 0.997731 | 0.998175 | 0.999983 | 0.040464 | fail | pass | no |
| `z0` | 0.1 | 21.677253 | 0.995952 | 0.996272 | 0.998076 | 0.040339 | fail | pass | no |
| `zcal` | 0.003 | 0.070103 | 0.036245 | 0.016093 | 0.430138 | 0.040270 | fail | fail | no |
| `zcal` | 0.01 | **0.056954** | **0.026938** | 0.037413 | 1.000000 | 0.040270 | fail | fail | no |
| `zcal` | 0.03 | 0.067823 | 0.041272 | 0.015272 | 0.408211 | 0.040283 | fail | fail | no |
| `zcal` | 0.1 | 0.070517 | 0.046319 | **0.012842** | 0.343245 | 0.040290 | fail | fail | no |

All cells pass the 1.1 relative-bias rule. That does not make any cell
eligible: every `z0` cell fails absolute health, and every `zcal` cell fails
both absolute health and persistent oscillation. Consequently, the formal
one-standard-error preference and candidate ranking cannot be invoked.

If an unhealthy configuration must be named for descriptive comparison,
`zcal` with rate `0.01` is the least-bad choice: it has the smallest overall
mean final target gap and the smallest overall mean final bias, and only one
of its six endpoint seed/stratum gates fails. It is not a production-ready
winner. Seeds 1 and 2 are persistently oscillatory, and seed 2 exceeds both
fixed-uniform bias thresholds. Rate `0.003` has fewer persistent seeds but
more endpoint health failures; rate `0.1` has better fixed-uniform values but
substantially worse fresh-on-policy bias. There is no defensible constant-rate
setup that satisfies the experiment's rules.

## Paired `bc0` versus Dalu comparison

The paired seed-0/1 endpoint comparison answers the motivating replication
question. Deltas are `dalu - bc0`; lower gaps are better. Hypervolume values
are on circuit-specific objective distributions, so their absolute
cross-circuit difference should not be interpreted as one circuit training
better than the other.

| initialization | rate | `bc0` fixed gap | `dalu` fixed gap | delta | `bc0` fresh gap | `dalu` fresh gap | delta | `bc0` HV | `dalu` HV |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `z0` | 0.003 | 42.70979 | 40.25924 | -2.45055 | 38.93160 | 36.95124 | -1.98037 | 0.244134 | 0.039033 |
| `z0` | 0.01 | 41.32239 | 38.86460 | -2.45778 | 37.54477 | 35.60995 | -1.93483 | 0.244134 | 0.039325 |
| `z0` | 0.03 | 37.45511 | 34.88096 | -2.57414 | 33.77631 | 32.00488 | -1.77143 | 0.244002 | 0.039294 |
| `z0` | 0.1 | 25.24998 | 22.78561 | -2.46437 | 22.22160 | 20.54859 | -1.67301 | 0.243881 | 0.039106 |
| `zcal` | 0.003 | 0.00536 | 0.06166 | +0.05630 | 0.01198 | 0.07922 | +0.06724 | 0.243698 | 0.039273 |
| `zcal` | 0.01 | 0.01040 | 0.05887 | +0.04847 | 0.01199 | 0.05608 | +0.04408 | 0.243526 | 0.039273 |
| `zcal` | 0.03 | 0.02455 | 0.03170 | +0.00714 | 0.02084 | 0.09385 | +0.07301 | 0.243526 | 0.039294 |
| `zcal` | 0.1 | 0.01461 | 0.04861 | +0.03400 | 0.00778 | 0.13233 | +0.12455 | 0.243617 | 0.039304 |

The approximately two-unit reduction in the `z0` gap on `dalu` is explained
by its smaller normalization target: the `bc0` seed targets are about
`41.56`--`41.69`, while the `dalu` targets are about `39.05`--`39.36`.
Learning dynamics are not repaired; `z0` still advances at roughly the same
rate and remains tens of log units away after 200 updates.

For `zcal`, `dalu` is generally harder at the endpoint. Its paired gaps are
larger than `bc0` for every rate and both validation strata at trajectory 800.
The increase is especially pronounced for fresh-on-policy evaluation at rate
`0.1` (`+0.12455`). The earlier checkpoints sometimes favor `dalu`, but the
advantage does not persist: by trajectory 800, seed sensitivity and
oscillation dominate. Thus the original pathology is not specific to `bc0`.
If anything, the three-seed `dalu` replication exposes it more clearly.

Within `dalu`, final archive hypervolume spans only `0.040270`--`0.040485`
across all eight cells, about 0.54% of the lower value. Within `zcal` alone,
the spread is about 0.05%. There is no material search-quality gain from
choosing a different constant `logZ` rate, and hypervolume cannot override
failed TB health.

## Overall conclusion and recommended setup

The combined evidence supports the following conclusions:

1. **No tested constant `logZ` learning rate is acceptable under the
   prespecified rules.** The original `bc0` screen has zero eligible cells,
   and the forced `dalu` replication also has zero descriptively eligible
   cells. There is therefore no healthy setup to promote.

2. **Initialization matters much more than the constant rate.** `z0` is
   unusably slow over 800 trajectories even at rate `0.1`; `zcal` immediately
   reaches the correct scale. Any practical setup should retain calibration
   or an equivalent normalization estimate.

3. **Among the tested options, `zcal` with rate `0.01` is the best descriptive
   baseline, not a validated winner.** It minimizes the `dalu` mean endpoint
   gap and mean bias and was also the prespecified control. Its major problems
   are persistent oscillation in two of three `dalu` seeds, a decisive fixed
   bias failure in seed 2, and no measurable search-quality advantage.

4. **The failure is not caused simply by too large a rate.** Rate `0.003`
   still crosses the target roughly 90 times per seed and is persistently
   oscillatory in seed 2. Raising the rate changes which validation stratum or
   seed fails but does not produce robust health.

5. **The behavior is circuit-general over the two tested circuits.** The
   absolute target changes between `bc0` and `dalu`, but the qualitative
   `z0` under-normalization and `zcal` oscillation remain. The replication
   rejects the hypothesis that the `bc0` result was merely circuit-specific.

6. **There is a reproducibility issue in addition to the scientific issue.**
   GPU evaluation produced a `3.8e-6` calibration-target discrepancy in one
   otherwise paired run. Before another factorial study, calibration should be
   made bitwise reproducible or the protocol should define a justified
   tolerance plus a checksum over rate-independent inputs. The current exact
   check should not be weakened silently.

For subsequent work, use `zcal` and rate `0.01` only as the control condition.
The next experiment should change the time dynamics rather than test another
nearby constant rate: freeze `logZ` after calibration, use a decay schedule,
apply damping or an exponential moving average, or unfreeze only after a
controlled warm-up. Any candidate must be evaluated on both fixed-uniform and
fresh-on-policy strata, retain the longitudinal oscillation gate, and use at
least three seeds. Until such a mechanism passes those checks, Experiment 4
does not justify changing the production `logZ` optimizer setup.

## Best configuration across `bc0` and `dalu`

No tested configuration is a healthy winner under the prespecified Experiment
4 rules. Every `z0` cell fails the absolute TB-health gates, while every
`zcal` cell is persistently oscillatory in at least one seed. Therefore the
experiment does not authorize promoting any constant `logZ` learning rate as
a validated production setting.

If one of the tested configurations must be used as the cross-circuit
baseline, the best practical choice is:

- initialization: `zcal`;
- `logZ` learning rate: `0.01`;
- policy learning rate: `0.001`.

This is the least-bad baseline rather than a successful experimental winner.
It passes every `bc0` endpoint health gate and, on `dalu`, has the smallest
overall mean final absolute target gap (`0.056954`) and mean bias fraction
(`0.026938`). Five of its six `dalu` seed/validation-stratum endpoint gates
pass, which is the closest any calibrated rate comes to complete endpoint
health. Archive hypervolume is essentially insensitive to the tested `logZ`
rates, so there is no search-quality reason to choose a different rate.

The recommendation has important limitations. Rate `0.01` is persistently
oscillatory in both `bc0` seeds and in two of three `dalu` seeds. On `dalu`,
seed 2 also fails the final fixed-uniform bias gates: its bias fraction is
`0.065677`, above the `0.05` limit, and its standardized bias is `0.265129`,
above the `0.25` limit. It consequently remains unsuitable as a validated
production configuration.

Rate `0.003` is the closest alternative because it has fewer persistent seeds
across the two circuits and excellent `bc0` endpoint values. It is not a
better general choice: on `dalu` it has a larger mean endpoint gap and bias
than rate `0.01` and passes only four of six endpoint seed/stratum gates.
Rates `0.03` and `0.1` are still less attractive because they fail the `bc0`
relative-bias safeguard and introduce additional `dalu` health or
fresh-on-policy bias problems.

Accordingly, future experiments should retain `zcal` with rate `0.01` as the
control but test a non-constant optimization mechanism, such as freezing
`logZ` after calibration, controlled unfreezing, learning-rate decay, or
damping. Until one of those mechanisms passes both validation strata and the
longitudinal oscillation gate, there is no production-ready `logZ` setup from
Experiment 4.
