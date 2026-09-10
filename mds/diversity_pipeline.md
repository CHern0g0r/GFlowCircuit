# Running persistent Pareto diversity experiments

The implementation follows `mds/baselines_w_diversity.md`. The new campaign is
`cfg/exp/diversity_evaluation/protocol.yaml`; existing campaigns remain intact.
It covers REINFORCE, DRiLLS-A2C, PPO, and GFlowNet on the same eight circuits,
with ten training seeds, 800 training trajectories per seed, and ten final
evaluation seeds. Each evaluation seed uses prefixes of one 200-sample sequence.

## Runtime

Use the repository's training dependencies, its circuit-enabled `pyspiel` build
with `save_circuit`, and an ABC executable supporting `if -K`. The evaluator
itself needs NumPy and PyYAML, but does not import a policy or require pyspiel
when AIG manifests already exist. Commands below run from the repository root.
`ABC` denotes the executable path configured in your shell.

Validate all 32 method/circuit configurations without training:

```bash
python -m src.diversity_campaign validate
```

Run one full method/circuit task, with ten fresh training seeds:

```bash
python -m src.diversity_campaign run-task \
  --method gflownet_tb --circuit C1355 \
  --artifact-root outputs/pareto-diversity-v1 \
  --abc-path "$ABC" --device cuda
```

Methods are `reinforce`, `drills`, `ppo`, and `gflownet_tb`. Circuit names are
`C1355`, `C5315`, `adder`, `apex1`, `bc0`, `dalu`, `k2`, and `max`. This command
runs locally on the invoking host; on Martin invoke it within an allocated GPU
job and use the NFS artifact base specified by the protocol. There is one task
per method/circuit, 32 tasks in total. No submission is performed by the CLI.
Runtime and disk requirements should be measured before setting full-job limits.

The new campaign enables `seed_training_rng=true`, seeding model initialization
and Torch sampling for each training seed. That flag and
`discovery_metrics.archive_enabled` both default to false in legacy configs.
Archive capture is independent of RNG seeding. For a custom existing training
configuration, enable capture with:

```bash
python -m src.run --config-name baseline_configs/ppo \
  dataset_cfg=cfg/data/zhu2020/C1355.yaml \
  discovery_metrics.archive_enabled=true seed_training_rng=true
```

## Mapping and recovery

Evaluate existing manifests without loading models:

```bash
python -m src.diversity_evaluation evaluate \
  --input-root outputs/pareto-diversity-v1/gflownet_tb_C1355/attempt_000/train/pareto_archives \
  --input-root outputs/pareto-diversity-v1/gflownet_tb_C1355/attempt_000/final_sampling \
  --output-dir outputs/manual-diversity --abc-path "$ABC" --k 6
```

Alternatively pass repeated `--manifest <path>` arguments. Add `--resume` to
reuse validated mappings after a failure. Resume requires the same manifests,
mapper binary, k, budgets, and metric settings; use a new output directory when
these change. Every occurrence remains in reports even when mappings share a
cached artifact. Corrupt mapped outputs are regenerated. Failed mappings never
produce a complete-success status, and partial metrics report mapping coverage.

For a campaign attempt whose training and sampling finished, resume mapping and
update the attempt's parent status together:

```bash
python -m src.diversity_campaign resume-mapping \
  --attempt outputs/pareto-diversity-v1/gflownet_tb_C1355/attempt_000 \
  --abc-path "$ABC"
```

Interrupted training or final sampling starts a fresh attempt. Historical
archives are never reconstructed from checkpoints. To evaluate final sampling
alone from an old experiment with `.hydra/config.yaml` and `saved_models`:

```bash
python -m src.diversity_campaign sample-checkpoints \
  --experiment outputs/old-experiment \
  --circuit data/hdl-benchmarks/mcnc/Combinational/blif/C1355.blif \
  --output-dir outputs/old-experiment-samples --device cuda
python -m src.diversity_evaluation evaluate \
  --input-root outputs/old-experiment-samples \
  --output-dir outputs/old-experiment-diversity --abc-path "$ABC"
```

Such reports mark the training archive unavailable, rather than reporting zero
historical diversity.

## Artifacts and interpretation

```text
<artifact-root>/<method>_<circuit>/attempt_000/
  status.json                         # independent training/sampling/mapping status
  working_tree.patch
  train.log
  train/
    .hydra/config.yaml
    saved_models/run_0/last.pt
    discovery_front.csv
    discovery_metrics.csv
    pareto_archives/run_0/0_C1355/
      reference.aig
      aig/<sha256>.aig
      events.jsonl
      training_terminals.jsonl
      front.json
      manifest.json
      status.json
  final_sampling/run_0/seed_0/
    reference.aig
    aig/000000.aig
    manifest.json
  samples_seed_0.csv
  samples_seed_0_n10.csv
  diversity/
    identity.json
    mapping_cache/<key>.blif
    mapping_cache/<key>.json
    mapping_cache/<key>.log
    mapping_records.json
    metrics.json
    stats.csv
    fronts.json
    status.json
```

`metrics.json` separates all training terminals, the historical training
archive, the final training front, training archive milestones, and final
sampling budgets. All-training terminal records provide recipe and coordinate
metrics; their graph and LUT diversity is unavailable because dominated
circuits are intentionally not exported. Historical-archive metrics describe
admission-selected circuits, not the full training distribution.

AIG fronts minimize node count and level count. Different serialized circuits
at identical coordinates remain distinct artifacts. Hash-based counts describe
serialization identity, not a graph-isomorphism classification. The original
reference is excluded from generated-artifact diversity and best generated LUT
values; `front_includes_reference` explicitly identifies reference-inclusive
front counts. Training hypervolume uses the fixed original circuit reference
normalized to `(1, 1)`. Out-of-reference trade-offs stay in the archive/front but
contribute zero area. Milestones use chronological trajectory indices, while
recipe unique-fraction curves use labeled permutation means and quantiles.

LUT reports include unique artifact/coordinate counts and fractions, front
counts, best size/depth/product with artifact paths, and rank correlations.
Fractions over mapped samples use successful mapping occurrences as their
denominator; coverage must be checked for incomplete reports. `sample_index`
links winners/front members to stable manifest rows. Original circuits are
mapped separately for context. Archived AIGs remain on disk after retirement.

Aggregate complete reports, averaging evaluation seeds within each model before
computing mean and population standard deviation across training seeds:

```bash
python -m src.diversity_evaluation aggregate \
  --report outputs/pareto-diversity-v1/ppo_C1355/attempt_000/diversity/metrics.json \
  --report outputs/pareto-diversity-v1/gflownet_tb_C1355/attempt_000/diversity/metrics.json \
  --output outputs/pareto-diversity-v1/summary.json
```

## Verification

Run the focused regression suite:

```bash
python -m pytest tests/test_persistent_diversity.py tests/test_diversity_campaign.py \
  tests/test_discovery_metrics.py tests/test_sampling_diversity.py \
  tests/test_baseline_evaluation.py tests/test_gfn_checkpoint_evaluation.py \
  tests/test_exploration_tuning.py
```

Run the actual exporter, four trainers, sampler, mapper, and equivalence checker
on C1355 with tiny budgets:

```bash
python -m scr.diversity_smoke --output-dir outputs/diversity-smoke --abc-path "$ABC"
```

The smoke test trains each method with capture disabled and enabled from the
same RNG seed, requires identical learned tensors, verifies GFlowNet calibration
reuse counts, samples four endpoints, reloads saved AIG statistics, and checks
AIG/LUT equivalence with `cec -n` (port order, because AIGER renames ports).
It writes an example artifact tree and `summary.json`. An unavailable pyspiel
exporter or ABC executable is a real test failure, not a mocked success.

## CIAI: four method jobs and separate mapping

The GPU campaign has four SLURM array tasks, one for each method in protocol order
(REINFORCE, DRiLLS-A2C, PPO, GFlowNet). Each task supervises at most two circuit
processes sharing one allocated GPU. Circuits remain in protocol order; training
seeds remain sequential inside each circuit. GPU tasks stop after final sampling.
The separately submitted CPU array maps up to two circuits per method without
loading policies or initializing CUDA.

| Script | Array | Resources per task | Time |
|---|---|---|---|
| `ciai_diversity_campaign.slurm` | `0-3%2` | 1 GPU, 8 CPUs, 32 GiB | 48h |
| `ciai_diversity_mapping.slurm` | `0-3%3` | 8 CPUs, 32 GiB | 72h |

Scripts live in `/Users/fedor.chernogorskii/workspace/local/scripts/projects/gflowcircuit/scr`.
Use Conda `/home/fedor.chernogorskii/envs/ospiel` and ABC
`/home/fedor.chernogorskii/heap/gflowcircuit/abc/abc`.

After reviewing, committing, and pushing both repositories, run the updated
preflight through `myhpc`. It includes the actual two-circuit supervisor, CUDA
training/sampling and separate mapping. Its previous readiness result is stale
after these code changes. These are explicit submissions, not automatic steps:

```bash
myhpc run ciai gflowcircuit /Users/fedor.chernogorskii/workspace/local/scripts/projects/gflowcircuit/scr/ciai_diversity_preflight.slurm
# After preflight succeeds:
myhpc run ciai gflowcircuit /Users/fedor.chernogorskii/workspace/local/scripts/projects/gflowcircuit/scr/ciai_diversity_campaign.slurm
# After the GPU array ends (successful circuits can be mapped even if another failed):
myhpc run ciai gflowcircuit /Users/fedor.chernogorskii/workspace/local/scripts/projects/gflowcircuit/scr/ciai_diversity_mapping.slurm
```

Do not use force synchronization without explicit overwrite authorization. The
remote project tree is shared; do not change it while a campaign is running.
Four GPU tasks use the current four-submitted-job allowance. Other GPU jobs can
consume this allowance. The CPU mapping array has independent scheduler logs
under `ciai-diversity-mapping-v1/log`; evaluation outputs stay beside their
selected training attempts under `ciai-diversity-v1`.

### Direct commands inside an allocation

```bash
python -m src.diversity_campaign validate
python -m src.diversity_campaign run-method --method reinforce \
  --stage train-sample --workers 2 --device cuda \
  --artifact-root "$CAMPAIGN_ART" --abc-path "$ABC_PATH"
python -m src.diversity_campaign run-method --method reinforce \
  --stage mapping --workers 2 --device cuda \
  --artifact-root "$CAMPAIGN_ART" --abc-path "$ABC_PATH"
```

`--device cuda` in the mapping command selects attempts originally trained on
CUDA; it does not use a GPU. `--circuits C1355 C5315` selects a subset, scheduled
in protocol order. Workers can be 1 or 2. `run-task --stage all` retains the
single-circuit combined pipeline; `--stage train-sample` stops before mapping.
`training_device=auto` retains legacy standalone training behavior; campaigns
explicitly select `cpu` or `cuda` for both training and sampling.

### Restart and failure behavior

`methods/<method>/status.json` summarizes the latest invocation, selected
attempts, log paths, states and exit codes. Timestamped invocation directories
retain earlier summaries and worker logs. Attempt status separately records
training, sampling and mapping completion; overall completion requires mapping.

A nonblocking method lock prevents overlapping method supervisors across both
stages. A failed circuit does not cancel other circuits. The method exits
nonzero if any selected circuit failed or was interrupted. SIGTERM/SIGINT stops
new work and terminates worker process groups, including training subprocesses.

Resubmission reuses only matching, validated attempts: source/config/protocol,
circuit, device and native binary identities must match. Checkpoints, manifests,
AIGs, CSVs and completed evaluation outputs are verified against saved digests.
Missing or corrupt claimed-complete inputs fail visibly. Interrupted training or
sampling gets a fresh attempt; previous files remain. Mapping resumes its own
validated cache. Old attempts without the new identity/inventory are not reused
by method orchestration; legacy standalone mapping commands remain available.
Each evaluation directory has a single writer. Never run manual mapping against
an attempt concurrently with its method supervisor.

### Reduced concurrent smoke

Run on a compute allocation, using a fresh output directory:

```bash
python -m scr.campaign_smoke --device cuda --abc-path "$ABC_PATH" \
  --output-dir "$SMOKE_ART"
```

This uses all four methods, C1355 and C5315 concurrently, one training seed,
four terminal trajectories, one evaluation seed, and four final samples. It
runs the same supervisor and stage commands as production, then repeats both
stages to verify reuse, reloads AIGs, checks ABC equivalence, nested prefixes,
and saved winner paths. `--device cpu` supports CPU validation; a successful
CPU smoke does not certify the GPU environment. `--smoke` is an explicit
CLI-only reduction; normal production protocol validation remains strict.
