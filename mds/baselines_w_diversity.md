# Baselines and GFlowNet: persistent Pareto archives and LUT diversity

## Implementation objective

Extend REINFORCE, DRiLLS-A2C, PPO, and trajectory-balance GFlowNet so that new
training runs preserve discovered Pareto circuits as `.aig` artifacts. Add a
post-training evaluation stage that maps the complete historical archive and
the final stochastic samples to k-LUT networks and reports diversity and quality.
Implement the pipeline, campaign configuration, validation, and tests described
below. This document specifies future implementation; it does not record a
completed implementation or authorize submitting the campaign as part of writing
this instruction.

## Decisions established in the requirements interview

| Question | Decision |
| --- | --- |
| Which training states are candidates? | Terminal circuits only. |
| Which candidates are admitted? | Every currently nondominated circuit, including new trade-offs and distinct circuits at an existing Pareto coordinate. |
| Must a candidate dominate an existing point? | No. That condition would miss incomparable trade-offs and tied-coordinate diversity. |
| May size or depth exceed the original? | Yes, if the candidate is nondominated. |
| What happens to replaced circuits? | Retain their AIG files and provenance permanently in the historical archive. |
| Which circuits are mapped? | Every historical archive circuit and every final sample; report these populations separately. |
| Mapping setting | Configurable k, default 6, fixed for a comparison. |
| Main mapped quality metric | Minimum LUT count × LUT depth; also report minimum count and minimum depth. |
| AIG hypervolume reference | Original circuit, normalized to (1, 1). |

## Existing implementation to extend

- `src/discovery_metrics.py`: `CircuitDiscoveryArchive` and
  `TrainingDiscoveryTracker` currently retain feasible terminal coordinates,
  initialize the front with the original circuit, reject tied coordinates and
  points worse than the original in either dimension, and discard replaced
  coordinates. They do not persist circuits. Extend or share these utilities
  rather than maintaining inconsistent Pareto rules in each algorithm.
- `src/algorithms/{reinforce,drills_a2c,ppo,gflownet_tb}/trainer.py` already call
  discovery tracking. Their episode/sampler modules own the circuit state or
  recipe required for export. `src/run.py` constructs runs and writes reports.
- `src/sampling_diversity.py` already implements recipe metrics, coordinate
  diversity, deterministic best-value selection, LUT fronts, replay/export,
  SHA-256 digests, and ABC 6-LUT mapping. Factor reusable functions out as needed;
  generalize `map_to_lut6` without breaking its existing callers.
- `src/exploration_analysis.py` contains 2-D minimization Pareto and hypervolume
  helpers. Preserve a single tested metric convention across reports.
- `src/baseline_evaluation.py`, `src/gfn_checkpoint_evaluation.py`, and
  `src/sample_exp.py` implement the campaign's paired final sampling and nested
  budgets. Integrate artifact capture into that sampling path.
- Existing experiment definitions are `mds/baselines_run.md`,
  `cfg/exp/baseline_evaluation/protocol.yaml`,
  `cfg/exp/gfn_checkpoint_evaluation/protocol.yaml`, and
  `cfg/tb_zhuDOP_baseline_backbone.yaml`. Metric precedents are documented in
  `mds/sampling_diversity_exp.md`.

## Archive semantics

Maintain an independent archive for each method, circuit, training seed, and
run/attempt. Use raw integer AIG node count S and level count D; minimize both.
Point x dominates y exactly when `Sx <= Sy` and `Dx <= Dy`, with at least one
strict inequality. Equality is not dominance.

Maintain two separate data structures:

1. **Current front:** nondominated coordinates, each linked to every admitted
   serialized circuit realizing it.
2. **Historical archive:** all circuits ever admitted, with immutable discovery
   provenance, even after their coordinates leave the current front.

Persist the original circuit as a reference artifact and use it to initialize
the front, preserving the existing reference convention. Mark it explicitly as
`origin=reference`; exclude it from generated-circuit diversity and generated
sample counts. Report whether a front count includes that reference. Map the
reference separately for context, excluding it from generated best-LUT metrics.

For every newly generated terminal training circuit, in deterministic discovery
order:

1. Record its trajectory identity, complete ordered action sequence, and terminal
   size/depth. Count a new rollout once, whether or not admitted.
2. Reject failed/invalid states with an explicit failure record. Do not reject a
   valid state merely because S or D exceeds the original value.
3. If any current point dominates the candidate, do not archive its AIG.
4. Otherwise export and identify the circuit. If the same serialized circuit
   was already saved, reuse the artifact and record the repeat occurrence.
5. Save a new circuit as `.aig`, including a circuit at a tied coordinate, and
   append its admission event. Remove any dominated coordinates from the
   current front, recording their retirement event; never delete their files.
6. Commit the front update only after successful artifact persistence. An export
   failure must fail the archive stage visibly rather than report success with
   a missing circuit.

Use SHA-256 of the consistently serialized AIG as artifact identity. This is a
serialization identity, not a proof of graph-isomorphism equivalence. Equal
coordinates must never be used to deduplicate circuits. Do not claim that AIG
strashing or normalized BLIF text provides complete graph canonicalization.
Record serialization method/version so hash-based diversity is reproducible.

Save from an isolated state clone or replay the complete recipe into an isolated
environment, preserving source and environment configuration. Export must not
mutate the training state, consume policy RNG, or change learning. Verify replay
metrics against the recorded terminal metrics and verify the saved AIG can be
reloaded with matching statistics. A mismatch is an error.

Validation/in-training evaluation rollouts must not enter the training archive.
Include GFlowNet calibration rollouts when first generated. Its cached
calibration trajectories are reused for initial updates: rescoring/reusing them
must not create new discoveries or increase the sampling budget. Give batch
rollouts a stable order and record it; asynchronous completion must not decide
which circuit entered the historical archive.

The historical archive is a selected population. Its diversity is not the
diversity of all training outputs. Keep lightweight terminal occurrence records
for all training rollouts so recipe and coordinate diversity over all training
outputs can also be reported without saving every dominated circuit. LUT
diversity of training outputs is measured only over the historical archive.

## Post-training mapping and final sampling

Create a separately runnable, resumable evaluation stage consuming an archive
manifest and/or final-sampling records. It must run without loading a policy
when all required AIGs already exist. Training completion and mapping completion
must have separate status fields.

For each input AIG, run the selected ABC executable with the equivalent of:

```text
read <input.aig>; strash; if -K <k>; print_stats; write_blif <output.blif>
```

Validate k against supported mapper settings. Persist the mapped `.blif`, LUT
count, LUT depth, digest, ABC command, binary identity/version, and logs. Use
proper path quoting, a timeout, return-code checks, and output existence/statistics
validation. Normalize BLIF comments, blank lines, and model names consistently
with the existing pipeline before digesting. Do not overwrite the source AIG.

Map each unique input artifact once per mapping configuration; cache identity
must include AIG digest, k, ABC identity, and command/options. Preserve all
occurrence-to-artifact links when reusing a mapping. Never drop duplicate sample
rows: their frequencies matter for diversity. A mapping failure remains an
explicit failure and prevents a complete-success status. Partial reports must
identify failed inputs and denominators; unavailable metrics are null, not zero.

Use exactly the existing ordered final samples, with evaluation seeds 0–9 and
nested prefixes `[10, 50, 100, 200]` of 200 rollouts per evaluation seed and
trained model. Save/map every final terminal sample, including dominated ones.
Do not generate a separate 1,000-sample experiment for this campaign. Here 200
means rollout occurrences, not necessarily 200 distinct recipes or circuits.
Persist action sequences so existing sampling records can be replayed only when
their provenance and metrics can be verified.

Compute these populations explicitly:

| Population | Membership |
| --- | --- |
| All training terminals | Every training rollout, for recipe/coordinate metrics only. |
| Training historical archive | Every unique AIG admitted at discovery, including later-retired circuits. |
| Training final front | Generated circuits at the final nondominated coordinates, as a subset of the archive. |
| Final sampling | All ordered terminal occurrences, separately per evaluation seed and prefix budget. |
| Combined discovered set | Optional supplementary union of archive and final samples, clearly labeled. |

Map the whole historical archive before selecting mapped optima or the LUT
front. A retired AIG circuit can be the best LUT circuit. Do not filter mapping
inputs using only the final AIG front.

## Required metrics and reporting

Compute metrics within circuit and training seed; keep evaluation seeds and
budgets separate for final sampling. Never construct a front across circuits.

For each applicable population report:

- Number of occurrences, unique serialized AIG artifacts, distinct AIG (S,D)
  coordinates, unique mapped artifacts, and distinct LUT (count,depth)
  coordinates. Report unique fractions with explicit denominators. Artifact
  fractions use occurrences where available; archive-unique fractions use
  archived AIG count and are labeled accordingly.
- Number of distinct coordinates on the AIG Pareto front and number of generated
  AIG artifacts realizing those coordinates. Persist front coordinates and all
  associated circuit/occurrence identifiers. Include historical archive count,
  admission count, repeat count, failure count, and final active count separately.
- AIG hypervolume of the nondominated set normalized by the original positive
  size/depth, with fixed reference (1,1): the union area of rectangles from each
  normalized point to that reference. Points exceeding the reference in either
  coordinate contribute zero but remain eligible archive/front members. Empty
  sets have hypervolume zero. Undefined normalization must fail validation or be
  reported explicitly as null. Never choose a method-dependent reference.
- Best LUT count×depth, count, and depth, with values, all attaining circuit IDs,
  and a selected mapped artifact. For product ties choose count, then depth,
  then first discovery/sample order; for count ties choose depth then order;
  for depth ties choose count then order. Include best AIG values for context.
- LUT Pareto coordinates and all realizing mapped-artifact/source IDs; also
  report the LUT front coordinate count. AIG-to-LUT Spearman size and depth
  correlations use paired records with average ranks for ties; constant inputs
  produce null. State whether correlations use unique archive inputs or sample
  occurrences.
- Recipe unique fraction curves, recipe entropy in nats, and per-position and
  mean action entropy for all training terminal records and final samples, using
  existing metric definitions. Archive recipe summaries, if provided, must be
  labeled as admission-selected rather than policy-distribution estimates.

Emit training front hypervolume/count and cumulative archive diversity at the
existing 50-trajectory milestones and at completion. Persist event/trajectory
indices so post-hoc LUT archive metrics can be reconstructed at those same
milestones. Actual training-discovery curves use chronological order; any
permutation-based recipe curves are separate and explicitly labeled.

Report per-run values and per-circuit/method mean and standard deviation across
training seeds. For final sampling, first aggregate evaluation-seed results
within each trained model, then summarize across training seeds. Do not treat
all evaluation seeds as independent trained models. Pooled fronts, if included,
are supplementary and cannot replace per-run comparisons.

## Artifact and recovery contract

Use the following logical layout, adapted to existing attempt directories:

```text
<attempt>/
  train/<run>/<circuit>/pareto_archive/
    reference.aig
    aig/<artifact_id>.aig
    manifest.json
    events.jsonl
    training_terminals.jsonl
    front.json
  evaluation/diversity/<run>/<circuit>/
    training_archive/lut_k<k>/<artifact_id>.blif
    final_sampling/seed_<seed>/aig/<sample_id>.aig
    final_sampling/seed_<seed>/lut_k<k>/<sample_id>.blif
    mapping_records.jsonl
    metrics.json
    stats.csv
    fronts.json
    status.json
```

Manifests/records must carry schema version, method, circuit source and digest,
training seed/run/attempt, project/config provenance, checkpoint identity for
final sampling, evaluation seed/sample index where applicable, discovery index,
trajectory index, calibration/training origin, complete action sequence, raw
AIG metrics, relative artifact paths and hashes, admission reason, and current
or retired membership via events. Mapping records add mapper provenance, k,
mapped metrics/hash/path, completion state, and failure details.

Write artifacts and snapshots atomically; preserve committed events on failure.
Resume must validate identities and checksums before reusing work and reconcile
incomplete records without duplicate admissions or sample rows. Reject mixing
different seeds, sources, checkpoints, or mapper configurations. Implement
evaluation resume independently of training-checkpoint resume; if training
resume is supported, restore its archive at the matching discovery boundary.
Measure archive/export and mapping wall time and artifact bytes separately from
training/sampling throughput.

## Campaign integration

Use the same eight circuits, ten training seeds, method hyperparameters, 20-step
horizon, and action space as the baseline and backbone-matched GFlowNet
protocols. Baselines use 800 training trajectories per seed; preserve GFlowNet's
configured 200×4 budget and its calibration reuse, without adding discovery
rollouts. Make archive settings and LUT settings explicit in a versioned new
campaign protocol and resolved Hydra configuration.

The existing GFlowNet evaluator targets already-trained checkpoint artifacts.
Those checkpoints cannot recover discarded historical training circuits. The
new full campaign must train all four methods with archive capture enabled,
then run final sampling and mapping. Support final-sampling-only evaluation of
old checkpoints but mark their training archive unavailable. Do not claim an
empty archive means zero historical diversity. Preserve historical campaign
provenance instead of editing old commit/hash/job identifiers to describe new
runs.

Provide documented commands to validate the new protocol, run one method/circuit
training-and-sampling task, evaluate an existing archive, resume mapping, and
aggregate reports. Connect stage success to artifact validation so a job is not
reported fully complete while required mapping or metrics are missing. Prepare
cluster-compatible output paths and resource settings; avoid prescribing an
unmeasured runtime or launching the full campaign during implementation checks.

## Acceptance criteria

1. Unit tests cover initial reference, strict dominance, incomparable insertion,
   tied-coordinate distinct artifacts, exact artifact repetition, rejected
   dominated candidates, out-of-reference trade-offs, and permanent retention
   after retirement. For example, with original (100,100), (80,95) and (95,80)
   both enter; (90,90) enters without dominating either; (70,70) retires all
   three without deleting their files. (110,60) remains an eligible trade-off.
2. Verify hypervolume analytically: with original (100,100), points (80,90) and
   (90,80) give 0.03. Adding (110,60) changes front counts but not hypervolume.
   Tied coordinates do not change hypervolume.
3. Each of the four trainers saves accepted terminal circuits with replayable
   provenance. No intermediate state or evaluation rollout is admitted;
   GFlowNet calibration reuse is counted once. Archive capture preserves the
   action sequence and learning outputs of a fixed-seed run.
4. Mapping tests cover k propagation, artifact validation, timeout/error
   handling, duplicate caching with occurrence preservation, and resume with
   changed mapper/input identities. Include a retired AIG that maps to the best
   LUT product to prevent accidental final-front-only evaluation.
5. Metric tests cover coordinate versus artifact diversity, reference exclusion,
   empty/failed populations, duplicate sample frequencies, LUT ties, undefined
   correlations, and separation of seeds/populations. Every reported winner
   resolves to a saved mapped circuit.
6. Campaign tests prove prefix budgets reuse the same 200 ordered occurrences,
   all final samples are mapped, and incomplete stages fail completion checks.
   Existing discovery, sampling-diversity, baseline-evaluation, and
   GFlowNet-evaluation tests remain valid or receive explicit versioned updates
   for the changed archive semantics.
7. Run a small local end-to-end smoke test for all four methods with the actual
   pyspiel exporter and ABC mapper where available. Reload AIGs, validate mapped
   outputs, and check functional equivalence to the source on the smoke circuit
   using the available ABC equivalence flow. Report unavailable dependencies
   explicitly; mocked tests alone do not establish real export/mapping success.
8. Deliver an example artifact tree, machine-readable reports, and reproducible
   CLI instructions. Report changes, verification performed, and remaining
   environment limitations before any full campaign launch.
