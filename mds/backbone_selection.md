# Backbone selection for circuit GFlowNets

## Task

Implement and execute a staged backbone-selection study for the circuit policy.
The study must identify the smallest graph/vector encoder and policy head that
can robustly represent the exact reward-proportional next-action distribution
on enumerated horizon-four circuit trees.

The selected model will become the shared policy backbone for the controlled
GFlowNet, REINFORCE, PPO, DRiLLS-A2C, and PCN comparison described in
`mds/comparison.md`. The study must therefore select capacity, not select the
architecture on which GFlowNet has the largest downstream advantage.

The current Zhu-style contract model is the immutable reference:

- graph branch: four Zhu GCN layers, hidden dimension 12, graph output
  dimension 4, and global mean pooling;
- vector branch: the Zhu state projected linearly to 28 dimensions;
- fusion: graph/vector concatenation;
- policy head: one 32-unit hidden layer.

Job `13184` established that this model is numerically stable but does not pass
the supervised exact-conditional representability gates on `bc0`. Fixed and
cosine Adam both passed zero of three seeds. The best mean prefix TV was about
`0.0143` against a `0.005` gate, and the largest errors were concentrated at
depth three.

The backbone task must answer four questions in order:

1. Is the policy head alone too small?
2. If not, is capacity missing from the graph branch, vector branch, or both?
3. Does the vector state need the ordered action prefix rather than recent
   action counts alone?
4. If widening the Zhu family is insufficient, is a more expressive
   edge-aware graph encoder and graph readout required?

## Scientific constraints

- Keep the exact conditional targets, reward transformation, action set,
  horizon, evaluation frequency, optimizer definitions, seeds, and acceptance
  gates unchanged from the supervised experiment.
- Do not tune against `i10`; it remains the locked holdout circuit.
- Use `bc0` for screening and primary qualification and `dalu` as the
  architecture stress test.
- Never select a model using full-horizon holdout hypervolume.
- Preserve the original Zhu configuration as a reported baseline. New
  configurations must have distinct names.
- Keep each candidate's policy head fixed within a comparison stage so encoder
  conclusions are not confounded by decoder size.
- Record every attempted candidate, including numerical failures and rejected
  candidates.
- Choose the smallest robustly passing candidate. Lower loss is not sufficient
  reason to choose a larger model once the gates pass.

The exact supervised test is a necessary capacity qualification, not proof of
full-horizon generalization. A prefix-history encoder may fit a deterministic
per-circuit tree without learning a useful graph representation, so the final
candidate must also pass the `dalu` stress test and a short full-horizon
preflight.

## Shared supervised protocol

For every candidate:

- circuit: `bc0` during screening and qualification;
- secondary circuit: `dalu` for finalists;
- horizon: 4;
- nonterminal prefixes: 400 when all seven actions remain legal;
- terminal sequences: 2,401 when all seven actions remain legal;
- seeds: 0, 1, and 2 for confirmation;
- optimizer variants: fixed Adam and cosine-decayed Adam;
- initial learning rate: `0.001`;
- cosine minimum learning rate: `1e-5`;
- maximum updates: 20,000;
- evaluation interval: 100 updates;
- required consecutive passing evaluations: 3;
- full-prefix objective with prefix chunk size 128;
- exact `logZ`, excluded from optimization;
- identical initialization for fixed/cosine variants within each seed.

Use the existing target-construction validation:

- every legal conditional row sums to one within `1e-10`;
- illegal actions have target probability zero;
- products of target conditionals reconstruct all exact terminal
  probabilities within `1e-10`;
- target and reconstructed terminal mass are one within numerical tolerance.

Any failure of enumeration, target reconstruction, pairing, checkpoint reload,
or artifact writing is an execution failure rather than a failed architecture.

## Acceptance gates

A run passes only after three consecutive evaluations satisfying:

- finite logits, probabilities, loss, gradients, and parameters;
- legal-action normalization error at most `1e-6`;
- illegal-action probability at most `1e-6`;
- mean prefix TV at most `0.005`;
- maximum prefix TV at most `0.02`;
- terminal TV at most `0.02`;
- terminal maximum probability error at most `0.01`;
- exact-`logZ` RMS TB residual at most `0.05`;
- maximum absolute TB residual at most `0.15`;
- terminal probability-mass error at most `1e-6`.

Conditional KL, maximum conditional-action error, metrics by prefix depth, and
supervised cross-entropy remain diagnostic metrics.

For ranking nonpassing screening candidates, use the existing maximum
normalized gate violation:

\[
V_{\max} =
\max_j \frac{\text{observed metric}_j}{\text{gate}_j}
\]

over upper-bounded fit gates, with mean conditional KL as the tie-breaker. Do
not rank candidates only by supervised loss because the loss averages away
the worst-prefix errors that currently prevent acceptance.

## Common policy head

Before changing either encoder, introduce an adequate common policy decoder:

```yaml
head:
  type: mlp
  hidden_dims: [128, 128]
  activation: gelu
  layer_norm: true
  dropout: 0.0
```

Candidate `C1` tests this head with both original encoders. All later
candidates use this head unchanged. If `C1` passes robustly, select it and do
not modify the encoders.

The original 32-unit head remains part of `C0` only.

## Graph encoder candidates

### Zhu graph family

The first graph candidates retain the published Zhu-style node features,
message passing, four-layer depth, and mean pooling. Only width and graph
output dimension change.

| Name | Layers | Hidden dimension | Graph output | Pooling |
| --- | ---: | ---: | ---: | --- |
| `zhu_original` | 4 | 12 | 4 | mean |
| `zhu_medium` | 4 | 32 | 16 | mean |
| `zhu_large` | 4 | 64 | 32 | mean |

Do not initially test width 128. The immediate suspected bottleneck is the
four-dimensional graph output, and width 64/output 32 is already a material
increase. Only consider width 128 after every more informative candidate
fails.

### Edge-aware residual GINE

If the widened Zhu family does not qualify, implement a separate configurable
GINE encoder rather than silently changing `ZhuGCNEncoder`.

The GINE encoder must:

- consume the environment node features directly;
- consume edge attributes, including the inversion indicator;
- use a two-layer MLP inside each GINE convolution;
- use residual connections whenever input/output dimensions match;
- apply LayerNorm and GELU or ReLU after hidden message-passing layers;
- expose node representations from every layer for optional Jumping Knowledge;
- support mean, sum, max, and concatenated multi-pooling;
- project the final pooled representation to a configured graph-output size;
- remain permutation invariant at graph level.

Initial configurations:

| Name | Layers | Hidden/node dimension | Graph output | Readout |
| --- | ---: | ---: | ---: | --- |
| `gine_small` | 3 | 32 | 32 | mean/max/sum projected to 32 |
| `gine_medium` | 4 | 64 | 64 | mean/max/sum projected to 64 |

`gine_medium` is the primary strong graph candidate. Do not introduce graph
attention or a full graph Transformer in the initial study.

### Optional direction-aware GINE with Jumping Knowledge

Run this candidate only if `gine_medium` remains nonpassing or if it reduces
mean error while retaining a large worst-prefix error.

- represent each AIG edge with its inversion and direction attributes;
- add an explicitly typed reverse edge, or use separate forward/backward
  aggregation channels;
- combine representations from all four 64-dimensional GINE layers using
  concatenation or max Jumping Knowledge;
- project the combined node representation to 64;
- use concatenated mean/max/sum graph pooling and project to a 64-dimensional
  graph embedding.

Treat direction augmentation as part of the candidate definition and serialize
it in the resolved configuration. It must be shared by every later controlled
baseline.

## Vector-state encoder candidates

### Current Zhu vector

With seven actions, the current vector has 12 values:

- normalized current size and depth;
- normalized previous size and depth;
- normalized counts of the seven actions in the recent history window;
- normalized current step.

The current vector loses the order of recent actions. Preserve it as
`vector_original`.

### Nonlinear Zhu-vector MLPs

| Name | Input | Hidden dimensions | Output |
| --- | ---: | --- | ---: |
| `vector_original` | 12 | none | 28 |
| `vector_medium` | 12 | `[64]` | 32 |
| `vector_large` | 12 | `[128, 64]` | 64 |

Use GELU or ReLU, LayerNorm after hidden layers, and no dropout during the
representability study. Do not test outputs above 64 unless every ordered
history and graph candidate fails.

### Ordered action-history encoder

Implement an optional history-aware vector encoder. Extend observations in a
backward-compatible way with the exact ordered action prefix. Do not infer
order from the action-count vector.

The encoder has two branches:

1. Embed the ordered action sequence and encode it with a GRU.
2. Encode scalar circuit statistics with an MLP.

Concatenate the branches and project to the configured vector-output size.
Retain the current recent-action counts as optional auxiliary statistics.

Initial configurations:

| Name | Action embedding | GRU hidden | Statistics MLP | Vector output |
| --- | ---: | ---: | --- | ---: |
| `history_medium` | 16 | 32 | `[32]` | 64 |
| `history_large` | 32 | 64 | `[64]` | 128 |

Start with `history_medium`. Horizon 20 is short, so a GRU is the primary
sequence encoder; do not add a Transformer to the initial search.

The observation extension must:

- represent the empty prefix unambiguously;
- distinguish padding from every legal action;
- preserve complete order up to the environment horizon;
- move history tensors with `Observation.observation_to_device`;
- be constructed identically by exact enumeration, GFlowNet sampling,
  REINFORCE, PPO, DRiLLS-A2C, PCN, evaluation, and checkpoint reload;
- leave configurations that do not request history behaviorally unchanged.

### Vector-only diagnostic

Allow graph ablation in the supervised subsystem. A vector-only
`history_medium` policy is a diagnostic, not an eligible final shared backbone.
If it passes, ordered prefix identity is sufficient to fit the per-circuit
tree, but this does not establish transfer across circuits. Retain a graph
branch in the final candidate unless a separate cross-circuit protocol
justifies removing it.

## Candidate matrix

Use these canonical candidate identifiers:

| ID | Graph | Vector | Head | Purpose |
| --- | --- | --- | --- | --- |
| `C0_contract` | `zhu_original` | `vector_original` | `[32]` | Immutable job-13184 reference |
| `C1_head` | `zhu_original` | `vector_original` | `[128,128]` | Test decoder capacity |
| `C2_graph` | `zhu_large` | `vector_original` | `[128,128]` | Isolate graph capacity |
| `C3_vector` | `zhu_original` | `vector_large` | `[128,128]` | Isolate vector capacity |
| `C4_wide_zhu` | `zhu_large` | `vector_large` | `[128,128]` | Test larger original family |
| `C5_history_zhu` | `zhu_large` | `history_medium` | `[128,128]` | Test ordered prefix information |
| `C6_gine_history` | `gine_medium` | `history_medium` | `[128,128]` | Strong standard shared backbone |
| `C7_direction_jk` | direction-aware GINE + JK | `history_medium` | `[128,128]` | Conditional final escalation |

`C7_direction_jk` is not automatic. Run it only if the promotion criteria
below require it.

If `C2_graph` or `C3_vector` passes while `C4_wide_zhu` is unnecessary, select
the passing single-branch change. If `C4_wide_zhu` passes, optionally test the
medium sizes needed to find the smallest passing member:

- graph hidden/output `32/16` instead of `64/32`;
- vector hidden/output `[64]/32` instead of `[128,64]/64`.

This down-sizing occurs only after a robust passing architecture exists.

## Experimental stages

### Stage 0: reproduce and validate the reference

Do not spend GPU time rerunning `C0_contract` unless code changes invalidate
checkpoint reload or metric reproduction. Import job `13184` results into the
new aggregate report and record:

- project commit and configuration checksum;
- pass counts;
- best/final metrics;
- best update;
- wall time and memory;
- initialization checksums.

Confirm that the new aggregation code reproduces the authoritative
classification `representability_not_demonstrated`.

### Stage 1: head-only control

Run `C1_head` on `bc0`:

- seeds 0 and 1;
- fixed Adam only;
- full 20,000-update horizon.

If both seeds pass, immediately run seed 2 plus the paired cosine variants for
formal confirmation. If the fixed variant passes all three seeds, select
`C1_head`; additional encoders are unnecessary.

If neither seed passes, continue to Stage 2. Retain the full curves even when
the result is clearly nonpassing.

### Stage 2: Zhu component diagnosis

Run `C2_graph`, `C3_vector`, and `C4_wide_zhu` on `bc0`:

- seeds 0 and 1;
- fixed Adam only;
- full 20,000 updates.

This is a controlled `2 x 2` encoder-capacity diagnosis with the adequate head
fixed:

| | Original vector | Large vector |
| --- | --- | --- |
| Original graph | `C1_head` | `C3_vector` |
| Large graph | `C2_graph` | `C4_wide_zhu` |

Interpret the results by gate family:

- graph widening improves errors at every depth: graph compression is a
  bottleneck;
- vector widening especially improves depth-three metrics: vector/history
  processing is a bottleneck;
- only the combined candidate improves: both branches are limiting;
- mean TV improves but maximum TV remains high: increase representational
  discrimination or readout quality rather than width alone;
- terminal TV improves without conditional gates: errors remain locally
  unacceptable even if they partially cancel at terminal level.

Promote at most two candidates. A candidate is promotable when:

- it passes either screening seed; or
- its median best `V_max` over seeds 0 and 1 is at least 15% lower than
  `C1_head`, with no numerical failure and no more than 10% worsening of
  terminal TV.

Always promote the smallest passing candidate. If no candidate qualifies,
continue to Stage 3.

### Stage 3: ordered-history diagnosis

Run `C5_history_zhu` on `bc0`, seeds 0 and 1, fixed Adam, for 20,000 updates.
Also run the vector-only `history_medium` diagnostic for seed 0.

Promote `C5_history_zhu` if it passes a seed or meets the Stage 2 improvement
rule. If the vector-only diagnostic passes while `C5_history_zhu` does not,
diagnose the fusion or graph branch rather than concluding that ordered
history is ineffective.

### Stage 4: strong graph encoder

Run `C6_gine_history` on `bc0`, seeds 0 and 1, fixed Adam, for 20,000 updates.

Run `C7_direction_jk` only when:

- `C6_gine_history` is the best nonpassing candidate and its median best
  `V_max` is within 25% of passing; or
- its mean prefix TV is near the gate but maximum prefix TV remains the
  dominant violation; or
- inspection identifies systematic errors consistent with direction/readout
  ambiguity.

If `C6_gine_history` is far from every gate, stop and review the observation
definition and supervised implementation before adding more model capacity.
Do not proceed automatically to width 128 or a graph Transformer.

### Stage 5: full paired confirmation

Confirm at most the two best promoted candidates on `bc0`:

- seeds 0, 1, and 2;
- fixed and cosine optimizer variants;
- paired initialization checksum within each seed;
- full 20,000 updates;
- three consecutive passing evaluations.

Classify each candidate using the established definitions:

- `robust_representability`: at least one optimizer variant passes all three
  seeds;
- `representable_but_optimizer_sensitive`: at least one run passes but neither
  variant passes all three seeds;
- `representability_not_demonstrated`: no run passes;
- `execution_failure`: enumeration, configuration, numerical validation, or
  artifact writing fails.

Only `robust_representability` candidates are eligible to become the shared
backbone. A sensitive candidate may be retained as an ablation but must not be
selected while a robust candidate exists.

### Stage 6: `dalu` stress test

Run the smallest robust `bc0` candidate and, if different, the best alternative
on the exact horizon-four `dalu` tree with the same targets, gates, seeds,
optimizer variants, and update cap.

The shared backbone should have one optimizer variant pass all three `dalu`
seeds. If exact enumeration produces a different number of legal prefixes or
terminal sequences, record the actual counts; do not force `bc0` counts.

If no candidate is robust on `dalu`, do not use `i10` to choose among them.
Return to the architecture diagnosis using only `bc0` and `dalu`.

### Stage 7: full-horizon preflight

Before freezing the selected backbone:

- run `bc0` and `dalu`, seed 0;
- horizon 20;
- 100 newly collected trajectories per circuit;
- use the selected optimizer initialization/calibration settings available at
  that point;
- require finite logits, losses, gradients, parameters, and normalized legal
  probabilities;
- measure policy-forward time, observation-construction time, ABC time, GPU
  memory, CPU memory, and checkpoint size;
- verify checkpoint save/reload and deterministic fixed-sequence evaluation.

This preflight is a runtime and integration check, not a search-quality
selection stage. Do not reject a representationally adequate backbone merely
because a provisional TB optimizer is not yet calibrated.

## Selection rule

Among candidates robust on both `bc0` and `dalu`, select in this order:

1. smallest total trainable policy parameter count;
2. smallest graph/vector embedding dimensions;
3. lowest full-prefix supervised evaluation time;
4. lowest full-horizon policy-forward time;
5. lowest peak GPU memory;
6. lowest mean conditional KL as the final tie-breaker.

Prefer a fixed-LR 3/3 pass over a cosine-only 3/3 pass when candidates are
otherwise comparable, because it yields a simpler downstream protocol.

Do not choose by:

- lowest loss after 20,000 updates once multiple candidates pass;
- best `bc0` terminal TV far below the gate;
- full-horizon hypervolume on `i10`;
- the largest model;
- the model with the largest GFlowNet advantage over an RL baseline.

Freeze the chosen:

- graph construction and edge semantics;
- graph encoder class and dimensions;
- graph readout;
- vector statistics;
- history encoding;
- fusion operation;
- policy head;
- activation, normalization, and dropout;
- parameter initialization;
- resolved configuration name and checksum.

The frozen policy backbone must then be used without policy-side architectural
changes in the shared-backbone comparison. Actor-critic methods may add a
critic, and PCN may add its required return/horizon conditioning.

## Implementation requirements

### Configuration

- Add distinct Hydra encoder configurations for every implemented graph/vector
  candidate.
- Compose candidates from reusable graph, vector, fusion, and head
  configurations where practical.
- Do not edit `cfg/encoder/hybrid_zhu.yaml` in place.
- Validate positive dimensions, supported pooling, layer counts, history
  horizon, action vocabulary, and edge-feature dimensions at construction.
- Include the complete resolved architecture in every summary and checkpoint.

### Model interfaces

- Preserve the existing `encoder(obs_batch) -> [batch, embedding_dim]`
  contract.
- Expose the encoder output dimension through resolved configuration rather
  than hard-coded assumptions.
- Keep legal-action masking in the policy, not in the encoder.
- Support single observations and batches.
- Support CPU and CUDA.
- Avoid mutation of cached observations during forward passes.

### Initialization and pairing

- Initialize one base policy per candidate and seed.
- Clone the exact state dictionary into fixed and cosine variants.
- Hash model parameters before training.
- Record branch-specific and whole-policy parameter counts.
- Ensure `log_z` remains fixed to exact `true_log_z` and excluded from the
  optimizer during supervised training.

### Checkpoint compatibility

Every best/final checkpoint must store:

- candidate ID and resolved configuration;
- graph/vector/head class names and dimensions;
- action and history vocabulary;
- project commit;
- parameter checksum;
- model, optimizer, and scheduler state;
- exact tree and reward metadata;
- update and prefix-presentation counts;
- recorded metrics.

Reload every finalist's best and final checkpoints and reproduce the saved
metrics before classification.

## Tests

### Unit tests

- Wider Zhu configurations produce the declared embedding dimensions.
- GINE consumes node and edge features and supports graphs with no edges.
- GINE residual blocks preserve batch and node shapes.
- Multi-pooling equals independently computed mean, max, and sum pooling before
  projection.
- Jumping Knowledge uses every configured layer.
- Direction augmentation preserves inversion attributes and distinguishes
  forward/reverse edges.
- Ordered histories distinguish permutations with the same action counts.
- Empty, partial, and full-length histories are encoded without ambiguity.
- Padding is distinct from all seven actions.
- History tensors move correctly between CPU and CUDA.
- Batched and per-observation forwards agree in evaluation mode.
- Original configurations remain numerically unchanged when history is
  disabled.
- Candidate parameter counts and configuration hashes are deterministic.

### Supervised tests

- A small synthetic prefix tree can be fit by every candidate class.
- Chunked and unchunked gradients agree for each encoder family.
- Fixed/cosine variants receive identical initialization.
- `log_z` is excluded from optimization and remains exact.
- Best-checkpoint ordering uses maximum normalized gate violation and KL
  tie-breaking.
- Candidate classifications cover robust, sensitive, nonpassing, and execution
  failure.

### Circuit integration tests

- `bc0` exact observations include aligned graph, vector, and history inputs.
- Candidate forwards remain aligned with the ordered prefix list after
  batching.
- Best/final checkpoints reload and reproduce metrics.
- Short CUDA runs remain finite for every promoted candidate.
- `bc0` and `dalu` full-horizon samplers construct the same history semantics
  as exact enumeration.
- Existing REINFORCE, PPO, DRiLLS-A2C, PCN, and GFlowNet configurations still
  load after the observation extension.

## Artifacts

Write a top-level selection directory containing:

- `candidate_manifest.json`;
- `screening_summary.csv`;
- `screening_summary.json`;
- `selection_summary.json`;
- `selection_report.md`;
- `resolved_configs/<candidate>.json`;
- `parameter_counts.csv`;
- `runtime_summary.csv`;
- `plots/gate_violation_by_candidate.png`;
- `plots/metric_by_update.png`;
- `plots/depth_tv_by_candidate.png`;
- one subdirectory per candidate/circuit.

Each candidate/circuit directory must contain the standard supervised
artifacts:

- `tree_summary.json`;
- `observation_collisions.json`;
- `target_conditionals.csv`;
- `initialization_checksums.json`;
- `metrics.jsonl`;
- `summary.json`;
- conditional-prediction CSVs;
- terminal-distribution CSVs;
- best and final checkpoints.

The aggregate selection report must show:

- every attempted candidate and status;
- pass counts by circuit, optimizer, and seed;
- first passing update;
- best/final gate table;
- depth-specific conditional metrics;
- parameter counts and embedding dimensions;
- runtime and memory;
- promotion/rejection reason;
- selected candidate and exact selection-rule trace.

## Execution ordering

1. Implement the common head and configuration plumbing.
2. Run unit and synthetic supervised tests.
3. Execute `C1_head`.
4. Execute the Stage 2 Zhu `2 x 2` diagnosis.
5. Implement/run history only if widening is insufficient.
6. Implement/run GINE only if history/widening is insufficient.
7. Confirm at most two candidates on all seeds and both schedulers.
8. Stress-test robust finalists on `dalu`.
9. Run full-horizon integration preflights.
10. Freeze and document the selected shared backbone.
11. Rerun active-baseline/logZ diagnosis with the selected backbone before
    treating later optimizer and budget conclusions as authoritative.
12. Use the frozen backbone in the shared-backbone comparison contract.

The already completed active-baseline diagnosis for the original contract
model remains useful as a historical characterization. Its optimizer and
trajectory-budget conclusions do not automatically transfer to the selected
backbone.

## Completion criteria

The task is complete only when:

- all required tests pass;
- every executed candidate has complete, reload-verified artifacts;
- at least one candidate is robust on both `bc0` and `dalu`, or the report
  explicitly concludes that representability remains undemonstrated;
- the selection rule is applied without consulting `i10`;
- the selected configuration is frozen under a stable name and checksum;
- the original Zhu contract configuration remains available;
- `mds/comparison.md` and the optimizer protocol identify the selected
  backbone and the experiments that must be rerun.
