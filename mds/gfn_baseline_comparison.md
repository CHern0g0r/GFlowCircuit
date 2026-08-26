# GFlowNet baseline comparison

This campaign adds the calibrated Trajectory Balance GFlowNet to the final
eight-circuit baseline evaluation without retraining REINFORCE, DRiLLS-A2C, or
PPO. The governing configuration is
`cfg/exp/gfn_baseline_comparison/protocol.yaml`.

## Scientific status

The selected setup is the strongest cross-circuit descriptive control from
the `gfn_health` branch, not a validated healthy optimizer winner:

- calibrated `logZ` initialization (`zcal`);
- Adam policy learning rate `0.001`;
- separate `logZ` learning rate `0.01`;
- four trajectories per optimizer update;
- 64 calibration trajectories, optimized once in collection order;
- 736 subsequent on-policy trajectories;
- 800 total training trajectories and 200 optimizer updates;
- epsilon `0.5` for the first 20 updates, then decay to `0.01` over the
  800-trajectory schedule;
- difference-of-product reward and the selected C1 backbone.

Calibration trajectories count toward the scientific budget. Diagnostic
validation and post-training comparison rollouts do not.

## Local interfaces

Validate the immutable matrix and resolved Hydra configuration:

```bash
python -m src.gfn_baseline_comparison validate
```

Run one circuit task (ten training seeds followed by all evaluation samples):

```bash
python -m src.gfn_baseline_comparison run-task \
  --circuit-index 0 \
  --artifact-root /path/to/gfc-gfn-baseline-v1 \
  --device cuda
```

Every invocation creates a new `attempt_NNN` directory. A complete circuit
contains ten native health runs, ten raw evaluation-seed files, a canonical
`points.csv`, and exact nested-prefix views for `N = 10, 50, 100, 200`.

After all eight GFlowNet tasks and the three baseline campaigns are complete:

```bash
python -m src.gfn_baseline_comparison compare \
  --gfn-root /shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-baseline-comparison/gfc-gfn-baseline-v1 \
  --output-dir /shared/home/fedor.chernogorskii/agent/art/gflowcircuit-gfn-baseline-comparison/gfc-gfn-compare-v1/report
```

The baseline roots default to the three version-2 artifact paths frozen in the
protocol. They can be overridden, in protocol order, with repeated
`--baseline-root METHOD=PATH` arguments.

## Metrics and outputs

The primary metric is strict two-dimensional minimization hypervolume on
`(size / initial_size, depth / initial_depth)` with reference `(1, 1)`.
Secondary outputs include mean product improvement, best size and depth
reductions, distinct endpoints, pooled fronts, and best-of-N curves.

GFlowNet-minus-baseline comparisons pair the circuit, training seed, and
evaluation seed. Confidence intervals use 10,000 deterministic hierarchical
bootstrap repetitions, resampling training seeds and then evaluation seeds.
The report directory contains:

- `seed_metrics.csv`;
- `summary.csv`;
- `pairwise.csv`;
- `pooled_fronts.csv`;
- `artifact_manifest.json`;
- `comparison_report.md`;
- `comparison_summary.json`.

## Martin execution

The immutable project name is `gflowcircuit-gfn-baseline-comparison`. First run
the eight-element GPU training array and wait for every task to complete. Then
run the CPU comparison job. Use `myhpc`; do not submit either script until the
project and scripts repositories are reviewed, committed, pushed, clean, and
equal to their upstreams.
