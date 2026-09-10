# Backbone-matched GFlowNet rerun

This campaign uses the standard `python -m src.run` entrypoint on the eight
circuits from the baseline campaign. It transfers the selected `gfn_health`
optimization settings while restoring the effective policy backbone used by
REINFORCE, DRiLLS-A2C, and PPO.

The dedicated Hydra configuration is
`cfg/tb_zhuDOP_baseline_backbone.yaml`. Existing GFlowNet and selected-backbone
configurations are unchanged.

## Frozen scientific contract

- Hybrid Zhu encoder with graph output 4 and Zhu10 vector output 28.
- One-width-32 ReLU policy head, without layer normalization or dropout.
- Adam policy learning rate `0.001` and separate `logZ` rate `0.01`.
- Calibrated `logZ` initialization from 64 epsilon-0.5 trajectories.
- The calibration set is replayed once in collection order for the first 16
  optimizer updates; 736 new on-policy trajectories follow.
- Four trajectory presentations per update, 200 updates, 800 unique training
  trajectories, and 800 total optimizer presentations.
- Difference-of-product reward with equal size/depth coefficients, reward
  alpha `4`, epsilon `1e-8`, and improvement clip `2`.
- Exploration epsilon `0.5` for updates 1–20, then decay to `0.01` at update
  200.
- Training seeds `0–9` for `C1355`, `C5315`, `adder`, `apex1`, `bc0`, `dalu`,
  `k2`, and `max`.

## TensorBoard and discovery outputs

Each circuit is one `src.run` invocation with ten paper-mode runs. Event files
are written below:

```text
<artifact-root>/circuits/<circuit>/tensorboard/run_<0-9>/
```

The event files contain:

- `train/*`: TB loss, `logZ`, returns, terminal reward, trajectory length,
  exploration epsilon, calibration source indicator, and trajectory counters;
- `eval/*`: periodic return, size/depth, QoR, best-result, and resyn2 comparison
  metrics;
- `discovery/<circuit>/*`: training-only Pareto hypervolume and nondominated
  count at 50-trajectory milestones;
- `discovery/mean_*`: aggregate discovery metrics (identical to the single
  circuit stream in per-circuit mode).

The calibration target and assigned initial `logZ` are logged at TensorBoard
step 0. Calibration trajectories are included in discovery evaluation exactly
once; replay presentations are not counted again.

`src.run` also writes `discovery_front.csv`, `discovery_metrics.csv`, ten model
checkpoints, and `gflownet_tb_report.json` within each circuit artifact
directory.

Post-training paired samples are produced separately by
`python -m src.gfn_checkpoint_evaluation`. The checkpoint-only protocol evaluates
training seeds 0–9 with evaluation seeds 0–9, materializes 200 ordered samples
per pair, and publishes a validated `points.csv` in each circuit artifact. Raw
seed files and exact `N = 10, 50, 100, 200` prefixes remain under a fresh
`evaluation/attempt_NNN` directory.

## Local invocation

For example, run `C1355` with:

```bash
python -m src.run --config-name tb_zhuDOP_baseline_backbone \
  dataset_cfg=cfg/data/zhu2020/C1355.yaml \
  output_dir=/path/to/C1355 \
  hydra.run.dir=/path/to/C1355/hydra
```

## Martin preparation

The Martin project is `gflowcircuit-gfn-backbone-matched`, with job
`gfc-gfn-backbone-matched-v1`. Its four-element GPU array starts two independent
`src.run` processes per GPU. Each process handles one circuit and ten training
seeds. The script records the synchronized project commit and experiment-config
SHA256 before launching work.

Review, commit, and push both repositories before using `myhpc sync` or
`myhpc run`. This implementation does not synchronize or submit the campaign.
