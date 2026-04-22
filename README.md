# GFlowCircuit

This repository currently supports two independent training pipelines:

- `REINFORCE` (existing default path).
- `GFlowNet-TB` (new Trajectory Balance path in parallel).

## REINFORCE (unchanged)

```bash
python -m src.run
```

or with the project script:

```bash
bash scr/run.sh
```

## GFlowNet-TB (parallel pipeline)

```bash
python -m src.run_gflownet_tb --config-name gflownet_tb_mcnc
```

or with the project script:

```bash
bash scr/run_gflownet_tb.sh
```

## Run both in parallel

Because each pipeline has its own entrypoint and report names (`reinforce_report.json` vs `gflownet_tb_report.json`), they can be launched simultaneously in separate terminals:

```bash
python -m src.run --config-name zhu2020_size_mcnc run_name=reinforce_parallel
python -m src.run_gflownet_tb --config-name gflownet_tb_mcnc run_name=tb_parallel
```

Use different `run_name` values when launching multiple jobs to keep Hydra output directories distinct.
