#!/bin/bash
set -euo pipefail

openspiel_path="$HOME/workspace/iwls/open_spiel"
export PYTHONPATH="${PYTHONPATH:-}:${openspiel_path}"
export PYTHONPATH="${PYTHONPATH}:${openspiel_path}/build/python"

base="zhu2020_ablation"
stamp="$(date +%m%d_%H%M%S)"

# Fixed protocol knobs for fair comparison.
common=(
  --config-name zhu2020_size_mcnc
  paper_mode.num_runs=4
  paper_mode.infer_rollouts=10
)

echo "Running A0 control..."
python -m src.run \
  "${common[@]}" \
  run_name="${base}_A0_${stamp}" \
  value_learning_rate=3e-3 \
  clip_grad_norm_policy=null \
  clip_grad_norm_value=null \
  normalize_returns=false \
  entropy_beta=0.0

echo "Running A1 critic_stable..."
python -m src.run \
  "${common[@]}" \
  run_name="${base}_A1_${stamp}" \
  value_learning_rate=1e-3 \
  clip_grad_norm_policy=1.0 \
  clip_grad_norm_value=1.0 \
  normalize_returns=false \
  entropy_beta=0.0

echo "Running A2 critic_stable_plus_norm..."
python -m src.run \
  "${common[@]}" \
  run_name="${base}_A2_${stamp}" \
  value_learning_rate=1e-3 \
  clip_grad_norm_policy=1.0 \
  clip_grad_norm_value=1.0 \
  normalize_returns=true \
  entropy_beta=0.0

echo "Running A3 full_stable..."
python -m src.run \
  "${common[@]}" \
  run_name="${base}_A3_${stamp}" \
  value_learning_rate=1e-3 \
  clip_grad_norm_policy=1.0 \
  clip_grad_norm_value=1.0 \
  normalize_returns=true \
  entropy_beta=1e-3

echo "Ablation suite complete."
