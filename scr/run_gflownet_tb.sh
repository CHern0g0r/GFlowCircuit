openspiel_path="$HOME/workspace/iwls/open_spiel"

export PYTHONPATH=$PYTHONPATH:$openspiel_path
export PYTHONPATH=$PYTHONPATH:$openspiel_path/build/python

run_name="zhu2020_size_mcnc_tb_$(date +%m%d_%H%M%S)"

echo "Running GFlowNet-TB multi-run experiment..."
python -m src.run_gflownet_tb \
  --config-name gflownet_tb_mcnc \
  run_name="${run_name}" \
  seed=0 \
  paper_mode.num_runs=10 \
  learning_rate=0.001 \
  paper_mode.infer_rollouts=4 \
  +tb.reward_alpha=4.0 \
  +tb.reward_improvement_clip=2.0 \
  +tb.trajectories_per_episode=4 \
  json_out="output/${run_name}.json"

echo "GFlowNet-TB multi-run experiment complete."
