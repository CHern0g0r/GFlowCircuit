openspiel_path="$HOME/workspace/iwls/open_spiel"

export PYTHONPATH=$PYTHONPATH:$openspiel_path
export PYTHONPATH=$PYTHONPATH:$openspiel_path/build/python

base_name="gflownet_tb"
run_name="${base_name}-$(date +%m%d_%H%M%S)"

python -m src.run_gflownet_tb \
  --config-name gflownet_tb_mcnc \
  run_name=$run_name
