openspiel_path="$HOME/workspace/iwls/open_spiel"

export PYTHONPATH=$PYTHONPATH:$openspiel_path
export PYTHONPATH=$PYTHONPATH:$openspiel_path/build/python

python -m src.resyn2 \
  --file_path $HOME/workspace/circ/repos/EPFL_benchmarks/arithmetic/bar.aig \
  --output_path output/best.bench \
  --json_out output/result.json
