openspiel_path="$HOME/workspace/iwls/open_spiel"

export PYTHONPATH=$PYTHONPATH:$openspiel_path
export PYTHONPATH=$PYTHONPATH:$openspiel_path/build/python

python -m src.baselines.reinforce \
  --num_steps 10 \
  --episodes 200 \
  --eval_every 20 \
  --train_ratio 0.75 \
  --seed 0 \
  --dataset_cfg cfg/data/mcnc_seq.yaml \
  --json_out output/reinforce_report.json