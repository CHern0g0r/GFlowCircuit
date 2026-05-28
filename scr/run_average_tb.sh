#!/usr/bin/env bash
# Aggregate TensorBoard runs from Zhu 2020 reproduction experiments (2026-05-25).
set -euo pipefail

cd "$(dirname "$0")"

conda_env="/home/fedor.chernogorskii/envs/ospiel"
if [[ -f "/apps/local/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/apps/local/anaconda3/etc/profile.d/conda.sh"
  conda activate "${conda_env}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${conda_env}"
fi

date_root="../outputs/2026-05-25"
output_root="${date_root}/tb_aggregated"

declare -A best_runs=()
declare -A best_logdir=()

for exp_dir in "${date_root}"/*/; do
  tb="${exp_dir}tensorboard"
  [[ -d "${tb}" ]] || continue

  n_runs=0
  for run_dir in "${tb}"/run_*; do
    [[ -d "${run_dir}" ]] || continue
    ((n_runs++)) || true
  done
  [[ "${n_runs}" -gt 0 ]] || continue

  base="$(basename "${exp_dir%/}")"
  if [[ "${base}" =~ zhu2020_([^_]+)_reproduce ]]; then
    circuit="${BASH_REMATCH[1]}"
  else
    circuit="${base}"
  fi

  # If the same circuit was launched more than once, keep the folder with more runs.
  if [[ -z "${best_runs[${circuit}]:-}" || "${n_runs}" -gt "${best_runs[${circuit}]}" ]]; then
    best_runs["${circuit}"]="${n_runs}"
    best_logdir["${circuit}"]="${tb}"
  fi
done

if [[ "${#best_logdir[@]}" -eq 0 ]]; then
  echo "No TensorBoard runs found under ${date_root}" >&2
  exit 1
fi

logdir_args=()
name_args=()
mapfile -t circuits < <(printf '%s\n' "${!best_logdir[@]}" | sort)
for circuit in "${circuits[@]}"; do
  logdir_args+=(--logdir "${best_logdir[${circuit}]}")
  name_args+=(--name "zhu2020_${circuit}")
  echo "Circuit ${circuit}: ${best_runs[${circuit}]} runs -> ${best_logdir[${circuit}]}"
done

python average_tb.py \
  "${logdir_args[@]}" \
  "${name_args[@]}" \
  --output_root "${output_root}"
