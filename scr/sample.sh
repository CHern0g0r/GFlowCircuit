# python -m src.sample_exp \
  # --experiment 2026-05-26/12:06_tb_zhu_i10_79035 \
  # --num-samples 50 \
  # --circuit data/hdl-benchmarks/mcnc/Combinational/blif/i10.blif \
  # --seed 42 \
  # --device cuda

DATE="2026-05-27"
OUTPUTS_ROOT="outputs/${DATE}"
SKIP_DIR="tb_aggregated"
CIRCUIT_ROOT="data/hdl-benchmarks/mcnc/Combinational/blif"

NUM_SAMPLES=50
SEED=42
DEVICE="cuda"

resolve_circuit_path() {
  local circuit_name="$1"
  local circuit_path="${CIRCUIT_ROOT}/${circuit_name}.blif"

  if [[ -f "${circuit_path}" ]]; then
    echo "${circuit_path}"
    return 0
  fi

  local match
  match="$(find "${CIRCUIT_ROOT}" -maxdepth 1 -iname "${circuit_name}.blif" -print -quit)"
  if [[ -n "${match}" ]]; then
    echo "${match}"
    return 0
  fi

  echo "Circuit not found for name '${circuit_name}' under ${CIRCUIT_ROOT}" >&2
  return 1
}

for experiment_dir in "${OUTPUTS_ROOT}"/*/; do
  run_name="$(basename "${experiment_dir}")"
  if [[ "${run_name}" == "${SKIP_DIR}" ]]; then
    continue
  fi

  # Run name examples:
  # - {time}_zhu2020_{circuit_name}_reproduce_{job_id}
  # - {time}_tb_zhu2020_{circuit_name}_{job_id}
  # We extract the token after "zhu2020" to get {circuit_name}.
  circuit_name=""
  IFS='_' read -r -a _run_parts <<< "${run_name}"
  for ((i=0; i<${#_run_parts[@]}; i++)); do
    if [[ "${_run_parts[$i]}" == "zhu2020" ]]; then
      if (( i + 1 < ${#_run_parts[@]} )); then
        circuit_name="${_run_parts[$((i+1))]}"
      fi
      break
    fi
  done
  if [[ -z "${circuit_name}" ]]; then
    echo "Could not parse circuit name from run: ${run_name}" >&2
    exit 1
  fi

  circuit_path="$(resolve_circuit_path "${circuit_name}")"

  echo "=== Sampling experiment: ${DATE}/${run_name} (circuit=${circuit_name}) ==="
  python -m src.sample_exp \
    --experiment "${DATE}/${run_name}" \
    --num-samples "${NUM_SAMPLES}" \
    --circuit "${circuit_path}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
done
