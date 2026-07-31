#!/usr/bin/env bash
# TD-SOSA / signifier ablation over 5 diverse passing SimuHome scenarios.
#
# Flavors (each appends to its own results CSV):
#   notdsosa_clear       : --no-td-sosa, signifiers cleared before every run
#   notdsosa_signifiers  : --no-td-sosa, signifiers cleared once per seed block (runs 2..N reuse)
#   tdsosa_clear         : TD-SOSA,      signifiers cleared before every run
#   tdsosa_signifiers    : TD-SOSA,      signifiers cleared once per seed block (runs 2..N reuse)
#
# All outputs (CSVs, run logs, per-run result dirs, prompt dumps) are written
# to the current working directory at invocation time, so run it from wherever
# you want the results to land.
#
# Assumes HA_URL, HA_TOKEN, OPENAI_API_KEY (and optional OPENAI_MODEL) are already exported.
set -u

BENCHMARK_DIR="${BENCHMARK_DIR:-/Users/cristi/aiml/gits/SimuHome/data/benchmark}"
VIRTUAL_YAML_DIR="${VIRTUAL_YAML_DIR:-${HOME}/homeassistant/custom_components/virtual/}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS="${RUNS:-10}"

# Diverse picks from the 15 seeds that passed 2/2:
#   seed_1  temperature decrease (utility_room)
#   seed_45 illuminance increase (kitchen)
#   seed_87 humidity    increase (dining_room)
#   seed_25 pm10        decrease (bathroom)
#   seed_38 multi-goal: illuminance increase (living_room) + humidity increase (bathroom)
SEEDS=(1 45 87 25 38)

# Override with e.g.: FLAVORS="tdsosa_clear tdsosa_signifiers" ./run_ablation_chatgpt.sh
FLAVORS=(${FLAVORS:-notdsosa_clear notdsosa_signifiers tdsosa_clear tdsosa_signifiers})

for flavor in "${FLAVORS[@]}"; do
  csv="${OUT_DIR}/results_chatgpt_${flavor}.csv"
  extra_args=()
  case "${flavor}" in
    notdsosa_*) extra_args+=(--no-td-sosa) ;;
  esac

  for seed in "${SEEDS[@]}"; do
    scenario="${BENCHMARK_DIR}/qt2_feasible_seed_${seed}.json"
    if [ ! -f "${scenario}" ]; then
      echo "Missing scenario: ${scenario}" >&2
      continue
    fi

    for i in $(seq 1 "${RUNS}"); do
      run_args=(${extra_args[@]+"${extra_args[@]}"})
      case "${flavor}" in
        *_clear) run_args+=(--clear-signifiers) ;;
        *_signifiers) [ "${i}" -eq 1 ] && run_args+=(--clear-signifiers) ;;
      esac

      timestamp="$(date +%Y%m%d_%H%M%S)"
      model_tag="${OPENAI_MODEL:-chatgpt}"
      safe_model="${model_tag//[^A-Za-z0-9._-]/_}"
      run_tag="${timestamp}_${safe_model}_${flavor}_seed${seed}_run${i}"
      log="${OUT_DIR}/chatgpt_ablation_${run_tag}.log"
      results_dir="${OUT_DIR}/results/${run_tag}"
      prompt_dump_dir="${OUT_DIR}/prompt_dumps/${run_tag}"

      echo "===== [${flavor}] seed_${seed} run ${i}/${RUNS} ====="
      "${PYTHON_BIN}" "${SCRIPT_DIR}/run_simuhome_e2e.py" \
        --virtual-yaml-dir "${VIRTUAL_YAML_DIR}" \
        --discovery-timeout 1800 \
        --dump-prompts \
        --prompt-dump-dir "${prompt_dump_dir}" \
        --results-dir "${results_dir}" \
        --results-csv "${csv}" \
        ${run_args[@]+"${run_args[@]}"} \
        "${scenario}" \
        2>&1 | tee "${log}"
      echo "exit=${PIPESTATUS[0]}"
    done
  done
done

echo "ChatGPT ablation complete. Outputs in ${OUT_DIR}:"
for flavor in "${FLAVORS[@]}"; do
  echo "  results_chatgpt_${flavor}.csv"
done
echo "  results/<run_tag>/ (manifests, run_cases.log, work dirs)"
echo "  prompt_dumps/<run_tag>/ (LLM prompt/answer dumps)"
