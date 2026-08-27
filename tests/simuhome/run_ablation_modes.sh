#!/usr/bin/env bash
# JSON-IR vs direct-code generation-mode ablation over SimuHome scenarios.
#
# Compares plan-generation modes (behavior_tree = JSON IR tool call,
# python_code = builder-DSL script, py_trees_code = free py_trees code
# executed by CodeBTExecutor) across SLMs served by Ollama and OpenAI
# reference models, on the same seeds/flavors machinery as the earlier
# TD-SOSA ablations (run_ablation_chatgpt.sh / run_ablation_ollama.sh).
#
# All outputs (CSVs, run logs, per-run result dirs, prompt dumps) are written
# to the current working directory at invocation time:
#
#   cd ~/experiments/ir_vs_code
#   BENCHMARK_DIR=~/gits/SimuHome/data/benchmark \
#   OLLAMA_BASE_URL=http://10.205.0.124:11434/v1 \
#   RUNS=1 SEEDS="1" MODELS="qwen2.5-coder:3b" \
#   ~/path/to/llm-agents-for-ami/tests/simuhome/run_ablation_modes.sh
#
# Grid dimensions (env-overridable; defaults = the decided experiment grid:
# modes x models, TD-SOSA on, signifiers cleared every run):
#   MODES        behavior_tree python_code py_trees_code
#   MODELS       qwen2.5-coder:3b codegemma:7b-instruct deepseek-coder:1.3b gpt-5-mini gpt-4o
#   FLAVORS      tdsosa_clear   (also: notdsosa_clear tdsosa_signifiers notdsosa_signifiers)
#   QUERY_TYPES  qt2            (also: qt1)
#   SEEDS        1 45 87 25 38
#   RUNS         10
#
# Backend selection per model:
#   - OPENAI_BASE_URL already exported -> used as-is (explicit override wins).
#   - model tag contains ':' or '/'    -> Ollama; requires OLLAMA_BASE_URL
#     (defuses run_cases.py's silent openrouter.ai fallback for ':' tags).
#   - otherwise                        -> OpenAI; requires a real OPENAI_API_KEY.
#
# Requires: HA_URL, HA_TOKEN, BENCHMARK_DIR. DRY_RUN=1 prints the composed
# commands without executing anything.
set -u

BENCHMARK_DIR="${BENCHMARK_DIR:?set BENCHMARK_DIR to your SimuHome data/benchmark dir}"
VIRTUAL_YAML_DIR="${VIRTUAL_YAML_DIR:-${HOME}/homeassistant/custom_components/virtual/}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS="${RUNS:-10}"
DRY_RUN="${DRY_RUN:-0}"

MODES=(${MODES:-behavior_tree python_code py_trees_code})
MODELS=(${MODELS:-qwen2.5-coder:3b codegemma:7b-instruct deepseek-coder:1.3b gpt-5-mini gpt-4o})
FLAVORS=(${FLAVORS:-tdsosa_clear})
QUERY_TYPES=(${QUERY_TYPES:-qt2})
# Diverse picks from the 15 seeds that passed 2/2 (see run_ablation_chatgpt.sh):
#   seed_1  temperature decrease (utility_room)
#   seed_45 illuminance increase (kitchen)
#   seed_87 humidity    increase (dining_room)
#   seed_25 pm10        decrease (bathroom)
#   seed_38 multi-goal: illuminance increase (living_room) + humidity increase (bathroom)
SEEDS=(${SEEDS:-1 45 87 25 38})

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [ "${RUNS}" -lt 1 ]; then
  echo "RUNS must be a positive integer; got: ${RUNS}" >&2
  exit 1
fi

EXPLICIT_BASE_URL="${OPENAI_BASE_URL:-}"
ORIG_OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ORIG_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-}"

echo "Writing all outputs to: ${OUT_DIR}"
echo "Grid: models=(${MODELS[*]}) modes=(${MODES[*]}) flavors=(${FLAVORS[*]}) query_types=(${QUERY_TYPES[*]}) seeds=(${SEEDS[*]}) runs=${RUNS}"

for model in "${MODELS[@]}"; do
  safe_model="${model//[^A-Za-z0-9._-]/_}"

  # Per-model backend env (exported for the run_simuhome_e2e.py subprocess).
  # Restore the caller's key/effort first so an Ollama model's dummy export
  # cannot leak into a later OpenAI model in the same sweep.
  export OPENAI_API_KEY="${ORIG_OPENAI_API_KEY}"
  export OPENAI_REASONING_EFFORT="${ORIG_REASONING_EFFORT}"
  extra_runner_args=()
  if [ -n "${EXPLICIT_BASE_URL}" ]; then
    export OPENAI_BASE_URL="${EXPLICIT_BASE_URL}"
  elif [[ "${model}" == *:* || "${model}" == */* ]]; then
    export OPENAI_BASE_URL="${OLLAMA_BASE_URL:?local model '${model}' needs OLLAMA_BASE_URL}"
    export OPENAI_API_KEY="dummy"
    export OPENAI_REASONING_EFFORT=""
    export LLM_TIMEOUT_DEFAULT="${LLM_TIMEOUT_DEFAULT:-600}"
    extra_runner_args+=(--discovery-timeout 1800)
  else
    unset OPENAI_BASE_URL
    if [ -z "${OPENAI_API_KEY}" ] || [ "${OPENAI_API_KEY}" = "dummy" ]; then
      echo "OpenAI model '${model}' needs a real OPENAI_API_KEY" >&2
      exit 1
    fi
  fi
  export OPENAI_MODEL="${model}"

  for mode in "${MODES[@]}"; do
    for flavor in "${FLAVORS[@]}"; do
      csv="${OUT_DIR}/results_${safe_model}_${mode}_${flavor}.csv"
      flavor_args=()
      case "${flavor}" in
        notdsosa_*) flavor_args+=(--no-td-sosa) ;;
      esac

      for qt in "${QUERY_TYPES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          scenario="${BENCHMARK_DIR}/${qt}_feasible_seed_${seed}.json"
          if [ ! -f "${scenario}" ] && [ "${DRY_RUN}" != "1" ]; then
            echo "Missing scenario: ${scenario}" >&2
            continue
          fi

          for i in $(seq 1 "${RUNS}"); do
            run_args=(${flavor_args[@]+"${flavor_args[@]}"})
            case "${flavor}" in
              *_clear) run_args+=(--clear-signifiers) ;;
              *_signifiers) [ "${i}" -eq 1 ] && run_args+=(--clear-signifiers) ;;
            esac

            timestamp="$(date +%Y%m%d_%H%M%S)"
            run_tag="${timestamp}_${safe_model}_${mode}_${flavor}_${qt}_seed${seed}_run${i}"
            log="${OUT_DIR}/mode_ablation_${run_tag}.log"

            cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/run_simuhome_e2e.py" "${scenario}"
              --virtual-yaml-dir "${VIRTUAL_YAML_DIR}"
              --generation-mode "${mode}"
              --results-dir "${OUT_DIR}/results/${run_tag}"
              --results-csv "${csv}"
              --dump-prompts
              --prompt-dump-dir "${OUT_DIR}/prompt_dumps/${run_tag}"
              ${extra_runner_args[@]+"${extra_runner_args[@]}"}
              ${run_args[@]+"${run_args[@]}"})

            echo "[${run_tag}] model=${model} mode=${mode} flavor=${flavor} ${qt} seed=${seed} run=${i}/${RUNS}"
            if [ "${DRY_RUN}" = "1" ]; then
              printf '  DRY_RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
              continue
            fi
            "${cmd[@]}" 2>&1 | tee "${log}"
            status="${PIPESTATUS[0]}"
            echo "[${run_tag}] exit=${status}"
          done
        done
      done
    done
  done
done

echo "Done. CSVs in ${OUT_DIR}/results_*.csv"
