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
# NLU/planner role split (docs/multi_model_choice.md):
#   NLU_MODEL    unset (default) -> single-model protocol: each MODELS entry
#                runs both agents (UserAssistant intent extraction AND the
#                InteractionSolver planner).
#                set, e.g. NLU_MODEL=gpt-5-mini -> the UserAssistant always
#                runs NLU_MODEL (on its own backend, resolved by the rules
#                above) and each MODELS entry is the *planner* model only, so
#                the generation mode is exercised even by SLMs that cannot
#                extract intents. CSV names gain a "_nlu-<model>" tag and every
#                row records planner_model_name next to model_name.
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
# Per-invocation HASP port so co-located shards on one node neither collide
# on the bind nor kill each other's adapter in the pre-run sweep. Derived
# from the SLURM job id when present; override with HASP_PORT.
HASP_PORT="${HASP_PORT:-$(( 18000 + ${SLURM_JOB_ID:-0} % 1000 ))}"

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
NLU_MODEL="${NLU_MODEL:-}"

# resolve_backend <model-tag>: sets BACKEND_BASE_URL (empty = OpenAI default),
# BACKEND_API_KEY, BACKEND_REASONING_EFFORT and BACKEND_LOCAL (1 = Ollama).
resolve_backend() {
  local tag="$1"
  BACKEND_LOCAL=0
  BACKEND_API_KEY="${ORIG_OPENAI_API_KEY}"
  BACKEND_REASONING_EFFORT="${ORIG_REASONING_EFFORT}"
  if [ -n "${EXPLICIT_BASE_URL}" ]; then
    BACKEND_BASE_URL="${EXPLICIT_BASE_URL}"
  elif [[ "${tag}" == *:* || "${tag}" == */* ]]; then
    BACKEND_BASE_URL="${OLLAMA_BASE_URL:?local model '${tag}' needs OLLAMA_BASE_URL}"
    BACKEND_API_KEY="dummy"
    BACKEND_REASONING_EFFORT=""
    BACKEND_LOCAL=1
  else
    BACKEND_BASE_URL=""
    if [ -z "${BACKEND_API_KEY}" ] || [ "${BACKEND_API_KEY}" = "dummy" ]; then
      echo "OpenAI model '${tag}' needs a real OPENAI_API_KEY" >&2
      exit 1
    fi
  fi
}

echo "Writing all outputs to: ${OUT_DIR}"
echo "Grid: models=(${MODELS[*]}) modes=(${MODES[*]}) flavors=(${FLAVORS[*]}) query_types=(${QUERY_TYPES[*]}) seeds=(${SEEDS[*]}) runs=${RUNS}"
if [ -n "${NLU_MODEL}" ]; then
  echo "Role split: UserAssistant (NLU) = ${NLU_MODEL}; MODELS entries are planner models"
fi

for model in "${MODELS[@]}"; do
  safe_model="${model//[^A-Za-z0-9._-]/_}"

  # Per-model backend env (exported for the run_simuhome_e2e.py subprocess).
  # Restore the caller's key/effort first so an Ollama model's dummy export
  # cannot leak into a later OpenAI model in the same sweep; likewise drop any
  # PLANNER_* from a previous iteration.
  export OPENAI_API_KEY="${ORIG_OPENAI_API_KEY}"
  export OPENAI_REASONING_EFFORT="${ORIG_REASONING_EFFORT}"
  unset PLANNER_MODEL PLANNER_BASE_URL PLANNER_API_KEY PLANNER_REASONING_EFFORT
  extra_runner_args=()
  csv_tag="${safe_model}"

  resolve_backend "${model}"
  any_local="${BACKEND_LOCAL}"
  if [ -n "${NLU_MODEL}" ]; then
    # Role split: this MODELS entry drives only the InteractionSolver planner
    # (PLANNER_* -> planning.llm_planning in run_cases.py); the shared
    # OPENAI_* provider below is the UserAssistant's fixed NLU model.
    export PLANNER_MODEL="${model}"
    export PLANNER_BASE_URL="${BACKEND_BASE_URL:-https://api.openai.com/v1}"
    export PLANNER_API_KEY="${BACKEND_API_KEY}"
    export PLANNER_REASONING_EFFORT="${BACKEND_REASONING_EFFORT}"
    safe_nlu="${NLU_MODEL//[^A-Za-z0-9._-]/_}"
    csv_tag="${safe_model}_nlu-${safe_nlu}"
    resolve_backend "${NLU_MODEL}"
    [ "${BACKEND_LOCAL}" = "1" ] && any_local=1
    export OPENAI_MODEL="${NLU_MODEL}"
  else
    export OPENAI_MODEL="${model}"
  fi
  if [ -n "${BACKEND_BASE_URL}" ]; then
    export OPENAI_BASE_URL="${BACKEND_BASE_URL}"
  else
    unset OPENAI_BASE_URL
  fi
  export OPENAI_API_KEY="${BACKEND_API_KEY}"
  export OPENAI_REASONING_EFFORT="${BACKEND_REASONING_EFFORT}"
  if [ "${any_local}" = "1" ]; then
    export LLM_TIMEOUT_DEFAULT="${LLM_TIMEOUT_DEFAULT:-600}"
    extra_runner_args+=(--discovery-timeout 1800)
  fi

  for mode in "${MODES[@]}"; do
    for flavor in "${FLAVORS[@]}"; do
      csv="${OUT_DIR}/results_${csv_tag}_${mode}_${flavor}.csv"
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
            run_tag="${timestamp}_${csv_tag}_${mode}_${flavor}_${qt}_seed${seed}_run${i}"
            log="${OUT_DIR}/mode_ablation_${run_tag}.log"

            cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/run_simuhome_e2e.py" "${scenario}"
              --virtual-yaml-dir "${VIRTUAL_YAML_DIR}"
              --hasp-port "${HASP_PORT}"
              --generation-mode "${mode}"
              --results-dir "${OUT_DIR}/results/${run_tag}"
              --results-csv "${csv}"
              --dump-prompts
              --prompt-dump-dir "${OUT_DIR}/prompt_dumps/${run_tag}"
              ${extra_runner_args[@]+"${extra_runner_args[@]}"}
              ${run_args[@]+"${run_args[@]}"})

            echo "[${run_tag}] model=${model}${NLU_MODEL:+ nlu=${NLU_MODEL}} mode=${mode} flavor=${flavor} ${qt} seed=${seed} run=${i}/${RUNS}"
            if [ "${DRY_RUN}" = "1" ]; then
              printf '  DRY_RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
              continue
            fi
            # A run that aborted mid-flight can leak its HASP/adapter or
            # sidecar; a leftover bound to the HASP port makes every later
            # run die at startup ("HASP exited unexpectedly"). Sweep before
            # starting — but scope the patterns to THIS invocation (the
            # per-job HASP port and this script's repo path): unscoped
            # pkills SIGTERM a co-located shard's services on shared nodes
            # (observed: 71 runs, zero yield, sidecar killed with -15).
            pkill -f "uvicorn hasp:app --host 127.0.0.1 --port ${HASP_PORT}" 2>/dev/null || true
            pkill -f "uvicorn ygg_ha_adapter:app --host 127.0.0.1 --port ${HASP_PORT}" 2>/dev/null || true
            pkill -f "${SCRIPT_DIR}/simulate_simuhome_scenario[.]py" 2>/dev/null || true
            sleep 1
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
