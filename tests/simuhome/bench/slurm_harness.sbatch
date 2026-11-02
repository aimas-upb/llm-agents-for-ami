#!/bin/bash
#SBATCH --job-name=ami_harness
#SBATCH -A hria
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# One SimuHome harness run (tests/simuhome/bench/harness.py) on a GPU node,
# with the variant's local models served by an Ollama started on the node.
# No Home Assistant, SPADE or XMPP: the bench drives the agents in-process and
# starts its own simulator + SHTD per worker on free ports.
#
# Usage (from the repo root on fep, after `source` of nothing: the key is read
# from the repo's .env by run.py / evaluate.py):
#
#   sbatch --export=ALL,VARIANT=tests/simuhome/bench/variants/fep.M1.yaml \
#          tests/simuhome/bench/slurm_harness.sbatch
#   sbatch --export=ALL,VARIANT=...,ATTEMPT=2,EPISODES="qt2_*,qt4-1_*,qt4-2_*,qt4-3_*" ...
#   sbatch --export=ALL,VARIANT=...,ATTEMPT=smoke,LIMIT=1,WORKERS=2 ...
#
# Variables (all optional except VARIANT):
#   VARIANT   harness config; its local provider URL (http://localhost:11434/v1)
#             is rewritten to this job's Ollama port
#   ATTEMPT   records go to $OUT_ROOT/<attempt>/<variant name>/ (default 1);
#             repeated attempts of the same episodes = different ATTEMPT values
#   EPISODES, SEEDS, LIMIT, WORKERS   harness overrides (see harness.py)
#   NO_JUDGE=1                         score infeasible episodes without the LLM judge
#   OUT_ROOT  default $HOME/experiments/semantic/harness
#   OLLAMA_CTX  context length for every local model (default 32768)
#
# Ollama: one server per job on port 11500 + (job id mod 1000), every model
# kept loaded for the whole job (OLLAMA_KEEP_ALIVE=-1, several models resident
# at once) so phases never swap models in and out of VRAM, and parallel request
# slots so several workers' calls batch on the GPU.
set -euo pipefail

REPO="${REPO:-$HOME/repos/llm-agents-for-ami-semantic}"
PYTHON="${PYTHON:-$HOME/venvs/ami-semantic/bin/python}"
OLLAMA_DIR="${OLLAMA_DIR:-$HOME/opt/ollama}"
OUT_ROOT="${OUT_ROOT:-$HOME/experiments/semantic/harness}"
ATTEMPT="${ATTEMPT:-1}"
VARIANT="${VARIANT:?set VARIANT=<harness config yaml, relative to the repo or absolute>}"
case "$VARIANT" in /*) VARIANT_PATH="$VARIANT" ;; *) VARIANT_PATH="$REPO/$VARIANT" ;; esac

for path in "$REPO" "$REPO/.env" "$PYTHON" "$OLLAMA_DIR/bin/ollama" "$VARIANT_PATH" "$REPO/../SimuHome/data/benchmark"; do
  [ -e "$path" ] || { echo "ERROR: missing prerequisite $path"; exit 1; }
done

OUT_DIR="$OUT_ROOT/$ATTEMPT"
JOB_DIR="$OUT_ROOT/jobs/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR" "$JOB_DIR"
echo "host: $(hostname -f)  variant: $VARIANT  attempt: $ATTEMPT  out: $OUT_DIR"
nvidia-smi -L || true

# --- Ollama on a per-job port -------------------------------------------------
OLLAMA_PORT=$(( 11500 + SLURM_JOB_ID % 1000 ))
export OLLAMA_MODELS="$HOME/ollama_models"
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="-1"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-5}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CTX:-32768}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
"$OLLAMA_DIR/bin/ollama" serve > "$JOB_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
cleanup() { kill "$OLLAMA_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 30); do
  curl -s -m 2 "http://127.0.0.1:${OLLAMA_PORT}/api/version" >/dev/null && break
  kill -0 "$OLLAMA_PID" 2>/dev/null || break
  sleep 1
done
curl -s -m 2 "http://127.0.0.1:${OLLAMA_PORT}/api/version" >/dev/null \
  || { echo "ERROR: ollama did not start on port ${OLLAMA_PORT}"; tail -5 "$JOB_DIR/ollama.log"; exit 1; }
echo "ollama up on port ${OLLAMA_PORT} (ctx ${OLLAMA_CONTEXT_LENGTH}, parallel ${OLLAMA_NUM_PARALLEL})"

# --- the variant, with the local URL pointed at this job's server ------------
VARIANT_RUN="$JOB_DIR/$(basename "$VARIANT_PATH")"
sed "s#http://localhost:11434/v1#http://127.0.0.1:${OLLAMA_PORT}/v1#g; s#http://127.0.0.1:11434/v1#http://127.0.0.1:${OLLAMA_PORT}/v1#g" \
  "$VARIANT_PATH" > "$VARIANT_RUN"

# Load every local model now (and fail early on a missing one), so the first
# episodes are not timed with model loading inside them.
MODELS=$(grep -oE 'model: *"?[A-Za-z0-9_./:-]+"?' "$VARIANT_RUN" | sed -E 's/model: *"?([^"]+)"?/\1/' | sort -u)
for m in $MODELS; do
  "$OLLAMA_DIR/bin/ollama" list 2>/dev/null | awk '{print $1}' | grep -qx "$m" || continue   # OpenAI names are skipped
  echo "loading $m ..."
  curl -s -m 600 "http://127.0.0.1:${OLLAMA_PORT}/api/generate" \
    -d "{\"model\": \"$m\", \"keep_alive\": -1}" >/dev/null \
    || { echo "ERROR: could not load $m"; exit 1; }
done
"$OLLAMA_DIR/bin/ollama" ps || true

# --- the harness --------------------------------------------------------------
cd "$REPO"
ARGS=(--config "$VARIANT_RUN" --out "$OUT_DIR")
[ -n "${EPISODES:-}" ] && ARGS+=(--episodes "$EPISODES")
[ -n "${SEEDS:-}" ]    && ARGS+=(--seeds "$SEEDS")
[ -n "${LIMIT:-}" ]    && ARGS+=(--limit "$LIMIT")
[ -n "${WORKERS:-}" ]  && ARGS+=(--workers "$WORKERS")
[ -n "${NO_JUDGE:-}" ] && ARGS+=(--no-judge)
echo "=== harness ${ARGS[*]}  ($(date '+%F %T'))"
"$PYTHON" -m tests.simuhome.bench.harness "${ARGS[@]}" --dry-run
"$PYTHON" -m tests.simuhome.bench.harness "${ARGS[@]}"
echo "=== done ($(date '+%F %T'))"
