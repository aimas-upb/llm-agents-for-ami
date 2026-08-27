#!/bin/bash
#SBATCH --job-name=ami_ollama
#SBATCH -A hria
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=ami_ollama_%j.out
#SBATCH --error=ami_ollama_%j.err
#
# Serves the EACL 2027 experiment SLMs over Ollama's OpenAI-compatible API
# from a GPU node, for the IR-vs-code ablation
# (tests/simuhome/run_ablation_modes.sh).
#
# One GPU is deliberate: the SLMs are 1.3B-7B and the experiment is
# harness-bottlenecked (settling times, HA polling), not GPU-bottlenecked.
# Override anything at submit time -- CLI flags beat these directives:
#   sbatch -A prof -p dgxh100 --gres=gpu:8 --time=4:00:00 slurm_ollama_serve.sbatch
#
# The job prints the OLLAMA_BASE_URL to use from the harness side and then
# serves until the walltime ends. Models are pulled into $HOME/ollama_models
# once and reused by every later job.
set -euo pipefail

OLLAMA_DIR="${OLLAMA_DIR:-$HOME/opt/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama_models}"
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT:-11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
PULL_MODELS="${PULL_MODELS:-qwen2.5-coder:3b codegemma:7b-instruct deepseek-coder:1.3b deepseek-coder:1.3b-instruct}"

[ -x "$OLLAMA_DIR/bin/ollama" ] || { echo "ERROR: missing $OLLAMA_DIR/bin/ollama"; exit 1; }
mkdir -p "$OLLAMA_MODELS"

PORT="${OLLAMA_HOST##*:}"
echo "host:            $(hostname -f)"
echo "OLLAMA_BASE_URL: http://$(hostname -f):${PORT}/v1"
echo "models dir:      $OLLAMA_MODELS"
nvidia-smi -L || true

"$OLLAMA_DIR/bin/ollama" serve &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for the API to come up (the client honours OLLAMA_HOST for its target).
for i in $(seq 1 60); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/api/version" >/dev/null; then break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: ollama serve died"; exit 1; }
  sleep 1
done
curl -s -m 2 "http://127.0.0.1:${PORT}/api/version"; echo

# Pull experiment models (no-op when already in $OLLAMA_MODELS). A bad tag
# warns instead of killing the serving job.
export OLLAMA_HOST="127.0.0.1:${PORT}"
for model in ${PULL_MODELS}; do
  if "$OLLAMA_DIR/bin/ollama" show "$model" >/dev/null 2>&1; then
    echo "already present: $model"
  else
    echo "pulling: $model"
    "$OLLAMA_DIR/bin/ollama" pull "$model" || echo "WARNING: pull failed for $model"
  fi
done
"$OLLAMA_DIR/bin/ollama" list

# Warm-up + throughput sanity per model via the OpenAI-compatible endpoint.
for model in ${PULL_MODELS}; do
  "$OLLAMA_DIR/bin/ollama" show "$model" >/dev/null 2>&1 || continue
  echo "--- warmup: $model"
  t0=$(date +%s)
  curl -s -m 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${model}\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: ok\"}], \"max_tokens\": 20}" \
    | head -c 400
  echo
  echo "    (${model}: $(( $(date +%s) - t0 ))s round-trip)"
done

echo "READY -- serving until walltime. Use from fep / other nodes:"
echo "  export OLLAMA_BASE_URL=http://$(hostname -f):${PORT}/v1"
wait "$SERVER_PID"
