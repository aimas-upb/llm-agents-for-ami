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
# wn11: cuInit fails after a 30 s kernel wait since the driver upgrade of
# 2026-10-13 (error 919); wn16: idle but without the home mount (exit 53).
# dgxa100-ncit-wn01: the same missing home mount.
#SBATCH --exclude=ucsc-precis-h200-wn11,ucsc-precis-h200-wn16,dgxa100-ncit-wn01
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
#   EPISODES="qt2_*,qt4-1_*,qt4-2_*,qt4-3_*" ATTEMPT=2 VARIANT=... sbatch --export=ALL ...
#     (a value with commas cannot go in the --export list, which splits on them;
#      exported from the submitting shell it travels with --export=ALL)
#   sbatch --export=ALL,VARIANT=...,ATTEMPT=smoke,LIMIT=1,WORKERS=2 ...
#   sbatch -A prof -p dgxa100 --export=ALL,VARIANT=... ...   # A100 pool (prof account)
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
# The bundle is copied to node-local disk first: Ollama probes the GPU by
# starting a llama-server with its CUDA libraries, under a 30 s watchdog, and
# a cold read of those 2 GB over the NFS home misses it -- Ollama then falls
# back to the CPU without failing. The copy is kept per node and reused.
OLLAMA_LOCAL="${OLLAMA_LOCAL:-/tmp/ollama-$(basename "$OLLAMA_DIR")-$USER}"
if [ ! -x "$OLLAMA_LOCAL/bin/ollama" ] || [ "$OLLAMA_DIR/bin/ollama" -nt "$OLLAMA_LOCAL/bin/ollama" ]; then
  echo "copying ollama bundle to $OLLAMA_LOCAL ..."
  rm -rf "$OLLAMA_LOCAL"; mkdir -p "$OLLAMA_LOCAL"
  cp -a "$OLLAMA_DIR/." "$OLLAMA_LOCAL/"
fi
OLLAMA_BIN="$OLLAMA_LOCAL/bin/ollama"

OLLAMA_PORT=$(( 11500 + SLURM_JOB_ID % 1000 ))
export OLLAMA_MODELS="$HOME/ollama_models"
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="-1"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-5}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CTX:-32768}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cuda_v13}"   # skip the cuda_v12 probe (30 s on its own)
"$OLLAMA_BIN" serve > "$JOB_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
cleanup() {
  kill ${WATCH_PID:-} 2>/dev/null || true      # the watchdogs first: they would restart what follows
  kill "$OLLAMA_PID" $(cat "$JOB_DIR/genre.pid" 2>/dev/null) 2>/dev/null || true
}
trap cleanup EXIT
# Wait for the GPU probe to finish, then insist on the GPU: a CPU-only server
# would run the whole grid at a crawl and poison every latency number.
for _ in $(seq 1 150); do
  grep -q 'msg="inference compute"' "$JOB_DIR/ollama.log" 2>/dev/null && break
  kill -0 "$OLLAMA_PID" 2>/dev/null || break
  sleep 2
done
if ! grep 'msg="inference compute"' "$JOB_DIR/ollama.log" | grep -q 'library=CUDA'; then
  echo "ERROR: ollama did not find the GPU (see $JOB_DIR/ollama.log):"
  grep -E 'inference compute|discovery|watchdog' "$JOB_DIR/ollama.log" | cut -c1-200 | tail -6
  exit 1
fi
curl -s -m 5 "http://127.0.0.1:${OLLAMA_PORT}/api/version" >/dev/null \
  || { echo "ERROR: ollama not answering on port ${OLLAMA_PORT}"; tail -5 "$JOB_DIR/ollama.log"; exit 1; }
echo "ollama up on port ${OLLAMA_PORT} on $(grep 'msg="inference compute"' "$JOB_DIR/ollama.log" | grep -oE 'description="[^"]+"' | head -1) (ctx ${OLLAMA_CONTEXT_LENGTH}, parallel ${OLLAMA_NUM_PARALLEL})"

# --- the variant, with the local URL pointed at this job's server ------------
VARIANT_RUN="$JOB_DIR/$(basename "$VARIANT_PATH")"
sed "s#http://localhost:11434/v1#http://127.0.0.1:${OLLAMA_PORT}/v1#g; s#http://127.0.0.1:11434/v1#http://127.0.0.1:${OLLAMA_PORT}/v1#g" \
  "$VARIANT_PATH" > "$VARIANT_RUN"

# --- optional: a Hugging Face model behind genre_server.py (`decoding: genre`) ---
# Set GENRE_MODEL_DIR to the model's weights; a variant's `genre` provider
# (http://localhost:11600/v1) is rewritten to this job's server. GENRE_BEAMS: beams of
# the answer (default 4); GENRE_LENGTH_PENALTY: beam score = sum logprob / length^penalty
# (default 1.0, transformers'; 0 = plain sum, no reward for a longer answer). GENRE_MAX_TOKENS: cap of
# the thinking (default 4096; a Qwen3 model thinks several times what a DeepSeek-R1 distill does).
# Without GENRE_MODEL_DIR nothing here runs.
#
# A watchdog restarts the server when its process is gone (job 300191: it was killed by
# the memory limit after 34 episodes and every later episode failed in 4 s with
# `Connection error`). GENRE_RESTARTS restarts at most (default 5); past that the job is
# stopped, so it never writes a grid of failed records. The episode whose request was in
# flight when the server died fails with `Connection error` (the OpenAI client's retries
# end before a restart does); `mem.log` in the job directory says what ran out of memory.
start_genre() {
  ( cd "$REPO" && exec "$PYTHON" -m tests.simuhome.bench.genre_server \
      --model-dir "$GENRE_MODEL_DIR" --port "$GENRE_PORT" \
      --num-beams "${GENRE_BEAMS:-4}" \
      --max-tokens "${GENRE_MAX_TOKENS:-4096}" \
      --length-penalty "${GENRE_LENGTH_PENALTY:-1.0}" ) >> "$JOB_DIR/genre_server.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$JOB_DIR/genre.pid"
  for _ in $(seq 1 300); do
    curl -s -m 2 "http://127.0.0.1:${GENRE_PORT}/health" >/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || { echo "ERROR: genre server died:"; tail -20 "$JOB_DIR/genre_server.log"; return 1; }
    sleep 2
  done
  echo "ERROR: genre server not answering on ${GENRE_PORT}"; tail -20 "$JOB_DIR/genre_server.log"; return 1
}
genre_watchdog() {
  local restarts=0
  while sleep 10; do
    kill -0 "$(cat "$JOB_DIR/genre.pid")" 2>/dev/null && continue
    restarts=$(( restarts + 1 ))
    echo "=== WATCHDOG $(date '+%F %T'): genre server gone (restart $restarts of ${GENRE_RESTARTS:-5}); memory:"
    tail -3 "$JOB_DIR/mem.log" 2>/dev/null
    if [ "$restarts" -gt "${GENRE_RESTARTS:-5}" ] || ! start_genre; then
      echo "=== WATCHDOG: giving up, stopping the job"
      kill -TERM "$MAIN_PID"; return 1
    fi
    echo "=== WATCHDOG $(date '+%F %T'): genre server back up"
  done
}
# GENRE_HF_REPO (e.g. Qwen/Qwen3-8B): download the weights into GENRE_MODEL_DIR first when
# they are not there (for a home quota too small to keep them: point GENRE_MODEL_DIR at the
# node-local disk, /tmp/...; the copy is kept per node and reused, like the ollama bundle).
if [ -n "${GENRE_MODEL_DIR:-}" ] && [ -n "${GENRE_HF_REPO:-}" ] && [ ! -f "$GENRE_MODEL_DIR/.complete" ]; then
  echo "downloading $GENRE_HF_REPO to $GENRE_MODEL_DIR ..."
  mkdir -p "$GENRE_MODEL_DIR"
  "$PYTHON" -c 'import os; from huggingface_hub import snapshot_download as d; d(os.environ["GENRE_HF_REPO"], local_dir=os.environ["GENRE_MODEL_DIR"], allow_patterns=["*.safetensors", "*.json", "*.txt"])' \
    && touch "$GENRE_MODEL_DIR/.complete" || { echo "ERROR: download of $GENRE_HF_REPO failed"; exit 1; }
fi
if [ -n "${GENRE_MODEL_DIR:-}" ]; then
  [ -d "$GENRE_MODEL_DIR" ] || { echo "ERROR: missing GENRE_MODEL_DIR $GENRE_MODEL_DIR"; exit 1; }
  GENRE_PORT=$(( 11800 + SLURM_JOB_ID % 200 ))
  sed -i "s#http://localhost:11600/v1#http://127.0.0.1:${GENRE_PORT}/v1#g" "$VARIANT_RUN"
  MAIN_PID=$$
  start_genre || exit 1
  echo "genre server up on port ${GENRE_PORT} (beams ${GENRE_BEAMS:-4}, model $GENRE_MODEL_DIR)"
  genre_watchdog &
  WATCH_PID=$!
fi

# --- memory log: what holds the job's RAM, once a minute, in $JOB_DIR/mem.log ----
# (the job has a hard limit, --mem; sacct's MaxRSS says how much, not whose.)
# One line per minute: the cgroup's anonymous and file-backed memory (GB), then the
# resident size (GB) of the processes that matter, by name.
mem_log() {
  local cg
  cg="/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)"
  while sleep 60; do
    {
      printf '%s  cgroup anon=%s file=%s GB  ' "$(date '+%T')" \
        "$(awk '$1=="anon"{printf "%.1f", $2/1073741824}' "$cg/memory.stat" 2>/dev/null)" \
        "$(awk '$1=="file"{printf "%.1f", $2/1073741824}' "$cg/memory.stat" 2>/dev/null)"
      ps -u "$USER" -o rss=,args= | awk '
        /genre_server/ {n="genre_server"} /ollama serve/ {n="ollama_serve"}
        /llama-server|ollama runner/ {n="runner"} /bench\.(harness|run)/ {n="harness"}
        n {sum[n]+=$1; n=""}
        END {for (k in sum) printf "%s=%.1f ", k, sum[k]/1048576; print ""}'
    } >> "$JOB_DIR/mem.log" 2>/dev/null
  done
}
mem_log &
WATCH_PID="${WATCH_PID:-} $!"

# Load every local model now (and fail early on a missing one), so the first
# episodes are not timed with model loading inside them.
MODELS=$(grep -oE 'model: *"?[A-Za-z0-9_./:-]+"?' "$VARIANT_RUN" | sed -E 's/model: *"?([^"]+)"?/\1/' | sort -u)
for m in $MODELS; do
  "$OLLAMA_BIN" list 2>/dev/null | awk '{print $1}' | grep -qx "$m" || continue   # OpenAI names are skipped
  echo "loading $m ..."
  curl -s -m 600 "http://127.0.0.1:${OLLAMA_PORT}/api/generate" \
    -d "{\"model\": \"$m\", \"keep_alive\": -1}" >/dev/null \
    || { echo "ERROR: could not load $m"; exit 1; }
done
"$OLLAMA_BIN" ps || true

# --- the harness --------------------------------------------------------------
cd "$REPO"
export SHTD_LOG_DIR="$JOB_DIR/shtd"      # each worker's SHTD stderr (worker.py), for post-mortems
export SIMUHOME_STARTUP_TIMEOUT="${SIMUHOME_STARTUP_TIMEOUT:-300}"   # simulator / SHTD start, seconds (worker.py)
ARGS=(--config "$VARIANT_RUN" --out "$OUT_DIR")
[ -n "${EPISODES:-}" ] && ARGS+=(--episodes "$EPISODES")
[ -n "${SEEDS:-}" ]    && ARGS+=(--seeds "$SEEDS")
[ -n "${LIMIT:-}" ]    && ARGS+=(--limit "$LIMIT")
[ -n "${WORKERS:-}" ]  && ARGS+=(--workers "$WORKERS")
[ -n "${NO_JUDGE:-}" ] && ARGS+=(--no-judge)
# The GPU behind every latency number of this run, beside the records.
NAME=$(grep -E '^name:' "$VARIANT_RUN" | head -1 | sed -E 's/^name: *"?([^"#]+)"?.*/\1/' | xargs || true)
[ -n "$NAME" ] || NAME=$(basename "$VARIANT_RUN" .yaml)   # HomeBench configs have no `name:` key
mkdir -p "$OUT_DIR/$NAME"
printf 'job %s  node %s  partition %s  gpu %s\n' "$SLURM_JOB_ID" "$(hostname -s)" "${SLURM_JOB_PARTITION:-?}" \
  "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$OUT_DIR/$NAME/gpu.txt"
echo "=== harness ${ARGS[*]}  ($(date '+%F %T'))"
"$PYTHON" -m tests.simuhome.bench.harness "${ARGS[@]}" --dry-run
"$PYTHON" -m tests.simuhome.bench.harness "${ARGS[@]}"
echo "=== done ($(date '+%F %T'))"
