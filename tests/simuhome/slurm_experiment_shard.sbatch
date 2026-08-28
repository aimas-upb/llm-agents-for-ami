#!/bin/bash
#SBATCH --job-name=ami_shard
#SBATCH -A hria
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --output=ami_shard_%j.out
#SBATCH --error=ami_shard_%j.err
#
# Self-contained EACL 2027 experiment shard: Ollama + HomeAssistant Core +
# SPADE XMPP + the SimuHome mode-ablation runner, all on one GPU node over
# localhost. Multiple shards run in parallel on different nodes by splitting
# the grid via env vars:
#
#   sbatch --export=ALL,MODELS="qwen2.5-coder:3b",RUNS=10 slurm_experiment_shard.sbatch
#   sbatch -p dgxh100 --export=ALL,MODELS="codegemma:7b-instruct" ...
#
# Prerequisites (once, on fep -- already done):
#   ~/opt/ollama            Ollama bundle          ~/ollama_models  pulled models
#   ~/venvs/ami             agents venv            ~/venvs/ha       HA Core venv
#   ~/ha_config             onboarded HA config (patched hass-virtual, token
#                           user in .storage)      ~/.ha_token      long-lived token
#   ~/repos/llm-agents-for-ami                     ~/repos/SimuHome benchmark
#
# The HA config dir is COPIED to node-local /tmp per job, so concurrent shards
# never share HA state; the long-lived token stays valid because the auth
# storage is copied with it.
#
# CAUTION — do NOT schedule two shards onto the SAME node: Ollama and HASP
# ports are per-job now, but HomeAssistant (8123), SPADE XMPP (5222) and the
# notification webhook (8086) still bind fixed ports, so the second shard
# silently borrows the first one's services — and the stale-workspace sweep
# can then delete the other shard's live workspace when seeds coincide.
# Place concurrent shards on different partitions (or --exclude the busy node). Grid env vars (MODES/MODELS/FLAVORS/QUERY_TYPES/
# SEEDS/RUNS) pass straight through to run_ablation_modes.sh. Results land in
# ~/experiments/shard_${SLURM_JOB_ID} on shared home.
set -euo pipefail

REPO="${REPO:-$HOME/repos/llm-agents-for-ami}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/repos/SimuHome/data/benchmark}"
OLLAMA_DIR="${OLLAMA_DIR:-$HOME/opt/ollama}"
OUT_DIR="${OUT_DIR:-$HOME/experiments/shard_${SLURM_JOB_ID}}"
HA_SRC_CONFIG="${HA_SRC_CONFIG:-$HOME/ha_config}"
HA_RUN_CONFIG="/tmp/ha_config_${SLURM_JOB_ID}"

for path in "$REPO" "$BENCHMARK_DIR" "$OLLAMA_DIR/bin/ollama" "$HA_SRC_CONFIG" "$HOME/.ha_token" \
            "$HOME/venvs/ami/bin/python" "$HOME/venvs/ha/bin/hass"; do
  [ -e "$path" ] || { echo "ERROR: missing prerequisite $path"; exit 1; }
done

mkdir -p "$OUT_DIR"
echo "host: $(hostname -f)  out: $OUT_DIR"
nvidia-smi -L || true

# Node-local repo copy: the signifier storage lives inside the repo tree
# (ami_agents/shared/memory/storage), so concurrent shards sharing the NFS
# checkout would contaminate each other's signifier state.
REPO_RUN="/tmp/ami_repo_${SLURM_JOB_ID}"
rm -rf "$REPO_RUN"
cp -a "$REPO" "$REPO_RUN"

cleanup() {
  kill $(jobs -p) 2>/dev/null || true
  rm -rf "$HA_RUN_CONFIG" "$REPO_RUN"
}
trap cleanup EXIT

# ---- Ollama (GPU) ----
# Per-job port: with the fixed 11434, a second job on the same node fails to
# bind and silently *borrows* the first job's server through the health check
# — and loses its LLM endpoint mid-shard when that other job exits (observed:
# 50 zero-llm-call rows). A unique port makes the health check prove OUR
# server is alive, and the kill -0 check catches a dead serve process.
OLLAMA_PORT=$(( 11500 + SLURM_JOB_ID % 1000 ))
export OLLAMA_MODELS="$HOME/ollama_models"
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="30m"
"$OLLAMA_DIR/bin/ollama" serve > "$OUT_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 60); do
  curl -s -m 2 "http://127.0.0.1:${OLLAMA_PORT}/api/version" >/dev/null && break
  kill -0 "$OLLAMA_PID" 2>/dev/null || break
  sleep 1
done
kill -0 "$OLLAMA_PID" 2>/dev/null || { echo "ERROR: ollama serve died (port ${OLLAMA_PORT} taken?)"; tail -5 "$OUT_DIR/ollama.log"; exit 1; }
curl -s -m 2 "http://127.0.0.1:${OLLAMA_PORT}/api/version" >/dev/null || { echo "ERROR: ollama did not start"; exit 1; }
echo "ollama up on port ${OLLAMA_PORT}"

# ---- HomeAssistant Core (node-local copy of the onboarded config) ----
rm -rf "$HA_RUN_CONFIG"
cp -a "$HA_SRC_CONFIG" "$HA_RUN_CONFIG"
"$HOME/venvs/ha/bin/hass" -c "$HA_RUN_CONFIG" > "$OUT_DIR/hass.log" 2>&1 &
HA_TOKEN="$(cat "$HOME/.ha_token")"
for i in $(seq 1 120); do
  curl -s -m 3 -H "Authorization: Bearer $HA_TOKEN" http://127.0.0.1:8123/api/ 2>/dev/null | grep -q "API running" && break
  sleep 5
done
curl -s -m 3 -H "Authorization: Bearer $HA_TOKEN" http://127.0.0.1:8123/api/ | grep -q "API running" || { echo "ERROR: HA did not start"; tail -20 "$OUT_DIR/hass.log"; exit 1; }
echo "homeassistant up"

# ---- SPADE XMPP server ----
"$HOME/venvs/ami/bin/spade" run > "$OUT_DIR/spade.log" 2>&1 &
for i in $(seq 1 30); do
  (exec 3<>/dev/tcp/127.0.0.1/5222) 2>/dev/null && { exec 3>&-; break; }
  sleep 1
done
echo "spade up"

# ---- The ablation runner ----
export HA_URL="ws://localhost:8123/api/websocket"
export HA_TOKEN
export HA_VIRTUAL_CONFIG_PREFIX="$HA_RUN_CONFIG/custom_components/virtual"
export VIRTUAL_YAML_DIR="$HA_RUN_CONFIG/custom_components/virtual"
export OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}/v1"
export SPADE_SERVER=localhost
export SPADE_PASSWORD=password
export PYTHON_BIN="$HOME/venvs/ami/bin/python"
export BENCHMARK_DIR

cd "$OUT_DIR"
bash "$REPO_RUN/tests/simuhome/run_ablation_modes.sh"
echo "shard finished"
