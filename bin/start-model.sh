#!/usr/bin/env bash
# start-model.sh <scout|qwen|qwen-next> [--cpu-only]
#
# Launches llama-server for the given model, backgrounded, and waits until it
# is confirmed listening (or confirmed failed) before returning.
set -euo pipefail

usage() {
  echo "Usage: $0 <scout|qwen|qwen-next> [--cpu-only]" >&2
  exit 1
}

MODEL="${1:-}"
[[ -n "$MODEL" ]] || usage

case "$MODEL" in
  scout|qwen|qwen-next) ;;
  *)
    echo "Error: unknown model '$MODEL'. Must be 'scout', 'qwen', or 'qwen-next'." >&2
    exit 1
    ;;
esac

CPU_ONLY=0
if [[ $# -ge 2 ]]; then
  case "${2}" in
    --cpu-only) CPU_ONLY=1 ;;
    *)
      echo "Error: unknown option '${2}'. Only '--cpu-only' is supported." >&2
      exit 1
      ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_SERVER="$ROOT_DIR/llama.cpp/build/bin/llama-server"
LOG_DIR="$ROOT_DIR/logs"
RUN_DIR="$ROOT_DIR/run"

mkdir -p "$LOG_DIR" "$RUN_DIR"

case "$MODEL" in
  scout)
    PORT=8090
    MODEL_PATH="$ROOT_DIR/models/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf"
    GPU_CTX=2048
    ;;
  qwen)
    PORT=8091
    MODEL_PATH="$ROOT_DIR/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_M.gguf"
    GPU_CTX=4096
    ;;
  qwen-next)
    # General-purpose sibling of qwen (Qwen3-Coder-Next): same Qwen3-Next
    # 80B-A3B hybrid-attention MoE backbone, instruct-tuned generally
    # instead of fine-tuned for code. Same offload recipe applies.
    PORT=8092
    MODEL_PATH="$ROOT_DIR/models/qwen3-next-instruct/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf"
    GPU_CTX=4096
    ;;
esac

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "Error: llama-server binary not found or not executable: $LLAMA_SERVER" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: model file not found: $MODEL_PATH" >&2
  exit 1
fi

LOG_FILE="$LOG_DIR/$MODEL.log"
PID_FILE="$RUN_DIR/$MODEL.pid"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Error: $MODEL already appears to be running (PID $EXISTING_PID, see $PID_FILE)." >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

if [[ "$CPU_ONLY" -eq 1 ]]; then
  EXTRA_ARGS=(-ngl 0)
  CTX=4096
else
  EXTRA_ARGS=(-ngl 999 --cpu-moe)
  CTX="$GPU_CTX"

  if systemctl is-active --quiet nicehash-miner.service; then
    echo "nicehash-miner.service is active; stopping it to free the GPU for inference..."
    sudo -n systemctl stop nicehash-miner.service
    systemctl --user stop mining-idle-watcher.service || true
    echo
    echo "REMINDER: mining will NOT restart automatically."
    echo "When you are done with the model, run:"
    echo "  systemctl --user start mining-idle-watcher.service"
    echo
  fi
fi

# Truncate the log so our "listening on" / failure grep only ever sees this run.
: > "$LOG_FILE"

nohup "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  -c "$CTX" \
  --jinja \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
disown

echo "$SERVER_PID" > "$PID_FILE"
echo "Starting $MODEL (PID $SERVER_PID), logging to $LOG_FILE ..."

FAIL_PATTERN='error loading model|failed to load model|exiting due to model loading error|CUDA error|Segmentation fault|terminate called|Aborted'
TIMEOUT_SECS=120
ELAPSED=0
POLL_INTERVAL=0.5

while true; do
  if grep -qE "listening on" "$LOG_FILE" 2>/dev/null; then
    echo "$MODEL is up: http://127.0.0.1:$PORT"
    exit 0
  fi

  if grep -qE "$FAIL_PATTERN" "$LOG_FILE" 2>/dev/null; then
    echo "Error: $MODEL failed to start. Relevant log output:" >&2
    grep -E "$FAIL_PATTERN" "$LOG_FILE" >&2 || true
    rm -f "$PID_FILE"
    exit 1
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Error: $MODEL process exited before listening. Last log lines:" >&2
    tail -n 40 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi

  if awk "BEGIN{exit !($ELAPSED >= $TIMEOUT_SECS)}"; then
    echo "Error: timed out after ${TIMEOUT_SECS}s waiting for $MODEL to report listening. Last log lines:" >&2
    tail -n 40 "$LOG_FILE" >&2
    kill "$SERVER_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 1
  fi

  sleep "$POLL_INTERVAL"
  ELAPSED=$(awk "BEGIN{print $ELAPSED + $POLL_INTERVAL}")
done
