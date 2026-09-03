#!/usr/bin/env bash
# stop-model.sh <qwen|qwen-next|qwen2.5|wrn>
#
# Stops the llama-server process started by start-model.sh for the given
# model. Does not touch mining/systemd in any way.
set -euo pipefail

usage() {
  echo "Usage: $0 <qwen|qwen-next|qwen2.5|wrn>" >&2
  exit 1
}

MODEL="${1:-}"
[[ -n "$MODEL" ]] || usage

case "$MODEL" in
  qwen|qwen-next|qwen2.5|wrn) ;;
  *)
    echo "Error: unknown model '$MODEL'. Must be 'qwen', 'qwen-next', 'qwen2.5', or 'wrn'." >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$ROOT_DIR/run"
PID_FILE="$RUN_DIR/$MODEL.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Error: no PID file for $MODEL at $PID_FILE (is it running?)." >&2
  exit 1
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"

if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  echo "Warning: process for $MODEL (PID ${PID:-unknown}) is not running; removing stale PID file." >&2
  rm -f "$PID_FILE"
  exit 0
fi

echo "Stopping $MODEL (PID $PID)..."

# Kill any direct children first (process tree), then the main process.
CHILDREN="$(pgrep -P "$PID" 2>/dev/null || true)"
if [[ -n "$CHILDREN" ]]; then
  kill $CHILDREN 2>/dev/null || true
fi
kill "$PID" 2>/dev/null || true

for _ in $(seq 1 50); do
  if ! kill -0 "$PID" 2>/dev/null; then
    break
  fi
  sleep 0.2
done

if kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID did not exit gracefully; sending SIGKILL." >&2
  if [[ -n "$CHILDREN" ]]; then
    kill -9 $CHILDREN 2>/dev/null || true
  fi
  kill -9 "$PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "$MODEL stopped."
