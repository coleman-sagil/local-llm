#!/usr/bin/env bash
# gui/launch-picker.sh
#
# Starts gui/picker_server.py (the tiny local backend for picker.html's
# click-to-launch buttons) if it isn't already running, then opens the
# picker page in a detected browser's app-mode window -- this is the
# "front door" the user actually clicks day to day.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$ROOT_DIR/run"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$RUN_DIR/picker-server.pid"
LOG_FILE="$LOG_DIR/picker-server.log"

PICKER_PORT=8089
URL="http://127.0.0.1:$PICKER_PORT/"

mkdir -p "$RUN_DIR" "$LOG_DIR"

server_up() {
  curl -sf -o /dev/null -m 2 "http://127.0.0.1:$PICKER_PORT/health" 2>/dev/null
}

if server_up; then
  echo "picker_server already running at $URL"
else
  echo "Starting picker_server.py on port $PICKER_PORT ..."
  : > "$LOG_FILE"
  nohup python3 "$SCRIPT_DIR/picker_server.py" "$PICKER_PORT" >> "$LOG_FILE" 2>&1 &
  SERVER_PID=$!
  disown
  echo "$SERVER_PID" > "$PID_FILE"

  ELAPSED=0
  TIMEOUT_SECS=10
  until server_up; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Error: picker_server.py exited before becoming ready. Log:" >&2
      tail -n 40 "$LOG_FILE" >&2
      rm -f "$PID_FILE"
      exit 1
    fi
    if awk "BEGIN{exit !($ELAPSED >= $TIMEOUT_SECS)}"; then
      echo "Error: timed out after ${TIMEOUT_SECS}s waiting for picker_server.py." >&2
      tail -n 40 "$LOG_FILE" >&2
      exit 1
    fi
    sleep 0.2
    ELAPSED=$(awk "BEGIN{print $ELAPSED + 0.2}")
  done
  echo "picker_server up at $URL"
fi

BROWSER_BIN=""
BROWSER_ARGS=()
if command -v google-chrome >/dev/null 2>&1; then
  BROWSER_BIN="google-chrome"
  BROWSER_ARGS=(--app="$URL")
elif command -v chromium >/dev/null 2>&1; then
  BROWSER_BIN="chromium"
  BROWSER_ARGS=(--app="$URL")
elif command -v chromium-browser >/dev/null 2>&1; then
  BROWSER_BIN="chromium-browser"
  BROWSER_ARGS=(--app="$URL")
elif command -v firefox >/dev/null 2>&1; then
  BROWSER_BIN="firefox"
  BROWSER_ARGS=(--new-window --url "$URL")
else
  echo "Error: no supported browser found (checked google-chrome, chromium, chromium-browser, firefox)." >&2
  exit 1
fi

echo "Opening $URL in app-mode window via $BROWSER_BIN ..."
nohup "$BROWSER_BIN" "${BROWSER_ARGS[@]}" >/tmp/local-llm-gui-picker-browser.log 2>&1 &
disown

echo "Launched picker: $URL"
