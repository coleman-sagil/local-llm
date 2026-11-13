#!/usr/bin/env bash
# gui/launch.sh <model>   (qwen35, bolt, nano, qwythos, ...; ports match bin/start-model.sh)
#
# "Desktop app" launcher for one model's built-in llama-server chat webui:
#   1. Ensures the model's llama-server is up, via bin/start-model.sh
#      <model> (GPU mode by default; pass --cpu-only to force CPU).
#   2. Opens the model's webui URL in a detected browser's "app mode"
#      (no tabs / no address bar), which reads as a standalone desktop app.
#
# This is what gui/local-llm-scout.desktop and gui/local-llm-qwen.desktop
# point their Exec= at, and what gui/picker_server.py shells out to.
set -euo pipefail

usage() {
  echo "Usage: $0 <scout|qwen|qwen-next|qwen2.5|wrn|qwen35|bolt|nano|qwythos>" >&2
  exit 1
}

MODEL="${1:-}"
[[ -n "$MODEL" ]] || usage

case "$MODEL" in
  scout)     PORT=8090 ;;
  qwen)      PORT=8091 ;;
  qwen-next) PORT=8092 ;;
  qwen2.5)   PORT=8093 ;;
  wrn)       PORT=8094 ;;
  qwen35)    PORT=8095 ;;
  bolt)      PORT=8096 ;;
  nano)      PORT=8097 ;;
  qwythos)   PORT=8098 ;;
  *)
    echo "Error: unknown model '$MODEL'. Must be one of: scout, qwen, qwen-next, qwen2.5, wrn, qwen35, bolt, nano, qwythos." >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
URL="http://127.0.0.1:$PORT"

# Accept an optional --cpu-only flag (default: GPU mode).
START_FLAGS=()
if [[ "${2:-}" == "--cpu-only" ]]; then
  START_FLAGS+=(--cpu-only)
  MODE_LABEL="CPU-only"
else
  MODE_LABEL="GPU"
fi

# If the server is already up (user already launched it, or double-clicked
# the app icon), skip straight to opening a browser window instead of
# tripping start-model.sh's "already running" guard.
if curl -sf -o /dev/null -m 2 "$URL/health" 2>/dev/null; then
  echo "$MODEL is already running at $URL"
else
  echo "Starting $MODEL ($MODE_LABEL) ..."
  if ! "$ROOT_DIR/bin/start-model.sh" "$MODEL" "${START_FLAGS[@]+"${START_FLAGS[@]}"}"; then
    echo "Error: failed to start $MODEL. See $ROOT_DIR/logs/$MODEL.log for details." >&2
    exit 1
  fi
fi

# Detect an installed browser, in the documented preference order.
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
nohup "$BROWSER_BIN" "${BROWSER_ARGS[@]}" >/tmp/local-llm-gui-"$MODEL"-browser.log 2>&1 &
disown

echo "Launched $MODEL GUI: $URL"
