#!/usr/bin/env bash
# start-model.sh <qwen|qwen-next|qwen2.5|wrn> [--cpu-only]
#
# Launches llama-server for the given model, backgrounded, and waits until it
# is confirmed listening (or confirmed failed) before returning.
#
# Model families and the offload recipe each one needs:
#   qwen       Qwen3-Coder-Next  - Qwen3-Next 80B-A3B hybrid-attention MoE. The
#                                  experts are huge and mostly cold, so they go
#                                  to CPU RAM (--cpu-moe) while everything else
#                                  goes to GPU (-ngl 999).
#   qwen-next  Qwen3-Next-80B-A3B-Instruct - same MoE backbone as qwen, same
#                                  --cpu-moe + -ngl 999 recipe.
#   qwen2.5    Qwen2.5-7B-Instruct - DENSE 7B. Every layer fits on the 6GB card,
#                                  so all of it goes to GPU (-ngl 99) with NO
#                                  --cpu-moe (there are no experts to offload;
#                                  passing --cpu-moe here would be a no-op at
#                                  best and confusing at worst).
#   wrn        WhiteRabbitNeo-V3-7B - DENSE (Llama-3.1-8B based, ~8B params),
#                                  same dense GPU recipe as qwen2.5.
#
# Flash Attention (-fa on) is forced for the dense models: it shrinks the KV
# cache and was confirmed working on this Turing / sm_75 card in the Phase 1
# smoke test. The MoE models are left exactly as they were (no -fa) - changing
# their launch flags is out of scope for this revision.
set -euo pipefail

usage() {
  echo "Usage: $0 <qwen|qwen-next|qwen2.5|wrn|qwen35|bolt|nano|qwythos> [--cpu-only]" >&2
  exit 1
}

MODEL="${1:-}"
[[ -n "$MODEL" ]] || usage

case "$MODEL" in
  qwen|qwen-next|qwen2.5|wrn|qwen35|bolt|nano|qwythos) ;;
  *)
    echo "Error: unknown model '$MODEL'. Must be one of: qwen, qwen-next, qwen2.5, wrn, qwen35, bolt, nano, qwythos." >&2
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

# Per-model config. MOE=1 selects the expert-offload GPU recipe
# (-ngl 999 --cpu-moe); MOE=0 selects the dense all-on-GPU recipe
# (-ngl 99 -fa on). GPU_CTX is the context size used in GPU mode only
# (--cpu-only always falls back to a small 4096 ctx below).
case "$MODEL" in
  qwen)
    PORT="${LLAMA_PORT:-8091}"
    MODEL_PATH="$ROOT_DIR/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_M.gguf"
    GPU_CTX=4096
    MOE=1
    ;;
  qwen-next)
    # General-purpose sibling of qwen (Qwen3-Coder-Next): same Qwen3-Next
    # 80B-A3B hybrid-attention MoE backbone, instruct-tuned generally
    # instead of fine-tuned for code. Same offload recipe applies.
    #
    # GPU_CTX history: 4096 -> 8192 (2026-08-25) to fit the merged
    # filesystem+playwright MCP tool schema (~6637 tokens) plus room for
    # real conversation on top of it. That still overflowed mid-conversation
    # ("request (8198 tokens) exceeds the available context size (8192
    # tokens)") after ~8 exchanges, so raised again the same day to 131072
    # after empirically testing the real ceiling on this 8GB card: 32768 used
    # ~4.0GB VRAM, 131072 used ~6.4GB (verified stable across an actual
    # generation, not just at load), 262144 hard-OOMs at startup (tried to
    # cudaMalloc 6.1GB for KV cache alone, more than the remaining headroom).
    # 131072 leaves a real ~1.7GB safety margin against other GPU consumers
    # (idle mining, browser, desktop) -- don't raise further without redoing
    # this same empirical check, and don't trust linear extrapolation from a
    # smaller context (32768->131072 was a 4x context increase for only ~1.6x
    # more VRAM, i.e. sub-linear, right up until it wasn't).
    PORT="${LLAMA_PORT:-8092}"
    MODEL_PATH="$ROOT_DIR/models/qwen3-next-instruct/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf"
    GPU_CTX=131072
    MOE=1
    ;;
  qwen2.5)
    # DENSE Qwen2.5-7B-Instruct, Q4_K_M (~4.68GB of weights). General-purpose
    # dense worker for Phase 1. All layers on GPU (-ngl 99), -fa on, q8_0 KV.
    #
    # Measured on this card (RTX 2060 Max-Q, ~5746 MiB usable after desktop),
    # -ngl 99 --jinja -fa on --cache-type-k q8_0 --cache-type-v q8_0:
    #     -c 32768 -> 5389 MiB used / 359 MiB free   (works; too tight vs. desktop drift)
    #     -c 16384 -> ~4900 MiB used / ~850 MiB free (safe default, extrapolated + smoke-anchored)
    # q8_0 K+V roughly halves the KV cache (28 layers, 4 KV heads x 128); the
    # symmetric q8_0/q8_0 D128 fused FA kernel IS in this build (no
    # -DGGML_CUDA_FA_ALL_QUANTS needed). Push -c to 24576/32768 only when the
    # GPU is otherwise idle; re-measure with nvidia-smi during a real
    # generation before trusting a bigger number.
    PORT="${LLAMA_PORT:-8093}"
    MODEL_PATH="$ROOT_DIR/models/qwen2.5-7b-instruct/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    GPU_CTX=16384
    MOE=0
    ;;
  wrn)
    # DENSE WhiteRabbitNeo-V3-7B, Q4_K_M (~4.68GB of weights). Llama-3.1-8B
    # based (~8B params) security/offensive-tooling specialist. All layers on
    # GPU (-ngl 99), -fa on, q8_0 KV.
    #
    # Llama-3.1-8B geometry (32 layers, 8 KV heads x 128) makes the KV cache
    # ~2.3x qwen2.5's per token; f16 KV would OOM past ~4096 on this card, so
    # q8_0 K+V is effectively required here. Measured, -ngl 99 -fa on
    # --cache-type-k q8_0 --cache-type-v q8_0:
    #     -c 8192  -> 4651 MiB used / 1097 MiB free   (verified, comfortable)
    #     -c 16384 -> ~5200 MiB used / ~550 MiB free  (extrapolated; try + watch nvidia-smi)
    # 8192 is the safe default. Do NOT drop the q8_0 flags for this model.
    PORT="${LLAMA_PORT:-8094}"
    MODEL_PATH="$ROOT_DIR/models/whiterabbitneo-v3-7b/WhiteRabbitNeo_WhiteRabbitNeo-V3-7B-Q4_K_M.gguf"
    GPU_CTX=8192
    MOE=0
    ;;
  qwen35)
    # HYBRID Qwen3.5-27B (qwen35 arch): 65 layers, hybrid SSM-attention
    # (full_attention_interval=4, so ~17 attention layers, ~48 SSM layers).
    # Q4_K_P = ~17.9GB weights — DOES NOT FIT full GPU on this 6GB card.
    # Measured on RTX 2060 Max-Q (6144 MiB total, ~5750 MiB usable):
    #     -ngl 22 -> OOM (tried to allocate 5996 MiB for weights alone)
    #     -ngl 18 -> 5573 MiB used / 181 MiB free (verified stable, tight)
    # ponytail: partial offload, -ngl 999 when card has >=24GB VRAM
    #
    # SSM-attention hybrid means only ~17 layers need KV cache (the rest
    # are Mamba/SSM with state vectors, not KV). Of those, only ~5 are on
    # GPU (layers 0,4,8,12,16 within the 18 offloaded layers). This makes
    # large context cheap on VRAM, but the compute buffer for the PP
    # forward pass still scales with context (attention masks, pos embed).
    # Measured on this card (RTX 2060 Max-Q, 6144 MiB total):
    #   -np 1 -c 16384 -> 5537 MiB used / 217 MiB free (verified)
    #   -np 1 -c 32768 -> OOM (318 MiB compute buffer vs 181 MiB free)
    # ponytail: 16384 is the safe ceiling (217 MiB free); 24576 works but 89 MiB free is risky with desktop/browser
    PORT="${LLAMA_PORT:-8095}"
    MODEL_PATH="$ROOT_DIR/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
    GPU_CTX=16384
    MOE=0
    PARTIAL_NGL=18
    AGENT_MODE=1
    SINGLE_USER=1
    ;;
  bolt)
    # DENSE/MoE Nemotron 3.5 Bolt E4B (gemma4 arch): 42 layers, ~8-9B total
    # params with ~4B effective per token. Q4_K_M = 5.8GB file.
    # 8 attention heads, 2 KV heads, 65536 native context.
    # At 5.8GB weights, full GPU is tight on 6GB card — test first,
    # fall back to partial offload if OOM.
    # Measured on RTX 2060 Max-Q (6144 MiB total):
    #   -ngl 99 -c 16384 -> 3469 MiB used / 2285 MiB free (verified)
    #   -ngl 99 -c 65536 -> 4117 MiB used / 1637 MiB free (verified, native max)
    # ponytail: full native context fits comfortably
    PORT="${LLAMA_PORT:-8096}"
    MODEL_PATH="$ROOT_DIR/models/nemotron-bolt-e4b/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
    GPU_CTX=65536
    MOE=0
    AGENT_MODE=1
    ;;
  nano)
    # HYBRID Nemotron3 Nano 4B Uncensored (nemotron_h arch): 42 layers,
    # hybrid SSM-attention like qwen35 but 4B params. Q8_K_P = 4.4GB.
    # SSM (Mamba) layers + sparse attention. 1M native context.
    # 4.4GB fits fully on RTX 2060 Max-Q with room for KV cache.
    # Measured on RTX 2060 Max-Q (6144 MiB total):
    #   -ngl 99 -c 16384 -> 4743 MiB used / 1011 MiB free (verified)
    #   -ngl 99 -c 32768 -> 4951 MiB used / 803 MiB free (verified)
    # ponytail: 32K fits with ~800 MiB headroom; SSM layers don't need KV cache
    PORT="${LLAMA_PORT:-8097}"
    MODEL_PATH="$ROOT_DIR/models/nemotron-nano-4b-uncensored/Nemotron3-Nano-4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
    GPU_CTX=32768
    MOE=0
    AGENT_MODE=1
    ;;
  qwythos)
    # Qwythos-9B (Claude-Mythos finetune, reasoning model): 9.2B params,
    # Q4_K_M = 5.5GB file, 1M native context. Thinks before every answer
    # (reasoning_content), so budget tokens accordingly.
    # Measured on RTX 2060 Max-Q (6144 MiB total), -ngl 99 -fa on, q8_0 KV, -np 1:
    #   -c 16384 -> 5623 MiB used / 521 MiB free (verified through a generation)
    #   -c 24576 -> OOM (176 MiB compute buffer would not fit)
    #   -c 32768 -> OOM (216 MiB compute buffer would not fit)
    # ponytail: 16384 is the ceiling on this card; QWYTHOS_CTX overrides it for re-measuring
    PORT="${LLAMA_PORT:-8098}"
    MODEL_PATH="$ROOT_DIR/models/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"
    GPU_CTX="${QWYTHOS_CTX:-16384}"
    MOE=0
    AGENT_MODE=1
    SINGLE_USER=1
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

# Leave one core free rather than defaulting to nproc threads: llama.cpp's CPU
# backend rendezvous-barriers all worker threads at every compute-graph step,
# so if an unrelated process (e.g. a stuck root daemon) pins even one core,
# every thread stalls waiting on the one that can't get scheduled -- a single
# busy neighbor process was measured to cause a ~29x generation-speed
# collapse (26.5 -> 0.9 tok/s) on this 6-core/no-SMT machine. One idle core
# of headroom is cheap insurance against that failure class.
THREADS="$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))"

if [[ "$CPU_ONLY" -eq 1 ]]; then
  EXTRA_ARGS=(-ngl 0 --threads "$THREADS")
  CTX=4096
else
  if [[ "$MOE" -eq 1 ]]; then
    # MoE: non-expert weights on GPU, all expert weights on CPU RAM.
    EXTRA_ARGS=(-ngl 999 --cpu-moe --threads "$THREADS")
  elif [[ -n "${PARTIAL_NGL:-}" ]]; then
    # Partial offload: model too large for full GPU. Put PARTIAL_NGL layers
    # on GPU, rest on CPU. Still use FA + q8_0 KV for the attention layers.
    EXTRA_ARGS=(-ngl "$PARTIAL_NGL" -fa on --cache-type-k q8_0 --cache-type-v q8_0 --threads "$THREADS")
    EXTRA_ARGS+=(--threads-batch "$(nproc)")
    EXTRA_ARGS+=(--prio 2)
    # Per-model sampling defaults (from GGUF metadata); override with SAMPLING_ARGS
    if [[ "$MODEL" == "qwen35" ]]; then
      EXTRA_ARGS+=(--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.0)
      EXTRA_ARGS+=(--reasoning-preserve)
    fi
  else
    # Dense: every layer on GPU, Flash Attention on, q8_0 K+V cache. -fa and
    # the symmetric q8_0/q8_0 D128 fused kernel are both confirmed working on
    # this Turing / sm_75 build (no -DGGML_CUDA_FA_ALL_QUANTS needed). q8_0 KV
    # roughly halves the cache and is what keeps these 7-8B models inside 6GB
    # at a useful context. See the per-model GPU_CTX comments for measured VRAM.
    EXTRA_ARGS=(-ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --threads "$THREADS")
  fi
  # Single-user mode: 1 slot, all context for one session
  if [[ "${SINGLE_USER:-0}" -eq 1 ]]; then
    EXTRA_ARGS+=(-np 1)
  fi
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

AGENT_ARGS=()
if [[ "${AGENT_MODE:-0}" -eq 1 ]]; then
  # --agent enables built-in tools (read_file, write_file, grep_search,
  # file_glob_search, exec_shell_command, edit_file, apply_diff, get_datetime)
  # plus the MCP CORS proxy for the web UI.
  AGENT_ARGS=(--agent)
fi

nohup "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --alias "$MODEL" \
  --host "${LLAMA_HOST:-127.0.0.1}" \
  --port "$PORT" \
  -c "$CTX" \
  --jinja \
  "${AGENT_ARGS[@]}" \
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
