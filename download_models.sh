#!/usr/bin/env bash
# Direct HTTP download instead of huggingface_hub's Python client -- that
# client's newer Xet-transfer path hung indefinitely with zero bytes moved
# (huggingface.co only resolves to IPv6 here; a plain curl HEAD/range-GET
# worked fine at ~19 MB/s, so whatever the Xet path does differently is what
# stalled, not basic connectivity).
set -euo pipefail

download() {
    local url="$1"
    local out="$2"
    mkdir -p "$(dirname "$out")"
    echo "=== downloading $(basename "$out") ==="
    curl -L -C - --retry 5 --retry-delay 5 -o "$out" "$url"
    echo "=== done: $out ($(du -h "$out" | cut -f1)) ==="
}

BASE=/home/mateo/local-llm/models

download \
  "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf" \
  "$BASE/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf"

download \
  "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf" \
  "$BASE/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf"

download \
  "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_M.gguf" \
  "$BASE/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_M.gguf"

download \
  "https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF/resolve/main/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf" \
  "$BASE/qwen3-next-instruct/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf"

echo "ALL DOWNLOADS COMPLETE"
