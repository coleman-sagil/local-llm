# Phase 1 / Phase 2 llama.cpp Parameter Research

Machine: Zephy = ASUS GA401IV, RTX 2060 Max-Q, Turing sm_75, 6 GB VRAM
(~5746 MiB usable after desktop). llama.cpp pinned commit
`082b326fc76f6e9bbb835b3920a3022bfdb6691c`, binaries report v9951,
built `-DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=75`
(note: **NOT** built with `-DGGML_CUDA_FA_ALL_QUANTS=ON`).

Empirical anchor (smoke test, already run):
Qwen2.5-7B-Instruct Q4_K_M, `-ngl 99 -c 4096 --jinja -fa on`
-> 4633 MiB VRAM used, 1115 MiB free, GPU-resident, coherent.

All context math below assumes the KV cache is pre-allocated for the full
`-c` value at launch (llama.cpp does this), and that with flash attention
the CUDA compute buffer stays roughly constant as context grows (it scales
with `-ub`/`-b`, not with `-c`). Keep ~250-300 MiB VRAM free as headroom
because desktop/compositor VRAM use drifts.

---

## Cross-cutting answers

### Flash attention on Turing sm_75 -- YES, `-fa on`

Turing has first-class tensor-core (WMMA/MMA) flash-attention CUDA kernels
in llama.cpp and has for a long time; sm_75 is one of the better-supported
FA targets (Pascal sm_61 is the arch that lacks a good FA path, not Turing).
`-fa on` here:

- shrinks the CUDA compute buffer (no full attention matrix is materialised),
- speeds prefill 1.3-2x at longer context, ~neutral at <2k,
- is a hard prerequisite for quantizing the **V** cache.

Recent llama.cpp made `--flash-attn` a tri-state `on|off|auto` and moved the
default toward `auto` (which turns FA on for this config). Set `-fa on`
explicitly rather than trusting the default -- confirm your pinned build
accepts it with `llama-server --flash-attn on --help` (it does per the shared
context). `-fa off` would be a mistake for every model here.

### `--cache-type-k` / `--cache-type-v q8_0` -- worth it, with one caveat

q8_0 K+V roughly halves KV-cache VRAM (56 KiB/tok -> ~30 KiB/tok for a
4-KV-head model) at negligible quality cost, and only pays off **with FA on**
(without FA the cache is dequantized every step and it is slower than f16).

Caveat for this build: the default CUDA FA kernel set covers the
**symmetric** combos at head_dim 128 -- `f16/f16`, `q8_0/q8_0`, `q4_0/q4_0`.
All three Phase 1/2 models are head_dim 128 and you would use `q8_0/q8_0`, so
the fast fused kernel should be compiled in. Mixed/asymmetric types
(`q8_0` K + `f16` V, `q4_0`, `q5_0`, iq4_nl ...) need a rebuild with
`-DGGML_CUDA_FA_ALL_QUANTS=ON` or they silently fall back to a slow path.
Action: use `q8_0/q8_0` only; after first launch check the logs for a
CPU-attention / fallback warning and for expected tok/s. If it looks slow,
rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON`.

Do **not** quantize K only and leave V at f16 (asymmetric -> slow path).
Keep them equal: both `q8_0`.

### `--no-mmap` -- not needed for Phase 1

With `-ngl 99` the weights land entirely in VRAM. mmap (default on) just
controls how the file is paged from disk into that transfer; it gives **no
VRAM benefit**. `--no-mmap` forces a full host-RAM copy first (Zephy has
38 GB, so harmless) and can marginally speed a cold load / avoid page-cache
churn, but adds ~5 GB resident RSS. Recommendation: leave mmap default on
for single-model Phase 1. `--no-mmap` is only interesting on the Phase 2
**head** node (see part C, gotcha 9).

### `--jinja` -- keep it

Both Phase 1 models ship a chat template in the GGUF; `--jinja` uses it.
Already in the smoke recipe, no VRAM cost. Keep for both.

---

## A) Qwen2.5-7B-Instruct Q4_K_M (dense, 28 layers, 4 KV heads, D=128)

KV cache per token: f16 = 56 KiB, q8_0/q8_0 = ~30 KiB.
Non-KV footprint (weights + compute + CUDA graph + output), back-computed
from the smoke point: 4633 - (4096 * 56 KiB) = **~4409 MiB**.
KV budget before headroom: 5746 - 4409 ~= 1337 MiB; keep ~250 MiB back
=> ~1080-1130 MiB usable for KV.

| Setting | Value | Notes |
|---|---|---|
| `-ngl` | **99** | all 28 layers on GPU, no `--cpu-moe` (dense model) |
| `-fa` | **on** | see above |
| `--jinja` | on | GGUF chat template |
| `-c` (f16 KV) | **16384 safe**, 20480 aggressive | 16k f16 KV ~= 896 MiB; 20k ~= 1120 MiB (thin margin) |
| `-c` (q8_0 KV) | **32768** (native ctx ceiling) | 32k q8_0 KV ~= 952 MiB, fits; `--cache-type-k q8_0 --cache-type-v q8_0` |
| `--no-mmap` | omit | no benefit here |
| `-b` / `-ub` | leave default (2048 / 512) | lowering `-ub` to 256 saves a little compute-buffer VRAM if you need to claw back ~50-100 MiB |

Recommended default: `-ngl 99 -c 32768 -fa on --jinja --cache-type-k q8_0 --cache-type-v q8_0`
(full native context, ~5.6 GB total, ~150-250 MiB free -- verify on first run;
if it OOMs from desktop drift, drop to `-c 24576`).
Conservative default if you want f16 KV: `-ngl 99 -c 16384 -fa on --jinja`.

---

## B) WhiteRabbitNeo-V3-7B Q4_K_M (Llama-3.1-8B base: 32 layers, 8 KV heads, D=128)

This model is meaningfully heavier than Qwen2.5-7B on **both** axes:
- ~8B params + 32 layers -> non-KV footprint estimated **~4650-4800 MiB**
  (Q4_K_M file ~4.9 GB vs 4.68 GB, plus 4 extra layers of buffers). Uncertain
  +/- ~150 MiB -- measure on first launch.
- 8 KV heads (double Qwen2.5's 4) -> **KV f16 = 128 KiB/token**, q8_0/q8_0
  ~= 68 KiB/token. KV cost is ~2.3x Qwen2.5 at equal context.

KV budget before headroom: 5746 - ~4720 ~= ~1026 MiB; keep ~250 back
=> ~750-820 MiB usable for KV.

| Setting | Value | Notes |
|---|---|---|
| `-ngl` | **99** | all 32 layers on GPU, no `--cpu-moe` |
| `-fa` | **on** | mandatory -- also enables the q8_0 V cache you need here |
| `--jinja` | on | Llama-3.1 chat template in GGUF |
| KV type | **`--cache-type-k q8_0 --cache-type-v q8_0` (effectively required)** | f16 KV leaves almost no usable context on 6 GB |
| `-c` (f16 KV) | 4096 safe, 6144 tight, **8192 likely OOM** | 8k f16 KV = 1024 MiB > budget |
| `-c` (q8_0 KV) | **8192 safe**, 12288 aggressive, 16384 risky | 12k q8_0 KV ~= 816 MiB |
| `--no-mmap` | omit | no benefit |
| `-ub` | consider **256** | frees ~50-100 MiB compute buffer; helps push `-c` up on this tighter model |

Recommended default: `-ngl 99 -c 8192 -fa on --jinja --cache-type-k q8_0 --cache-type-v q8_0`
(~5.4-5.5 GB total). Try `-c 12288` and watch `free` on nvidia-smi; back off
to 8192 if under ~250 MiB free. Do **not** run this model with f16 KV beyond
`-c 4096`.

---

## C) Phase 2 RPC pool: Qwen3-14B Q4_K_M across Zephy (6 GB) + pop-os (8 GB, ~1.5 GB used)

**Office-mode only** -- wired GbE LAN, never the tailnet.
pop-os = head (runs `llama-server`, holds the model file). Zephy = pure RPC
compute peer (`ggml-rpc-server`, needs no model file).

Qwen3-14B: 40 layers, 8 KV heads, D=128 -> KV **f16 = 160 KiB/token**,
q8_0/q8_0 ~= 85 KiB/token. Weights Q4_K_M ~= 8.5 GB (~8700 MiB).

### Per-node usable budget (weights shard + KV shard)

| Node | Raw VRAM | Minus | Usable for shard |
|---|---|---|---|
| Zephy (RPC peer) | ~5746 MiB | RPC-side compute buffer ~400-600 MiB | **~5100-5300 MiB** |
| pop-os (head) | 8188 MiB | 1500 already used + head overhead (context state, logits/output buffer, compute buffer, CUDA graph) ~1200-1600 MiB | **~5000-5400 MiB** |

Combined shard budget ~= **10,200 MiB**. Weights 8700 -> **~1500 MiB total
left for KV across both nodes.**

### `--tensor-split`

Two numbers, one per device, **in llama.cpp's device-enumeration order**.
Usable budgets are ~equal, so start at **`--tensor-split 0.5,0.5`** (equiv.
`1,1`). The map is `<pop-os/CUDA0>,<zephy/RPC0>` **only if** that is the
enumeration order -- which is NOT guaranteed with RPC. So:

1. Run `llama-server --list-devices --rpc <zephy-lan-ip>:50052` first and read
   the printed order.
2. Pin it explicitly: `--device CUDA0,RPC0` (CUDA0 = pop-os 5060,
   RPC0 = Zephy). Then `--tensor-split 0.5,0.5` = `<pop-os>,<zephy>`.
3. If pop-os OOMs during load (it carries head overhead), shift weight toward
   Zephy: `--tensor-split 0.45,0.55` (pop-os,zephy). If Zephy OOMs, the other
   way: `0.55,0.45`.

### Where the KV cache lives

With the **default `--split-mode layer`** (which is what RPC uses), the KV
cache is **NOT** all on the head. Each node holds the KV for the layers
assigned to it, in the same proportion as `--tensor-split`. There is no
"KV entirely on head" option except `-sm none` (single device), which
defeats the pool. Plan VRAM for a KV shard on **each** node
(~750 MiB each at the numbers below).

Quantized KV (`q8_0/q8_0`) **is** compatible with `--split-mode layer`.
The incompatibility reported upstream (issues #21788, #23567) is
quantized KV + `--split-mode row`/tensor-parallel, which does not apply here.

### Safe `-c` for the pool

| KV type | KV VRAM at ctx | Recommended `-c` |
|---|---|---|
| f16 | 8k = 1280 MiB, 9.6k ~= budget | **8192** (thin), 6144 comfortable |
| q8_0/q8_0 | 8k = 680 MiB, 16k = 1360 MiB | **16384** comfortable, 12288 safe |

Recommended pool default:
`-ngl 99 -c 16384 -fa on --jinja --cache-type-k q8_0 --cache-type-v q8_0
--device CUDA0,RPC0 --tensor-split 0.5,0.5 --rpc <zephy-lan-ip>:50052`
on the pop-os head. Start here, watch both nodes' `nvidia-smi`, rebalance
`--tensor-split` per above.

### Known RPC gotchas (commit 082b326 / v9951 / July 2026)

1. **Protocol/build lock-step.** Both ends must be the *same* llama.cpp
   commit (082b326). RPC has no protocol-version negotiation across
   mismatched builds -> connection refused / crash / garbage. pop-os
   (WhiteOut) must build the identical pinned commit.
2. **Zero auth, zero TLS.** Bind `ggml-rpc-server` to the office LAN IP
   only: `ggml-rpc-server -H <office-lan-ip> -p 50052`. Never `0.0.0.0`,
   never the `tailscale0` / `100.64.0.0/10` address. Field access is via
   pop-os's HTTPS endpoint as a normal client and never touches port 50052.
3. **Device order is not guaranteed** with RPC (see `--tensor-split` above).
   Always `--list-devices` then pin `--device`.
4. **Prefill is bandwidth/latency-bound over RPC.** Prompt processing tok/s
   drops hard vs a single GPU; token generation is less affected. This is
   exactly why it is wired-LAN / office-mode only.
5. **RPC weight cache.** Enable it on Zephy so the ~4 GB weight shard is not
   re-streamed every launch: `ggml-rpc-server -c <cache-dir>` (or set
   `LLAMA_CACHE`; default `$HOME/.cache/llama.cpp/rpc`). First load still
   transfers the shard once. Threshold tunable via `GGML_RPC_CACHE_MIN_SIZE`.
6. **`-ngl` must cover every layer** (99/999). Any layer left unoffloaded
   runs on the pop-os CPU and tanks throughput.
7. **OOM on the peer is not graceful.** If Zephy's shard overflows,
   `ggml-rpc-server` dies and the head hangs or errors -- restart the RPC
   server first, then the head.
8. **FA must work on both GPUs.** `-fa` is global. Turing (Zephy) and
   pop-os's 5060 both support FA; keep `-fa on` on the head.
9. **`--no-mmap` on the head can help here.** It avoids double-buffering the
   model through the pop-os page cache while streaming shards to RPC; worth
   A/B testing on pop-os cold-load time. (Still no effect on Zephy, which
   gets tensors over the wire.)
10. **Leave ~1.5 GB slack on pop-os.** The head needs the full logits/output
    buffer + compute buffer + CUDA graph on top of its weight+KV shard;
    multi-node budgets are tighter than a naive "8 GB - 1.5 GB used" suggests.
11. `GGML_RPC_DEBUG=1` on the peer and `--verbose` on the head are the only
    real diagnostics; turn them on for first bring-up.

---

## Quick-reference recipes

| Model | Recipe |
|---|---|
| Qwen2.5-7B (default) | `-ngl 99 -c 32768 -fa on --jinja --cache-type-k q8_0 --cache-type-v q8_0` |
| Qwen2.5-7B (f16 KV) | `-ngl 99 -c 16384 -fa on --jinja` |
| WhiteRabbitNeo-V3-7B | `-ngl 99 -c 8192 -fa on --jinja --cache-type-k q8_0 --cache-type-v q8_0` |
| Qwen3-14B RPC pool (head=pop-os) | `-ngl 99 -c 16384 -fa on --jinja --cache-type-k q8_0 --cache-type-v q8_0 --device CUDA0,RPC0 --tensor-split 0.5,0.5 --rpc <zephy-lan-ip>:50052` |
| Qwen3-14B RPC peer (Zephy) | `ggml-rpc-server -H <office-lan-ip> -p 50052 -c ~/.cache/llama.cpp/rpc` |

All `-c` / `--tensor-split` values are starting points to verify against
`nvidia-smi` on first launch -- the non-KV footprints for the 8B model and
the RPC head overhead are estimates (+/- ~150-300 MiB).

---

## Sources

- https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
- https://github.com/ggml-org/llama.cpp/discussions/16625 (RPC device order / tensor-split)
- https://github.com/ggml-org/llama.cpp/discussions/12714 (RPC per-host layer allocation)
- https://github.com/ggml-org/llama.cpp/issues/21788 , https://github.com/ggml-org/llama.cpp/issues/23567 (quantized KV vs split-mode row)
- https://github.com/ggml-org/llama.cpp/discussions/22411 (symmetric KV quant enables fused FA kernel)
- https://github.com/ggml-org/llama.cpp/issues/24485 (GGML_CUDA_FA_ALL_QUANTS default / fallback warning)
- https://github.com/ggml-org/llama.cpp/pull/24122 (RPC small-message overhead / cache probing)
- https://github.com/ggml-org/llama.cpp/discussions/9646 (FA quality/behavior discussion)
- https://sharedllm.org/blog/llama-cpp-tensor-split.html , https://sharedllm.org/blog/llama-cpp-rpc-distributed-inference.html , https://sharedllm.org/blog/llama-cpp-rpc-two-macs.html
- https://inventivehq.com/blog/flash-attention-llama-cpp-benchmark (FA default-on, ~neutral perf/VRAM on modern NVIDIA)
- https://vramcalculator.com/llama-cpp-performance-flags/ , https://notes.itsvasugrover.com/kb/ai/llama-cpp/performance-tuning/
