# POOL.md — Phase 2 VRAM pooling (Zephy + pop-os)

Distributed inference across two GPUs using llama.cpp's RPC backend:
pop-os (RTX 5060, 8 GB) is the **head**, Zephy (RTX 2060 Max-Q, ~5.7 GB usable)
is a **pure compute peer**. Together ~14 GB of VRAM, enough to hold a dense
14B Q4_K_M model + KV cache fully on GPU that neither card fits alone.

---

## Office mode vs. Field mode

| | **Field mode (default, 99% of the time)** | **Office mode (Zephy physically docked at pop-os)** |
|---|---|---|
| Topology | Zephy and pop-os are independent. Zephy serves its own Phase-1 dense 7B models locally. pop-os serves Qwen3-Next-80B over HTTPS. | Zephy carried to the pop-os site, plugged into the **wired** office LAN (10.1.10.x). RPC compute peer for pop-os. |
| Link | Tailnet WAN, ~50 ms, DERP-relayed. | Gigabit wired LAN, sub-millisecond. |
| VRAM pool | **No.** RPC over the tailnet is far too slow — every layer boundary is a synchronous tensor round-trip; 50 ms × hundreds of ops per token = unusable. | **Yes.** This is the only time `ggml-rpc-server` runs on Zephy. |
| How the field talks to big models | Plain HTTPS client against `https://pop-os.tail1d1c9c.ts.net` (Tailscale Serve, valid cert). Never touches the RPC port. | Same HTTPS endpoint still works; the pool just makes the model behind it bigger/faster. |
| Zephy RPC service | `ggml-rpc-server.service` **stopped & disabled**. | `sudo systemctl start ggml-rpc-server.service` — manual, per session. |

**Rule:** the RPC port (50052) is a wired-LAN-only, session-scoped thing. In the
field it does not exist. The pooled model is reached exactly like any other
pop-os model — as an HTTP client of the pop-os head.

---

## Startup order (office mode)

RPC peers must be listening **before** the head starts — llama.cpp resolves and
allocates across all `--rpc` targets at model-load time and will abort if one is
unreachable.

1. **Zephy** — dock on the wired office LAN, then:
   ```
   sudo systemctl start ggml-rpc-server.service
   ss -ltnp | grep ':50052'      # confirm it bound the 10.1.10.x LAN IP, not 0.0.0.0
   ```
   Note the LAN IP it printed (`journalctl -u ggml-rpc-server -n 5`). Call it `<ZEPHY-LAN-IP>`.

2. **pop-os** — start the head pointing at Zephy (command below).

3. Verify from pop-os: `curl -s localhost:8095/health` → `{"status":"ok"}`.

Shut down in reverse: stop the pop-os head first, then
`sudo systemctl stop ggml-rpc-server.service` on Zephy, then undock.

---

## pop-os head command (pooled Qwen3-14B dense)

> pop-os needs its **own local copy** of the GGUF. The head loads the model file;
> RPC only distributes *compute*, never the weights-on-disk. Zephy as a pure RPC
> compute peer does **not** need the file.
>
> Model: `unsloth/Qwen3-14B-GGUF`, file `Qwen3-14B-Q4_K_M.gguf` (~8.5 GB).
> Put it at `~/models/qwen3-14b/Qwen3-14B-Q4_K_M.gguf` on pop-os.

```bash
# ON POP-OS (the head). Run AFTER Zephy's ggml-rpc-server is listening, and
# AFTER `llama-server --list-devices --rpc <ZEPHY-LAN-IP>:50052` to read the
# real device-enumeration order (RPC order is NOT guaranteed).
llama-server \
  -m ~/models/qwen3-14b/Qwen3-14B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8095 \
  -c 16384 \
  -ngl 99 \
  --jinja \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --device CUDA0,RPC0 \
  --tensor-split 0.5,0.5 \
  --rpc <ZEPHY-LAN-IP>:50052
```

### Starting values (from `docs/PARAMS.md` — verify on first real two-GPU launch)

- **`--device CUDA0,RPC0`** — pin the device order explicitly. `CUDA0` = pop-os's
  5060, `RPC0` = Zephy over the wire. Without this, `--tensor-split`'s two numbers
  map to whatever order RPC enumerates, which is not guaranteed. Run
  `llama-server --list-devices --rpc <ZEPHY-LAN-IP>:50052` first and confirm.

- **`--tensor-split 0.5,0.5`** — `<pop-os CUDA0>,<Zephy RPC0>`, valid only with the
  `--device` pin above. Per-node usable budgets are ~equal (~5.1–5.4 GB each after
  overhead), so start even. If pop-os OOMs on load (it also carries head overhead),
  shift toward Zephy: `0.45,0.55`. If Zephy OOMs: `0.55,0.45`.

- **`-c 16384` with `--cache-type-k q8_0 --cache-type-v q8_0`** — combined shard
  budget after ~8.7 GB weights is ~1.5 GB for KV *across both nodes*. With
  `--split-mode layer` (what RPC uses) the KV cache is **not** all on the head —
  each node holds the KV for its assigned layers, in the same proportion as the
  tensor-split. q8_0 K+V keeps 16k comfortable (~680 MiB total KV); f16 KV only
  fits ~8k. Push `-c` up only while watching `nvidia-smi` on **both** nodes.

Record the real VRAM readings once measured, the way `start-model.sh` records its
GPU_CTX history. See `docs/PARAMS.md` §C for the full per-node math and the RPC
gotcha list (lock-step build, weight cache, non-graceful peer OOM, etc.).

---

## Security rationale

- **The `ggml-rpc-server` protocol has ZERO auth and ZERO TLS.** Upstream README:
  *"the functionality is fragile and insecure. Never run the RPC server on an
  open network or in a sensitive environment."* Anyone who can open a TCP
  connection to port 50052 can allocate GPU memory and execute arbitrary compute
  graphs on Zephy.
- Therefore the server binds **only the wired LAN IP**, resolved dynamically at
  start (`ExecStartPre` helper picks the non-loopback / non-`tailscale0` /
  non-`docker0` / non-`virbr` IPv4). **Never `0.0.0.0`. Never the `tailscale0`
  address. Never a `100.64.0.0/10` CGNAT address.** The helper hard-refuses if
  the only address it can find is in the tailnet range.
- **ufw** allows `50052/tcp` only from `10.0.0.0/8`, `172.16.0.0/12`,
  `192.168.0.0/16`, and adds an explicit `deny in on tailscale0` (prepended, so
  it wins on any rule ordering). Tailnet peers use `100.64.0.0/10`, which matches
  none of the allow rules anyway — the interface deny is belt-and-braces.
  - Caveat: `docker0` (172.17.0.1/16) is inside `172.16.0.0/12`, so local
    containers can also reach the port. Fine on a single-user laptop; tighten to
    the exact office/home `/24` if that assumption breaks.
- **The unit is installed `disabled`.** It never starts at boot. It runs only
  when a human runs `systemctl start` while docked, and gets stopped before
  undocking. Field exposure of the RPC port is structurally impossible because
  the service isn't running and, even if it were, the tailnet interface is
  denied and the bind helper won't pick a tailnet IP.
- **Field access to big models is unchanged and unaffected:** clients hit
  `https://pop-os.tail1d1c9c.ts.net` over TLS. That path never involves Zephy's
  RPC port.

---

## Files

- `bin/` (repo) — Phase 1 local model launchers (`start-model.sh`, `stop-model.sh`).
- `zephy-rpc-setup.sh` (this draft set) — root installer for
  `ggml-rpc-server.service` + ufw rules on Zephy. Idempotent; installs disabled.
- Pinned llama.cpp: commit `082b326fc76f6e9bbb835b3920a3022bfdb6691c`, built with
  `-DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=75`.
  RPC server binary is **`ggml-rpc-server`** (not `rpc-server`), protocol `v3.0.0`.
