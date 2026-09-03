# CLUSTER.md — the longer road for local-llm

This is a planning document, not a pitch. It exists so that the phases after
"get inference working" have a written shape before anyone starts building
them, and so the constraints that will bite are written down while they're
still cheap to remember.

The tone to keep when editing this file: concrete first steps, honest about
what the hardware can't do, no roadmap theater. Each phase says what it
*unlocks*, what it *needs*, and the *first commit* that starts it — a real
commit, small enough to land in a sitting, not an epic.

---

## Where we are

- **Phase 1 — Zephy inference foundation.** Done / in progress. `llama.cpp`
  `llama-server` is the one inference engine everywhere; `ollama` is being
  retired on Zephy. Two dense 7–8B models (`qwen2.5` port 8093, `wrn` port
  8094) run locally on Zephy's 6GB RTX 2060 Max-Q. The 80B MoE models
  (`qwen`, `qwen-next`) run on the pop-os head node and are reached from
  Zephy as a plain HTTPS client at `https://pop-os.tail1d1c9c.ts.net`. See
  `docs/TUTORIAL.md` sections 8–10.
- **Phase 2 — VRAM pool (office mode).** Prep stage. `ggml-rpc-server` is
  built (`-DGGML_RPC=ON`, arch 75). The mechanism: Zephy runs
  `ggml-rpc-server` bound to the office LAN NIC, pop-os runs `llama-server
  --rpc <zephy-lan-ip>:50052 ...` as head, and a mid-size dense model
  (Qwen3-14B `Q4_K_M`, ~8.5GB; later WhiteRabbitNeo-13B) runs split across
  both GPUs. **Only works on the pop-os wired LAN** — RPC over the tailnet
  WAN is too slow to be usable. pop-os needs its own copy of the pooled
  model file; Zephy as a pure compute peer does not.

Everything below is Phase 3 and later.

---

## Cross-cutting constraints (true for every phase)

- **One GPU on Zephy, and it's shared.** 6GB, ~5.75GB usable. Inference,
  training, and (on the other machine) mining cannot co-reside. Any
  training run means inference is down for the duration.
- **`ggml-rpc-server` has zero auth and zero TLS.** Bind it to the office
  LAN interface only — never `0.0.0.0`, never `tailscale0`, never the
  `100.64.0.0/10` range. Off-box access is always via the pop-os HTTPS
  endpoint as a normal client, which never touches the RPC port.
- **pop-os is a different site, different user, not ours to drive.** The
  WhiteOut session operates it. Cross-host coordination is by message, and
  live claims about pop-os get verified against the real endpoint before
  anything depends on them.
- **Starlink CGNAT blocks inbound port-forwarding.** Anything that must be
  reachable off-site goes through Tailscale (Serve / Funnel), same as the
  pop-os head does today. No raw public ports.
- **Portability is not optional.** No hardcoded `/home/mateo`,
  `/home/mateo_c-s`, or LAN IPs in committed code. Derive paths from a
  script-location `$ROOT_DIR`; put host addresses in one config table.
- **The MacBook nodes are old and weak.** MBP2019Node3 (.206), MBP2010Node2
  (.54), AcerNode1 (pending). Little or no usable GPU. Treat them as
  CPU-only, non-latency-critical workers, and assume any one of them can be
  offline without the system breaking.

---

## Phase 3 — Unified backend + settle the frontend

### What it unlocks

One coherent view of every model across Zephy and pop-os: what exists, where
it runs, whether it's healthy, and how to start or stop it — without
hand-maintaining the same port/path facts in `bin/start-model.sh`,
`cli/llmcli.py`, and the GUI separately. Today those three files each carry
their own copy of the model→port contract; that's fine for four models on
two hosts and will not stay fine.

### What it needs

- **A multi-host model registry.** One declarative file — `registry/models.toml`
  is the natural choice — that is the single source of truth. One entry per
  model: key, host (`localhost` or a named remote), port, on-disk path
  *relative to that host's `$ROOT_DIR`*, launch recipe (`moe` / `dense` /
  `cpu-only`), default context, model family. A companion host table maps
  host names to base URLs (`localhost` → `http://127.0.0.1`, `pop-os` →
  `https://pop-os.tail1d1c9c.ts.net`). `start-model.sh` and `llmcli.py`
  both read this instead of embedding the facts.
- **Health.** A small poller that hits `/health` for local servers and
  `/v1/models` for the pop-os endpoint (the tailnet Serve front end may not
  proxy `/health`), and reports `up` / `down` / `loading`, plus local VRAM
  from `nvidia-smi`. Read-only; no lifecycle authority.
- **Lifecycle.** A thin CLI — `bin/llmctl` — wrapping the existing
  `start-model.sh` / `stop-model.sh` with the registry: `llmctl start
  qwen2.5`, `llmctl status`, `llmctl stop --all`. **Local lifecycle only in
  Phase 3.** Remote start/stop on pop-os needs an agent or SSH trust that
  doesn't exist yet — Phase 3 does *remote health, local control*, and
  that's a deliberate scope line, not an oversight.

### The frontend decision

Two options on the table:

1. **The bundled llama.cpp Svelte webui.** Every `llama-server` already
   serves its own chat UI on its own port. Zero maintenance, tracks
   upstream, always matches the server's capabilities.
2. **The repo's own picker page** (`gui/picker.html` + `picker_server.py`)
   as a bespoke chat frontend.

Recommendation: **keep the picker as a launcher/router only, send actual
chat to the bundled webui.** The picker's job becomes "show me every model
in the registry, its health, and a button that either starts it or opens
its webui." Don't build a bespoke multi-model chat UI unless a concrete need
for a *unified cross-host conversation view* shows up (one thread that
hops models mid-conversation) — that need would justify it; nothing short of
it does.

### First concrete commit

Add `registry/models.toml` with the four current models and the host table,
plus `registry/registry.py` (loader + validation + tests). Then change
`cli/llmcli.py` to build `MODEL_PORTS` / `base_url_for()` from the registry
instead of the hardcoded dict. No behavior change — pure de-duplication,
with the test proving the loaded values match what's hardcoded today.

---

## Phase 4 — Training (QLoRA / LoRA on the Qwen models)

### What it unlocks

Fine-tuned local variants: a `qwen2.5` adapter on domain data (e.g. the
Kearney & O'Banion habitability corpus already synced via Dropbox), a `wrn`
adapter on a curated security corpus. Adapters are small (tens of MB), swap
in at load time, and don't require re-hosting a whole model.

### What it needs

- **A QLoRA toolchain.** `unsloth` is the first choice — it's the most
  VRAM-efficient single-GPU path and explicitly targets cards this size.
  `axolotl` is the fallback and the better option for any multi-GPU run on
  pop-os. Pin versions; CUDA/toolkit coupling here is as fragile as it was
  for the `llama.cpp` build.
- **Dataset prep.** A pipeline: raw source → chat-format JSONL
  (`{messages: [...]}`) → dedupe → train/eval split → a frozen held-out
  set that never enters training. Deterministic, re-runnable, checked-in
  config (not checked-in data).
- **An eval harness.** Two parts: a task-specific rubric (does the adapter
  actually do the thing it was trained for) and a general-capability
  regression check (did it forget how to be a normal assistant) — a small
  slice of `lm-eval-harness` or a bespoke scored set. Always base vs adapter,
  same prompts, side by side.

### The hard constraint

**Zephy's 6GB does QLoRA on ≤7–8B only** — `qwen2.5-7B` and `wrn-7B`, at
modest sequence length, batch size 1 + gradient accumulation. That's the
whole envelope. **14B+ QLoRA, or any non-quantized LoRA, needs pop-os**
(8GB — still tight, still short-seqlen) **or cloud.** And training on Zephy
means local inference is fully down while it runs: schedule it as an
offline activity, not something that shares the day with serving.

### First concrete commit

A `training/` directory with `prepare_dataset.py` (raw → chat JSONL →
split, with a tiny checked-in sample corpus so the script is runnable) and
a documented `unsloth` environment spec (requirements + CUDA notes). Plus
one **smoke config**: Qwen2.5-7B QLoRA, LoRA rank 8, 50 steps, that proves
the loop completes on 6GB and writes a loadable adapter. No real training
run in the commit — just proof the pipeline turns over.

Second commit: `training/eval/` — the held-out set + a runner that scores
base vs adapter on the same prompts.

---

## Phase 5 — Multi-node cluster + RAG

### What it unlocks

The MacBook / Acer nodes earn their keep doing the parts of the system that
*aren't* token generation: embeddings, the vector store, retrieval, batch
jobs, and orchestration glue. On top of that: RAG (retrieval-augmented
answers grounded in the K&S corpus and project docs) and then recursive
model-calling-model orchestration (a planner model calling a worker model
calling a critic model, each hop a plain API call to whichever node hosts
that model).

### What it needs

- **Node roles, honestly assigned.** The MacBooks run CPU-only embedding
  models, the vector DB, the `/retrieve` service, and orchestration
  control. They do **not** generate tokens — that stays on Zephy (small
  models) and pop-os (big model). A 2010 MacBook is a background worker, not
  a latency-path component.
- **A vector store.** Start with `sqlite-vec` — a file, no server, no extra
  daemon to babysit — and only graduate to Qdrant / Chroma if corpus size
  or concurrent retrieval actually forces it.
- **Retrieval.** An `ingest` step (corpus → chunks → embeddings → store) and
  a `retrieve` client that inference front ends call *before* hitting
  `/v1/chat/completions`, splicing the hits into the prompt. The K&S Dropbox
  folder is already synced locally, so ingestion has a source on day one.
- **Recursive orchestration.** A controller that addresses models through
  the Phase 3 registry and lets one model's output drive another model's
  call. Keep the hop count small — every hop that crosses a machine
  boundary pays the network cost, and over the tailnet that's ~50ms each
  way before any compute.

### Constraints

The MacBook nodes are unreliable by assumption — design retrieval and
orchestration to degrade (fall back to no-RAG, fall back to single-model)
rather than fail when a node is down. Cross-node recursive calls are network
-bound; prefer keeping a planner→worker chain on the same host when both
models fit. Anything exposed off-site is Tailscale-only (Starlink CGNAT).

### First concrete commit

A `rag/` directory with `ingest.py` (corpus → chunked → embedded →
`sqlite-vec` file) and `retrieve.py` (query → top-k chunks), running
entirely on **one** machine first — no cluster, no network, just prove
retrieval quality is worth building on. Wire it into `llmcli.py` as an
opt-in `--rag` flag as the second commit. Orchestration comes only after
retrieval is solid on its own.
