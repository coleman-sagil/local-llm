# Local LLM Setup: A Tutorial for Understanding the Files

This document exists so that when you come back to this project later, you can
look at any file in the repo and know why it's there and what it does, without
having to reconstruct the reasoning from scratch. It's organized around the
shape of the system first, then the files, then day-to-day usage, then the
two pieces of engineering context (the GPU/CPU split, and the two problems
this session actually ran into) that make the launch scripts make sense.

## 1. The big picture

Everything in this project sits on top of one idea: **a model file on disk
becomes a live HTTP server, and everything else is a client of that server.**

Concretely, there are three layers:

1. **The models.** Three GGUF files (`llama-4-scout`, `qwen3-coder-next`, and
   `qwen3-next-instruct`) sitting in `models/`. A GGUF file is just weights
   plus metadata, it does nothing by itself. `qwen3-coder-next` and
   `qwen3-next-instruct` are actually the same underlying architecture
   (Alibaba's "Qwen3-Next" 80B-total/~3B-active hybrid-attention MoE
   backbone) - one fine-tuned for code, one instruct-tuned generally - which
   is why they share the same offload recipe and land in the same speed
   ballpark on this hardware.
2. **The inference engine.** `llama-server` (from the `llama.cpp` project,
   built from source in `llama.cpp/`) loads a GGUF file into memory and
   exposes it as an HTTP API on a fixed port, one server process per model.
   Critically, this API is **OpenAI-compatible**: the endpoint that matters is
   `POST /v1/chat/completions`, the same shape used by the OpenAI API and by
   basically every LLM tool in the ecosystem. That compatibility is why a
   generic HTTP client (`requests` in Python, `fetch` in JavaScript, `curl` on
   the command line) can talk to a local model exactly the way it would talk
   to a cloud one.
3. **The front ends.** `cli/llmcli.py` and everything in `gui/` are two
   independent, interchangeable clients of that same API. Neither one knows
   or cares that the other exists. They don't share code with each other;
   they share a *contract*: a model name maps to a fixed port
   (`scout` -> 8090, `qwen` -> 8091, `qwen-next` -> 8092, and the GUI's own
   picker page -> 8089), and the way to talk to a running model is always
   `POST http://127.0.0.1:<port>/v1/chat/completions`.

Once that shape is in your head, any new front end you add later (a phone
app, a Slack bot, a different TUI) is just "another client of the same API on
the same port." You never need to touch `llama-server` itself or the launch
scripts to add one; you only need to know which port to POST to.

The scripts in `bin/` are the one piece that bridges layers 1 and 2: they are
what actually starts and stops the `llama-server` process for a given model,
and they're deliberately the *only* place that knows the on-disk model paths,
the port numbers, and the GPU/CPU offload flags. The CLI and GUI both shell
out to `bin/start-model.sh` rather than duplicating that knowledge, which is
why changing, say, a context-length default only has to happen in one place.

## 2. Directory layout, and why each piece exists

| Path | Why it exists |
|---|---|
| `bin/` | The one place that knows how to start/stop a `llama-server` process for a given model: paths, ports, GPU/CPU flags. Everything else defers to it instead of re-implementing that knowledge. |
| `cli/` | A terminal chat client. Talks to whatever model server is already running; does not start or stop servers itself. |
| `gui/` | A point-and-click front end: a local picker web page plus the scripts that launch a model's own built-in chat webui in an app-mode browser window. |
| `docs/` | This file, and anywhere else project documentation belongs. |
| `models/` | The actual GGUF weight files. Gitignored, because a single model here is tens of gigabytes; git is the wrong tool for this, and the download scripts at the repo root can always re-fetch them. |
| `llama.cpp/` | The inference engine itself, built from source (with CUDA support) into `llama.cpp/build/bin/llama-server`. Gitignored for the same reason as `models/`: it's a large, independently-versioned upstream project vendored locally, not project source. |
| `logs/` | Where each running model's stdout/stderr goes (`logs/<model>.log`), written by `bin/start-model.sh`. This is how the launch scripts know a server is actually up (they poll the log for `"listening on"`) and it's the first place to look when something fails to start. Gitignored; it's runtime output, not source. |
| `run/` | PID files (`run/<model>.pid`), one per running model, written by `bin/start-model.sh` and consumed by `bin/stop-model.sh` so it knows which process (and process tree) to kill. Gitignored for the same reason as `logs/`. |

Two things worth knowing that don't fit the table:

**Why `llama.cpp/` is a whole vendored project inside this repo, not a pip
package:** `llama-server` needs to be compiled with CUDA support to actually
use the GPU, and that compilation is specific to this machine's CUDA
toolkit and GPU. There's no equivalent of `pip install llama-cpp-server` that
gives you a working CUDA build reliably across machines, so it's built from
source here once and then just invoked as a binary by everything else.

**The stray `.log` files at the repo root** (`build.log`, `download.log`,
`meta_probe.log`, `moe_probe.log`, `server_scout.log`, `server_qwen.log`) are
leftovers from manual, one-off debugging during setup (building llama.cpp,
downloading models, probing the GGUF metadata for chat templates and MoE
tensor names). They're gitignored and harmless clutter, not part of anything
the scripts produce on purpose. The scripts' own logs always live under
`logs/`, not at the root; feel free to delete the root ones whenever.

There are also two model-download scripts at the repo root,
`download_models.py` and `download_models.sh`. Only the `.sh` one actually
works on this machine: the `.py` version uses `huggingface_hub`'s newer
"Xet" transfer path, which hung indefinitely at zero bytes transferred here
(this box only resolves `huggingface.co` over IPv6, and something about that
combination stalls the Xet client). `download_models.sh` just does plain
`curl` range-downloads instead, which worked fine. Both are kept, but if
you're re-downloading a model, use the `.sh` one.

## 3. Starting and stopping a model, day to day

The everyday commands, run from the repo root:

```bash
bin/start-model.sh scout     --cpu-only     # start Scout, CPU-only
bin/start-model.sh qwen      --cpu-only     # start Qwen (coder), CPU-only
bin/start-model.sh qwen-next --cpu-only     # start Qwen3-Next (general), CPU-only
bin/stop-model.sh  scout                    # stop whichever one you started
bin/stop-model.sh  qwen
bin/stop-model.sh  qwen-next
```

`start-model.sh` backgrounds `llama-server`, writes its output to
`logs/<model>.log`, records its PID in `run/<model>.pid`, and then *waits*,
polling the log file rather than sleeping a fixed amount of time, until
either `"listening on"` shows up (success: it prints the URL and exits 0) or
a failure pattern shows up (model load error, CUDA error, segfault, etc:
it prints the relevant log lines and exits 1). That means the script's exit
code is a real signal you can script against, not a guess. It also refuses
to start a model that already has a live PID in `run/`, so you won't
accidentally spawn two servers fighting over the same port.

`stop-model.sh` reads `run/<model>.pid`, kills that process's direct children
first (there can be a small process tree, not just one PID), then the main
process, waits up to ten seconds for a clean exit, and falls back to
`SIGKILL` only if it has to. It never touches mining or systemd; it only
knows about the one process tree it started.

There are two genuinely different ways to run a model, controlled by
`--cpu-only`:

**`--cpu-only`** runs entirely on CPU (`-ngl 0`, meaning "offload zero layers
to GPU"), with a 4096-token context for any of the three models. This is slow
(CPU token generation is on the order of one token per several seconds for
Scout and Qwen3-Next, and considerably worse for Qwen3-Coder-Next
specifically - its 512-expert routing is expensive to do from CPU RAM,
sometimes dropping under 1 token/sec) but it never touches the GPU at all,
so it's safe to run any time, including while mining is active. This is the
mode the CLI and GUI use by default, and it's the mode you should use for
your own casual testing of this setup.

**Without `--cpu-only`** (GPU offload) is the real, intended fast path:
`-ngl 999 --cpu-moe` (explained in detail in section 6), with a 2048-token
context for Scout and 4096 for both Qwen variants. This is where the mining
interaction matters, covered next.

### The mining interaction

The GPU has 8GB of VRAM, and the NiceHash idle-miner
(`nicehash-miner.service`, arbitrated by `mining-idle-watcher.service`, which
starts mining only when the desktop is idle) will happily be sitting in that
VRAM when you go to use a model. GPU-offload mode needs that VRAM, so before
launching in that mode, `start-model.sh` checks whether the miner is active
(`systemctl is-active nicehash-miner.service`) and, if so, stops it for you:

```bash
sudo -n systemctl stop nicehash-miner.service
systemctl --user stop mining-idle-watcher.service
```

The `sudo -n` relies on a scoped, passwordless `NOPASSWD` sudo rule set up
specifically for that one command; `-n` means it fails immediately rather
than hanging waiting for a password if that rule is ever missing, so if you
ever see this step fail, that's the first thing to check.

**The important part: this is one-directional.** The script stops mining for
you automatically, but it does **not** restart it when you're done, because
it has no reliable way to know when "done" is. When it stops the miner for
you, it prints a reminder; take it seriously. When you're finished with a
GPU-mode model session, run:

```bash
systemctl --user start mining-idle-watcher.service
```

If you forget, mining simply stays off until you remember, silently. It
won't hurt anything, but it does mean lost mining time, so it's worth making
this the actual last step of a GPU-mode session, not an afterthought.

`--cpu-only` mode never goes near any of this. It doesn't check the miner,
doesn't stop it, doesn't touch systemd at all, which is exactly why it's the
"safe anytime" mode.

## 4. Using the CLI day to day

```bash
python3 cli/llmcli.py --model scout
python3 cli/llmcli.py --model qwen        # coding specialist
python3 cli/llmcli.py --model qwen-next   # general-purpose sibling of qwen
```

This assumes the corresponding server is already running (via
`bin/start-model.sh`); the CLI is a pure client, it never starts a server
itself. If the server isn't reachable, it fails fast with one clear line
telling you exactly which command to run, and exits nonzero, rather than
hanging or printing a stack trace.

Once connected, it's a straightforward REPL: type at the `you>` prompt, the
reply streams token-by-token as it's generated, and conversation history is
kept in memory across turns so the model has context from earlier in the
session. Two commands: `/clear` wipes the conversation history (useful when
you want to start a fresh topic without restarting the server), and `/exit`
(or `/quit`, or Ctrl+D) ends the session. Ctrl+C works cleanly at any point,
including mid-stream while a slow CPU-only generation is still producing
tokens, not just at the input prompt.

Under the hood this is exactly the shape described in section 1: it POSTs to
`/v1/chat/completions` with `"stream": true`, reads the response as
server-sent events, and prints each chunk's text as it arrives.

### The `--tools` layer: a real MCP client

```bash
uv run cli/llmcli.py --model bolt --tools
uv run cli/llmcli.py --model nano --tools --servers terminal,filesystem
```

`uv` reads the dependency header at the top of `llmcli.py` (`requests`, `mcp`),
so there is no venv to manage. From fish, `llama-agent <model>` loads the
model (via `llama-swap`) and starts this in one step.

This is an opt-in layer kept separate from the core chat path (non-streaming,
best-effort). On startup it reads `cli/mcp_servers.json`, connects over stdio
to every server with `"enabled": true` (or exactly the ones named in
`--servers a,b,c`), calls `list_tools()` on each, and merges the results into
the single tool schema sent to the model. Every tool the model sees is
namespaced `{server}__{tool}`, e.g. `terminal__run_command`,
`filesystem__read_file`.

**What's configured.** There are no hardcoded tools any more: the old builtin
`run_shell` is now the `terminal` server.

| Server | Tools | Schema | What it is |
|---|---|---|---|
| `terminal` | `run_command` | ~120 tok | `bash -lc` with a timeout and truncated output |
| `filesystem` | 8 | ~1,500 tok | official `@modelcontextprotocol/server-filesystem`, rooted at `$HOME`, cut from 16 tools |
| `system` | `system_info`, `list_processes`, `network_info`, `http_probe` | ~310 tok | OS, GPU and network inspection |
| `fetch` | `fetch` | ~360 tok | official `mcp-server-fetch`, a web page as markdown |
| `github` | `gh` | ~90 tok | the already-logged-in `gh` CLI, argument list, no shell |
| `claude` | `ask_claude` | ~110 tok | escalates a question to Claude (see below) |
| `desktop` | `open_path`, `notify` | ~150 tok | `xdg-open` and `notify-send` |
| `memory` | `remember`, `recall` | ~180 tok | long-term memory backed by graphify (see below) |
| `dropbox` | `dropbox_find`, `dropbox_recent`, `dropbox_status` | ~270 tok | the locally synced `~/*Dropbox*` folders |
| `toolsuite` | `hackstack`, `hackstack_targets` | ~190 tok | `~/toolsuite/hackstack`'s scope-gated `hs-*` wrappers |
| `browser` | 9 | ~940 tok | opt-in: headless Chromium via `@playwright/mcp`, cut from 25 tools |
| `computer` | 5 | ~770 tok | opt-in: desktop control via `cua-driver`, cut from 60 tools |
| `google-drive` | | | disabled scaffold until OAuth exists, see `docs/GOOGLE_SETUP.md` |

The enabled set is about 3,300 tokens of schema. `browser` and `computer` are
opt-in: `--servers +browser,+computer` adds them to the enabled set (about
5,000 tokens together), while `--servers a,b` without a `+` uses exactly
those.

`terminal`, `system`, `github`, `claude`, `desktop`, `memory`, `dropbox` and
`toolsuite` are all one file, `cli/local_mcp.py <group>`, written against the
official `mcp` Python SDK. They are ordinary stdio MCP servers, so any other
MCP client can use them too. The official GitHub MCP server was measured and
rejected for these models: 46 tools / ~13K tokens by default and 22 tools /
~7K tokens at its smallest read-only setting, against one tool / ~90 tokens
for `gh`.

**Cutting big servers down.** An entry's optional `"tools"` map in
`mcp_servers.json` exposes only the tools it names, and for each one: `keep`
lists the parameters the model sees, `set` forces arguments on every call,
`description` replaces the text, `append` copies fields that a server returns
only as structured data into the text result, and `squeeze_indent` shrinks
deeply indented output. That is how `cua-driver`'s 60 tools / ~24K tokens
became 5 tools / ~770 tokens.

**`memory`.** `remember(title, content)` writes a markdown note to
`~/.local/share/local-llm/memory/notes/`. `recall(query)` returns the best
matching notes in full, plus related topics from a graphify knowledge graph
built over the notes. The graph is rebuilt when a `--tools` session ends
(`local_mcp.py memory-sync`, run detached, log in
`~/.cache/local-llm/memory-sync.log`): graphify's semantic extraction runs
against the model that is still loaded, which takes one to four minutes per
new note. Recall does not wait for it, because it also reads the notes
directly. Only one build runs at a time (a lock file), each request to the
model may take up to 30 minutes, and the whole build is killed after an
hour; notes it did not reach are picked up after the next session. On a
single-slot model (`qwythos`, `qwen35`) the build occupies the model, so a
session started straight afterwards waits for it: on `qwen35` that wait was
11 minutes. Swapping models right after a session ends just makes the build
fail and retry later. The installed `graphify` lacks the `openai` client its
OpenAI-compatible backend needs, so the sync runs the same version through
`uvx --from "graphifyy[openai]==<installed version>"`.

**`computer`.** Text-only models cannot read screenshots, so this drives apps
through the accessibility tree: `launch_app`, `list_windows`,
`get_window_state` (controls as an indexed text tree), `click`, `set_value`.
Verified on this GNOME Wayland session with the calculator: enter `6*7` with
`set_value`, click `=`, read back `42`. Limits found here: `cua-driver` 0.28
refuses keyboard input (`type_text`, `press_key`, `hotkey`) on GNOME Wayland,
so those tools are left out and text entry is `set_value`; an app that
exposes no accessibility tree cannot be driven; the Wayland backend is
experimental and is switched on by `CUA_DRIVER_RS_ENABLE_WAYLAND=1` in the
entry's `env`.

**`browser`.** `@playwright/mcp` is pinned to a version whose Chromium is
installed under `~/.cache/ms-playwright`. After changing the pin, run
`npx -y @playwright/mcp@<version> install-browser chrome-for-testing`.

Paths in `mcp_servers.json` use `${HOME}` and `${CLI_DIR}`, substituted by
`llmcli.py`, so the file holds nothing machine-specific. Servers launch with
the system `npx`/`uvx`/`uv`.

**`ask_claude`.** Runs the `claude` CLI in print mode with every Claude tool,
MCP server and customization disabled, so Claude can only answer in text. It
starts from a clean environment (only `HOME` and `PATH`): `claude-local`
exports `ANTHROPIC_BASE_URL` and related variables to point Claude Code at a
local model, and inheriting them would make the tool ask the local model
itself. Limits: 10 calls per session, 20,000 characters of question plus
context, 180 s. It uses plan usage. `ASK_CLAUDE_MODEL` changes the model
(default `sonnet`).

**Unreachable servers don't take the CLI down with them.** Each server gets a
bounded connect attempt; if it fails, `llmcli.py` prints one warning line
naming it and moves on with the rest.

**The safety gate did not change.** Every tool call still requires a manual
y/n confirmation before it executes, with no auto-execute path. That gate is
the real safety boundary: `terminal` runs arbitrary commands and `filesystem`
can read and write anything under its root. The servers themselves never ask;
a different MCP client pointed at `local_mcp.py` gets no confirmation unless
it provides its own.

**Context budget.** The enabled servers' merged schema is about 3,300 tokens,
sent on every request. `llmcli.py` compares the schema against the server's
real context size at startup and warns if it is tight. A single tool result
larger than 8,000 characters is cut (head and tail kept) before it goes back
to the model. A turn may chain up to 20 tool rounds. MCP servers' own stderr
goes to `~/.cache/local-llm/mcp-servers.log`.

**What's verified.** With no model involved, through `llmcli.py`'s real
schema and execute path: all twelve servers connect (about 8 s) and every
local tool was called for real, including the `run_command` timeout path,
`ask_claude`, `remember`/`recall`, the Dropbox listings, a hackstack scope
refusal, a browser navigate and snapshot, and the full calculator loop with
`computer`. With live models and the first seven servers: `qwythos`, `bolt`
and `nano` each chose the right tool and answered correctly in about a
minute, `qwen35` in 4.5 minutes; `qwythos` also completed a multi-step turn
with `ask_claude` and `gh` in parallel. The newer servers were then run end to end on every model, one task each:

| Task | `qwythos` | `bolt` | `nano` | `qwen35` |
|---|---|---|---|---|
| `computer`: calculator, 6*7 = 42 | pass, 190 s | pass, 243 s | pass, 485 s | fail: lost track, timed out at 30 min |
| `browser`: read a page's link text | pass | pass | fail: never read the page, guessed | pass, 276 s |
| `dropbox` + `toolsuite` status | not run | pass | pass | pass, 231 s |
| `remember` | pass | pass | pass | pass, 164 s |
| `recall` in a later session | pass | pass | pass | pass, 21 min (waited on the graph build) |

`bolt` is the best fit for browser and desktop work: it passed everything and
has the context to spare. `qwen35` is too slow for multi-step desktop tasks
on this card. Not verified: long sessions near the context limit, and any
desktop app other than the calculator.

## 5. Using the GUI day to day

The GUI is a small stack of pieces, and it's worth knowing all three ways
into it, because they're genuinely different in what they require:

**The picker page** (`gui/launch-picker.sh`) is the real front door. It
starts a tiny local backend, `gui/picker_server.py`, listening on port 8089
(stdlib-only Python, no dependencies), which serves `gui/picker.html`, a
three-button page ("Launch Qwen" / "Launch Scout" / "Launch Qwen3-Next").
Clicking a button `POST`s
to `/launch/<model>` on that backend, which shells out to `gui/launch.sh
<model>` in the background and returns immediately (it doesn't wait for the
model to finish loading, since that can take a while on CPU). This is
functionally verified: the POST endpoint was confirmed to correctly launch
`launch.sh` as a detached background process, including against the live
qwen-next server (already-running detection skipped the restart, PID
unchanged). The rendering was also visually confirmed on 2026-07-15: no
gnome-screenshot/scrot/import/maim/flameshot on this machine, but Python's
`mss` (already installed, no sudo needed) grabbed the full virtual display
and a crop of the app-mode Chrome window shows the picker page rendering
correctly, dark theme included, screenshot at
`docs/screenshots/picker-verified-2026-07-15.png`.

**Direct per-model launch** (`gui/launch.sh <scout|qwen|qwen-next>`) is what
the picker button calls, and you can also call it yourself directly, or via
the three `.desktop` files (`gui/local-llm-scout.desktop`,
`gui/local-llm-qwen.desktop`, `gui/local-llm-qwen-next.desktop`). It checks
whether the model's server is
already up (a quick `/health` check, so it doesn't trip over
`start-model.sh`'s "already running" guard), starts it CPU-only via
`bin/start-model.sh <model> --cpu-only` if not, then opens the model's own
built-in llama.cpp chat webui in an app-mode browser window (no tabs, no
address bar, so it reads as a standalone app rather than a browser tab). It
tries `google-chrome`, then `chromium`, then `chromium-browser`, then
`firefox`, in that order, and fails clearly if none are installed. The
three `.desktop` files were installed into `~/.local/share/applications` on
this machine on 2026-07-15 (all three validate cleanly under
`desktop-file-validate`, and `Icon=web-browser` resolves fine against the
Pop icon theme), so double-clicking an app-launcher icon works here now. A
fresh checkout on another machine still needs the manual step:
`cp gui/local-llm-*.desktop ~/.local/share/applications/` followed by
`update-desktop-database ~/.local/share/applications/` if that command
exists. Also note this GUI layer only ever launches CPU-only; GPU-offload mode
(with its mining interaction) is something you invoke directly via
`bin/start-model.sh <model>` yourself, on purpose, not through the GUI.

**The copyable command fallback** lives directly on the picker page itself,
under each button: if the JavaScript `fetch` to the picker backend fails for
any reason (backend not running, browser blocking local requests, whatever),
the page shows the exact `./gui/launch.sh <model>` command to run by hand.
This isn't a "the real thing wasn't built" placeholder, the real click-to-launch
backend genuinely exists and is genuinely wired up; the fallback text is just
a deliberate safety net for the case where it doesn't respond.

Either way, once a model's chat window opens, that page *is* just a plain
web page talking to `POST /v1/chat/completions` on that model's port,
same as the CLI, same as anything else. Loading is slow the first time on
CPU; give it up to a minute after the button says "launching" before judging
whether the chat window looks right.

## 6. How it fits together: what `-ngl` and `--cpu-moe` actually mean

This is the part that makes the launch scripts' flags legible instead of
cargo-culted.

All three models are Mixture-of-Experts (MoE) architectures. A dense model
uses every one of its weights on every single token it processes. An MoE
model instead has a large bank of "expert" feed-forward sub-networks, and a
small routing mechanism that picks only a handful of them to actually run
for each token. Most of an MoE model's weights sit idle for any given token;
only the attention layers and a small "always-active" routing/shared portion
run on literally everything.

That distinction is exactly what makes an 8GB GPU usable here at all. Scout's
two GGUF parts total about 65GB, Qwen3-Coder-Next's single file is about
49GB, and Qwen3-Next-Instruct's single file is about 43GB; none of them
remotely fit in 8GB of VRAM as a whole. But because most of an MoE
model's bulk is those sparsely-used expert weights, you don't actually need
all of them sitting in fast VRAM to get most of the speed benefit. `-ngl
999` tells `llama-server` "offload as many transformer layers to GPU as the
model has" (999 is just a stand-in for "all of them," it's not a real layer
count), and `--cpu-moe` specifically pins the MoE expert feed-forward
tensors to CPU RAM regardless. The combined effect: the attention layers and
shared weights, the parts that run on every single token and therefore
benefit most from GPU speed, live in the scarce, fast VRAM, while the bulk
of the parameter count, the expert weights that any given token mostly
doesn't touch, live in cheap, plentiful CPU RAM. That's the whole trick that
lets a consumer 8GB card meaningfully accelerate models many times larger
than its own memory.

`-ngl 0` in `--cpu-only` mode is the same flag turned all the way off:
nothing goes to GPU, everything runs on CPU, which is why it's slow but also
why it never touches VRAM at all and can safely coexist with mining.

The context-length numbers matter for the same VRAM-budget reason: the
context window determines the size of the KV cache, which is additional
memory that has to sit alongside the model's GPU-resident weights. All
three models use a 4096 context in CPU-only mode, where there's no VRAM
budget to protect. In GPU mode, Scout's context is trimmed to 2048
specifically to leave more VRAM headroom for its larger (17B active
parameter) attention layers, while both Qwen variants stay at 4096 in GPU
mode too, since their much smaller (~3B active parameter) attention/shared
layers leave more headroom to begin with. Same mechanism (context size
trades off against VRAM for weights), just applied more conservatively for
Scout than for either Qwen model in this configuration.

One more small design detail worth understanding while you're in this
territory: `start-model.sh` doesn't just fire off `llama-server` and assume
it worked. It polls `logs/<model>.log` in a loop, checking for the literal
string `"listening on"` (which `llama-server` prints once its HTTP server is
actually up) versus a set of known failure strings (`CUDA error`, `error
loading model`, `Segmentation fault`, and so on). That's why the script's
exit code is trustworthy for scripting against, and why `logs/<model>.log`
is always the right first place to look if a start ever fails: it's the same
thing the script itself is watching.

## 7. Troubleshooting

Two real problems came up while building this, and both are worth knowing
in advance.

### Out-of-memory from the miner holding VRAM

The NiceHash idle-miner sits in GPU VRAM whenever it's running, and an 8GB
card doesn't have much room to spare once a model's GPU-resident weights and
KV cache are added on top. If you ever see a CUDA out-of-memory error in
`logs/<model>.log` after launching without `--cpu-only`, the first thing to
check is what's actually holding VRAM right now:

```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

If `memory.used` is unexpectedly high before you've even started a model,
something (usually the miner, sometimes just a lingering process from a
previous session that didn't fully exit) is still sitting on the card. In
the normal case, `start-model.sh` already stops the miner for you before
launching in GPU mode, so this shouldn't come up; if it does anyway, confirm
the miner is actually stopped (`systemctl is-active nicehash-miner.service`
should say `inactive`), and if `nvidia-smi` still shows the memory held even
with the miner service inactive, check for a stray process
(`nvidia-smi` lists processes by PID at the bottom of its full output, not
just the `--query-gpu` summary) that didn't exit cleanly, and kill it before
retrying. `--cpu-only` mode sidesteps this whole class of problem entirely
since it never touches the GPU.

### Template-leakage garbage output from hitting the wrong endpoint

`llama-server` exposes two different ways to generate text: the raw
`/completion` endpoint, which takes a bare prompt string and does nothing
else to it, and the OpenAI-compatible `/v1/chat/completions` endpoint, which
takes a structured list of `{role, content}` messages and automatically
formats them through the model's embedded Jinja chat template (both GGUFs
here have a proper embedded chat template, confirmed via `gguf_dump.py`,
and `--jinja` is on by default in `llama-server`, which is what actually
applies it).

If you ever hit `/completion` directly with a hand-written prompt string
instead of going through `/v1/chat/completions`, you skip that templating
entirely, and what comes back can include literal template control tokens
and role markers leaking into the output as visible garbage text. This isn't
a model defect; it's a mismatch between the endpoint you called and the
input shape it expects. It happened once during this project's own ad hoc
testing, and the fix was simply switching the test to hit
`/v1/chat/completions` with a proper messages array, exactly what both
`cli/llmcli.py` and the browser-based webui already do. The rule going
forward is simple: always talk to a running model through
`/v1/chat/completions`, never `/completion`, and this class of problem
doesn't come up.

### Mojibake (corrupted emoji/accents) in the CLI's streamed output

This one actually shipped briefly and was caught after the fact: `cli/llmcli.py`'s
core streaming path used to garble any non-ASCII character (emoji, accented
letters, curly quotes) into corrupted-looking bytes, and the corruption
polluted in-memory conversation history, not just what was printed to the
terminal. Root cause: `llama-server`'s streaming response has `Content-Type:
text/event-stream` with no `charset` parameter, so Python's `requests`
library falls back to guessing `ISO-8859-1` for the response encoding, and
`resp.iter_lines(decode_unicode=True)` decodes using that wrong guess,
mangling every multi-byte UTF-8 sequence. The fix (already applied in
`stream_chat()`) is one line: explicitly set `resp.encoding = "utf-8"` right
after the request, before iterating. Confirmed fixed with a forced
round-trip test (asking a model to echo back `Café naïve 🌟 résumé` and
verifying the captured bytes decode as valid, correct UTF-8). If you ever
see garbled non-ASCII text out of the CLI again, this line is the first
place to check.
# Local LLM Setup, continued: the multi-host picture

> These sections extend `docs/TUTORIAL.md`. They pick up where section 7
> (Troubleshooting) leaves off, and they assume you've read sections 1 and 6 —
> the "a model file becomes an HTTP server, everything else is a client"
> framing, and what `-ngl` / `--cpu-moe` actually do. Nothing below changes
> the original contract; it widens it from "one machine" to "two machines,
> and eventually more."

## 8. The multi-host reality

The original tutorial was written when this repo lived on a single box and
that box *was* the whole system. That's no longer true. There are now two
machines in play, they are physically in different places, and they do
different jobs.

**Zephy** — the machine this checkout is on — is an ASUS ROG G14 (GA401IV)
running CachyOS. Its GPU is an RTX 2060 Max-Q with **6GB** of VRAM (about
5746 MiB actually usable once the desktop has taken its share). CPU is a
Ryzen 9 4900HS, 8 cores / 16 threads, 38GB RAM. Zephy runs small models
*locally* on its own GPU, and the rest of the time it is a **client** of the
other machine.

> Note on the older sections: every "8GB", "NiceHash miner", and
> "`nicehash-miner.service`" mention in sections 3–7 describes the *other*
> machine's earlier configuration, not Zephy. Zephy has a 6GB card and no
> miner. The miner-stop block is still in `bin/start-model.sh` because it's
> harmless where the service doesn't exist (`systemctl is-active` just
> returns non-zero and the block is skipped), and keeping one script that
> works on both machines is worth more than trimming a dead branch.

**pop-os** is a separate physical machine at a different site. Different
user account (`mateo`, not `mateo_c-s`), its own disks, its own GPU (an RTX
5060, 8GB). It is the **head node**: the big model lives there. You reach it
only over the Tailscale tailnet — there is no LAN path from wherever Zephy
usually is — and that link is roughly 50ms and DERP-relayed (relayed, not a
direct WireGuard path), so it behaves like a modest-latency internet hop,
not like localhost.

pop-os runs **Qwen3-Next-80B-A3B-Instruct** under `llama-server`, and
exposes it through Tailscale Serve at:

```
https://pop-os.tail1d1c9c.ts.net
```

That endpoint has a real, valid TLS cert (Tailscale issues it) and answers
`POST /v1/chat/completions` with the same OpenAI-compatible shape as any
local `llama-server`. pop-os also keeps a copy of Qwen3-Coder-Next, and is
currently pulling down Qwen3-14B (`Q4_K_M`) for the Phase 2 VRAM-pool work.

A peer session (called "WhiteOut") operates pop-os. This checkout cannot
start, stop, or reconfigure anything on pop-os — coordination with that
machine happens by message, and any claim about what's live on pop-os
should be verified against the actual endpoint before you build on it.

### Field mode vs office mode

There are two completely different ways Zephy participates, and which one
you're in is a physical fact about where the laptop is, not a config flag.

**Field mode** is the default and covers everywhere except one wired desk.
Zephy is away from pop-os. In this mode:

- Zephy runs its **two local dense 7B models** on its own 6GB GPU (details
  below), for anything that needs to be fast, offline, or private to the
  laptop.
- For the big model, Zephy is a **plain HTTPS client** of
  `https://pop-os.tail1d1c9c.ts.net` — exactly the "just another client of
  the same API" idea from section 1, with the base URL pointed at the
  tailnet instead of `127.0.0.1`.
- There is **no RPC, no VRAM pooling**. The tailnet WAN hop is far too slow
  for llama.cpp's RPC compute protocol, which assumes a LAN.

**Office mode** only exists when Zephy is physically carried to pop-os and
plugged into the same wired LAN. Then, and only then, the two GPUs can be
**pooled** over llama.cpp RPC: Zephy runs `ggml-rpc-server` bound to the
office LAN interface, pop-os runs `llama-server --rpc <zephy-lan-ip>:50052`
as the head, and a model too big for either card alone (Qwen3-14B dense,
WhiteRabbitNeo-13B) runs split across both. This is Phase 2, and it is
covered — with its security constraints, which are not optional — in
`CLUSTER.md`.

### The model / port table, as it stands now

`scout` is gone (it was dropped — see the repo history). The current set:

| Key | Model | Runs on | Port | Recipe | Family |
|---|---|---|---|---|---|
| `qwen` | Qwen3-Coder-Next (80B-A3B) | pop-os (head) | 8091 | `-ngl 999 --cpu-moe` | MoE |
| `qwen-next` | Qwen3-Next-80B-A3B-Instruct | pop-os (head) | 8092 | `-ngl 999 --cpu-moe` | MoE |
| `qwen2.5` | Qwen2.5-7B-Instruct | **Zephy (local GPU)** | **8093** | `-ngl 99 -fa on` | dense |
| `wrn` | WhiteRabbitNeo-V3-7B | **Zephy (local GPU)** | **8094** | `-ngl 99 -fa on` | dense |

The two dense entries are the new part. `qwen2.5` is the general-purpose
local workhorse; `wrn` (WhiteRabbitNeo-V3-7B, a Llama-3.1-8B-based
security-focused fine-tune, so ~8B parameters despite the "7B" in the name)
is the local security specialist. Both are `Q4_K_M`, about 4.68GB on disk,
and both live in `models/` (gitignored, byte-verified after download).

Ports 8093 and 8094 continue the same fixed-port contract the original
tutorial describes: a model key maps to exactly one port, that mapping lives
in `bin/start-model.sh` and `cli/llmcli.py`, and any new front end just
needs to know which port to POST to. The MoE ports (8091 / 8092) refer to a
server on pop-os, reached via the tailnet URL rather than `127.0.0.1` —
same contract, different host.

## 9. MoE vs dense: why the launch recipe is not one-size-fits-all

Section 6 explained `-ngl 999 --cpu-moe` for the big models. That recipe is
**correct for MoE backbones and wrong for dense models**, and now that both
kinds are in the repo it's worth stating the split plainly.

**MoE models (`qwen`, `qwen-next`)** — the Qwen3-Next 80B-total / ~3B-active
hybrid-attention MoE backbone. The quantized weights are tens of GB and do
not fit in any GPU here. The trick from section 6 applies: `-ngl 999` puts
every transformer *layer* on the GPU, and `--cpu-moe` then pulls the bulky,
sparsely-used **expert feed-forward tensors** back out to CPU RAM, leaving
only the attention and shared/routing weights — the parts that run on every
token — in fast VRAM. Without `--cpu-moe` the server would try to place the
full expert bank on the GPU and OOM immediately. These models run on pop-os,
which has both the VRAM headroom and the RAM for the expert bank.

**Dense models (`qwen2.5`, `wrn`)** — a dense model uses *every* weight on
*every* token. There are no expert tensors, so there is nothing for
`--cpu-moe` to offload; passing it on a dense model is a no-op at best and a
misleading signal to the next reader at worst. Because these are only
7–8B parameters at `Q4_K_M` (~4.7GB), the **whole model fits in Zephy's
6GB VRAM**. So the recipe is the straightforward one:

```
-ngl 99        # every layer on the GPU (99 is "all of them", like 999 was)
-fa on         # flash attention: Turing / sm_75 supports it, smoke-tested
               # working on this card; it shrinks the KV-cache footprint
# NO --cpu-moe
```

`-ngl 99` rather than `-ngl 999` is just cosmetic honesty — these models
have far fewer than 99 layers, so either value means "all" — but it reads as
"this is the small-model path," which is the point.

**The empirical anchor** (from the Phase 1 smoke test, real numbers, so you
don't have to guess): Qwen2.5-7B-Instruct `Q4_K_M`, launched with
`-ngl 99 -c 4096 --jinja -fa on`, produced:

- **4633 MiB VRAM used**, fully GPU-resident (no CPU-side layers)
- **1115 MiB free** afterward
- HTTP server **listening in ~6s** from launch
- first coherent generation in **~4.7s**

That's the basis for the default `GPU_CTX=8192` on the dense entries: the
KV cache at 4096 context with flash attention is only ~230–250 MiB of that
4633, so doubling to 8192 costs roughly another ~235 MiB and still leaves
~880 MiB free. Pushing past 8192 is possible but starts competing with the
browser and desktop for the last few hundred MiB — if you raise it, watch
`nvidia-smi` across an actual generation (peak VRAM is higher than
load-time VRAM) and back off if free drops under ~300 MiB.

WhiteRabbitNeo-V3-7B is Llama-3.1-8B-based, so it has a few more layers than
Qwen2.5-7B (32 vs 28). Treat the numbers above as a **lower bound** for
`wrn` and verify with `nvidia-smi` if you tune its context.

`--cpu-only` mode is unchanged for both families: `-ngl 0`, 4096 context,
nothing touches the GPU, safe to run any time. Slow, but it's the fallback
that always works.

## 10. Reaching the pop-os head from Zephy

From Zephy, the big model is just a remote HTTP API. Everything section 1
says about clients applies unchanged — the only difference is the base URL.

**The endpoint:**

```
https://pop-os.tail1d1c9c.ts.net/v1/chat/completions
```

**Prerequisite: Tailscale must be up on Zephy.** Check with `tailscale
status`; you should see `pop-os` in the peer list. If Zephy isn't on the
tailnet, this endpoint simply doesn't resolve — there is no other route to
that machine.

**A quick sanity check that won't mislead you:**

```bash
curl -s -H 'Accept-Encoding: identity' \
  https://pop-os.tail1d1c9c.ts.net/v1/models
```

A real request with sane headers gets `HTTP 200` and a JSON model list. Note
the header: a *bare* `curl` with no `Accept-Encoding` gets back a benign
**`HTTP 415`** from the Serve front end — that is **not an outage**, it's an
artifact of how the proxy negotiates encoding. Don't take a 415 from a
header-less probe as "pop-os is down"; send a proper request before
concluding anything.

**A real chat call looks exactly like a local one:**

```bash
curl -s https://pop-os.tail1d1c9c.ts.net/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-next",
    "messages": [{"role": "user", "content": "one-sentence sanity check please"}],
    "stream": true
  }'
```

Same `POST /v1/chat/completions`, same `{role, content}` message array, same
server-sent-event stream back. `cli/llmcli.py` can talk to it by pointing
`base_url_for()` at the tailnet URL instead of `http://127.0.0.1:<port>`
(the `qwen` / `qwen-next` keys are the pop-os models).

**What the network hop costs you.** The link is ~50ms and DERP-relayed, so:

- Time-to-first-token has a visible network tax on top of pop-os's own
  prompt-processing time. Fine for interactive chat; noticeable when you
  shove a large prompt (e.g. a big merged MCP tool schema) across it.
- Throughput of streamed tokens is generally fine — they're small SSE
  frames — but a flaky tailnet (café Wi-Fi, Starlink weather) will show up
  as stutter, not as a clean error.
- **Do not** try to run llama.cpp RPC (`--rpc`) across this link. RPC ships
  raw tensor data every compute step and assumes LAN latency; over the
  tailnet it collapses. RPC / VRAM pooling is office-mode only (Phase 2).

**Coordinating changes.** If the pop-os model set changes (a new port, a
model swapped, the 14B coming online), that's a change made by the WhiteOut
session on pop-os, not something this checkout can do. Confirm the live
state against `/v1/models` on the endpoint before assuming a port contract
still holds.
