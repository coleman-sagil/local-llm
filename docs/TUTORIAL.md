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

### The `--tools` layer: honest status

```bash
python3 cli/llmcli.py --model qwen --tools
```

This is an experimental, opt-in layer bolted onto the side of the CLI, kept
deliberately separate from the core chat path (it's non-streaming, and it's
clearly commented in the source as best-effort). It wires up two tools,
`read_file(path)` and `run_shell(command)`, using the standard OpenAI
`tools`/`tool_calls` function-calling format, and **every single tool call
requires a manual y/n confirmation before it executes**, with no
auto-execute path at all. That confirmation gate is not a placeholder; it's
the actual safety boundary, since these tools can read arbitrary files or run
arbitrary shell commands unsandboxed on your real machine.

What's verified and what isn't, plainly: the wire format is correct (a raw
API call confirmed the request is well-formed and the tool schemas are sent
correctly), and the safety fallback is verified (when a model responds with
plain text instead of a structured tool call, the CLI correctly treats it as
a normal reply and never fires the confirm/execute path). What is **not**
verified is the actual happy path, model emits a real structured tool call,
you approve it, the tool runs, the result feeds back into the conversation.
Neither model available in this setup reliably produced a structured
`tool_calls` response under CPU-only quantized inference during testing;
Scout emitted plain-text pseudo-syntax instead of a real tool call, and
Qwen3-Coder-Next (which is the more tool-use-tuned of the two) ran at well
under one token per second CPU-only, too slow to wait out a live test. So
treat `--tools` as a demonstration of the mechanism and the safety gate, not
as something to rely on yet. If you want to actually push on this later, it
would be worth trying `--tools` with Qwen in GPU-offload mode, where it will
run fast enough to actually observe several rounds of tool-calling behavior.

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
functionally verified this session: the POST endpoint was confirmed to
correctly launch `launch.sh` as a detached background process. What's *not*
verified is what the page actually looks like rendered in a browser: this
sandbox had no working screenshot tool available (none of
gnome-screenshot/scrot/import/maim/flameshot/spectacle were installed, `xwd`
failed with X errors, and GNOME's own D-Bus screenshot method returned
access-denied), so the visual layout is unconfirmed. The HTML/CSS is
straightforward and there's no reason to expect it looks broken, but the
honest status is "should work, never actually seen rendered."

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
`firefox`, in that order, and fails clearly if none are installed. Note that
the `.desktop` files are **not currently installed** into
`~/.local/share/applications`, so double-clicking an app-launcher icon
doesn't work yet out of the box; you'd need to copy them there yourself
(`cp gui/local-llm-*.desktop ~/.local/share/applications/`) if you want that.
Also note this GUI layer only ever launches CPU-only; GPU-offload mode
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
