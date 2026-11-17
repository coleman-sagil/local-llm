#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "mcp>=2,<3"]
# ///
"""
llmcli.py - a small, hackable REPL for chatting with locally-served LLMs
via llama.cpp's OpenAI-compatible /v1/chat/completions endpoint.

Core design goals (read this before you "improve" it):
  - stdlib + `requests` only. No openai package, no framework.
  - The CORE PATH (plain streaming chat) is meant to be simple and solid.
  - Tool-calling (--tools) is a separate, clearly-optional, best-effort layer
    bolted on the side. See the big comment block above run_tools_repl() for
    why it behaves differently (non-streaming, confirm-before-execute, etc).

Usage (uv reads the dependency header above; no venv or pip needed):
    uv run cli/llmcli.py --model qwythos
    uv run cli/llmcli.py --model bolt --tools
    uv run cli/llmcli.py --model nano --tools --servers terminal,filesystem

Fixed port contract (do not change without updating bin/start-model.sh too):
    qwen       -> http://127.0.0.1:8091   (Qwen3-Coder-Next, coding specialist)
    qwen-next  -> http://127.0.0.1:8092   (Qwen3-Next-80B-A3B-Instruct)
    qwen2.5    -> http://127.0.0.1:8093   (Qwen2.5-7B-Instruct)
    wrn        -> http://127.0.0.1:8094   (WhiteRabbitNeo-V3-7B)
    qwen35     -> http://127.0.0.1:8095   (Qwen3.8-27B uncensored, partial GPU offload)
    bolt       -> http://127.0.0.1:8096   (Gemma-4-E4B uncensored)
    nano       -> http://127.0.0.1:8097   (Nemotron3-Nano-4B uncensored)
    qwythos    -> http://127.0.0.1:8098   (Qwythos-9B, reasoning model)

Multi-host: set LLM_HOST (default 127.0.0.1) to aim the port-based targets above
at another machine's llama-server ports (LAN or tailnet). The separate target
"popos" talks to the remote pop-os head node over its https Tailscale Serve
endpoint (no port, LLM_HOST unused); its wire model id is read from
GET /v1/models at run time.
"""

import argparse
import asyncio
import contextlib
import json
import os
import re
import string
import subprocess
import sys
from pathlib import Path

import requests

MODEL_PORTS = {
    "qwen": 8091,
    "qwen-next": 8092,
    "qwen2.5": 8093,
    "wrn": 8094,
    "qwen35": 8095,
    "bolt": 8096,
    "nano": 8097,
    "qwythos": 8098,
}

# Host for the port-based targets above. Override to aim this CLI at another
# machine running the same llama-server ports (LAN or tailnet). REMOTE_TARGETS
# entries ignore this - they carry a full base URL.
LLM_HOST = (os.environ.get("LLM_HOST") or "127.0.0.1").strip() or "127.0.0.1"

# Fixed-URL targets that are NOT "host:port on LLM_HOST". The pop-os head node
# is reached over its Tailscale Serve HTTPS endpoint (valid cert, no port). Its
# wire model id is not hardcoded - resolve_wire_model() reads it at run time.
REMOTE_TARGETS = {
    "popos": "https://pop-os.tail1d1c9c.ts.net",
}

# Everything --model accepts: local model servers + remote aliases.
ALL_TARGETS = sorted({*MODEL_PORTS, *REMOTE_TARGETS})

REQUEST_TIMEOUT_CONNECT = 5  # seconds, just for the initial reachability probe


def is_remote(model: str) -> bool:
    return model in REMOTE_TARGETS


def base_url_for(model: str) -> str:
    if model in REMOTE_TARGETS:
        return REMOTE_TARGETS[model]
    return f"http://{LLM_HOST}:{MODEL_PORTS[model]}"


def resolve_wire_model(base_url: str):
    """Ask the server which model id it actually serves (GET /v1/models) so the
    request's "model" field can match it. A single-model llama-server is lenient
    here, but a proxied/multi-model remote may not be. Returns None on any
    failure; callers then just omit the field entirely."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=REQUEST_TIMEOUT_CONNECT)
        resp.raise_for_status()
        data = resp.json().get("data") or []
    except (requests.exceptions.RequestException, ValueError):
        return None
    if data and isinstance(data[0], dict):
        return data[0].get("id")
    return None


def server_reachable(base_url: str) -> bool:
    """Cheap reachability probe. Only swallows connection-level failures;
    a real HTTP error response still means "reachable"."""
    try:
        requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_CONNECT)
        return True
    except requests.exceptions.RequestException:
        return False


def unreachable_message(model: str, base_url: str) -> str:
    if is_remote(model):
        return (
            f"Error: can't reach {model} server at {base_url}. "
            f"It's a remote head node - check the tailnet ('tailscale status') "
            f"and that llama-server is running on the other end."
        )
    return (
        f"Error: can't reach {model} server at {base_url}. "
        f"Start it with: bin/start-model.sh {model} --cpu-only "
        f"(or without --cpu-only for GPU speed)."
    )


# ---------------------------------------------------------------------------
# Core streaming chat
# ---------------------------------------------------------------------------

def stream_chat(base_url: str, messages: list, wire_model=None) -> str:
    """POST /v1/chat/completions with stream: true, print tokens as they
    arrive, and return the full assistant reply text (so it can be added
    back into conversation history).

    Raises requests.exceptions.RequestException on connection failure -
    caller decides how to present that to the user.
    """
    payload = {"messages": messages, "stream": True}
    if wire_model:
        payload["model"] = wire_model
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(REQUEST_TIMEOUT_CONNECT, None),  # no read timeout; tokens can be slow on CPU
    )
    resp.raise_for_status()
    # llama-server's SSE response has no charset in its Content-Type
    # (text/event-stream), so requests falls back to guessing ISO-8859-1 for
    # text/* bodies without one. iter_lines(decode_unicode=True) decodes
    # using that guess, silently mangling any non-ASCII UTF-8 byte (emoji,
    # accents, curly quotes) into mojibake - and that corrupted text is what
    # gets stored in conversation history, not just printed. Force the
    # correct encoding before iterating.
    resp.encoding = "utf-8"

    full_reply = []
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue  # SSE keep-alive / blank separator lines
        if not raw_line.startswith("data:"):
            continue
        payload = raw_line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue  # be defensive about stray/partial lines
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            print(content, end="", flush=True)
            full_reply.append(content)
    print()  # newline after the streamed reply
    return "".join(full_reply)


def run_chat_repl(model: str) -> None:
    base_url = base_url_for(model)

    if not server_reachable(base_url):
        print(unreachable_message(model, base_url), file=sys.stderr)
        sys.exit(1)

    wire_model = resolve_wire_model(base_url) if is_remote(model) else None
    wire_note = f" [wire model: {wire_model}]" if wire_model else ""
    print(f"Connected to {model} at {base_url}{wire_note}. Commands: /clear, /exit (or Ctrl+C / Ctrl+D)\n")

    messages = []
    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print("\nGoodbye.")
            return
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye.")
            return

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("Goodbye.")
            return
        if user_input == "/clear":
            messages = []
            print("(conversation cleared)")
            continue

        messages.append({"role": "user", "content": user_input})
        print(f"{model}> ", end="", flush=True)
        try:
            reply = stream_chat(base_url, messages, wire_model)
        except requests.exceptions.RequestException as exc:
            print(f"\nError talking to {model} server: {exc}", file=sys.stderr)
            print(unreachable_message(model, base_url), file=sys.stderr)
            # Drop the user message we just appended so a retry after
            # restarting the server isn't polluted by a half-turn.
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# EXPERIMENTAL / BEST-EFFORT: tool-calling layer, gated behind --tools.
#
# This is off by default and is NOT part of the "core path" contract above.
# Small local Q4 models are not reliable at structured tool-calling - they
# can emit malformed JSON arguments, hallucinate tool names, or loop. This
# layer is a minimal demonstration, not a hardened agent framework.
#
# Architecture (real MCP client, not hardcoded tools):
#   - Every tool comes from an MCP server configured in mcp_servers.json
#     (stdio transport, via the `mcp` SDK). There are no hardcoded tools:
#     shell access is the "terminal" server in cli/local_mcp.py, which also
#     provides the system, github, claude (escalation) and desktop servers.
#     --servers a,b,c picks a subset for one session.
#   - Every tool is exposed to the model under a namespaced name:
#     "{server}__{original_name}", e.g. "terminal__run_command",
#     "filesystem__read_file". This is a hard requirement, not cosmetic: two
#     servers can expose tools with the same name. One dispatch table, one
#     namespacing rule, no per-server special-casing.
#   - On startup, every *enabled* server in mcp_servers.json is connected
#     with a bounded timeout. A server that's configured but unreachable
#     (bad path, no credentials yet, hangs on init - e.g. the google-drive
#     scaffold entry before OAuth setup is done) gets ONE warning line and
#     is skipped; it never crashes the CLI or blocks the other servers.
#   - The `mcp` package is only imported lazily, inside this tools-only code
#     path, so plain `llmcli.py --model X` (no --tools) still needs nothing
#     beyond stdlib + requests, per the core design goals at the top of this
#     file. If `mcp` isn't installed, --tools fails fast with one clear line
#     instead of an ImportError traceback.
#   - Uses stream: false (non-streaming) instead of SSE streaming. Parsing
#     incrementally-streamed tool_call argument fragments is finicky and
#     not worth the complexity for a best-effort demo layer.
#   - Every single tool call is confirmed with the user (y/n) before it is
#     ever executed - no exceptions, no auto-execute
#     path. This is the actual safety boundary (these tools read/write
#     arbitrary files under the configured root and drive a real unsandboxed
#     browser) and it is not up for negotiation by a future edit.
#   - Honest known limitation, found while wiring this up (not a bug in the
#     code, a fact about these models): the *merged tool schema itself* can
#     be larger than these models' context windows. With both filesystem
#     (~14 tools) and playwright (~24 tools) enabled, the schema alone is
#     ~6600 prompt tokens against a 4096-token context (measured directly
#     against the live qwen-next server: "request (6637 tokens) exceeds the
#     available context size (4096 tokens)" - before a single word of
#     conversation). See the startup warning below and docs/TUTORIAL.md.
# ---------------------------------------------------------------------------

CLI_DIR = Path(__file__).resolve().parent
MCP_SERVERS_CONFIG_PATH = CLI_DIR / "mcp_servers.json"
MCP_CONNECT_TIMEOUT_S = 30  # generous: first npx/uvx run per package can cold-check its cache

TOOLS_SYSTEM_PROMPT = (
    "You are a helpful assistant running on the user's own Linux machine, "
    "with tools for the shell, files, the system, the network and more. "
    "Call a tool only when it is needed to answer; every tool call needs the "
    "user's manual approval before it runs, so don't call tools "
    "speculatively. If a problem is beyond you and an ask_claude tool is "
    "available, use it rather than guessing. "
    "You have persistent memory backed by graphify knowledge graphs shared "
    "across all models on this machine. Use recall to search notes and the "
    "knowledge graph when the user refers to earlier sessions or to something "
    "you don't know. Use graph_search for deeper traversal of the knowledge "
    "graph (code, docs, relationships). Use remember to save durable facts. "
    "Use graph_status to see what knowledge is available."
)

# MCP servers write their own logging to stderr (npm notices, driver warnings).
# It goes here instead of into the middle of the conversation.
MCP_SERVER_LOG = Path.home() / ".cache" / "local-llm" / "mcp-servers.log"

# Browser and desktop tasks are a chain of small steps (look, act, look again).
MAX_TOOL_ROUNDS = 20
# A single tool result bigger than this is cut (head + tail kept) before it goes
# back to the model: one unbounded file read or web page would otherwise fill
# the whole context window of these models.
MAX_TOOL_RESULT_CHARS = 8000


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def load_mcp_server_configs(only: list[str] | None = None,
                            path: Path = MCP_SERVERS_CONFIG_PATH) -> list[dict]:
    """Load servers from mcp_servers.json as a list of dicts with keys:
    name, command, args, env.

    With only=None this is every *enabled* server. With only=[names] it is
    exactly those servers, enabled or not (naming a server is an explicit
    request for it); names written "+name" are added to the enabled set
    instead of replacing it. An unknown name is a fatal error rather than a
    silent skip, so a typo can't quietly leave the model without a tool.

    An entry's optional "tools" map is passed through as cfg["tools"]; see
    build_tool_schemas for what it does.

    ${VAR}-style references in command, args and env values are substituted
    from the current environment plus CLI_DIR (this file's directory), so the
    config holds no machine-specific absolute paths. Each server inherits the
    full os.environ with the entry's own "env" block layered on top.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: no MCP server config at {path}; running with zero MCP servers.", file=sys.stderr)
        return []
    except json.JSONDecodeError as exc:
        print(f"Warning: {path} is not valid JSON ({exc}); running with zero MCP servers.", file=sys.stderr)
        return []

    servers = data.get("servers", {})
    enabled = [name for name, entry in servers.items() if entry.get("enabled", False)]
    if only is None:
        names = enabled
    else:
        extra = [name[1:] for name in only if name.startswith("+")]
        exact = [name for name in only if not name.startswith("+")]
        unknown = [name for name in extra + exact if name not in servers]
        if unknown:
            print(f"Error: unknown MCP server(s) {', '.join(unknown)}. "
                  f"Configured in {path.name}: {', '.join(servers)}", file=sys.stderr)
            sys.exit(1)
        names = exact + [name for name in (enabled if extra else []) + extra if name not in exact]
        names = list(dict.fromkeys(names))

    variables = {**os.environ, "CLI_DIR": str(CLI_DIR)}

    def substitute(value):
        return string.Template(value).safe_substitute(variables) if isinstance(value, str) else value

    configs = []
    for name in names:
        entry = servers[name]
        env = {**os.environ}
        for key, value in (entry.get("env") or {}).items():
            env[key] = substitute(value)
        configs.append({
            "name": name,
            "command": substitute(entry["command"]),
            "args": [substitute(arg) for arg in entry.get("args", [])],
            "env": env,
            "tools": entry.get("tools"),
        })
    return configs


def _text_of(content_blocks) -> str:
    """Flatten an MCP CallToolResult's content blocks down to plain text."""
    return "".join(getattr(block, "text", "") for block in content_blocks)


async def _connect_one_server(cfg: dict, stack: "contextlib.AsyncExitStack"):
    """Start one MCP server over stdio, initialize it, and list its tools.
    Registers its teardown on `stack` as soon as it's opened - even if the
    initialize()/list_tools() step below times out, whatever was already
    entered into `stack` (the subprocess + stdio streams + session) is still
    cleaned up correctly when the caller eventually closes `stack`, so a
    slow/wedged server can never leak an orphaned subprocess.

    Deliberately does NOT wrap the stack.enter_async_context() calls in any
    timeout (anyio.fail_after or asyncio.wait_for): both stdio_client and
    ClientSession open anyio cancel scopes/task groups internally that are
    meant to stay open until `stack` closes them much later, and any
    timeout wrapper around the enter_async_context() calls would try to
    close its own (nested) scope before those inner ones are done - anyio
    requires strict LIFO nesting of cancel scopes and raises a RuntimeError
    ("Attempted to exit a cancel scope that isn't the current task's
    current cancel scope") if that's violated. This was caught by
    cli/_test_mcp_integration.py. The bounded timeout below only wraps
    initialize()/list_tools(), which are plain request/response awaits that
    fully resolve (or raise) within the `with` block - nothing is left
    dangling either way, so bounding them is safe."""
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=cfg["command"], args=cfg["args"], env=cfg["env"])
    MCP_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    errlog = stack.enter_context(open(MCP_SERVER_LOG, "a", encoding="utf-8"))
    read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
    session = await stack.enter_async_context(ClientSession(read, write))
    with anyio.fail_after(MCP_CONNECT_TIMEOUT_S):
        await session.initialize()
        listing = await session.list_tools()
    return session, listing.tools


async def connect_configured_servers(configs: list[dict], stack: "contextlib.AsyncExitStack"):
    """Connect to every configured server, skipping (with a one-line
    warning) any that fail to start, fail to import `mcp`/`anyio`, or don't
    respond within MCP_CONNECT_TIMEOUT_S (see _connect_one_server). Returns
    (sessions, tools_by_server), both keyed by server name, containing only
    the servers that connected successfully."""
    sessions = {}
    tools_by_server = {}
    for cfg in configs:
        name = cfg["name"]
        try:
            session, tools = await _connect_one_server(cfg, stack)
        except Exception as exc:
            print(f"Warning: MCP server '{name}' unavailable, skipping ({exc!r}).", file=sys.stderr)
            continue
        sessions[name] = session
        tools_by_server[name] = tools
        print(f"  MCP server '{name}': connected, {len(tools)} tool(s).")
    return sessions, tools_by_server


SLIM_TOOL_DESCRIPTION_CHARS = 240
SLIM_PARAM_DESCRIPTION_CHARS = 120


def _slim_schema(tool, rule: dict) -> tuple[str, dict]:
    """Cut one tool's schema down to the parameters named in rule["keep"],
    with descriptions shortened. Big third-party servers describe every tool
    for a frontier model with a huge context; these models need the short
    version."""
    schema = tool.input_schema or {"type": "object", "properties": {}}
    keep = rule.get("keep")
    properties = {}
    for name, spec in (schema.get("properties") or {}).items():
        if keep is not None and name not in keep:
            continue
        spec = dict(spec) if isinstance(spec, dict) else {}
        if isinstance(spec.get("description"), str):
            spec["description"] = spec["description"][:SLIM_PARAM_DESCRIPTION_CHARS]
        properties[name] = spec
    slim = {"type": "object", "properties": properties,
            "required": [name for name in schema.get("required") or [] if name in properties]}
    description = rule.get("description") or (tool.description or "")[:SLIM_TOOL_DESCRIPTION_CHARS]
    return description, slim


def build_tool_schemas(tools_by_server: dict, filters: dict | None = None) -> tuple[list[dict], dict]:
    """Merge every connected MCP server's tools into one OpenAI-style tool
    schema list, namespaced as "{server}__{tool}". Returns (schemas, dispatch)
    where dispatch maps namespaced name -> (server, original_name, forced_args).

    filters maps a server name to that server's "tools" map from
    mcp_servers.json. A server with such a map exposes ONLY the tools named in
    it, each slimmed by its rule: {"keep": [params shown to the model],
    "set": {param: value forced on every call}, "description": override,
    "append": [structured-result fields to add to the text result],
    "squeeze_indent": bool}. A server with no map exposes everything unchanged."""
    schemas = []
    dispatch = {}
    for server_name, tools in tools_by_server.items():
        rules = (filters or {}).get(server_name)
        for tool in tools:
            if rules is not None and tool.name not in rules:
                continue
            rule = (rules or {}).get(tool.name) or {}
            if rules is None:
                description = tool.description or ""
                parameters = tool.input_schema or {"type": "object", "properties": {}}
            else:
                description, parameters = _slim_schema(tool, rule)
            namespaced = f"{server_name}__{tool.name}"
            schemas.append({
                "type": "function",
                "function": {"name": namespaced, "description": description, "parameters": parameters},
            })
            dispatch[namespaced] = (server_name, tool.name, rule)
    return schemas, dispatch


def clip_tool_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    half = MAX_TOOL_RESULT_CHARS // 2
    cut = len(text) - MAX_TOOL_RESULT_CHARS
    return f"{text[:half]}\n[... {cut} chars truncated by the client ...]\n{text[-half:]}"


async def execute_tool(name: str, args: dict, dispatch: dict, sessions: dict) -> str:
    """Execute a tool call after user confirmation, routing it to the right
    MCP server session. Always returns a string (even on error/refusal) so
    it can go straight back to the model as the tool result content."""
    entry = dispatch.get(name)
    if entry is None:
        return f"Error: unknown tool '{name}'."
    source, original_name, rule = entry
    args = {**args, **(rule.get("set") or {})}

    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if not confirm(f"  Model wants to call {source}.{original_name}({args_str}). Allow?"):
        return "User denied this tool call."

    session = sessions.get(source)
    if session is None:
        return f"Error: MCP server '{source}' is not connected."
    try:
        result = await session.call_tool(original_name, args)
    except Exception as exc:
        return f"Error calling {source}.{original_name}: {exc}"
    text = _text_of(result.content)
    if rule.get("squeeze_indent"):
        # Accessibility trees nest 30+ spaces deep; keep the nesting visible at a sixth of the cost.
        text = re.sub(r"^ +", lambda m: " " * (len(m.group(0)) // 6), text, flags=re.M)
    structured = getattr(result, "structured_content", None) or {}
    for field in rule.get("append") or []:
        if field in structured:
            # Some servers return ids only in structuredContent; the model sees only text.
            text = f"{field}={structured[field]}\n{text}"
    text = clip_tool_result(text)
    return f"[tool error] {text}" if result.is_error else text


def start_memory_sync(base_url: str) -> None:
    """Detached `local_mcp.py memory-sync`: a no-op unless a note is newer than the graph."""
    log = open(MCP_SERVER_LOG.with_name("memory-sync.log"), "a", encoding="utf-8")
    subprocess.Popen(["uv", "run", "-q", "--script", str(CLI_DIR / "local_mcp.py"), "memory-sync", base_url],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)


def chat_once_with_tools(base_url: str, messages: list, tool_schemas: list, wire_model=None) -> dict:
    """Single non-streaming request, tools attached. Returns the response
    message dict (may contain tool_calls)."""
    payload = {"messages": messages, "tools": tool_schemas, "stream": False}
    if wire_model:
        payload["model"] = wire_model
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=(REQUEST_TIMEOUT_CONNECT, None),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]


def warn_if_schema_too_big(base_url: str, tool_schemas: list) -> None:
    """Best-effort, non-fatal heads-up: query the live server's actual
    context size (/slots) and compare it against a rough estimate of the
    merged tool schema's token footprint (bytes/4, the usual ballpark for
    English/JSON text). Both playwright (~24 tools) and filesystem+playwright
    together are known to be tight or over budget on the 2048-4096 ctx these
    models currently run with - see the big comment above and
    docs/TUTORIAL.md. This can't be exact (real tokenization isn't a fixed
    bytes/token ratio) and isn't meant to be - it's a cheap early warning so
    a real 'exceeds context size' error isn't the first the user hears of it."""
    try:
        resp = requests.get(f"{base_url}/slots", timeout=REQUEST_TIMEOUT_CONNECT)
        resp.raise_for_status()
        slots = resp.json()
        n_ctx = slots[0]["n_ctx"]
    except Exception:
        return  # best-effort only; /slots may be disabled or shaped differently

    estimated_tokens = len(json.dumps(tool_schemas)) // 4
    if estimated_tokens >= n_ctx:
        print(
            f"Warning: the merged tool schema is an estimated ~{estimated_tokens} tokens, "
            f"which already meets or exceeds this server's context size ({n_ctx} tokens) - "
            "on its own, before any conversation. Expect 'exceeds context size' errors on "
            "the first message. Pass a smaller set with --servers (filesystem and playwright "
            "are the biggest contributors) or raise GPU_CTX/CTX in bin/start-model.sh.",
            file=sys.stderr,
        )
    elif estimated_tokens >= n_ctx * 0.7:
        print(
            f"Note: the merged tool schema is an estimated ~{estimated_tokens} tokens against "
            f"a {n_ctx}-token context - there's not much room left for actual conversation.",
            file=sys.stderr,
        )


async def run_tools_repl(model: str, only: list[str] | None = None) -> None:
    base_url = base_url_for(model)

    if not server_reachable(base_url):
        print(unreachable_message(model, base_url), file=sys.stderr)
        sys.exit(1)

    wire_model = resolve_wire_model(base_url) if is_remote(model) else None

    try:
        import mcp  # noqa: F401 (import-guard only; real imports happen where used)
    except ImportError:
        print(
            "Error: --tools mode requires the `mcp` package. Run this script with uv, which "
            "installs it from the header: uv run cli/llmcli.py --model <name> --tools",
            file=sys.stderr,
        )
        sys.exit(1)

    configs = load_mcp_server_configs(only)
    async with contextlib.AsyncExitStack() as stack:
        sessions, tools_by_server = await connect_configured_servers(configs, stack)
        filters = {cfg["name"]: cfg["tools"] for cfg in configs if cfg.get("tools") is not None}
        tool_schemas, dispatch = build_tool_schemas(tools_by_server, filters)
        warn_if_schema_too_big(base_url, tool_schemas)
        if "memory" in sessions:
            # Notes saved this session get indexed into the graphify memory graph
            # after the chat, by the model that is still loaded and now idle.
            stack.callback(start_memory_sync, base_url)

        print(
            f"Connected to {model} at {base_url} [EXPERIMENTAL tools mode].\n"
            f"{len(tool_schemas)} tool(s) available: {', '.join(sorted(dispatch))}\n"
            "Every tool call will always ask for y/n confirmation.\n"
            "Commands: /clear, /exit (or Ctrl+C / Ctrl+D)\n"
        )

        messages = [{"role": "system", "content": TOOLS_SYSTEM_PROMPT}]
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print("\nGoodbye.")
                return
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye.")
                return

            if not user_input:
                continue
            if user_input in ("/exit", "/quit"):
                print("Goodbye.")
                return
            if user_input == "/clear":
                messages = [{"role": "system", "content": TOOLS_SYSTEM_PROMPT}]
                print("(conversation cleared)")
                continue

            turn_start = len(messages)  # so a mid-turn failure can roll back cleanly
            messages.append({"role": "user", "content": user_input})

            try:
                for _ in range(MAX_TOOL_ROUNDS):
                    message = chat_once_with_tools(base_url, messages, tool_schemas, wire_model)
                    tool_calls = message.get("tool_calls")

                    if not tool_calls:
                        reply = message.get("content") or ""
                        print(f"{model}> {reply}")
                        messages.append({"role": "assistant", "content": reply})
                        break

                    # Model wants to call one or more tools. Record its request,
                    # then execute each (with confirmation) and feed results back.
                    messages.append(message)
                    for call in tool_calls:
                        fn = call.get("function", {})
                        name = fn.get("name", "")
                        raw_args = fn.get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            args = {}
                            print(f"  (warning: model sent malformed arguments for {name!r}: {raw_args!r})")

                        result = await execute_tool(name, args, dispatch, sessions)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id", ""),
                                "content": result,
                            }
                        )
                else:
                    print(f"(hit {MAX_TOOL_ROUNDS}-round tool-call limit; stopping this turn)")

            except requests.exceptions.RequestException as exc:
                print(f"\nError talking to {model} server: {exc}", file=sys.stderr)
                print(unreachable_message(model, base_url), file=sys.stderr)
                # Drop the whole dangling turn, not just the last message: a
                # failure on round 2+ happens after the user turn already grew
                # to include an assistant tool_calls message and one or more
                # tool results, and messages.pop() only undid the last of those,
                # leaving orphaned tool_call_ids in history for the next turn.
                del messages[turn_start:]
                continue


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a locally-served LLM (llama.cpp).")
    parser.add_argument(
        "--model",
        choices=ALL_TARGETS,
        required=True,
        help=(
            "Target: a local model server (on LLM_HOST, default 127.0.0.1) or "
            "'popos' for the remote head node's https endpoint."
        ),
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help=(
            "EXPERIMENTAL: enable best-effort MCP-backed tool-calling (every enabled "
            "server in cli/mcp_servers.json) with y/n confirmation per call."
        ),
    )
    parser.add_argument(
        "--servers",
        metavar="A,B,C",
        help=(
            "With --tools: use exactly these servers from cli/mcp_servers.json instead of "
            "every enabled one. Fewer servers means a smaller tool schema and more context "
            "left for the conversation. Write +name to add a disabled server to the enabled "
            "set instead, e.g. --servers +browser,+computer."
        ),
    )
    args = parser.parse_args()

    if args.tools:
        only = [name.strip() for name in args.servers.split(",") if name.strip()] if args.servers else None
        asyncio.run(run_tools_repl(args.model, only))
    else:
        run_chat_repl(args.model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Catches Ctrl+C at any point, including mid-stream while waiting on
        # a token from a slow CPU-only generation - not just at the input()
        # prompt (which the REPL loops already handle on their own).
        print("\nInterrupted. Goodbye.")
        sys.exit(0)
