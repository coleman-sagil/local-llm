#!/usr/bin/env python3
"""
llmcli.py - a small, hackable REPL for chatting with locally-served LLMs
via llama.cpp's OpenAI-compatible /v1/chat/completions endpoint.

Core design goals (read this before you "improve" it):
  - stdlib + `requests` only. No openai package, no framework.
  - The CORE PATH (plain streaming chat) is meant to be simple and solid.
  - Tool-calling (--tools) is a separate, clearly-optional, best-effort layer
    bolted on the side. See the big comment block above run_tools_repl() for
    why it behaves differently (non-streaming, confirm-before-execute, etc).

Usage:
    python3 llmcli.py --model scout
    python3 llmcli.py --model qwen --tools

Fixed port contract (do not change without updating bin/start-model.sh too):
    scout      -> http://127.0.0.1:8090   (Llama-4-Scout, general-purpose)
    qwen       -> http://127.0.0.1:8091   (Qwen3-Coder-Next, coding specialist)
    qwen-next  -> http://127.0.0.1:8092   (Qwen3-Next-80B-A3B-Instruct, general-purpose
                                            sibling of qwen - same architecture/backbone)
"""

import argparse
import asyncio
import contextlib
import json
import os
import string
import subprocess
import sys
from pathlib import Path

import requests

MODEL_PORTS = {
    "scout": 8090,
    "qwen": 8091,
    "qwen-next": 8092,
}

REQUEST_TIMEOUT_CONNECT = 5  # seconds, just for the initial reachability probe


def base_url_for(model: str) -> str:
    return f"http://127.0.0.1:{MODEL_PORTS[model]}"


def server_reachable(base_url: str) -> bool:
    """Cheap reachability probe. Only swallows connection-level failures;
    a real HTTP error response still means "reachable"."""
    try:
        requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_CONNECT)
        return True
    except requests.exceptions.RequestException:
        return False


def unreachable_message(model: str, base_url: str) -> str:
    return (
        f"Error: can't reach {model} server at {base_url}. "
        f"Start it with: bin/start-model.sh {model} --cpu-only "
        f"(or without --cpu-only for GPU speed)."
    )


# ---------------------------------------------------------------------------
# Core streaming chat
# ---------------------------------------------------------------------------

def stream_chat(base_url: str, messages: list) -> str:
    """POST /v1/chat/completions with stream: true, print tokens as they
    arrive, and return the full assistant reply text (so it can be added
    back into conversation history).

    Raises requests.exceptions.RequestException on connection failure -
    caller decides how to present that to the user.
    """
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": messages,
            "stream": True,
        },
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

    print(f"Connected to {model} at {base_url}. Commands: /clear, /exit (or Ctrl+C / Ctrl+D)\n")

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
            reply = stream_chat(base_url, messages)
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
#   - Tools come from two sources: a tiny always-on "builtin" source (just
#     run_shell - no MCP server here does arbitrary shell execution, so it
#     stays a hardcoded local tool) and zero or more MCP servers configured
#     in mcp_servers.json (stdio transport, via the `mcp` SDK). The old
#     hardcoded read_file is GONE - the filesystem MCP server's own
#     read_file/read_text_file/etc. replace it, sandboxed to whatever root
#     that server is configured with (see mcp_servers.json).
#   - Every tool, from every source, is exposed to the model under a
#     namespaced name: "{source}__{original_name}", e.g. "builtin__run_shell",
#     "filesystem__read_file", "playwright__browser_navigate". This is a
#     hard requirement, not cosmetic: the filesystem MCP server itself
#     exposes a tool literally named "read_file", which would silently
#     collide with the old builtin of the same name if both were ever
#     un-namespaced at once. One dispatch table, one namespacing rule, no
#     per-source special-casing.
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
#     ever executed - MCP-routed or builtin, no exceptions, no auto-execute
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

# Machine-level Node 22 toolchain (downloaded standalone, no sudo, gitignored
# - see cli/mcp_smoke_test.py for the full story). Hardcoded as an absolute
# path on purpose: it's tied to this machine, not to any particular git
# worktree/branch checkout of this repo, so every entry-point that needs it
# (this file, mcp_smoke_test.py) should resolve it the same fixed way.
NODE22_BIN_DIR = "/home/mateo/local-llm/.tools/node22/bin"

MCP_SERVERS_CONFIG_PATH = Path(__file__).resolve().parent / "mcp_servers.json"
MCP_CONNECT_TIMEOUT_S = 30  # generous: first npx run per package can cold-check its cache

BUILTIN_SOURCE = "builtin"

BUILTIN_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": f"{BUILTIN_SOURCE}__run_shell",
            "description": "Run a shell command on the local machine and return its stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]
BUILTIN_DISPATCH = {f"{BUILTIN_SOURCE}__run_shell": (BUILTIN_SOURCE, "run_shell")}

TOOLS_SYSTEM_PROMPT = (
    "You are a helpful local assistant. You have optional access to tools - "
    "always a local run_shell, plus whatever MCP servers are currently "
    "connected (e.g. filesystem access, browser automation). Only call a "
    "tool when it is actually necessary to answer the user; every tool call "
    "requires the user's manual approval before it runs, so don't call "
    "tools speculatively."
)

MAX_TOOL_ROUNDS = 8


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def load_mcp_server_configs(path: Path = MCP_SERVERS_CONFIG_PATH) -> list[dict]:
    """Load the *enabled* servers from mcp_servers.json as a list of dicts
    with keys: name, command, args, env.

    Every server's env is built as: full inherited os.environ, with node22's
    bin dir forced onto the front of PATH, then the entry's own "env" block
    (if any) layered on top with ${VAR}-style substitution against the
    current environment. Forcing node22-first PATH centrally (rather than
    trusting each entry to do it) matters because at least one entry in this
    file (google-drive) launches bare "npx" rather than an absolute node22
    binary; this puts node22 first on PATH for anything that entry (or a
    future one) spawns via bare command-name resolution. The entries this
    file actually enables by default (filesystem, playwright) sidestep the
    question entirely by invoking the node22 binary directly by absolute
    path - see their "notes" in mcp_servers.json.
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

    configs = []
    for name, entry in data.get("servers", {}).items():
        if not entry.get("enabled", False):
            continue
        env = {**os.environ}
        env["PATH"] = f"{NODE22_BIN_DIR}:{os.environ.get('PATH', '')}"
        for key, value in (entry.get("env") or {}).items():
            env[key] = string.Template(value).safe_substitute(os.environ) if isinstance(value, str) else value
        configs.append({
            "name": name,
            "command": entry["command"],
            "args": entry.get("args", []),
            "env": env,
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
    read, write = await stack.enter_async_context(stdio_client(params))
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


def build_tool_schemas(tools_by_server: dict) -> tuple[list[dict], dict]:
    """Merge builtin + every connected MCP server's tools into one OpenAI-
    style tool schema list, namespaced as "{server}__{tool}". Returns
    (schemas, dispatch) where dispatch maps namespaced name -> (server, original_name)."""
    schemas = list(BUILTIN_TOOL_SCHEMAS)
    dispatch = dict(BUILTIN_DISPATCH)
    for server_name, tools in tools_by_server.items():
        for tool in tools:
            namespaced = f"{server_name}__{tool.name}"
            schemas.append({
                "type": "function",
                "function": {
                    "name": namespaced,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
            dispatch[namespaced] = (server_name, tool.name)
    return schemas, dispatch


def _run_shell(args: dict) -> str:
    command = args.get("command", "")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        output += f"\n[exit code: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60s."
    except OSError as exc:
        return f"Error running command: {exc}"


async def execute_tool(name: str, args: dict, dispatch: dict, sessions: dict) -> str:
    """Execute a tool call after user confirmation, routing it to the
    correct source (builtin, or the right MCP server session). Always
    returns a string (even on error/refusal) so it can go straight back to
    the model as the tool result content."""
    entry = dispatch.get(name)
    if entry is None:
        return f"Error: unknown tool '{name}'."
    source, original_name = entry

    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if not confirm(f"  Model wants to call {source}.{original_name}({args_str}). Allow?"):
        return "User denied this tool call."

    if source == BUILTIN_SOURCE:
        if original_name == "run_shell":
            return _run_shell(args)
        return f"Error: unknown builtin tool '{original_name}'."

    session = sessions.get(source)
    if session is None:
        return f"Error: MCP server '{source}' is not connected."
    try:
        result = await session.call_tool(original_name, args)
    except Exception as exc:
        return f"Error calling {source}.{original_name}: {exc}"
    text = _text_of(result.content)
    return f"[tool error] {text}" if result.isError else text


def chat_once_with_tools(base_url: str, messages: list, tool_schemas: list) -> dict:
    """Single non-streaming request, tools attached. Returns the response
    message dict (may contain tool_calls)."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": messages,
            "tools": tool_schemas,
            "stream": False,
        },
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
            "the first message. Disable an MCP server in cli/mcp_servers.json (playwright is "
            "the biggest single contributor) or raise GPU_CTX/CTX in bin/start-model.sh.",
            file=sys.stderr,
        )
    elif estimated_tokens >= n_ctx * 0.7:
        print(
            f"Note: the merged tool schema is an estimated ~{estimated_tokens} tokens against "
            f"a {n_ctx}-token context - there's not much room left for actual conversation.",
            file=sys.stderr,
        )


async def run_tools_repl(model: str) -> None:
    base_url = base_url_for(model)

    if not server_reachable(base_url):
        print(unreachable_message(model, base_url), file=sys.stderr)
        sys.exit(1)

    try:
        import mcp  # noqa: F401 (import-guard only; real imports happen where used)
    except ImportError:
        print(
            "Error: --tools mode requires the `mcp` package (pip install --user mcp).",
            file=sys.stderr,
        )
        sys.exit(1)

    configs = load_mcp_server_configs()
    async with contextlib.AsyncExitStack() as stack:
        sessions, tools_by_server = await connect_configured_servers(configs, stack)
        tool_schemas, dispatch = build_tool_schemas(tools_by_server)
        warn_if_schema_too_big(base_url, tool_schemas)

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
                    message = chat_once_with_tools(base_url, messages, tool_schemas)
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
    parser.add_argument("--model", choices=sorted(MODEL_PORTS), required=True, help="Which model server to talk to.")
    parser.add_argument(
        "--tools",
        action="store_true",
        help=(
            "EXPERIMENTAL: enable best-effort MCP-backed tool-calling (builtin run_shell "
            "plus every enabled server in cli/mcp_servers.json) with y/n confirmation per call."
        ),
    )
    args = parser.parse_args()

    if args.tools:
        asyncio.run(run_tools_repl(args.model))
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
