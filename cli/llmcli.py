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
import json
import subprocess
import sys

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
# Design notes / deliberate simplifications vs. the core chat path:
#   - Uses stream: false (non-streaming) instead of SSE streaming. Parsing
#     incrementally-streamed tool_call argument fragments is finicky and
#     not worth the complexity for a best-effort demo layer.
#   - Every single tool call is confirmed with the user (y/n) before it is
#     ever executed. There is no auto-execute path, on purpose - these
#     models run unsandboxed on the user's real machine.
#   - Tools: read_file(path) and run_shell(command). Both are genuinely
#     dangerous (arbitrary file read, arbitrary shell execution) if you
#     skip the confirmation prompt. Don't skip it.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the UTF-8 text contents of a file on the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
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

TOOLS_SYSTEM_PROMPT = (
    "You are a helpful local assistant. You have optional access to tools "
    "(read_file, run_shell). Only call a tool when it is actually necessary "
    "to answer the user; every tool call requires the user's manual approval "
    "before it runs, so don't call tools speculatively."
)

MAX_TOOL_ROUNDS = 8


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call after user confirmation. Always returns a string
    (even on error/refusal) so it can go straight back to the model as the
    tool result content."""
    if name == "read_file":
        path = args.get("path", "")
        if not confirm(f"  Model wants to read_file(path={path!r}). Allow?"):
            return "User denied this tool call."
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError as exc:
            return f"Error reading file: {exc}"

    if name == "run_shell":
        command = args.get("command", "")
        if not confirm(f"  Model wants to run_shell(command={command!r}). Allow?"):
            return "User denied this tool call."
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            output += f"\n[exit code: {result.returncode}]"
            return output
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 60s."
        except OSError as exc:
            return f"Error running command: {exc}"

    return f"Error: unknown tool '{name}'."


def chat_once_with_tools(base_url: str, messages: list) -> dict:
    """Single non-streaming request, tools attached. Returns the response
    message dict (may contain tool_calls)."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
        },
        timeout=(REQUEST_TIMEOUT_CONNECT, None),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]


def run_tools_repl(model: str) -> None:
    base_url = base_url_for(model)

    if not server_reachable(base_url):
        print(unreachable_message(model, base_url), file=sys.stderr)
        sys.exit(1)

    print(
        f"Connected to {model} at {base_url} [EXPERIMENTAL tools mode].\n"
        "Tool calls (read_file, run_shell) will always ask for y/n confirmation.\n"
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
                message = chat_once_with_tools(base_url, messages)
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

                    result = execute_tool(name, args)
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
        help="EXPERIMENTAL: enable best-effort tool-calling (read_file, run_shell) with y/n confirmation.",
    )
    args = parser.parse_args()

    if args.tools:
        run_tools_repl(args.model)
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
