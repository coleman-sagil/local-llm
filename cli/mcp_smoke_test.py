#!/usr/bin/env python3
"""
mcp_smoke_test.py - minimal standalone proof that the Python `mcp` SDK can
drive a real MCP server over stdio, independent of any LLM. Not part of
llmcli.py; a later agent wires real MCP support into the CLI. This just
proves the plumbing (SDK + npx-launched server + tool call) works
mechanically.

Why this hardcodes a Node path (read before "fixing"):
  The system Node (`/usr/bin/node`, v12.22.9 from apt, EOL) cannot run
  current MCP servers - they use ES2022 top-level await and die with
  `SyntaxError: Unexpected reserved word` under Node 12. A modern Node
  (v22.23.1 LTS) was downloaded standalone - no sudo, not installed
  system-wide - to /home/mateo/local-llm/.tools/node22/. That path is
  hardcoded as an absolute path (not derived from this file's location)
  on purpose: .tools/ is a local, gitignored toolchain cache tied to the
  machine, not to any particular git worktree/branch checkout of this
  repo, so it should resolve the same way no matter where a copy of this
  script is run from.

  Also critical: `npx`/`npm` in that distribution are JS files with a
  `#!/usr/bin/env node` shebang. Executing them by absolute path is NOT
  enough to pin the Node version - `env node` still re-resolves "node"
  via PATH, so if the caller's shell has system Node first on PATH you
  silently get v12 again and the SyntaxError. This script sidesteps that
  entirely by invoking `node22/bin/node <path-to-npx-cli.js> ...`
  directly (no shebang involved), and additionally passes an env with
  node22/bin prepended to PATH for anything npx spawns internally.

Usage:
    python3 mcp_smoke_test.py filesystem [DIR]   # default DIR: /home/mateo/local-llm/docs
    python3 mcp_smoke_test.py playwright [URL]   # default URL: https://example.com
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

NODE22_BIN_DIR = "/home/mateo/local-llm/.tools/node22/bin"
NODE_BIN = f"{NODE22_BIN_DIR}/node"
NPX_CLI_JS = "/home/mateo/local-llm/.tools/node22/lib/node_modules/npm/bin/npx-cli.js"

ENV = {**os.environ, "PATH": f"{NODE22_BIN_DIR}:{os.environ.get('PATH', '')}"}

# google-chrome (installed at /usr/bin/google-chrome) is used for the
# playwright smoke test instead of letting @playwright/mcp download its own
# Chromium. Playwright's own cached chromium also exists at
# ~/.cache/ms-playwright/chromium-1228/ if that's ever preferred instead.
GOOGLE_CHROME = "/usr/bin/google-chrome"

DEFAULT_FS_DIR = "/home/mateo/local-llm/docs"


def npx_server_params(package: str, extra_args: list[str]) -> StdioServerParameters:
    """StdioServerParameters that run `npx -y <package> <extra_args>` via
    the bundled modern Node directly - not via the npx/npm shebang - so
    this is correct regardless of what's first on the caller's shell PATH.
    """
    return StdioServerParameters(
        command=NODE_BIN,
        args=[NPX_CLI_JS, "-y", package, *extra_args],
        env=ENV,
    )


def _text_of(content_blocks) -> str:
    return "".join(getattr(c, "text", "") for c in content_blocks)


async def smoke_test_filesystem(directory: str) -> None:
    print(f"=== filesystem MCP server smoke test (root={directory}) ===")
    params = npx_server_params("@modelcontextprotocol/server-filesystem", [directory])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"tools available ({len(names)}): {names}")

            if "list_directory" not in names:
                raise RuntimeError(f"expected 'list_directory' tool, got: {names}")

            result = await session.call_tool("list_directory", {"path": directory})
            text = _text_of(result.content)
            print(f"call_tool(list_directory, path={directory!r}) ->")
            print(text)
            if result.isError:
                raise RuntimeError("server reported isError=True on list_directory call")

    print("=== filesystem smoke test PASSED ===")


async def smoke_test_playwright(url: str) -> None:
    print(f"=== playwright MCP server smoke test (url={url}) ===")
    params = npx_server_params(
        "@playwright/mcp",
        ["--headless", "--isolated", "--executable-path", GOOGLE_CHROME],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"tools available ({len(names)}): {names}")

            nav_tool = "browser_navigate"
            if nav_tool not in names:
                raise RuntimeError(f"expected {nav_tool!r} tool, got: {names}")

            result = await session.call_tool(nav_tool, {"url": url})
            text = _text_of(result.content)
            print(f"call_tool({nav_tool}, url={url!r}) ->")
            print(text[:3000])
            if result.isError:
                raise RuntimeError(f"server reported isError=True on {nav_tool} call")

    print("=== playwright smoke test PASSED ===")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("filesystem", "playwright"):
        print("Usage: mcp_smoke_test.py {filesystem|playwright} [arg]", file=sys.stderr)
        sys.exit(2)

    which = sys.argv[1]
    if which == "filesystem":
        directory = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FS_DIR
        asyncio.run(smoke_test_filesystem(directory))
    else:
        url = sys.argv[2] if len(sys.argv) > 2 else "https://example.com"
        asyncio.run(smoke_test_playwright(url))


if __name__ == "__main__":
    main()
