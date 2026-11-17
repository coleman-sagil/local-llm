# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=2,<3"]
# ///
"""local_mcp.py <group> - local stdio MCP servers for small local models.

One file, several servers: the <group> argument picks which tools this process
exposes, so each group can be enabled on its own in mcp_servers.json and shows
up to the model under its own namespace (terminal__run_command, ...).

    terminal  run_command
    system    system_info, list_processes, network_info, http_probe
    github    gh            (the already-logged-in gh CLI)
    claude    ask_claude    (escalate a question to Claude via the claude CLI)
    desktop   open_path, notify
    memory    remember, recall   (notes on disk, indexed as a graphify knowledge graph)
    dropbox   dropbox_find, dropbox_recent, dropbox_status   (the locally synced folders)
    toolsuite hackstack, hackstack_targets   (~/toolsuite/hackstack scope-gated wrappers)

`local_mcp.py memory-sync <llama-server base url>` is not a server: it rebuilds
the memory graph from new notes using that model (llmcli.py runs it on exit).

Run:  uv run --script /abs/path/local_mcp.py <group>

Tool descriptions are deliberately one line each: every byte of schema is sent
to the model on every request, and these models have 16K-64K of context.

Rules this file follows (each one breaks the server when ignored):
  - nothing is written to stdout except by the SDK (stdout is the JSON-RPC pipe)
  - every subprocess gets stdin=DEVNULL (otherwise it eats the JSON-RPC pipe)
  - tools return plain str with structured_output=False (otherwise the text is sent twice)

This server does not ask for confirmation. The client does: llmcli.py confirms
every tool call with the user before it reaches here.
"""
import datetime
import fcntl
import glob
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

GROUPS = ("terminal", "system", "github", "claude", "desktop", "memory", "dropbox", "toolsuite")
GROUP = sys.argv[1] if len(sys.argv) > 1 else ""
if GROUP not in GROUPS + ("memory-sync",):
    sys.exit(f"usage: local_mcp.py <{'|'.join(GROUPS)}>  |  local_mcp.py memory-sync <base-url>")

OUT_CAP = 8000  # bytes kept per stream (head half + tail half), about 2K tokens
MAX_TIMEOUT_S = 600

mcp = MCPServer(GROUP, log_level="WARNING")


def tool(group):
    """Register the function as a text-only tool, but only in its own group's server."""
    def register(fn):
        return mcp.tool(structured_output=False)(fn) if group == GROUP else fn
    return register


def _clip(text: str) -> str:
    """Keep the head and tail of long text, with a marker saying how much was cut."""
    if len(text) <= OUT_CAP:
        return text
    half = OUT_CAP // 2
    return f"{text[:half]}\n[... {len(text) - OUT_CAP} chars truncated ...]\n{text[-half:]}"


def _capture(pipe, box):
    """Drain a pipe to EOF keeping only the first and last OUT_CAP/2 bytes (bounded memory)."""
    half = OUT_CAP // 2
    head, tail, total = b"", b"", 0
    for chunk in iter(lambda: pipe.read(65536), b""):
        total += len(chunk)
        if len(head) < half:
            take = half - len(head)
            head, chunk = head + chunk[:take], chunk[take:]
        tail = (tail + chunk)[-half:]
    pipe.close()
    cut = total - len(head) - len(tail)
    mark = f"\n[... {cut} bytes truncated ...]\n".encode() if cut > 0 else b""
    box.append((head + mark + tail).decode("utf-8", "replace"))


def _run(argv, timeout_s, cwd=None, env=None, stdin_text=None):
    """Run argv in its own process group; return (status line, stdout, stderr), output capped."""
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,  # own process group, so the timeout can kill children too
    )
    out, err = [], []
    readers = [threading.Thread(target=_capture, args=(p, b), daemon=True)
               for p, b in ((proc.stdout, out), (proc.stderr, err))]
    for t in readers:
        t.start()
    if stdin_text is not None:
        try:
            proc.stdin.write(stdin_text.encode())
            proc.stdin.close()
        except BrokenPipeError:
            pass
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
    # Kill the group either way: on timeout it stops the command; after a normal exit it reaps
    # background children (`cmd &`) that would otherwise hold the pipes open forever.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    for t in readers:
        t.join(timeout=5)  # ponytail: a setsid-escaped child can still hold the pipe; stop waiting after 5s
    status = f"TIMEOUT after {timeout_s}s (process group killed)" if timed_out else f"exit code: {proc.returncode}"
    return status, "".join(out), "".join(err)


def _sh(argv, timeout=10):
    """Run a trusted helper command (no shell); return stdout, or '' if missing/failed."""
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------------- terminal

@tool("terminal")
def run_command(command: str, cwd: str = "", timeout_s: int = 60) -> str:
    """Run a bash command. Returns exit code, stdout, stderr (long output is truncated). cwd defaults to home."""
    timeout_s = max(1, min(timeout_s, MAX_TIMEOUT_S))
    workdir = os.path.expanduser(cwd) if cwd else os.path.expanduser("~")
    if not os.path.isdir(workdir):
        raise ToolError(f"cwd is not a directory: {workdir}")
    status, out, err = _run(["bash", "-lc", command], timeout_s, cwd=workdir)
    return f"{status}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"


# --------------------------------------------------------------------------- system

@tool("system")
def system_info() -> str:
    """OS, kernel, CPU, RAM, disk, GPU/VRAM, uptime and load of this machine."""
    osname = next((l.split("=", 1)[1].strip().strip('"') for l in open("/etc/os-release")
                   if l.startswith("PRETTY_NAME=")), platform.system())
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}  # kB
    gib = 1024 ** 3
    lines = [
        f"os: {osname} | kernel {platform.release()} | host {platform.node()}",
        f"cpu: {cpu} ({os.cpu_count()} threads)",
        f"ram: {(mem['MemTotal'] - mem['MemAvailable']) / 1048576:.1f} / {mem['MemTotal'] / 1048576:.1f} GiB used",
    ]
    for path in ("/", "/home"):
        d = shutil.disk_usage(path)
        lines.append(f"disk {path}: {d.used / gib:.0f} / {d.total / gib:.0f} GiB used ({d.free / gib:.0f} free)")
    for g in _sh(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                  "--format=csv,noheader,nounits"]).splitlines():
        name, used, total = (x.strip() for x in g.split(","))
        lines.append(f"gpu: {name} | vram {used} / {total} MiB used")
    up = int(float(open("/proc/uptime").read().split()[0]))
    lines.append(f"uptime: {up // 86400}d {up % 86400 // 3600}h {up % 3600 // 60}m | "
                 f"load: {' '.join(f'{x:.2f}' for x in os.getloadavg())}")
    return "\n".join(lines)


@tool("system")
def list_processes(filter: str = "", limit: int = 25) -> str:
    """Top processes by CPU, or only those whose command line contains filter. Columns: PID %CPU %MEM RSS_MB CMD."""
    rows = []
    for line in _sh(["ps", "-eo", "pid=,pcpu=,pmem=,rss=,args=", "--sort=-pcpu"]).splitlines():
        pid, cpu, pmem, rss, args = line.split(None, 4)
        if int(pid) != os.getpid() and filter.lower() in args.lower():
            rows.append(f"{pid} {cpu} {pmem} {int(rss) // 1024} {args[:120]}")
    limit = max(1, min(limit, 200))
    more = f"\n[{len(rows) - limit} more not shown]" if len(rows) > limit else ""
    return "PID %CPU %MEM RSS_MB CMD\n" + "\n".join(rows[:limit]) + more if rows else "no matching processes"


@tool("system")
def network_info() -> str:
    """Network interfaces and addresses, default route, DNS servers, listening TCP ports."""
    dns = [l.split()[1] for l in open("/etc/resolv.conf") if l.startswith("nameserver")]
    ports = []
    for line in _sh(["ss", "-Htlnp"]).splitlines():  # ponytail: owner process shown only for our own uid (no sudo)
        f = line.split()
        owner = f[-1].split('"')[1] if '"' in f[-1] else "?"
        ports.append(f"{f[3]} ({owner})")
    return "\n".join([
        "interfaces:", _sh(["ip", "-brief", "addr"]),
        "default route: " + (_sh(["ip", "route", "show", "default"]) or "none"),
        "dns: " + (", ".join(dns) or "none"),
        "listening tcp: " + ", ".join(sorted(set(ports))),
    ])


class _NoRaise(urllib.request.HTTPErrorProcessor):
    """Report 4xx/5xx as normal responses instead of raising."""
    def http_response(self, request, response):
        return response
    https_response = http_response


@tool("system")
def http_probe(url: str) -> str:
    """HTTP GET a URL and report status, final URL, key headers and timing. The body is not downloaded."""
    if not url.startswith(("http://", "https://")):  # urllib would also open file:// and ftp://
        raise ToolError("url must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": "local-mcp-probe/1"})
    opener = urllib.request.build_opener(_NoRaise)  # redirects are still followed (max 10)
    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=15) as r:
            ms = (time.perf_counter() - t0) * 1000
            keep = ("content-type", "content-length", "server", "location", "last-modified")
            hdrs = "\n".join(f"{k}: {v}" for k, v in r.headers.items() if k.lower() in keep)
            return f"status: {r.status} {r.reason}\nfinal url: {r.url}\ntime to headers: {ms:.0f} ms\n{hdrs}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise ToolError(f"request failed after {(time.perf_counter() - t0) * 1000:.0f} ms: {e}")


# --------------------------------------------------------------------------- github

@tool("github")
def gh(args: list[str]) -> str:
    """Run the GitHub CLI. args is the argument list after "gh", e.g. ["issue","list","--repo","owner/name"]."""
    if not shutil.which("gh"):
        raise ToolError("the gh CLI is not installed")
    env = {**os.environ, "GH_PAGER": "cat", "NO_COLOR": "1", "GH_PROMPT_DISABLED": "1"}
    status, out, err = _run(["gh", *args], 60, env=env)
    return f"{status}\n{out}" + (f"\n--- stderr ---\n{err}" if err.strip() else "")


# --------------------------------------------------------------------------- claude

ASK_CLAUDE_MODEL = os.environ.get("ASK_CLAUDE_MODEL", "sonnet")
ASK_CLAUDE_MAX_CALLS = 10      # per server process, so a looping local model cannot drain the plan's usage limit
ASK_CLAUDE_MAX_CHARS = 20000   # question + context
ASK_CLAUDE_TIMEOUT_S = 180
_ask_claude_calls = 0


@tool("claude")
def ask_claude(question: str, context: str = "") -> str:
    """Ask Claude, a much stronger AI model, a hard question. It sees only what you pass in: put needed facts in context."""
    global _ask_claude_calls
    claude = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    if not os.path.exists(claude):
        raise ToolError("the claude CLI is not installed")
    if _ask_claude_calls >= ASK_CLAUDE_MAX_CALLS:
        raise ToolError(f"ask_claude limit reached ({ASK_CLAUDE_MAX_CALLS} calls this session). Answer on your own.")
    prompt = f"{question}\n\n<context>\n{context}\n</context>" if context.strip() else question
    if len(prompt) > ASK_CLAUDE_MAX_CHARS:
        raise ToolError(f"question + context is {len(prompt)} chars; the limit is {ASK_CLAUDE_MAX_CHARS}. Shorten it.")
    _ask_claude_calls += 1
    # A clean environment on purpose: `claude-local` exports ANTHROPIC_BASE_URL and friends to
    # route Claude Code to a local model. Inheriting them would make this tool ask the local
    # model itself. With only HOME and PATH, the CLI uses the normal login in ~/.claude.
    env = {"HOME": os.path.expanduser("~"), "PATH": f"{os.path.dirname(claude)}:/usr/bin:/bin"}
    status, out, err = _run(
        [claude, "-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
         "--no-session-persistence", "--max-turns", "1", "--model", ASK_CLAUDE_MODEL,
         "--system-prompt", "You are answering a question escalated by a smaller local model. "
                            "Answer directly and concisely in plain text.",
         "--output-format", "text"],
        ASK_CLAUDE_TIMEOUT_S, cwd=tempfile.gettempdir(), env=env, stdin_text=prompt)
    if status != "exit code: 0":
        raise ToolError(f"claude failed ({status}): {(out or err).strip()[:500]}")
    return out.strip()


# --------------------------------------------------------------------------- desktop

@tool("desktop")
def open_path(target: str) -> str:
    """Open a file, folder or URL in its default desktop application."""
    if not target.startswith(("http://", "https://")):
        target = os.path.expanduser(target)
        if not os.path.exists(target):
            raise ToolError(f"no such file or folder: {target}")
    subprocess.Popen(["xdg-open", target], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return f"opened {target}"


@tool("desktop")
def notify(title: str, body: str = "") -> str:
    """Show a desktop notification."""
    status, _, err = _run(["notify-send", "--", title, body], 10)
    if status != "exit code: 0":
        raise ToolError(f"notify-send failed: {err.strip()}")
    return "notification shown"


# --------------------------------------------------------------------------- memory
#
# Notes are plain markdown files, so nothing is ever lost to a failed index build. graphify
# turns them into a knowledge graph (entities + relations) with the local model doing the
# extraction; that takes about a minute per note, so it runs after the chat (memory-sync),
# not inside remember(). recall() reads the notes directly as well as the graph, so a note
# is findable the moment it is written.

MEMORY_DIR = os.path.expanduser("~/.local/share/local-llm/memory")
NOTES_DIR = os.path.join(MEMORY_DIR, "notes")
NOTES_GRAPH = os.path.join(NOTES_DIR, "graphify-out", "graph.json")
SYSTEM_GRAPH = os.path.expanduser("~/graphify-out/graph.json")
RECALL_NOTES = 3          # full notes returned per recall
RECALL_NOTE_CHARS = 1500
# A slow model (the 27B at ~2 tok/s) needs more than graphify's default 600 s per request;
# with the default it timed out, split the chunk and retried without end.
SYNC_REQUEST_TIMEOUT_S = 1800
SYNC_TOTAL_TIMEOUT_S = 3600   # then the whole build is killed; unindexed notes are retried next session


def _note_paths():
    return sorted(glob.glob(os.path.join(NOTES_DIR, "*.md")), reverse=True)  # newest first (date-prefixed names)


@tool("memory")
def remember(title: str, content: str) -> str:
    """Save a fact or note to long-term memory so it is available in future sessions. One topic per note."""
    if not title.strip() or not content.strip():
        raise ToolError("title and content are both required")
    os.makedirs(NOTES_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "note"
    path = os.path.join(NOTES_DIR, f"{datetime.datetime.now():%Y-%m-%d-%H%M%S}-{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title.strip()}\n\n{content.strip()}\n")
    return f"saved as {os.path.basename(path)}"


@tool("memory")
def recall(query: str) -> str:
    """Search long-term memory from earlier sessions. Returns matching notes and how their topics are related."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    if not words:
        raise ToolError("query needs at least one word of 3+ letters")
    scored = []
    for path in _note_paths():
        text = open(path, encoding="utf-8", errors="replace").read()
        low = text.lower()
        score = sum(low.count(w) for w in words)
        if score:
            scored.append((score, path, text))
    scored.sort(key=lambda item: -item[0])  # stable: ties stay newest first
    parts = [f"--- {os.path.basename(path)} ---\n{text.strip()[:RECALL_NOTE_CHARS]}"
             for _, path, text in scored[:RECALL_NOTES]]
    if len(scored) > RECALL_NOTES:
        parts.append("other matching notes: " + ", ".join(os.path.basename(p) for _, p, _ in scored[RECALL_NOTES:12]))
    if shutil.which("graphify"):
        for label, gpath in [("notes", NOTES_GRAPH), ("system", SYSTEM_GRAPH)]:
            if not os.path.exists(gpath):
                continue
            out = _sh(["graphify", "query", query, "--graph", gpath, "--budget", "400"], timeout=30)
            lines = [l for l in out.splitlines() if l.startswith(("NODE", "EDGE"))]
            if lines:
                parts.append(f"--- {label} knowledge graph ---\n" + "\n".join(lines))
    return "\n\n".join(parts) if parts else "nothing in memory matches that query"


@tool("memory")
def graph_search(query: str, mode: str = "bfs") -> str:
    """Search the graphify knowledge graph for concepts, code, or connections. mode: bfs (broad) or dfs (trace a path)."""
    if not shutil.which("graphify"):
        raise ToolError("graphify is not installed")
    graphs = [(l, g) for l, g in [("notes", NOTES_GRAPH), ("system", SYSTEM_GRAPH)] if os.path.exists(g)]
    if not graphs:
        raise ToolError("no knowledge graph built yet")
    flag = ["--dfs"] if mode == "dfs" else []
    results = []
    for label, gpath in graphs:
        out = _sh(["graphify", "query", query, "--graph", gpath, "--budget", "800"] + flag, timeout=30)
        lines = [l for l in out.splitlines() if l.startswith(("NODE", "EDGE", "HYPER"))]
        if lines:
            results.append(f"--- {label} graph ---\n" + "\n".join(lines))
    return "\n\n".join(results) if results else "no matching nodes in any knowledge graph"


@tool("memory")
def graph_status() -> str:
    """Show available knowledge graphs and memory notes."""
    parts = []
    for label, gpath in [("notes", NOTES_GRAPH), ("system", SYSTEM_GRAPH)]:
        if os.path.exists(gpath):
            size_mb = os.path.getsize(gpath) / (1024 * 1024)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(gpath)).strftime("%Y-%m-%d %H:%M")
            parts.append(f"{label} graph: {gpath} ({size_mb:.0f} MB, updated {mtime})")
        else:
            parts.append(f"{label} graph: not built")
    notes = _note_paths()
    parts.append(f"notes: {len(notes)} in {NOTES_DIR}")
    return "\n".join(parts)


def memory_sync(base_url: str) -> int:
    """Rebuild the graph if any note is newer than it. Incremental: graphify only re-reads changed notes."""
    notes = _note_paths()
    if not notes:
        return 0
    if os.path.exists(NOTES_GRAPH) and os.path.getmtime(NOTES_GRAPH) >= max(map(os.path.getmtime, notes)):
        return 0
    # One build at a time: two of them on a single-slot server starve each other. The lock is
    # held by this process and released by the kernel when it exits, however it exits.
    lock = open(os.path.join(MEMORY_DIR, "sync.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("memory-sync: another build is already running; leaving it to finish", file=sys.stderr)
        return 0
    # The installed graphify has no `openai` client, which its OpenAI-compatible backend needs.
    # uvx runs the same version with that extra, without touching the installed tool.
    version = re.search(r"^graphifyy v?(\S+)", _sh(["uv", "tool", "list"]), re.M)
    spec = f"graphifyy[openai]=={version.group(1)}" if version else "graphifyy[openai]"
    env = {**os.environ, "OPENAI_BASE_URL": base_url.rstrip("/") + "/v1", "OPENAI_API_KEY": "none",
           "OPENAI_MODEL": "local"}
    status, out, err = _run(["uvx", "-q", "--from", spec, "graphify", "extract", NOTES_DIR,
                             "--backend", "openai", "--max-concurrency", "1",
                             "--api-timeout", str(SYNC_REQUEST_TIMEOUT_S)],
                            SYNC_TOTAL_TIMEOUT_S, env=env)
    print(f"memory-sync: {status}\n{out}\n{err}", file=sys.stderr)
    return 0 if status == "exit code: 0" else 1


# --------------------------------------------------------------------------- dropbox

def _dropbox_roots():
    return sorted(d for d in glob.glob(os.path.expanduser("~/*Dropbox*")) if os.path.isdir(d))


def _dropbox_files(find_args, limit):
    """Newest-first file listing across the synced Dropbox folders via find(1)."""
    roots = _dropbox_roots()
    if not roots:
        raise ToolError("no Dropbox folder found in the home directory")
    out = _sh(["find", *roots, "-type", "f", "-not", "-path", "*/.dropbox.cache/*", *find_args,
               "-printf", "%T@\t%TY-%Tm-%Td %TH:%TM\t%s\t%p\n"], timeout=60)
    rows = sorted((l.split("\t") for l in out.splitlines() if l.count("\t") == 3),
                  key=lambda r: -float(r[0]))
    limit = max(1, min(limit, 200))
    lines = [f"{when}  {int(size) // 1024:>8} KB  {path}" for _, when, size, path in rows[:limit]]
    more = f"\n[{len(rows) - limit} more not shown]" if len(rows) > limit else ""
    return "\n".join(lines) + more if lines else "no matching files"


@tool("dropbox")
def dropbox_find(name_contains: str, limit: int = 30) -> str:
    """Find files in the synced Dropbox folders whose path contains the text (case-insensitive). Newest first."""
    if not name_contains.strip():
        raise ToolError("name_contains is required")
    return _dropbox_files(["-ipath", f"*{name_contains}*"], limit)


@tool("dropbox")
def dropbox_recent(days: int = 7, limit: int = 30) -> str:
    """List files in the synced Dropbox folders modified in the last N days. Newest first."""
    # Some synced files carry timestamps years in the future; without the upper bound they top every listing.
    return _dropbox_files(["-mtime", f"-{max(1, days)}", "-not", "-newermt", "now"], limit)


@tool("dropbox")
def dropbox_status() -> str:
    """Whether the Dropbox sync client is running, and where the synced folders are."""
    running = bool(_sh(["pgrep", "-f", "dropbox-dist"]))
    roots = _dropbox_roots()
    return "\n".join([f"sync client: {'running' if running else 'NOT running'}",
                      "synced folders:"] + [f"  {r}" for r in roots] +
                     ["read or write the files themselves with the filesystem tools"])


# --------------------------------------------------------------------------- toolsuite

HACKSTACK_DIR = os.path.expanduser("~/toolsuite/hackstack")
HACKSTACK_COMMANDS = ("recon", "webscan", "capture", "findings", "engagement", "pivot")


@tool("toolsuite")
def hackstack(command: str, args: list[str]) -> str:
    """Run a hackstack wrapper: recon|webscan|capture|findings|engagement|pivot, e.g. command="recon", args=["nmap","zephy"]. Only in-scope target labels are accepted."""
    if command not in HACKSTACK_COMMANDS:
        raise ToolError(f"command must be one of: {', '.join(HACKSTACK_COMMANDS)}")
    wrapper = os.path.join(HACKSTACK_DIR, "bin", f"hs-{command}")
    if not os.path.exists(wrapper):
        raise ToolError(f"not installed: {wrapper}")
    # The wrapper is the scope gate (targets/*.env profiles, fail-closed); this tool adds no bypass.
    status, out, err = _run([wrapper, *args], MAX_TIMEOUT_S, cwd=HACKSTACK_DIR)
    return f"{status}\n{out}" + (f"\n--- stderr ---\n{err}" if err.strip() else "")


@tool("toolsuite")
def hackstack_targets() -> str:
    """List the target labels hackstack is allowed to scan, and each wrapper's usage."""
    labels = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(HACKSTACK_DIR, "targets", "*.env")))
    clients = sorted(os.path.basename(p)[:-4]
                     for p in glob.glob(os.path.join(HACKSTACK_DIR, "targets", "engagements", "*.env")))
    usage = []
    for command in HACKSTACK_COMMANDS:
        try:
            lines = open(os.path.join(HACKSTACK_DIR, "bin", f"hs-{command}"), encoding="utf-8").read().splitlines()
        except OSError:
            continue
        usage += [l[1:].strip() for l in lines[:40] if re.match(rf"#\s+hs-{command} ", l)]
    return "\n".join(["personal-lab targets: " + (", ".join(labels) or "none"),
                      "engagement clients (use <client>/<host>): " + (", ".join(clients) or "none"),
                      "usage (pass everything after the wrapper name as args):"] + [f"  {u}" for u in usage])


if __name__ == "__main__":
    if GROUP == "memory-sync":
        if len(sys.argv) < 3:
            sys.exit("usage: local_mcp.py memory-sync <llama-server base url>")
        sys.exit(memory_sync(sys.argv[2]))
    mcp.run()  # stdio transport
