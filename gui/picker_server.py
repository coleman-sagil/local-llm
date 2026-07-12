#!/usr/bin/env python3
"""gui/picker_server.py

Tiny, dependency-free (stdlib only) local HTTP backend for gui/picker.html's
"real click-to-launch" behavior.

Serves picker.html at GET / and handles POST /launch/<scout|qwen> by
shelling out to gui/launch.sh <model> in the background (which itself
starts the model via bin/start-model.sh <model> --cpu-only if needed, then
opens the model's webui in a browser app-mode window). This process does
NOT wait for the model to finish loading before responding -- launch.sh
runs detached and does its own readiness polling.

Binds to 127.0.0.1 only. No external dependencies, no CDN, nothing beyond
the Python standard library.

Usage:
    gui/picker_server.py [port]        # default port 8089
"""
from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
ROOT_DIR = GUI_DIR.parent
PICKER_HTML = GUI_DIR / "picker.html"
LAUNCH_SCRIPT = GUI_DIR / "launch.sh"

VALID_MODELS = {"scout": 8090, "qwen": 8091}

DEFAULT_PORT = 8089


class PickerHandler(BaseHTTPRequestHandler):
    server_version = "LocalLLMPicker/1.0"

    # Quiet down default logging noise; keep it minimal and useful.
    def log_message(self, fmt, *args):  # noqa: A003
        sys.stderr.write("[picker_server] %s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/picker.html", "/index.html"):
            try:
                body = PICKER_HTML.read_bytes()
            except OSError as exc:
                self._send_json(500, {"error": f"could not read picker.html: {exc}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        parts = self.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "launch":
            model = parts[1]
            if model not in VALID_MODELS:
                self._send_json(
                    400,
                    {"error": f"unknown model '{model}'; must be one of {sorted(VALID_MODELS)}"},
                )
                return

            if not LAUNCH_SCRIPT.exists():
                self._send_json(500, {"error": f"launch script not found: {LAUNCH_SCRIPT}"})
                return

            log_path = ROOT_DIR / "logs" / f"picker-launch-{model}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(log_path, "ab") as log_file:
                    # start_new_session detaches the child from this server
                    # process group, so it (and the browser it opens)
                    # keeps running even if picker_server.py is stopped.
                    subprocess.Popen(
                        [str(LAUNCH_SCRIPT), model],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        cwd=str(ROOT_DIR),
                    )
                port = VALID_MODELS[model]
                self._send_json(
                    202,
                    {
                        "status": "launching",
                        "model": model,
                        "url": f"http://127.0.0.1:{port}",
                        "log": str(log_path),
                    },
                )
            except OSError as exc:
                self._send_json(500, {"error": f"failed to launch {model}: {exc}"})
        else:
            self._send_json(404, {"error": "not found"})


def main() -> None:
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    server = HTTPServer(("127.0.0.1", port), PickerHandler)
    print(f"local-llm picker server listening on http://127.0.0.1:{port}")
    print(f"serving: {PICKER_HTML}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
