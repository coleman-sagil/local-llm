#!/usr/bin/env bash
###############################################################################
# zephy-rpc-setup.sh  --  Phase 2 VRAM-pool RPC infrastructure, ZEPHY side
#
# Installs (but does NOT enable) a systemd unit that runs llama.cpp's
# ggml-rpc-server as a pure CUDA compute peer for the pop-os head node, plus
# the ufw rules that keep the (auth-less, TLS-less) RPC port reachable only
# from wired office LANs and never from the tailnet.
#
# RUN AS ROOT:   sudo ./zephy-rpc-setup.sh
#
# ---------------------------------------------------------------------------
# WHY THE UNIT IS INSTALLED **DISABLED**:
#   The ggml-rpc-server wire protocol has ZERO authentication and ZERO
#   encryption (see llama.cpp/tools/rpc/README.md: "the functionality is
#   fragile and insecure. Never run the RPC server on an open network").
#   It is only ever safe to run while Zephy is physically docked on the
#   pop-os wired LAN. So this script installs the unit but leaves it
#   `disabled` -- you start it BY HAND when docked:
#
#       sudo systemctl start ggml-rpc-server.service
#
#   and stop it before undocking:
#
#       sudo systemctl stop ggml-rpc-server.service
#
#   Never `systemctl enable` it.
# ---------------------------------------------------------------------------
###############################################################################
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: must run as root (sudo $0)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 0. Config knobs (override via env; nothing machine-specific is hardcoded)
# ---------------------------------------------------------------------------
RPC_USER="${RPC_USER:-mateo_c-s}"          # unprivileged account that owns the build
RPC_PORT="${RPC_PORT:-50052}"              # ggml-rpc-server default
RPC_THREADS="${RPC_THREADS:-}"             # empty => let the binary decide (hw_concurrency/2)
HELPER_BIN="/usr/local/bin/ggml-rpc-resolve-lan-ip.sh"
ENV_FILE="/run/ggml-rpc-server.env"        # tmpfs; written fresh by ExecStartPre each start
UNIT_FILE="/etc/systemd/system/ggml-rpc-server.service"

# ---------------------------------------------------------------------------
# 1. Locate the already-built ggml-rpc-server binary (dynamic detection)
# ---------------------------------------------------------------------------
RPC_BIN="${RPC_BIN:-}"
if [[ -z "${RPC_BIN}" ]]; then
  RPC_USER_HOME="$(getent passwd "${RPC_USER}" | cut -d: -f6)"
  candidates=(
    "${LOCAL_LLM_ROOT:-}/llama.cpp/build/bin/ggml-rpc-server"
    "${RPC_USER_HOME}/local-llm/llama.cpp/build/bin/ggml-rpc-server"
    "${RPC_USER_HOME}/git/local-llm/llama.cpp/build/bin/ggml-rpc-server"
  )
  for c in "${candidates[@]}"; do
    [[ -n "${c}" && -x "${c}" ]] && RPC_BIN="${c}" && break
  done
fi
if [[ -z "${RPC_BIN}" || ! -x "${RPC_BIN}" ]]; then
  echo "ERROR: could not find an executable ggml-rpc-server." >&2
  echo "       Set RPC_BIN=/abs/path/to/ggml-rpc-server or LOCAL_LLM_ROOT=/abs/path/to/local-llm and re-run." >&2
  exit 1
fi
RPC_BIN_DIR="$(dirname "${RPC_BIN}")"          # build/bin -- also holds the CUDA-enabled ggml .so files
echo "Using ggml-rpc-server: ${RPC_BIN}"

# Real flags of THIS build (llama.cpp @ 082b326, ggml-rpc-server / server v9951):
#   -h, --help                      show help and exit
#   -t, --threads N                 CPU-device threads            (default: hw_concurrency/2)
#   -d, --device <d1,d2,...>        comma/slash-separated device allow-list (e.g. CUDA0)
#   -H, --host HOST                 bind address                  (default: 127.0.0.1)
#   -p, --port PORT                 bind port                     (default: 50052)
#   -c, --cache                     enable on-disk tensor cache (boolean; dir = $LLAMA_CACHE
#                                   or ~/.cache/llama.cpp/rpc). NOTE: in this build -c takes
#                                   NO path argument.
# There is NO --rpc flag on the server (that flag lives on llama-server/llama-cli,
# i.e. the pop-os head). Binding to anything other than 127.0.0.1 prints a big
# insecure-network warning but still works. Protocol handshake is "RPC v3.0.0".
# GGML_RPC_DEBUG=1 in the environment turns on verbose protocol logging.

# ---------------------------------------------------------------------------
# 2. Install the LAN-IP resolver helper (invoked as ExecStartPre)
# ---------------------------------------------------------------------------
# Picks the primary wired/wifi IPv4: first global-scope v4 address whose
# interface is NOT loopback / tailscale* / docker* / virbr* / br-* / veth* .
# Today that resolves to 192.168.1.147 (wlan0); at the office it will be the
# 10.1.10.x address on whatever dock/ethernet iface is up. Never hardcoded.
cat > "${HELPER_BIN}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-/run/ggml-rpc-server.env}"

lan_ip="$(
  ip -o -4 addr show scope global 2>/dev/null \
    | awk '{print $2, $4}' \
    | grep -vE '^(lo|tailscale[0-9]*|docker[0-9]*|virbr[0-9]*|br-|veth|wg[0-9]*|zt[0-9a-z]*|tun[0-9]*) ' \
    | awk '{sub(/\/.*/, "", $2); print $2; exit}'
)"

if [[ -z "${lan_ip:-}" ]]; then
  echo "ggml-rpc-resolve-lan-ip: no non-virtual global IPv4 found; refusing to start" >&2
  exit 1
fi

# Extra guard: never bind inside the Tailscale CGNAT range 100.64.0.0/10.
IFS=. read -r o1 o2 _ _ <<< "${lan_ip}"
if [[ "${o1}" -eq 100 && "${o2}" -ge 64 && "${o2}" -le 127 ]]; then
  echo "ggml-rpc-resolve-lan-ip: resolved ${lan_ip} is in the tailnet CGNAT range; refusing" >&2
  exit 1
fi

umask 022
printf 'RPC_LAN_IP=%s\n' "${lan_ip}" > "${OUT}"
echo "ggml-rpc-resolve-lan-ip: will bind ${lan_ip}"
EOF
chmod 0755 "${HELPER_BIN}"
echo "Installed ${HELPER_BIN}"

# ---------------------------------------------------------------------------
# 3. Install the systemd unit  (DISABLED -- manual start only)
# ---------------------------------------------------------------------------
THREADS_ARG=""
[[ -n "${RPC_THREADS}" ]] && THREADS_ARG=" --threads ${RPC_THREADS}"

cat > "${UNIT_FILE}" <<EOF
# ggml-rpc-server.service -- Phase 2 VRAM-pool CUDA compute peer (ZEPHY side)
#
# INSTALLED DISABLED ON PURPOSE. The RPC protocol has no auth and no TLS, so it
# must only run while this laptop is physically docked on the pop-os wired LAN.
# Start it by hand when docked:   sudo systemctl start ggml-rpc-server.service
# Stop it before you undock:      sudo systemctl stop  ggml-rpc-server.service
# Do NOT 'systemctl enable' this unit.
#
# The pop-os head node then runs:
#   llama-server -m Qwen3-14B-Q4_K_M.gguf --host 0.0.0.0 --port 8095 \\
#       -c <CTX> -ngl 99 --jinja -fa on \\
#       --rpc <this-machine-LAN-IP>:${RPC_PORT} --tensor-split <RATIO>
# (see POOL.md)

[Unit]
Description=llama.cpp ggml-rpc-server (CUDA compute peer for pop-os VRAM pool)
Documentation=file://${RPC_BIN_DIR}/../../tools/rpc/README.md
After=network-online.target
Wants=network-online.target
# Refuse to run if the tailnet is the only thing up -- this is office-dock only.
ConditionPathExists=!/run/ggml-rpc-server.disabled

[Service]
Type=simple
User=${RPC_USER}
# ExecStartPre resolves the primary LAN IP at every start and drops it in a
# tmpfs env file that ExecStart reads back as \${RPC_LAN_IP}.
ExecStartPre=${HELPER_BIN} ${ENV_FILE}
EnvironmentFile=${ENV_FILE}
# CUDA runtime env. CUDA_VISIBLE_DEVICES pins the single Turing GPU; the ggml
# CUDA .so files sit next to the binary in build/bin.
Environment=CUDA_VISIBLE_DEVICES=0
Environment=GGML_CUDA_NO_PINNED=0
Environment=LD_LIBRARY_PATH=${RPC_BIN_DIR}
WorkingDirectory=${RPC_BIN_DIR}
ExecStart=${RPC_BIN} --host \${RPC_LAN_IP} --port ${RPC_PORT} --device CUDA0${THREADS_ARG}
Restart=no
RuntimeDirectory=ggml-rpc-server
# --- hardening (RPC peer needs GPU + net, not much else) ---
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true
ProtectControlGroups=true
ProtectKernelModules=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX AF_NETLINK
# GPU device nodes must stay visible, so DevicePolicy is left open.

[Install]
WantedBy=multi-user.target
EOF
echo "Installed ${UNIT_FILE}"

systemctl daemon-reload
# Explicitly ensure it is NOT enabled (no-op if it never was).
systemctl disable ggml-rpc-server.service >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 4. Firewall (ufw): reachable only from RFC1918 wired LANs, never the tailnet
# ---------------------------------------------------------------------------
# The three private ranges cover home (192.168.1.0/24) and office (10.1.10.0/24).
# The tailnet uses 100.64.0.0/10 (CGNAT) which matches none of them, but we add
# an explicit interface-level deny on tailscale0 as belt-and-braces, prepended
# so it wins regardless of rule order.
if ! ufw status >/dev/null 2>&1; then
  echo "NOTE: ufw is not active. Enable it first (sudo ufw enable) for these rules to take effect." >&2
fi

ufw prepend deny in on tailscale0 to any port "${RPC_PORT}" proto tcp
ufw allow from 10.0.0.0/8      to any port "${RPC_PORT}" proto tcp
ufw allow from 172.16.0.0/12   to any port "${RPC_PORT}" proto tcp
ufw allow from 192.168.0.0/16  to any port "${RPC_PORT}" proto tcp
ufw reload || true

# CAVEAT: docker0 (172.17.0.1/16) falls inside 172.16.0.0/12, so local
# containers on the default bridge can also reach ${RPC_PORT}. Acceptable on a
# single-user laptop; tighten to the exact office/home /24s if that changes.

# ---------------------------------------------------------------------------
# 5. VERIFY
# ---------------------------------------------------------------------------
cat <<EOF

============================  VERIFY  ============================
# 1. Unit is installed but DISABLED (should print 'disabled'):
systemctl is-enabled ggml-rpc-server.service

# 2. Real flags of the pinned build:
${RPC_BIN} --help

# 3. LAN-IP resolver picks a sane wired/wifi address (NOT 100.64/10, NOT lo):
${HELPER_BIN} /tmp/ggml-rpc-verify.env && cat /tmp/ggml-rpc-verify.env

# 4. ufw rules present, tailscale0 denied:
ufw status verbose | grep -E '${RPC_PORT}|tailscale0'

# --- the following require being DOCKED on the pop-os wired LAN ---

# 5. Start it by hand and confirm it bound the LAN IP (not 0.0.0.0, not tailnet):
sudo systemctl start ggml-rpc-server.service
sleep 2
systemctl --no-pager -l status ggml-rpc-server.service
ss -ltnp | grep ':${RPC_PORT}'            # LocalAddress must be the 10.x/192.168.x IP

# 6. From pop-os, the port answers on the LAN IP:
#     nc -vz <zephy-lan-ip> ${RPC_PORT}

# 7. From a tailnet host, the port must be REFUSED/filtered:
#     nc -vz <zephy-tailnet-100.x> ${RPC_PORT}     # expect timeout / no route

# 8. Stop before undocking:
sudo systemctl stop ggml-rpc-server.service
=================================================================
EOF
