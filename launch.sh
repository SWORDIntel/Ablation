#!/bin/bash

# AEGIS-LAB System Launcher
# Orchestrates background services and optional GUI

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

ORCH_PORT="${AEGIS_ORCH_PORT:-5555}"
API_PORT="${AEGIS_API_PORT:-18000}"
ORCH_LOG="${AEGIS_ORCH_LOG:-orchestrator.log}"
API_LOG="${AEGIS_API_LOG:-api.log}"

if [ "$API_PORT" = "8000" ]; then
    API_PORT=18000
fi

cleanup() {
    trap - EXIT INT TERM

    for pid in "${GUI_PID:-}" "${API_PID:-}" "${ORCH_PID:-}"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done

    for pid in "${GUI_PID:-}" "${API_PID:-}" "${ORCH_PID:-}"; do
        if [ -n "${pid:-}" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local label="$3"
    local timeout="${4:-30}"

    python3 - "$host" "$port" "$timeout" "$label" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
label = sys.argv[4]
deadline = time.time() + timeout

while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex((host, port)) == 0:
            sys.exit(0)
    time.sleep(0.5)

print(f"Timed out waiting for {label} on {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

trap cleanup EXIT INT TERM

echo "============================================================"
echo "AEGIS-LAB: Surgical Ablation Suite (MTL-P Optimized)"
echo "============================================================"

./bootstrap.sh

echo "[1/3] Starting orchestrator on port $ORCH_PORT..."
python3 aegis.py orchestrator --port "$ORCH_PORT" >"$ORCH_LOG" 2>&1 &
ORCH_PID=$!
wait_for_port 127.0.0.1 "$ORCH_PORT" "orchestrator"
echo "Orchestrator up (PID: $ORCH_PID)."

echo "[2/3] Starting REST API on port $API_PORT..."
python3 aegis.py api --api-port "$API_PORT" --url "tcp://127.0.0.1:$ORCH_PORT" >"$API_LOG" 2>&1 &
API_PID=$!
wait_for_port 127.0.0.1 "$API_PORT" "REST API"
echo "REST API up (PID: $API_PID)."

echo "------------------------------------------------------------"
read -r -p "Launch the GUI dashboard? (y/n): " launch_gui

if [[ "$launch_gui" == "y" || "$launch_gui" == "Y" ]]; then
    echo "[3/3] Launching GUI dashboard..."
    python3 aegis.py gui
else
    echo "[3/3] Running headless."
    echo "Use 'aegis.py submit' or the REST API to manage jobs."
    wait "$ORCH_PID" "$API_PID"
fi
