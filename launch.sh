#!/bin/bash

# AEGIS-LAB System Launcher
# Orchestrates background services and optional GUI

PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src

# Run Bootstrap (Hardware activation, build, etc.)
./bootstrap.sh

echo "============================================================"
echo "🚀 AEGIS-LAB: Surgical Ablation Suite (MTL-P Optimized)"
echo "============================================================"

# 1. Start Orchestrator (Background)
echo "[1/3] Starting QIHSE Orchestrator..."
python3 aegis.py orchestrator > orchestrator.log 2>&1 &
ORCH_PID=$!

# Wait for orchestrator to bind
sleep 2
if ! netstat -tulpn 2>/dev/null | grep :5555 > /dev/null; then
    echo "❌ Error: Orchestrator failed to start. Check orchestrator.log"
    kill $ORCH_PID 2>/dev/null
    exit 1
fi
echo "✅ Orchestrator UP (PID: $ORCH_PID)"

# 2. Start REST API (Background with randomized port)
echo "[2/3] Starting REST API Server..."
# Capture the randomized port from output
API_OUTPUT=$(python3 aegis.py api &)
echo "$API_OUTPUT" | grep "Randomized API Port"
echo "✅ API Server starting in background..."

# 3. Optional Web Interface (GUI)
echo "------------------------------------------------------------"
read -p "❓ Would you like to launch the Web Interface/GUI? (y/n): " launch_gui

if [[ $launch_gui == "y" || $launch_gui == "Y" ]]; then
    echo "[3/3] Launching GUI Dashboard..."
    python3 aegis.py gui
else
    echo "[3/3] Skipping GUI. System is running in Headless Mode."
    echo "🔗 Use 'aegis.py submit' or the REST API to manage jobs."
    echo "Press Ctrl+C to shut down all services."
    # Wait for background processes
    wait $ORCH_PID
fi

# Cleanup on exit
trap "kill $ORCH_PID; echo 'Stopping services...'; exit" SIGINT SIGTERM
