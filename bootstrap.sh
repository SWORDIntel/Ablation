#!/bin/bash

# AEGIS-LAB Bootstrap Script
# Automates hardware activation, library compilation, and environment setup.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CONFIG_DIR="$PROJECT_ROOT/configs"
MODEL_CACHE_DIR="$PROJECT_ROOT/model_cache"
QIHSE_SRC_DIR="$PROJECT_ROOT/QIHSE/qihse"
QIHSE_LIB_PATH="$QIHSE_SRC_DIR/libqihse.so"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "============================================================"
echo "AEGIS-LAB: Bootstrap & Hardware Activation"
echo "============================================================"

cd "$PROJECT_ROOT"

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

count_movidius_devices() {
    if ! have_cmd lsusb; then
        echo 0
        return
    fi
    lsusb 2>/dev/null | awk '/03e7:2485/ {count++} END {print count+0}'
}

check_system_ready() {
    echo "--- Verifying system readiness ---"
    local all_ready=true
    local vpu_count
    vpu_count="$(count_movidius_devices)"

    if [ "$vpu_count" -gt 0 ]; then
        if [ "$vpu_count" -lt 2 ]; then
            echo "Warning: only $vpu_count Movidius stick(s) detected; 2+ is recommended."
        fi

        local udev_rule_file="/etc/udev/rules.d/99-movidius.rules"
        local expected_rule='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'

        if [ ! -f "$udev_rule_file" ] || ! grep -q "$expected_rule" "$udev_rule_file"; then
            echo "Warning: Movidius udev rules are missing or incorrect."
            if have_cmd sudo && sudo -n true 2>/dev/null; then
                if printf '%s\n' "$expected_rule" | sudo tee "$udev_rule_file" >/dev/null && \
                    sudo udevadm control --reload-rules >/dev/null && \
                    sudo udevadm trigger >/dev/null; then
                    echo "Applied Movidius udev rules."
                else
                    echo "Warning: unable to apply udev rules non-interactively."
                    all_ready=false
                fi
            else
                echo "Warning: sudo unavailable for automatic udev rule installation."
                all_ready=false
            fi
        else
            echo "Movidius udev rules are present."
        fi
    else
        echo "No Movidius VPUs detected; continuing in simulation mode."
    fi

    if ! python3 -c "import openvino" >/dev/null 2>&1; then
        echo "OpenVINO is not installed."
        all_ready=false
    else
        echo "OpenVINO is installed."
    fi

    if [ ! -f "$QIHSE_LIB_PATH" ]; then
        echo "QIHSE native library not found at $QIHSE_LIB_PATH."
        all_ready=false
    else
        echo "QIHSE native library found."
    fi

    if [ "$all_ready" = true ]; then
        echo "--- System is ready for AEGIS-LAB. ---"
        return 0
    fi

    echo "--- System is NOT ready. Please address the issues above. ---"
    return 1
}

echo "[1/5] Ensuring necessary directory structures exist..."
mkdir -p \
    "$CONFIG_DIR/quantization" \
    "$CONFIG_DIR/runtime_profiles" \
    "$CONFIG_DIR/scheduler" \
    "$CONFIG_DIR/verification" \
    "$MODEL_CACHE_DIR" \
    "$QIHSE_SRC_DIR"
echo "Directories created or verified."

echo "[2/5] Checking Movidius VPU status."
vpu_count="$(count_movidius_devices)"
if [ "$vpu_count" -gt 0 ]; then
    echo "Detected $vpu_count Movidius VPU(s)."
else
    echo "No Movidius VPUs detected."
fi

echo "[3/5] Verifying Python dependencies."
if ! python3 -c "import openvino" >/dev/null 2>&1; then
    echo "Installing OpenVINO for NPU/VPU support."
    python3 -m pip install --quiet openvino
fi
echo "Python environment verified."

echo "[4/5] Compiling QIHSE native library."
if [ -f "$QIHSE_SRC_DIR/Makefile" ] || [ -f "$QIHSE_SRC_DIR/makefile" ]; then
    make -C "$QIHSE_SRC_DIR" clean >/dev/null 2>&1 || true
    if make -C "$QIHSE_SRC_DIR" lib >/dev/null 2>&1; then
        echo "QIHSE library compiled successfully."
    else
        echo "QIHSE compilation failed. Check the native build in $QIHSE_SRC_DIR."
        exit 1
    fi
else
    echo "QIHSE Makefile not found; skipping native compilation."
fi

echo "[5/5] Performing comprehensive system readiness check."
if ! check_system_ready; then
    echo "Bootstrap aborted due to system readiness issues."
    exit 1
fi

python3 -c "from aegis_lab.hardware.discovery import HardwareDiscovery; d = HardwareDiscovery(); d.print_capabilities()"

echo "============================================================"
echo "AEGIS-LAB bootstrap complete."
echo "============================================================"
