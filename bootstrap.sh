#!/bin/bash

# AEGIS-LAB Bootstrap Script
# Automates hardware activation, library compilation, and environment setup.

PROJECT_ROOT=$(pwd)
CONFIG_DIR="$PROJECT_ROOT/configs"
MODEL_CACHE_DIR="$PROJECT_ROOT/model_cache"
QIHSE_SRC_DIR="$PROJECT_ROOT/QIHSE/qihse" # Source directory for QIHSE native library

echo "============================================================"
echo "🔧 AEGIS-LAB: Bootstrap & Hardware Activation"
echo "============================================================"

# 0. Directory Setup
echo "[0/5] Ensuring necessary directory structures exist..."
mkdir -p "$CONFIG_DIR/quantization" "$CONFIG_DIR/runtime_profiles" "$CONFIG_DIR/scheduler" "$CONFIG_DIR/verification"
mkdir -p "$MODEL_CACHE_DIR"
mkdir -p "$QIHSE_SRC_DIR" # Ensure QIHSE source directory exists
echo "✅ Directories created/verified."

# 1. VPU Hardware Activation (Movidius MyriadX)
echo "[1/5] Checking Movidius VPU status and applying udev rules..."
VPU_COUNT=$(lsusb | grep -i "03e7:2485" | wc -l)
if [ "$VPU_COUNT" -gt 0 ]; then
    echo "🔍 $VPU_COUNT Movidius VPU(s) detected. Ensuring udev rules are active..."
    if [ "$VPU_COUNT" -lt 2 ]; then
        echo "⚠️  Alert: Only $VPU_COUNT Movidius stick(s) detected. Aegis-Lab 'Flex Fabric' works best with 2+ sticks."
    fi
    UDEV_RULE_FILE="/etc/udev/rules.d/99-movidius.rules"
    EXPECTED_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
    
    if [ ! -f "$UDEV_RULE_FILE" ] || ! grep -q "$EXPECTED_RULE" "$UDEV_RULE_FILE"; then
        echo "⚠️  VPU udev rules missing or incorrect."
        # Attempt to apply rules non-interactively
        echo "Attempting to apply udev rules non-interactively..."
        if sudo -n tee "$UDEV_RULE_FILE" > /dev/null && sudo -n udevadm control --reload-rules > /dev/null && sudo -n udevadm trigger > /dev/null; then
            echo "✅ VPU udev rules applied successfully."
        else
            echo "❌ Failed to apply udev rules non-interactively. Please run manually:"
            echo "   echo '$EXPECTED_RULE' | sudo tee $UDEV_RULE_FILE && sudo udevadm control --reload-rules && sudo udevadm trigger"
            echo "------------------------------------------------------------"
        fi
    else
        echo "✅ VPU udev rules are correctly configured."
    fi
else
    echo "ℹ️  No Movidius VPUs detected via USB. Skipping VPU-specific udev checks."
fi

# 2. Python Dependencies
echo "[2/5] Verifying Python dependencies..."
if ! python3 -c "import openvino" 2>/dev/null; then
    echo "📦 Installing 'openvino' (required for NPU/VPU support)..."
    pip install --quiet openvino
fi
echo "✅ Python environment verified (OpenVINO detected)."

# 3. QIHSE Native Library Compilation
echo "[3/5] Compiling modular QIHSE native library..."
if [ -d "$QIHSE_SRC_DIR" ]; then
    cd "$QIHSE_SRC_DIR"
    make clean > /dev/null 2>&1
    if make benchmark-a00 > /dev/null 2>&1; then
        echo "✅ QIHSE library compiled successfully (AVX-512/AMX/VNNI enabled)."
    else
        echo "❌ Error: QIHSE compilation failed. Check manual build in $QIHSE_SRC_DIR"
    fi
    cd "$PROJECT_ROOT"
else
    echo "⚠️  QIHSE source directory '$QIHSE_SRC_DIR' not found. Skipping native compilation."
fi

# 4. System Readiness Check
echo "[4/5] Performing comprehensive system readiness check..."
check_system_ready() {
    echo "--- Verifying system readiness ---"
    local all_ready=true

    # Check udev rules presence for Movidius
    local UDEV_RULE_FILE="/etc/udev/rules.d/99-movidius.rules"
    local EXPECTED_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
    local VPU_COUNT=$(lsusb | grep -i "03e7:2485" | wc -l)
    if [ "$VPU_COUNT" -gt 0 ]; then
        if [ "$VPU_COUNT" -lt 2 ]; then
            echo "⚠️  Alert: Only $VPU_COUNT Movidius stick detected. Aegis-Lab works best with 2+."
        fi
        if [ ! -f "$UDEV_RULE_FILE" ] || ! grep -q "$EXPECTED_RULE" "$UDEV_RULE_FILE"; then
            echo "❌ Udev rules for Movidius are missing or incorrect."
            all_ready=false
        else
            echo "✅ Udev rules for Movidius are present."
        fi
    else
        echo "⚠️  No Movidius VPUs detected. Aegis-Lab 'Flex Fabric' will run in simulation mode for VPU."
    fi

    # Check OpenVINO installation
    if ! python3 -c "import openvino" 2>/dev/null; then
        echo "❌ OpenVINO Python package is not installed."
        all_ready=false
    else
        echo "✅ OpenVINO is installed."
    fi

    # Check QIHSE build status (presence of compiled library)
    local QIHSE_LIB_PATH="$QIHSE_SRC_DIR/libqihse.so"
    if [ ! -f "$QIHSE_LIB_PATH" ]; then
        echo "❌ QIHSE native library not found at $QIHSE_LIB_PATH."
        echo "   Consider running 'make' in $QIHSE_SRC_DIR."
        all_ready=false
    else
        echo "✅ QIHSE native library found."
    fi

    if [ "$all_ready" = true ]; then
        echo "--- System is ready for AEGIS-LAB. ---"
        return 0
    else
        echo "--- System is NOT ready. Please address the issues above. ---"
        return 1
    fi
}

if ! check_system_ready; then
    echo "------------------------------------------------------------"
    echo "🚀 Bootstrap aborted due to system readiness issues."
    echo "============================================================"
    exit 1
fi

# 5. Final Capability Check
echo "[5/5] Finalizing capability discovery..."
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src
python3 -c "from aegis_lab.hardware.discovery import HardwareDiscovery; d = HardwareDiscovery(); d.print_capabilities()"

echo "============================================================"
echo "🚀 Bootstrap Complete. AEGIS-LAB is ready for launch."
echo "============================================================"
