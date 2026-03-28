#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

echo "--- AEGIS-LAB: Hardware-Aware Mission Launch ---"

./bootstrap.sh

echo "Verifying hardware tiers..."
python3 -c "from aegis_lab.hardware.discovery import HardwareDiscovery; d = HardwareDiscovery.discover(); print(f'Detected Tier: {d.get(\"hardware_tier\", \"UNKNOWN\")}')"

echo "Starting ablation pipeline..."
python3 aegis.py mission "$@"
