#!/bin/bash
echo "--- AEGIS-LAB: Hardware-Aware Mission Launch ---"
./bootstrap.sh
echo "Verifying Hardware Tiers..."
python3 -c "from aegis_lab.hardware.discovery import HardwareDiscovery; d = HardwareDiscovery.discover(); print(f'Detected Tier: {d.get(\"hardware_tier\", \"UNKNOWN\")}');"
echo "Starting Ablation Pipeline..."
python3 aegis.py mission "$@"
