#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

"$ROOT_DIR/scripts/vpu_env.sh" "$ROOT_DIR/.venvs/openvino2022/bin/python" - <<'PY'
from aegis_lab.hardware.discovery import HardwareDiscovery
from aegis_lab.workers.vpu_worker import VpuWorker

caps = HardwareDiscovery.discover()
print("hardware", {
    "openvino_import": caps["openvino_import"],
    "vpu_present": caps["vpu_present"],
    "vpu_runtime_usable": caps["vpu_runtime_usable"],
    "vpu_runtime_details": caps["vpu_runtime_details"],
    "vpu_usb_details": caps["vpu_usb_details"],
})

worker = VpuWorker(orchestrator_url="tcp://127.0.0.1:1")
try:
    print("worker", {
        "vpus": list(worker.vpus.keys()),
        "mapping": {
            name: {
                "ov_device": data["ov_device"],
                "serial": data.get("usb_device", {}).get("serial"),
            }
            for name, data in worker.vpus.items()
        },
    })
finally:
    worker.stop()
PY
