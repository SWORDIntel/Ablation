# Optional Movidius VPU integration

The model brain surgery kit runs directly through PyTorch; a Movidius VPU is not required. VPU support belongs to the optional AEGIS-LAB worker platform.

## What is implemented

`src/aegis_lab/workers/vpu_worker.py` discovers OpenVINO MYRIAD runtime targets, matches hardware profiles and can execute prepared inference tasks. It also has simulation fallbacks when a device or model is unavailable. Those fallbacks return synthetic telemetry, including estimated SHAVE occupancy.

Before claiming native inference, verify a usable runtime target, successful model compilation and task telemetry for a real device. A connected USB stick, discovery entry or simulated success is insufficient. Test the workload on the selected device; do not infer language-model support from a worker interface.

## Environment qualification

```bash
./scripts/vpu_probe.sh
./scripts/vpu_env.sh --shell
```

The helper targets the legacy OpenVINO 2022.3 MYRIAD environment. The project's `hardware` extra declares OpenVINO >=2024.0; it is a separate dependency path and does not guarantee MYRIAD support. Inspect scripts before setup and isolate incompatible runtimes.

QIHSE native acceleration and heterogeneous scheduling also require the installed library, drivers and executed backend path. There is no current measured 2–3x power-efficiency result established by this guide. Capture device/runtime versions, power, latency, workload and baseline for such a claim.

See [operations](OPERATIONS.md) and the [historical VPU design](archive/VPU_INTEGRATION_20261004.md).
