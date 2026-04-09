# AEGIS-LAB

AEGIS-LAB is a hardware-aware model editing and ablation framework built around a staged orchestrator, a QIHSE-backed state layer, worker-based execution, and optional operator-facing API and GUI surfaces.

## Status

Status as of 2026-04-08:

- **100% Unit Test Success:** All 49 unit tests pass in the default environment, including complex orchestrator and hardware discovery contracts.
- **Hardware-Aware Quantization:** The orchestrator now passes hardware capabilities (NPU/GPU/CPU features) to workers, enabling optimized device-specific quantization (e.g., INT8 on NPU with 128MB cache awareness).
- **Atomic Promotion Logic:** Implemented and verified Milestone 7 (Part B) for atomic bundle promotion. Final artifacts are hashed (SHA256) and moved to the exports directory with collision-resistant naming and integrity manifests.
- **System Stability:** Improved IPC/socket management and thermal safety handling, ensuring robust execution across varying hardware environments.
- **Native Integration:** QIHSE natively links with `libvpu_core.so` and supports dynamic migration policies under thermal or bandwidth pressure.
- **Advanced Model Intervention Framework:** A comprehensive suite of surgical, feature-space, and dynamic inference-time intervention modules (Causal Tracing, SAE Clamping, RepE Steering, Adversarial Hardening).
- **Interactive Intervention UI:** A categorized, multi-select CLI/GUI interface for chaining intervention techniques with real-time impact validation.

Verified commands:

```bash
bash ./ci_smoke.sh
PYTHONPATH=src python3 -m unittest discover -s tests/unit
PYTHONPATH=src python3 -m unittest discover -s tests/integration
PYTHONPATH=src python3 -m unittest discover -s tests/performance
PYTHONPATH=src python3 -m unittest tests.unit.test_hardware_discovery_contract -v
./scripts/vpu_probe.sh
```

## Repository Layout

- `src/aegis_lab/`: main Python package.
- `src/aegis_lab/orchestrator/`: job lifecycle, IPC, optimizer, and scheduling integration.
- `src/aegis_lab/workers/`: CPU, NPU, VPU, and P2P worker logic.
- `src/aegis_lab/state/`: QIHSE wrapper, in-memory fallback, and persistent state model.
- `src/aegis_lab/editing/` and `src/aegis_lab/verification/`: ablation pipeline and validation contracts.
- `src/native/vpu_core/`: Rust native VPU extension.
- `tests/unit/`, `tests/integration/`, `tests/recovery/`, `tests/performance/`: regression coverage by scope.
- `configs/`, `data/`, and `docs/`: runtime configuration, sample datasets, and project documentation.

## Installation

For a reproducible local environment, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

Optional extras:

```bash
python3 -m pip install -e ".[api]"
python3 -m pip install -e ".[gui]"
python3 -m pip install -e ".[hardware]"
```

## Real VPU Runtime

If you want real Myriad execution instead of simulation, use the repo-local launcher after preparing the OpenVINO 2022.3 archive runtime and Python 3.10 environment on a non-root volume:

```bash
./scripts/vpu_probe.sh
./scripts/vpu_env.sh --shell
```

The probe reports both hardware discovery and worker mapping. A healthy real-device result includes `vpu_runtime_usable: True` and an `ov_device` of `MYRIAD`. On the current host, `./scripts/vpu_probe.sh` resolves `vpu_stick_17 -> MYRIAD` with USB serial `03e72485`.

## Startup

Bootstrap validates dependencies, prepares directories, and checks for native prerequisites:

```bash
bash bootstrap.sh
```

Launch the orchestrator and API together:

```bash
bash launch.sh
```

Focused entry points:

```bash
python3 aegis.py orchestrator
python3 aegis.py api --api-port 18000
python3 aegis.py worker
python3 aegis.py gui
```

## Model Editing Workflow

Submit a staged ablation job through the unified launcher:

```bash
python3 aegis.py train \
  --project My-Ablation-Project \
  --model models/qwen2.5.gguf \
  --target refusal \
  --device auto
```

Run the canned mission wrapper:

```bash
./run_mission.sh --model models/qwen2.5.gguf --target refusal
```

Monitor progress:

```bash
python3 aegis.py list
python3 aegis.py status --job-id job-xxxxxxxx
curl http://127.0.0.1:18000/jobs
curl http://127.0.0.1:18000/hardware/sitrep
```

## Runtime Model

- Native QIHSE is used when `QIHSE/qihse/libqihse.so` is present.
- If native QIHSE is missing, the state layer falls back to an in-memory implementation unless `AEGIS_REQUIRE_NATIVE_QIHSE=1` is set.
- Level Zero telemetry is disabled by default in generic environments and can be enabled with `AEGIS_ENABLE_LEVEL_ZERO=1`.
- IPC authentication is controlled by `AEGIS_AUTH_TOKEN`.
- VPU discovery now reports both `vpu_usb_present` and `vpu_runtime_usable`.
- VPU and NPU workers run in simulation mode when OpenVINO devices are not available.
- For Myriad-class VPUs, the generic PyPI OpenVINO path may still stay in simulation; use the local archive-backed launcher in `scripts/vpu_env.sh` when `MYRIAD` runtime support is required.

## Current Limitations

- The GUI and hardware paths still assume optional desktop and Intel runtime dependencies.
- The editing and validation stack reports explicit fallback mode, but it continues to evolve toward a fully production-grade semantic evaluation pipeline.
- High-fidelity QIHSE editing requires a compatible hardware backend for maximum performance.

## Documentation

- `docs/architecture/architecture.md`
- `docs/VPU_INTEGRATION.md`
- `docs/schemas/schemas.md`
- `AGENTS.md`
