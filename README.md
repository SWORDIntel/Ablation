# AEGIS-LAB

AEGIS-LAB is a hardware-aware model editing and ablation framework built around a staged orchestrator, a QIHSE-backed state layer, worker-based execution, and optional operator-facing API and GUI surfaces.

## Status

Status as of 2026-03-28:

- The repository is installable from the `src/` layout via `pyproject.toml`.
- Core unit, integration, recovery, and performance test suites pass in a non-hardware environment using the current simulation and in-memory fallback paths.
- Native QIHSE and Intel-specific accelerator integrations are still optional at runtime; when unavailable, the code now degrades to explicit fallback modes instead of failing immediately.

Verified commands:

```bash
bash ./ci_smoke.sh
PYTHONPATH=src python3 -m unittest discover -s tests/unit -p 'test_*.py'
PYTHONPATH=src python3 -m unittest discover -s tests/integration -p 'test_*.py'
PYTHONPATH=src python3 -m unittest discover -s tests/recovery -p 'test_*.py'
PYTHONPATH=src python3 -m unittest discover -s tests/performance -p 'test_*.py'
PYTHONWARNINGS=error::ResourceWarning PYTHONPATH=src python3 -m unittest discover -s tests/integration -p 'test_*.py'
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
- VPU and NPU workers can run in simulation mode when OpenVINO devices are not available.

## Current Limitations

- The GUI and hardware paths still assume optional desktop and Intel runtime dependencies.
- The editing and validation stack now reports explicit fallback mode, but it does not yet represent a fully production-grade semantic evaluation pipeline.
- There is still no license file in the repository root.

## Documentation

- `docs/architecture/architecture.md`
- `docs/VPU_INTEGRATION.md`
- `docs/schemas/schemas.md`
- `AGENTS.md`
