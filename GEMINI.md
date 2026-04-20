# GEMINI.md

AEGIS-LAB is a hardware-aware model editing and ablation framework designed for secure, reproducible, and performant LLM experimentation.

## Project Overview

AEGIS-LAB architecture centers on:
- **Orchestrator:** Manages job lifecycles, IPC, optimization, and scheduling.
- **State Layer:** Utilizes QIHSE (Quantum/High-Fidelity State Editing) with in-memory fallback.
- **Workers:** Supports CPU, NPU, and VPU execution.
- **Surface:** Optional API and GUI modules for operator interaction.

## Building and Running

### Development
```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Testing
bash ./ci_smoke.sh
PYTHONPATH=src python3 -m unittest discover -s tests/unit
```

### Execution
```bash
# Bootstrap
bash bootstrap.sh

# Launch Orchestrator and API
bash launch.sh

# Run Model Ablation
./run_mission.sh --model <path_to_model> --target <target_behavior>
```

### VPU/Hardware Integration
If using real Myriad VPU hardware:
```bash
./scripts/vpu_probe.sh
./scripts/vpu_env.sh --shell
```

## Development Conventions

- **Python:** Source code resides in `src/aegis_lab/`.
- **Environment:** Use `PYTHONPATH=src` when running commands outside the installed environment.
- **Native Extensions:** Includes Rust native VPU core (`src/native/vpu_core/`).
- **Dependencies:** Optional hardware support is available via `[hardware]` extra (requires OpenVINO 2022.3).
- **Graceful Degradation:** The system should prefer simulation mode if native hardware/libraries (e.g., QIHSE, Myriad VPU) are not present, unless explicitly forced.
- **Testing:** Comprehensive test suites are categorized by scope: `unit`, `integration`, `recovery`, and `performance`.

## Environment Variables
- `AEGIS_REQUIRE_NATIVE_QIHSE`: Enforce native QIHSE usage.
- `AEGIS_ENABLE_LEVEL_ZERO`: Enable Level Zero telemetry.
- `AEGIS_AUTH_TOKEN`: IPC authentication.
- `PYTHONPATH`: Must be set to `src` when running local modules.
