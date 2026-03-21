# AEGIS-LAB

AEGIS-LAB is a crash-tolerant, hardware-aware model editing platform built around a staged orchestration pipeline, QIHSE-backed state storage, Intel-oriented hardware discovery, and a PyQt operator GUI.

This repository is an early-stage implementation rather than a finished product. The codebase already contains a real project structure, CLI entry points, a REST API, GUI surfaces, tests, and a QIHSE native dependency, but several paths are still partially stubbed or rough around the edges. This README is written to reflect the code that is actually present.

## What It Does

AEGIS-LAB is organized around a job pipeline for model-editing and ablation-style workflows. At a high level it provides:

- A ZeroMQ orchestrator that tracks jobs, stages, workers, approvals, and evaluation results.
- A QIHSE-backed state layer used for jobs, stages, artifacts, atoms, logs, and telemetry snapshots.
- Hardware discovery and telemetry for CPU, iGPU, NPU, and Intel Level Zero environments.
- A PyQt dashboard with system status, job views, hardware charts, artifact browsing, leaderboard, chat, and diff panes.
- A small FastAPI service that exposes job submission and job status over HTTP.
- Unit, integration, performance, and recovery tests covering parts of the orchestration and state layers.

## Current Status

The repository is usable as a development sandbox, but not everything is production-complete.

Known caveats from the current tree:

- `README.md` was previously a stub; this document is now the top-level guide.
- The orchestrator code includes unfinished references and rough edges in some paths.
- Some launch flows are optimistic and assume local dependencies already exist.
- Hardware acceleration paths are strongest on Linux with Intel-oriented runtimes installed.
- Several components simulate or stub behavior rather than implementing a full backend.

If you are evaluating this repo, treat it as an active prototype with real structure, not a polished release.

## Architecture

The main subsystems are:

### Orchestrator

The orchestrator manages job lifecycle, stage assignment, stage completion, approvals, and leaderboard updates.

Relevant files:

- [src/aegis_lab/orchestrator/service.py](/home/john/Ablation/src/aegis_lab/orchestrator/service.py)
- [src/aegis_lab/orchestrator/ipc.py](/home/john/Ablation/src/aegis_lab/orchestrator/ipc.py)
- [src/aegis_lab/scheduler/engine.py](/home/john/Ablation/src/aegis_lab/scheduler/engine.py)

The default training-style workflow currently creates these stages:

1. `intake`
2. `probe`
3. `atom_extract`
4. `atom_clean`
5. `adversarial_eval`
6. `quantize`
7. `verify`
8. `promote`

Communication with workers and clients is handled through ZeroMQ REQ/REP IPC on port `5555` by default.

### State Layer

The persistent state layer wraps QIHSE vector storage and uses timestamp-based upserts to maintain the latest version of records.

Relevant files:

- [src/aegis_lab/state/db.py](/home/john/Ablation/src/aegis_lab/state/db.py)
- [src/aegis_lab/state/qihse_wrapper.py](/home/john/Ablation/src/aegis_lab/state/qihse_wrapper.py)

Tracked entities include:

- Jobs
- Stages
- Atoms
- Artifacts
- Sentinel runs
- Hardware snapshots
- Evaluation scores
- Logs

### Hardware and Telemetry

Hardware detection and runtime telemetry are focused on Intel systems, especially Meteor Lake-style heterogeneous environments.

Relevant files:

- [src/aegis_lab/hardware/discovery.py](/home/john/Ablation/src/aegis_lab/hardware/discovery.py)
- [src/aegis_lab/hardware/telemetry.py](/home/john/Ablation/src/aegis_lab/hardware/telemetry.py)
- [src/aegis_lab/hardware/thermal.py](/home/john/Ablation/src/aegis_lab/hardware/thermal.py)
- [src/aegis_lab/hardware/cuda_bridge.py](/home/john/Ablation/src/aegis_lab/hardware/cuda_bridge.py)

Discovery covers:

- AVX-512
- AMX
- AVX-VNNI
- Hybrid CPU topology
- OpenVINO-discovered GPU and NPU devices
- NPU BAR conflict detection via `/proc/iomem`
- Optional CUDA compatibility detection via ZLUDA-style paths

### GUI

The GUI is a PyQt dashboard for operators.

Relevant files:

- [src/aegis_lab/gui/main_window.py](/home/john/Ablation/src/aegis_lab/gui/main_window.py)
- [src/aegis_lab/gui/widgets/graph_view.py](/home/john/Ablation/src/aegis_lab/gui/widgets/graph_view.py)
- [src/aegis_lab/gui/widgets/chat_view.py](/home/john/Ablation/src/aegis_lab/gui/widgets/chat_view.py)
- [src/aegis_lab/gui/widgets/leaderboard_view.py](/home/john/Ablation/src/aegis_lab/gui/widgets/leaderboard_view.py)

The current GUI includes:

- A `Systems` command deck
- Job dashboard and stage view
- Hardware charts
- Artifact store browser
- Leaderboard
- Interrogation/chat panel
- Diff view

### API

The REST API is a thin FastAPI layer over the orchestrator IPC.

Relevant file:

- [src/aegis_lab/api/server.py](/home/john/Ablation/src/aegis_lab/api/server.py)

Currently exposed endpoints:

- `POST /jobs/submit`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /hardware/sitrep`

## Repository Layout

Top-level structure:

- [aegis.py](/home/john/Ablation/aegis.py): unified launcher for orchestrator, worker, GUI, API, submit, train, list, and status.
- [launch.sh](/home/john/Ablation/launch.sh): convenience launcher for orchestrator, API, and optional GUI.
- [src/aegis_lab](/home/john/Ablation/src/aegis_lab): Python application package.
- [tests](/home/john/Ablation/tests): unit, integration, performance, promotion, and recovery tests.
- [docs](/home/john/Ablation/docs): architecture and schema documentation.
- [QIHSE](/home/john/Ablation/QIHSE): native/vector engine sources, docs, and artifacts.
- [NOT_STISLA](/home/john/Ablation/NOT_STISLA): companion C code and benchmarking assets.
- [models](/home/john/Ablation/models): local model artifacts if present.

Important Python package areas:

- [src/aegis_lab/app](/home/john/Ablation/src/aegis_lab/app)
- [src/aegis_lab/artifacts](/home/john/Ablation/src/aegis_lab/artifacts)
- [src/aegis_lab/atoms](/home/john/Ablation/src/aegis_lab/atoms)
- [src/aegis_lab/cli](/home/john/Ablation/src/aegis_lab/cli)
- [src/aegis_lab/editing](/home/john/Ablation/src/aegis_lab/editing)
- [src/aegis_lab/evaluation](/home/john/Ablation/src/aegis_lab/evaluation)
- [src/aegis_lab/gui](/home/john/Ablation/src/aegis_lab/gui)
- [src/aegis_lab/hardware](/home/john/Ablation/src/aegis_lab/hardware)
- [src/aegis_lab/orchestrator](/home/john/Ablation/src/aegis_lab/orchestrator)
- [src/aegis_lab/quantization](/home/john/Ablation/src/aegis_lab/quantization)
- [src/aegis_lab/sentinel](/home/john/Ablation/src/aegis_lab/sentinel)
- [src/aegis_lab/state](/home/john/Ablation/src/aegis_lab/state)
- [src/aegis_lab/workers](/home/john/Ablation/src/aegis_lab/workers)

## Requirements

Minimum baseline from the current project metadata:

- Python 3.9+
- `pyyaml`
- `pyzmq`

In practice, the repo also uses additional libraries that are not fully declared in `pyproject.toml`, including:

- `fastapi`
- `uvicorn`
- `pydantic`
- `psutil`
- `PyQt6`
- `pyqtgraph`

Optional hardware/runtime dependencies:

- Intel Level Zero runtime
- OpenVINO
- Intel GPU/NPU-capable Linux environment
- QIHSE shared library at `QIHSE/qihse/libqihse.so`

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

At minimum:

```bash
pip install -e .
```

For GUI and API development you will likely also need:

```bash
pip install fastapi uvicorn pydantic psutil PyQt6 pyqtgraph
```

### 3. Ensure the native QIHSE library exists

Several code paths assume this file is available:

```text
QIHSE/qihse/libqihse.so
```

If it is missing or incompatible, state initialization and orchestrator startup may fail.

### 4. Export `PYTHONPATH` if you are running files directly

```bash
export PYTHONPATH=$PWD/src
```

The launcher scripts already do this for normal runs.

## Running The System

### Unified launcher

The main entry point is [aegis.py](/home/john/Ablation/aegis.py).

Examples:

```bash
python3 aegis.py orchestrator
python3 aegis.py worker
python3 aegis.py gui
python3 aegis.py api
python3 aegis.py list
python3 aegis.py status --job-id job-12345678
```

### CLI script entry point

The package defines an `aegis` console script in `pyproject.toml`.

Examples:

```bash
aegis orchestrator
aegis worker
aegis submit --project demo
aegis train --project demo --model models/qwen2.5.gguf --target refusal
```

### Convenience launcher

You can also use:

```bash
bash launch.sh
```

That script attempts to:

1. Start the orchestrator in the background
2. Start the REST API
3. Prompt to launch the GUI

## CLI Commands

The current launcher supports these commands:

- `orchestrator`
- `worker`
- `gui`
- `api`
- `submit`
- `train`
- `mission`
- `list`
- `status`

### Submit a generic job

```bash
python3 aegis.py submit --project my-project --type ablation
```

### Submit a training pipeline

```bash
python3 aegis.py train \
  --project Qwen-Mission-Alpha \
  --model models/qwen2.5.gguf \
  --target refusal \
  --device auto
```

### List jobs

```bash
python3 aegis.py list
```

### Inspect a job

```bash
python3 aegis.py status --job-id job-xxxxxxxx
```

## REST API

Run the API:

```bash
python3 aegis.py api
```

Example requests:

### Submit a job

```bash
curl -X POST http://localhost:8000/jobs/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo-project",
    "job_type": "ablation",
    "parameters": {}
  }'
```

### List jobs

```bash
curl http://localhost:8000/jobs
```

### Get job status

```bash
curl http://localhost:8000/jobs/job-xxxxxxxx
```

Note: the API defaults to talking to `tcp://localhost:5555` unless `ORCHESTRATOR_URL` is set.

## GUI

Run the GUI with:

```bash
python3 aegis.py gui
```

The GUI currently provides:

- A top-level systems overview
- A dashboard for active jobs and stage progress
- Hardware telemetry charts
- Artifact browsing
- A leaderboard tab
- An interrogation panel for atom/context workflows
- A diff view for before/after comparison

Because the GUI talks to the orchestrator over IPC, it is most useful when the orchestrator is already running.

## Testing

The repository includes test suites under [tests](/home/john/Ablation/tests).

Examples:

```bash
python3 -m pytest tests/unit
python3 -m pytest tests/integration
python3 -m pytest tests/recovery
python3 -m pytest tests/performance
```

Not every test path is guaranteed to pass in a clean environment. Some suites assume the native library, hardware-specific behavior, or additional dependencies are available.

## Development Notes

### IPC ports

Current defaults:

- Orchestrator IPC: `5555`
- Log server: `5556`
- API: `8000` or a randomized port in `aegis.py api`

### Local state and artifacts

Important runtime locations:

- State root: `~/.aegis_lab/state`
- Artifact root: `~/.aegis_lab/artifacts`

### Hardware assumptions

The codebase is clearly optimized for Intel heterogeneous systems. On non-Intel or non-Linux environments, most CPU-only orchestration logic may still be useful, but telemetry and acceleration behavior will degrade or noop.

## Documentation

Additional documentation in this repository:

- [docs/architecture/architecture.md](/home/john/Ablation/docs/architecture/architecture.md)
- [docs/schemas/schemas.md](/home/john/Ablation/docs/schemas/schemas.md)
- [QIHSE/docs/README.md](/home/john/Ablation/QIHSE/docs/README.md)
- [QIHSE/qihse/docs/user/README.md](/home/john/Ablation/QIHSE/qihse/docs/user/README.md)
- [QIHSE/qihse/docs/development/README.md](/home/john/Ablation/QIHSE/qihse/docs/development/README.md)

## Limitations

The current codebase has some mismatches between intent and implementation. Before relying on it heavily, expect to inspect and potentially patch:

- Incomplete dependency declarations in `pyproject.toml`
- Partially stubbed evaluation and chat behavior
- Launch-script assumptions around ports and background processes
- Hardware-specific paths that fail softly or require Intel runtimes
- Areas where tests and implementation have drifted

## License

No license file is currently present in the repository root. Until one is added, treat reuse and redistribution status as undefined.
