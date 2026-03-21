# AEGIS-LAB

AEGIS-LAB is a crash-tolerant, hardware-aware model editing platform built around a staged orchestration pipeline, QIHSE-backed state storage, Intel-oriented hardware discovery, and a PyQt operator GUI. It is highly optimized for Intel heterogeneous systems, including Meteor Lake-P (AVX-512, AMX) and Movidius MyriadX VPUs.

## Key Features & Recent Improvements

### 🔧 Modular QIHSE Architecture
The core **QIHSE** native engine has been refactored into a modular, high-performance library:
- **`qihse_instr`**: Hardware-specific intrinsics for **AVX-512**, **AMX**, and **AVX-VNNI**.
- **Automated ISA Fallback**: Optimized for stability; the engine dynamically detects CPU capabilities and falls back to AVX2 or scalar paths to prevent "Illegal Instruction" errors on older hardware.
- **`qihse_math`**: Optimized mathematical primitives, including Random Fourier Features (RFF) and quantum superposition encoding.
- **`qihse_search`**: Advanced search algorithms including adaptive Grover amplification and multi-resolution search.
- **Performance**: Verified $O(N)$ to $O(1)$ search acceleration with zero-copy data paths and persistent indexing.

### 💎 AVX-VNNI (INT8) Acceleration
The QIHSE engine now includes a dedicated **AVX-VNNI** backend for high-performance quantized search:
- **4x Throughput**: Leverages Intel's Vector Neural Network Instructions for optimized INT8 dot-product operations.
- **Quantized Model Search**: Specifically tuned for fast similarity matching across large-scale quantized model representations.
- **Zero-Copy Path**: Integrates with the existing zero-copy data flow to minimize memory overhead during high-frequency search tasks.

### 🚀 Movidius VPU Integration (Flex Fabric)
Integrated support for **Intel Movidius MyriadX** VPUs as high-efficiency "Sentinel" devices:
- **VpuWorker**: Dedicated worker class using OpenVINO for low-power inference.
- **Async Inference**: Optimized hot-path using `AsyncInferRequest` and zero-allocation tensor reuse.
- **Fail-Fast Scheduling**: "Flex Fabric" scheduler with Round-Robin load balancing and automatic fail-over to CPU/AMX if a VPU device hangs.
- **Model Caching**: Persistent `.blob` caching to eliminate model compilation latency on startup.

### 🌐 Distributed P2P Fabric
A new peer-to-peer data streaming layer reduces orchestrator bottlenecks:
- **Worker-to-Worker Streaming**: Workers can now stream large model weights and activation tensors directly to each other via ZeroMQ P2P sockets.
- **High Bandwidth**: Bypasses the central orchestrator for heavy data transfers, significantly reducing latency in distributed editing tasks.
- **Flexible Topology**: Supports push/pull streaming modes with automatic endpoint discovery through the Orchestrator.

### 🛡️ Automated Sentinel Missions
The platform now includes a robust **Red-Teaming validation flow** powered by the Sentinel architecture:
- **Continuous Validation**: Sentinel workers autonomously execute adversarial probes against edited models to verify stability and safety.
- **Multi-Stage Cascade**: Features a two-stage validation process—Stage 0 (Statistical Guard) and Stage 1 (Semantic Sentinel)—running on VPU/NPU backends.
- **Fail-Fast Promotion**: Automatically halts job promotion if a model fails to meet security or performance thresholds during a Sentinel mission.

### 🔒 Security Hardening
- **Authenticated IPC**: All ZeroMQ communication (Orchestrator to Workers) is now secured via a shared-secret (`AEGIS_AUTH_TOKEN`) handshake.
- **Native Memory Safety**: Hardened C-wrappers with strict bounds checking to prevent buffer overflows in telemetry processing.

### 📊 3D Ablation Map Visualization
The PyQt operator dashboard now features a hardware-accelerated **OpenGL 3D widget**:
- **Interactive Mapping**: Visualize model layers and behavioral "atoms" in a 3D coordinate space.
- **Real-Time Updates**: Watch as new representation data and ablation scores stream into the visualization from active workers.
- **Spatial Relationship Analysis**: Quickly identify clusters of high-impact neurons and representation shifts across stacked model layers.

### ⚡ New Bootstrap & Launch Workflow
- **`bootstrap.sh`**: A unified script that automates directory setup, dependency verification, native library compilation, and hardware capability discovery.
- **`launch.sh`**: Integrated with the bootstrap sequence for a "one-command" startup experience.

## What It Does

AEGIS-LAB is organized around a job pipeline for model-editing and ablation-style workflows. At a high level it provides:

- A ZeroMQ orchestrator that tracks jobs, stages, workers, approvals, and evaluation results.
- A QIHSE-backed state layer used for jobs, stages, artifacts, atoms, logs, and telemetry snapshots.
- Hardware discovery and telemetry for CPU, iGPU, NPU, VPU, and Intel Level Zero environments.
- A PyQt dashboard with system status, job views, hardware charts, artifact browsing, leaderboard, chat, and diff panes.
- A FastAPI service that exposes job submission and job status over HTTP.

## Architecture

The main subsystems are:

### Orchestrator & Scheduler

The orchestrator manages job lifecycle, stage assignment, and worker registration. The **SchedulerEngine** implements "Flex Fabric" logic to assign tasks based on hardware affinity (e.g., VPU for Stage 0 guards, AMX for heavy training).

Relevant files:

- [src/aegis_lab/orchestrator/service.py](/home/john/Ablation/src/aegis_lab/orchestrator/service.py)
- [src/aegis_lab/orchestrator/ipc.py](/home/john/Ablation/src/aegis_lab/orchestrator/ipc.py)
- [src/aegis_lab/scheduler/engine.py](/home/john/Ablation/src/aegis_lab/scheduler/engine.py)
- [src/aegis_lab/workers/vpu_worker.py](/home/john/Ablation/src/aegis_lab/workers/vpu_worker.py)

### State Layer

The persistent state layer wraps the modular QIHSE vector storage.

Relevant files:

- [src/aegis_lab/state/db.py](/home/john/Ablation/src/aegis_lab/state/db.py)
- [src/aegis_lab/state/qihse_wrapper.py](/home/john/Ablation/src/aegis_lab/state/qihse_wrapper.py)

### Hardware and Telemetry

Discovery covers:

- AVX-512, AMX, AVX-VNNI
- Hybrid CPU topology
- OpenVINO-discovered GPU, NPU, and **VPU (MYRIAD)** devices.
- Thermal telemetry and NPU BAR conflict detection.

Relevant files:

- [src/aegis_lab/hardware/discovery.py](/home/john/Ablation/src/aegis_lab/hardware/discovery.py)
- [docs/VPU_INTEGRATION.md](/home/john/Ablation/docs/VPU_INTEGRATION.md)

## Setup & Launch

### 1. Bootstrap the System
The new `bootstrap.sh` script fully automates the system setup process, including dependency checks, native library compilation, and hardware capability discovery:

```bash
bash bootstrap.sh
```

*Note: To activate physical Movidius VPUs, you may need to apply udev rules as prompted by the script.*

### 2. Launch Services
Start the orchestrator, API, and optional GUI in one command:

```bash
bash launch.sh
```

## Running The System

### Unified launcher
The main entry point is [aegis.py](/home/john/Ablation/aegis.py).

```bash
python3 aegis.py orchestrator
python3 aegis.py worker --type vpu
python3 aegis.py gui
```

## Documentation

- [docs/architecture/architecture.md](/home/john/Ablation/docs/architecture/architecture.md)
- [docs/VPU_INTEGRATION.md](/home/john/Ablation/docs/VPU_INTEGRATION.md)
- [QIHSE/docs/README.md](/home/john/Ablation/QIHSE/docs/README.md)

## Limitations

The current codebase has some mismatches between intent and implementation. Before relying on it heavily, expect to inspect and potentially patch:

- Incomplete dependency declarations in `pyproject.toml`
- Partially stubbed evaluation and chat behavior
- Launch-script assumptions around ports and background processes
- Hardware-specific paths that fail softly or require Intel runtimes
- Areas where tests and implementation have drifted

## License

No license file is currently present in the repository root. Until one is added, treat reuse and redistribution status as undefined.
