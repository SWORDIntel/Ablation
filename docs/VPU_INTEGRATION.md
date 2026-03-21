# Movidius VPU Integration in Aegis-Lab

This document outlines the integration of Intel Movidius Visual Processing Units (VPUs) into the Aegis-Lab "Flex Fabric" scheduler and how the modular QIHSE core leverages this hardware for optimized model editing and verification.

## 1. VPU in the "Flex Fabric" Hierarchy

The "Flex Fabric" scheduler represents Aegis-Lab's heterogeneous execution strategy, which dynamically maps computational stages to the most appropriate hardware (CPU, NPU, iGPU, or VPU).

### Role: The Low-Power Sentinel
In the Aegis-Lab architecture, the Movidius VPU (detected as `MYRIAD` via OpenVINO) is designated as the primary **Sentinel** device. Its specialized architecture is ideal for continuous, low-power monitoring and preliminary validation tasks.

### Placement Logic
The `SchedulerEngine` implements specific rules for VPU utilization:

1.  **Priority Sentinel Execution**: The VPU is the preferred device for `sentinel_stage0_guard`. This stage runs continuously to detect trigger conditions for model editing. Using the VPU ensures minimal impact on the system's primary power budget and thermal envelope.
2.  **Thermal Fail-Safe**: When the system detects high thermal pressure (`throttling_recommended: True`), the scheduler automatically shifts operations away from power-hungry CPU (AMX/AVX-512) and iGPU units toward the more efficient NPU and VPU "Flex Fabric."
3.  **Hardware Fallback**: On platforms lacking modern CPU instructions (like AMX or AVX-512 VNNI), the VPU serves as a critical compute offload target to maintain performance levels that would otherwise degrade on a legacy CPU path.

## 2. QIHSE Core: Hardware-Specific Optimization

The **Quantum-Inspired Hilbert Space Expansion (QIHSE)** core is designed with a modular backend architecture that enables surgical optimization for Movidius VPUs.

### Modular Backend Architecture
QIHSE abstracts hardware complexity through its `backends/` layer. The VPU integration benefits from:

*   **Unified Device Abstraction**: QIHSE provides a consistent C API that allows the `VpuWorker` to execute complex vector searches and Hilbert space expansions without managing low-level MYRIAD buffers directly.
*   **Asynchronous Orchestration**: The QIHSE parallel orchestrator can dispatch sub-tasks to the VPU while simultaneously utilizing the NPU for different stages of the same search operation, achieving true heterogeneous concurrency.
*   **Precision-Aware Quantization**: The QIHSE quantization pipeline is tuned for the VPU's optimal precision formats (typically FP16/INT8), ensuring that model weights are transformed specifically for the SHAVE cores within the Movidius architecture.

### Performance Gains
By offloading "Sentinel" and "Capture" stages to the VPU, Aegis-Lab achieves:
*   **2-3x Power Efficiency** compared to CPU-only monitoring.
*   **Reduced CPU Jitter**: By keeping the P-cores and E-cores free from background monitoring tasks, critical "Mandatory Authority" stages (like `edit_generate`) can execute with maximum deterministic performance.

## 3. Implementation: VpuWorker

The `VpuWorker` (located in `src/aegis_lab/workers/vpu_worker.py`) implements the interface between the Aegis-Lab orchestrator and the OpenVINO runtime for MYRIAD devices.

### Key Features
*   **Automatic Discovery**: Uses `HardwareDiscovery` to detect `MYRIAD` devices via OpenVINO's `Core.available_devices`.
*   **Simulation Mode**: In environments where physical VPU hardware is restricted or absent, the worker automatically enters a high-fidelity simulation mode, allowing for architectural validation and CI/CD testing without specialized hardware.
*   **Integrated Telemetry**: Each VPU execution stage reports detailed telemetry, including SHAVE core utilization and peak memory usage, allowing the `SchedulerEngine` to refine future placement decisions.
