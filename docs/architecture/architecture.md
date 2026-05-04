# AEGIS-LAB Architecture

## Overview
AEGIS-LAB is a crash-tolerant, hardware-aware heterogeneous model editing platform optimized for Intel Meteor Lake-P.

## Core Components

### 1. Orchestrator
- **Service**: Manages job lifecycles and stage transitions.
- **Promotion Controller**: Handles atomic promotion of quantized INT8 bundles. Uses `os.replace` for atomicity and `shutil.move` as a cross-device fallback.
- **Thermal Guardian**: Monitors system temperatures and provides safety triggers. Implements a 1-second cache to minimize scheduler jitter.

### 2. State Management (QIHSE DB)
- **StateDatabase**: Fully integrated with the QIHSE vector engine for metadata persistence.
- **Upsert Logic**: Uses versioned metadata entries where the latest timestamp wins during deduplication.
- **Hardware Acceleration**: Leverages Intel SHA instructions via `qihse_intel_hw_hash` for fast integrity verification.

### 3. Hardware Optimization
- **Zero-Copy UMA**: Uses `QIHSE_UMA_MIGRATE_EXPLICIT` to ensure truly zero-copy operations without implicit background migrations.
- **Lying E820 Protection**: Automatically detects NPU BAR conflicts (0x5010000000) and enables `QIHSE_E820_SAFE_MODE` to avoid BIOS misreporting issues.
- **SIMD Hashing**: Uses 1MB chunk sizes in `hash_file` to maximize AVX-512/AMX throughput.

### 4. Quantization & Verification
- **QuantizationValidator**: Performs semantic drift analysis using Cosine Similarity and KL Divergence.
- **Atomic Promotion**: Follows a strict 6-step algorithm (Temp stage -> Hash -> Manifest -> fsync -> Rename -> DB mark) to ensure filesystem integrity.
