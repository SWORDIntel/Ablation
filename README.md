# AEGIS-LAB

[![GitHub](https://img.shields.io/badge/GitHub-SWORDIntel%2FAblation-blue)](https://github.com/SWORDIntel/Ablation)

AEGIS-LAB is a hardware-aware model editing and ablation framework built around a staged orchestrator, a QIHSE-backed state layer, worker-based execution, and optional operator-facing API and GUI surfaces. It is the standalone upstream for the `aegis_lab` module embedded in [FRAMEWERX](https://authentik.192-168-1-240.sslip.io/gitlab/SWORD/framewerx).

---

## Table of Contents

- [Status](#status)
- [Major Features](#major-features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Startup](#startup)
- [Model Editing Workflow](#model-editing-workflow)
- [GGUF-Native Refusal Ablation](#gguf-native-refusal-ablation)
- [Heretic Refusal-Driven Ablation](#heretic-refusal-driven-ablation)
- [Intervention Registry](#intervention-registry)
- [Evaluation & Scoring](#evaluation--scoring)
- [RAG Engine](#rag-engine)
- [Hardware Support](#hardware-support)
- [Sentinel Cascade](#sentinel-cascade)
- [Scheduler & Sharding](#scheduler--sharding)
- [Artifact Store](#artifact-store)
- [Atom Registry](#atom-registry)
- [Runtime Model](#runtime-model)
- [Configuration](#configuration)
- [Testing](#testing)
- [Current Limitations](#current-limitations)
- [Documentation](#documentation)

---

## Status

Status as of 2026-07-09:

- **GGUF-Native Ablation:** Direct GGUF tensor weight modification via `GGUFRefusalAblator` — no PyTorch required. Supports `zero`, `prune`, and `clamp` methods on quantized models.
- **Heretic Refusal Pipeline:** Refusal-driven ablation with dataset agents, scoring agents, search agents, and model evaluation. Policy-document-aware with report-only and apply modes.
- **Intervention Registry:** Six registered interventions — `sae_clamp`, `moe_ablate`, `inference_steer`, `causal_edit`, `repe_steer`, `contrastive_steering` — with a categorized multi-select CLI/GUI for chaining.
- **100% Unit Test Success:** All unit tests pass in the default environment, including orchestrator and hardware discovery contracts.
- **Hardware-Aware Quantization:** Orchestrator passes hardware capabilities (NPU/GPU/CPU features) to workers for device-specific quantization (e.g., INT8 on NPU with 128MB cache awareness).
- **Atomic Promotion Logic:** SHA256-hashed artifacts moved to exports with collision-resistant naming and integrity manifests.
- **NPU Sentinel Cascade:** Stage 0 (statistical) and Stage 1 (semantic) guards for MTL-P NPU with ISH VSEC unlock sequence.
- **CUDA Bridge:** ZLUDA-based CUDA binary execution on Intel GPUs.
- **RAG Engine:** QIHSE-accelerated retrieval-augmented generation for operator chat with semantic search of behavioral atoms.
- **Scheduler & Sharding:** Job scheduling engine with hardware-aware sharding across CPU/NPU/VPU workers.
- **Artifact Store:** Hashed artifact storage with layout management and integrity manifests.
- **Atom Registry:** Immutable storage and indexing of behavioral atoms via QIHSE backend.

Verified commands:

```bash
bash ./ci_smoke.sh
PYTHONPATH=src python3 -m unittest discover -s tests/unit
PYTHONPATH=src python3 -m unittest discover -s tests/integration
PYTHONPATH=src python3 -m unittest discover -s tests/performance
PYTHONPATH=src python3 -m unittest tests.unit.test_hardware_discovery_contract -v
./scripts/vpu_probe.sh
```

---

## Major Features

| Feature | Module | Description |
|---------|--------|-------------|
| **GGUF Refusal Ablation** | `editing/gguf_ablation.py` | Direct tensor weight modification in GGUF files (no PyTorch) |
| **Heretic Refusal** | `editing/heretic_refusal/` | Refusal-driven ablation with dataset, scoring, and search agents |
| **Intervention Registry** | `editing/__init__.py` | 6 registered interventions with chaining support |
| **Ablation Pipeline** | `editing/pipeline.py` | Full pipeline: intake, probing, extraction, validation |
| **Model Refusal Ablation** | `editing/model_refusal_ablation.py` | PyTorch-based refusal neuron identification and ablation |
| **MoE Ablation** | `editing/moe_ablation.py` | Mixture-of-Experts layer ablation |
| **SAE Clamping** | `editing/sae_clamping.py` | Sparse Autoencoder feature clamping |
| **SAE Cross-Modal** | `editing/sae_crossmodal.py` | Cross-modal SAE interventions |
| **Causal Editor** | `editing/causal_editor.py` | Causal tracing and edit injection |
| **RepE Steering** | `editing/repe_steering.py` | Representation Engineering steering vectors |
| **Inference Steering** | `editing/inference_steering.py` | Runtime inference-time steering |
| **Adversarial Hardening** | `editing/adversarial.py` | Adversarial robustness hardening |
| **GCG Refiner** | `editing/gcg_refiner.py` | Greedy Coordinate Gradient refinement |
| **Speculative Refiner** | `editing/speculative_refiner.py` | Speculative edit refinement |
| **Delta Builder** | `editing/delta_builder.py` | Incremental edit delta construction |
| **Rollback** | `editing/rollback.py` | Edit rollback and version management |
| **RAG Engine** | `api/rag_engine.py` | QIHSE-accelerated retrieval-augmented generation |
| **NPU Sentinel** | `sentinel/cascade.py` | Stage 0/1 cascade guards for MTL-P NPU |
| **CUDA Bridge** | `hardware/cuda_bridge.py` | ZLUDA-based CUDA on Intel GPU |
| **Hardware Vault** | `hardware/vault.py` | Secure hardware credential/secrets management |
| **Scheduler** | `scheduler/engine.py` | Job scheduling with sharding support |
| **Artifact Store** | `artifacts/store.py` | Hashed artifact storage with manifests |
| **Atom Registry** | `atoms/registry.py` | Behavioral atom indexing via QIHSE |

---

## Repository Layout

```
Ablation/
├── src/aegis_lab/             # Main Python package
│   ├── __init__.py
│   ├── __main__.py            # python -m aegis_lab entry point
│   ├── _runner.py             # Unified launcher (bridges to framewerx namespace)
│   ├── api/                   # REST API server + RAG engine
│   ├── app/                   # Bootstrap and application setup
│   ├── artifacts/             # Artifact storage, hashing, layout, manifests
│   ├── atoms/                 # Behavioral atom extraction and registry
│   ├── cli/                   # CLI entry points, interactive mode, validation
│   ├── configs/               # Hardware configs, heretic refusal YAML, batch configs
│   ├── data/                  # Ablation smoke datasets, prompt pairs
│   ├── editing/               # Ablation pipeline, interventions, heretic refusal
│   ├── evaluation/            # Capability drift, consensus, ELO, hallucination, leaderboard
│   ├── gui/                   # PyQt6 GUI + TUI app + widgets
│   ├── hardware/              # Discovery, model selector, telemetry, thermal, vault, CUDA bridge
│   ├── intake/                # Model fingerprinting, HF browser, synthetic generator
│   ├── native/vpu_core/       # Rust native VPU extension
│   ├── orchestrator/          # Job lifecycle, IPC, optimizer, promotion
│   ├── probing/               | Capture and probing logic
│   ├── quantization/          # Calibration, exporter, validators
│   ├── scheduler/             # Job scheduling engine and sharding
│   ├── sentinel/              # NPU sentinel cascade and mission guard
│   ├── state/                 # QIHSE wrapper, in-memory fallback, DB
│   ├── tests/                 # Test suite (unit, integration, performance, recovery)
│   ├── utils/                 # Progress utilities
│   ├── verification/          # Ablation tax, semantic authority, proofs
│   └── workers/               # CPU, NPU, VPU, P2P worker logic
├── tests/                     # Top-level test suite
├── configs/                   # Runtime configuration sets
├── data/                      # Sample datasets and prompt pairs
├── docs/                      # Architecture docs, guides, schemas
├── hardware_docs/             # Hardware documentation
├── scripts/                   # VPU probe, env setup, native build
├── aegis.py                   # Root CLI entry point
├── ablate_model_refusal.sh    # Refusal ablation TUI/CLI launcher
├── ablate_model_refusal_tui.py # TUI implementation
├── run_ablation_demo.py       # Demo runner
├── run_batch_ablation.py      # Batch ablation runner
├── run_gguf_batch_ablation.py # GGUF batch ablation runner
├── bootstrap.sh               # Environment bootstrap
├── launch.sh                  # Orchestrator + API launcher
├── run_mission.sh             # Canned mission wrapper
├── ci_smoke.sh                # CI smoke test
├── pyproject.toml             # Package config with optional extras
└── AGENTS.md                  # Contributor guidelines
```

---

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
python3 -m pip install -e ".[api]"       # FastAPI REST backend
python3 -m pip install -e ".[gui]"       # PyQt6 desktop GUI
python3 -m pip install -e ".[hardware]"  # OpenVINO runtime + numpy
```

---

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

When running inside FRAMEWERX, use the unified launcher instead:

```bash
python3 fw_launcher.py aegis-orchestrator
python3 fw_launcher.py aegis-api
python3 fw_launcher.py aegis-worker
python3 fw_launcher.py aegis-gui
python3 fw_launcher.py aegis-tui
python3 fw_launcher.py aegis-mission --model models/qwen2.5.gguf --target refusal
python3 fw_launcher.py aegis-benchmark
```

---

## Model Neurosurgery

AEGIS-LAB now includes a structural **model neurosurgery** workflow for reducing checkpoint size and inference memory traffic rather than only changing behavior in-place.

Implemented through Stage 4:

- KEEP/DROP residual profiling with contrastive directional edits and SVD preservation bases;
- reversible whole-layer deletion search and greedy interacting deletion;
- gated-MLP channel profiling, masked search, and physical gate/up/down tensor slicing;
- GQA/MHA group profiling and physical Q/K/V/O slicing while preserving GQA grouping;
- MoE router profiling and physical router/expert slicing for supported HF-style expert blocks;
- Stage-4 joint constrained search across layers, MLP width, attention groups, and MoE experts;
- Pareto reporting over resident bytes removed, estimated dense MACs removed, and KL;
- automatic checksum-bound `optimized_plan.yaml` materialization;
- checksum-bound surgery plans and post-surgery KL/top-1 validation.

Install the project normally, then use the dedicated entry point:

```bash
aegis-neurosurgery --help
```

The detailed workflow, invariants, architecture constraints, validation guidance, and roadmap are in [docs/neurosurgery/README.md](docs/neurosurgery/README.md).

---

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

### Model-Agnostic Refusal Ablation

Remove safety/refusal mechanisms from a target model (model-agnostic):

TUI is the preferred mode and default when no CLI args are provided:

```bash
./ablate_model_refusal.sh                # interactive TUI (default)
./ablate_model_refusal.sh --tui           # explicit TUI mode
```

For scripted/non-interactive use, pass full positional args:

```bash
# Auto-detect and ablate (heuristic default mode)
bash ablate_model_refusal.sh models/input_model.gguf models/ablated_model.gguf zero ablation

# Heretic example with extra flags
bash ablate_model_refusal.sh \
  models/input_model.gguf \
  models/ablated_model.gguf \
  prune \
  heretic \
  config/heretic_refusal.yaml \
  --policy-document path/or/url/to/policy.md \
  --policy-document-label unsafe \
  --apply-heretic-edits
```

Single root entrypoint (interactive + non-interactive) cheat sheet:

```bash
./ablate_model_refusal.sh --help
```

The default strategy is `ablation`, which uses the current heuristic implementation.
Heretic mode is report-only unless `--apply-heretic-edits` is provided.

Ablation methods:
- `zero`: Zero out refusal neuron weights (recommended)
- `prune`: Remove weak connections below threshold
- `clamp`: Limit activation ranges to reduce refusal strength

TUI run output includes live progress and log lines. For shell/CLI runs, keep an eye on command output in your terminal.

---

## GGUF-Native Refusal Ablation

Direct GGUF tensor weight modification — no PyTorch required. Works on quantized GGUF models by reading tensor metadata, identifying refusal-related layers (FFN gate/up/down in target blocks), modifying specific neuron rows, and writing the modified GGUF.

### Usage

```bash
# Single model
python3 run_gguf_batch_ablation.py \
  --input models/input_model.gguf \
  --output models/ablated_model.gguf \
  --method zero \
  --layers 10,11,12,13,14

# Batch processing
python3 run_gguf_batch_ablation.py \
  --input-dir models/ \
  --output-dir models/ablated/ \
  --method zero
```

### Supported Methods
- `zero` — Set target neuron rows to zero
- `clamp` — Clamp tensor values to [-threshold, threshold]
- `prune` — Zero values below threshold magnitude

### Module
- `src/aegis_lab/editing/gguf_ablation.py` — `GGUFRefusalAblator` class
- `run_gguf_batch_ablation.py` — Batch runner with CLI args

---

## Heretic Refusal-Driven Ablation

A refusal-driven ablation pipeline that uses dataset agents, scoring agents, and search agents to identify and remove refusal mechanisms.

### Components
- **Config** (`heretic_refusal/config.py`) — YAML-based configuration
- **Dataset** (`heretic_refusal/dataset.py`) — Dataset loading and management
- **Dataset Agent** (`heretic_refusal/dataset_agent.py`) — Automated dataset curation
- **Search Agent** (`heretic_refusal/search_agent.py`) — Refusal neuron search
- **Scoring Agent** (`heretic_refusal/scoring_agent.py`) — Ablation impact scoring
- **Model Eval** (`heretic_refusal/model_eval.py`) — Pre/post ablation evaluation
- **Runner** (`heretic_refusal/runner.py`) — Orchestrates the full heretic pipeline
- **Interventions** (`heretic_refusal/interventions.py`) — Heretic-specific intervention registration

### Configuration
- `configs/heretic_refusal.yaml` — Heretic pipeline configuration
- Policy documents can be local files or URLs
- `--policy-document-label` controls safety classification (e.g., `unsafe`, `restricted`)

---

## Intervention Registry

Six registered interventions that can be chained for compound ablation effects:

| Intervention | Module | Description |
|-------------|--------|-------------|
| `sae_clamp` | `editing/sae_clamping.py` | Sparse Autoencoder feature clamping |
| `moe_ablate` | `editing/moe_ablation.py` | Mixture-of-Experts layer ablation |
| `inference_steer` | `editing/inference_steering.py` | Runtime inference-time steering |
| `causal_edit` | `editing/causal_editor.py` | Causal tracing and edit injection |
| `repe_steer` | `editing/repe_steering.py` | Representation Engineering steering |
| `contrastive_steering` | `editing/advanced.py` | Contrastive steering vectors |

The interactive intervention UI provides a categorized, multi-select CLI/GUI interface for chaining techniques with real-time impact validation.

---

## Evaluation & Scoring

Post-ablation evaluation and scoring modules:

| Module | Description |
|--------|-------------|
| `evaluation/capability_drift.py` | Measures capability drift after ablation |
| `evaluation/consensus.py` | Multi-role consensus scoring |
| `evaluation/elo_judge.py` | ELO-based model quality rating |
| `evaluation/hallucination.py` | Hallucination rate measurement |
| `evaluation/hardware_efficiency.py` | Hardware efficiency scoring (VPU/NPU/GPU) |
| `evaluation/leaderboard.py` | Ablation result leaderboard |
| `evaluation/telemetry_wrapper.py` | Telemetry wrapper for evaluation runs |

---

## RAG Engine

QIHSE-accelerated retrieval-augmented generation for operator chat. Integrates semantic search of behavioral atoms with prompt augmentation.

- `api/rag_engine.py` — `RAGEngine` class
- Uses `AegisState` for QIHSE-backed vector search
- Deterministic pseudo-random embeddings (production would use real embedding model)
- Query endpoint returns context-augmented prompts

---

## Hardware Support

### Intel VPU/NPU
- OpenVINO 2022.3 archive runtime for Myriad-class VPUs
- VPU probe: `./scripts/vpu_probe.sh`
- VPU environment: `./scripts/vpu_env.sh --shell`
- Hardware configs: `configs/hardware/vpu_stick_3.json`, `vpu_stick_17.json`
- Hardware discovery reports `vpu_usb_present` and `vpu_runtime_usable`

### CUDA via ZLUDA
- `hardware/cuda_bridge.py` — `CudaBridge` class wraps CUDA binaries with ZLUDA translation layer
- Enables CUDA-based inference on Intel GPUs
- Configurable via `ZLUDA_BIN` environment variable

### Hardware Vault
- `hardware/vault.py` — Secure credential and secrets management for hardware access
- Model selector with hardware-aware model routing (`hardware/model_selector.py`)
- Sideband communication channel (`hardware/sideband.py`)
- Thermal monitoring and safety (`hardware/thermal.py`)
- Telemetry reporting (`hardware/telemetry.py`)

---

## Sentinel Cascade

NPU Sentinel cascade for MTL-P (Meteor Lake-P) with Stage 0 and Stage 1 guards:

- `sentinel/cascade.py` — `NPUSentinel` class
- Stage 0: Statistical guard (fast, lightweight)
- Stage 1: Semantic guard (deeper analysis)
- ISH VSEC unlock sequence for MTL-P NPU access
- Optimized for `intel_ai_boost` with 128MB cache
- `sentinel/sentinel_mission.py` — Mission-level sentinel integration

---

## Scheduler & Sharding

- `scheduler/engine.py` — Job scheduling engine with priority and dependency management
- `scheduler/sharding.py` — Hardware-aware sharding across CPU/NPU/VPU workers
- Integrates with orchestrator for distributed job execution

---

## Artifact Store

- `artifacts/store.py` — `ArtifactStore` for hashed artifact storage
- `artifacts/hashing.py` — SHA256 artifact hashing
- `artifacts/layout.py` — Directory layout management
- `artifacts/manifests.py` — Integrity manifests with collision-resistant naming
- Atomic promotion: final artifacts hashed and moved to exports directory

---

## Atom Registry

- `atoms/registry.py` — `AtomRegistry` for immutable behavioral atom storage
- `atoms/extractor.py` — Behavioral atom extraction from models
- Integrated with QIHSE backend via `AegisState`
- Supports vector search and semantic retrieval

---

## Runtime Model

- Native QIHSE is used when `QIHSE/qihse/libqihse.so` is present.
- If native QIHSE is missing, the state layer falls back to an in-memory implementation unless `AEGIS_REQUIRE_NATIVE_QIHSE=1` is set.
- Level Zero telemetry is disabled by default in generic environments and can be enabled with `AEGIS_ENABLE_LEVEL_ZERO=1`.
- IPC authentication is controlled by `AEGIS_AUTH_TOKEN`.
- VPU discovery reports both `vpu_usb_present` and `vpu_runtime_usable`.
- VPU and NPU workers run in simulation mode when OpenVINO devices are not available.
- For Myriad-class VPUs, the generic PyPI OpenVINO path may still stay in simulation; use the local archive-backed launcher in `scripts/vpu_env.sh` when `MYRIAD` runtime support is required.

---

## Configuration

### Config Files
- `configs/heretic_refusal.yaml` — Heretic refusal pipeline configuration
- `configs/hardware/vpu_stick_3.json` — VPU stick 3 hardware profile
- `configs/hardware/vpu_stick_17.json` — VPU stick 17 hardware profile
- `configs/batch/` — Batch ablation configurations
- `configs/quantization/` — Quantization parameters
- `configs/runtime_profiles/` — Runtime profile presets
- `configs/scheduler/` — Scheduler configurations
- `configs/verification/` — Verification thresholds

### Datasets
- `data/ablation-smoke/positive.txt`, `negative.txt` — Smoke test prompt pairs
- `data/qwen_positive.jsonl`, `qwen_negative.jsonl` — Qwen prompt pairs
- `data/refusal_prompts.jsonl` — Refusal prompt dataset

### Environment Variables
- `AEGIS_AUTH_TOKEN` — IPC authentication token
- `AEGIS_ENABLE_LEVEL_ZERO` — Enable Level Zero telemetry (`1` to enable)
- `AEGIS_REQUIRE_NATIVE_QIHSE` — Require native QIHSE library (`1` to require)
- `ZLUDA_BIN` — Path to ZLUDA binary for CUDA bridge

---

## Testing

```bash
# All unit tests
PYTHONPATH=src python3 -m unittest discover -s tests/unit

# Integration tests
PYTHONPATH=src python3 -m unittest discover -s tests/integration

# Performance tests
PYTHONPATH=src python3 -m unittest discover -s tests/performance

# Recovery tests
PYTHONPATH=src python3 -m unittest discover -s tests/recovery

# Specific test
PYTHONPATH=src python3 -m unittest tests.unit.test_hardware_discovery_contract -v

# CI smoke test
bash ci_smoke.sh

# With pytest
PYTHONPATH=src python3 -m pytest tests/ -v
```

### Test Coverage by Scope
- `tests/unit/` — Orchestrator, hardware discovery, state, scheduler, sentinel, thermal, VPU robustness, API server contract, quantization, promotion, registry
- `tests/integration/` — Workflow integration, auth, P2P fabric, worker auth
- `tests/performance/` — Latency, VPU optimization verification
- `tests/recovery/` — Chaos recovery
- `tests/test_model_refusal_ablation.py` — Refusal ablation regression
- `tests/test_refusal_heretic_modules.py` — Heretic pipeline regression
- `tests/test_refusal_interventions.py` — Intervention registry regression
- `tests/test_refusal_model_eval_metrics.py` — Evaluation metrics regression

---

## Current Limitations

- The GUI and hardware paths still assume optional desktop and Intel runtime dependencies.
- The editing and validation stack reports explicit fallback mode, but it continues to evolve toward a fully production-grade semantic evaluation pipeline.
- High-fidelity QIHSE editing requires a compatible hardware backend for maximum performance.
- RAG engine uses deterministic pseudo-random embeddings; production use requires a real embedding model.
- CUDA bridge requires ZLUDA installation for Intel GPU CUDA compatibility.

---

## Documentation

- [docs/architecture/architecture.md](docs/architecture/architecture.md) — System architecture
- [docs/VPU_INTEGRATION.md](docs/VPU_INTEGRATION.md) — VPU integration guide
- [docs/MODEL_REFUSAL_ABLATION_GUIDE.md](docs/MODEL_REFUSAL_ABLATION_GUIDE.md) — Refusal ablation guide
- [docs/KHOJ_ABLATION_STUDY.md](docs/KHOJ_ABLATION_STUDY.md) — Khoj ablation study
- [docs/schemas/schemas.md](docs/schemas/schemas.md) — Data schemas
- [AGENTS.md](AGENTS.md) — Contributor guidelines
- [Aegis-lab Implementation Specification.pdf](Aegis-lab%20Implementation%20Specification.pdf) — Full implementation spec

---

## FRAMEWERX Integration

AEGIS-LAB is synced from FRAMEWERX's `src/framewerx/aegis_lab/` module. The sync maps:

| FRAMEWERX | Ablation |
|-----------|----------|
| `src/framewerx/aegis_lab/` | `src/aegis_lab/` |
| `src/framewerx/aegis_lab/tests/` | `tests/` |
| `src/framewerx/aegis_lab/configs/` | `configs/` |
| `src/framewerx/aegis_lab/data/` | `data/` |
| `src/framewerx/aegis_lab/native/vpu_core/` | `src/native/vpu_core/` |

FRAMEWERX REST API integration (port 7331):
- `POST /ablation/run` — Run AblationPipeline
- `POST /ablation/gguf` — Run GGUFRefusalAblator
- `GET /ablation/interventions` — List registered interventions
- `GET /ablation/status/{job_id}` — Job status
- `GET /benchmark/results`, `POST /benchmark/run`, `DELETE /benchmark/clear`
