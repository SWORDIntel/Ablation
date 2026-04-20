# MULTI_MODEL_ABLATION_PIPELINE.md

# AEGIS-LAB // Multi-Model Ablation Pipeline

## Purpose

This document defines the authoritative end-to-end pipeline for AEGIS-LAB. It turns the repository from a collection of strong subsystems into a single coherent platform for **hardware-adaptive, multi-model ablation** across dense models, mixture-of-experts systems, and multimodal architectures.

The pipeline is designed to:

- ingest unknown or semi-known model artifacts
- determine what the model actually is
- determine what hardware is actually available and usable
- choose an edit strategy appropriate to the model and target behavior
- schedule execution across the best available devices
- execute the ablation in resumable stages
- quantize final deployable artifacts toward **INT8** where appropriate
- verify semantic and operational quality
- promote only artifacts that pass explicit gates

This is the controlling doctrine for the repo. Where modules or docs disagree with this file, this file wins.

---

## Design Goals

1. **Model-agnostic first, model-specialized second**  
   The platform must not assume one architecture family, one runtime, or one accelerator path.

2. **Hardware-aware, not hardware-naive**  
   The system must exploit whatever hardware is present, but only after probing what is physically present, runtime-usable, precision-capable, and thermally safe.

3. **Ablation as a staged pipeline, not a monolithic function**  
   Each stage must produce explicit artifacts, manifests, hashes, logs, and resumable state.

4. **Truthful execution reporting**  
   Every run must clearly disclose whether each stage ran in:
   - native accelerated mode
   - native CPU mode
   - fallback mode
   - simulation mode

5. **INT8 as deployment target, not universal dogma**  
   Final deployable artifacts should target INT8 where supported and beneficial, but earlier stages may require higher precision for safe editing and validation.

6. **Promotion only after validation**  
   No artifact is promoted unless verification is complete, execution mode is disclosed, integrity is proven, and promotion gates are satisfied.

---

## Supported Model Classes

The pipeline must support the following model categories.

### 1. Dense models
Examples:
- standard decoder-only language models
- encoder-decoder models
- compact task-specific transformer variants

### 2. Mixture-of-Experts models
Examples:
- router-driven sparse expert models
- expert-partitioned decoder stacks

Additional concerns:
- expert routing stability
- expert collapse
- post-edit expert skew
- routing entropy degradation

### 3. Multimodal models
Examples:
- vision-language models
- text-image fused models
- audio-text models
- mixed encoder/decoder systems with modality adapters

Additional concerns:
- cross-modal consistency
- cross-attention behavior
- modality-specific regression
- calibration data diversity

### 4. Pre-quantized or partially quantized artifacts
Examples:
- GGUF
- already-INT8 OpenVINO IR
- mixed precision exported models

Additional concerns:
- whether edit operations remain feasible
- whether re-quantization is allowed
- whether dequantized working form is required

---

## Supported Hardware Classes

The pipeline must opportunistically exploit any detected and runtime-usable hardware, including:

- CPU
- iGPU
- dGPU
- NPU
- VPU
- OpenVINO-compatible devices
- vendor-specific accelerator backends where integration exists

The pipeline must distinguish:

- **physically present**
- **driver/runtime available**
- **usable for target workload**
- **safe under current thermal/power conditions**

No component may claim acceleration unless the runtime path was actually proven usable.

---

## Canonical Pipeline

The pipeline is divided into ten authoritative stages.

---

## Stage 1 — Model Intake and Fingerprinting

### Objective
Inspect the incoming model artifact and convert it into a normalized internal description that the rest of the pipeline can use without architecture-specific guessing.

### Inputs
- local model path
- model directory
- model manifest
- model file bundle
- optional operator metadata

### Outputs
- `model_profile.json`

### Responsibilities
The intake layer must determine:

- source format
- architecture family
- topology class
- modality support
- estimated parameter scale
- context length if derivable
- quantization state
- likely runtime compatibility
- edit feasibility class
- capture feasibility class

### Required normalized fields
At minimum, the model profile should include:

- `model_id`
- `source_path`
- `source_format`
- `family`
- `topology`
- `modalities`
- `parameter_estimate`
- `context_length`
- `quantization_state`
- `expert_topology`
- `runtime_support`
- `estimated_memory_by_precision`
- `supports_runtime_hooks`
- `supports_delta_edit`
- `supports_activation_capture`
- `notes`

### Rules
- Family detection must not rely on one fragile string.
- MoE detection must inspect topology fields where possible.
- Multimodal detection must inspect subcomponents and adapters, not branding alone.
- Quantized artifacts must be explicitly tagged as editable, partially editable, or non-editable.

### Existing repo anchors
- `src/aegis_lab/intake/fingerprint.py`

### Recommended repo changes
- split raw inspection from normalized profile generation
- add contracts under `src/aegis_lab/intake/contracts.py`
- add format-specific loaders under `src/aegis_lab/intake/loaders.py`

---

## Stage 2 — Hardware Discovery and Capability Mapping

### Objective
Build a truthful, runtime-grounded view of available compute resources.

### Inputs
- local hardware state
- drivers
- runtime libraries
- thermal/load telemetry

### Outputs
- `hardware_sitrep.json`
- `hardware_capability_matrix.json`

### Responsibilities
The hardware layer must detect:

- CPU capabilities
- iGPU presence and usable runtime path
- dGPU presence and usable runtime path
- NPU presence and usable runtime path
- VPU presence and usable runtime path
- precision support by device
- memory budget by device
- thermal state
- runtime confidence

### Required per-device fields
At minimum:

- `device_id`
- `device_class`
- `physically_present`
- `runtime_usable`
- `backend`
- `supported_precisions`
- `supported_stage_types`
- `memory_budget`
- `thermal_state`
- `runtime_notes`
- `confidence_score`

### Rules
- Presence is not usability.
- Usability is not suitability.
- Suitability can change dynamically with thermal state.
- All fallback and downgrade reasons must be recorded.

### Existing repo anchors
- `src/aegis_lab/hardware/discovery.py`
- `src/aegis_lab/hardware/telemetry.py`
- `src/aegis_lab/hardware/thermal.py`

### Recommended repo changes
- create `src/aegis_lab/hardware/contracts.py`
- split physical discovery, runtime probing, and matrix construction
- normalize telemetry outputs

---

## Stage 3 — Target Analysis and Edit Strategy Selection

### Objective
Translate an operator’s ablation intent into a model-aware edit plan.

### Inputs
- `model_profile.json`
- operator target request
- protected capability policy
- optional adversarial test plan

### Outputs
- `edit_plan.json`

### Supported target classes
Examples include:

- refusal suppression
- behavior cluster removal
- latent trait attenuation
- routing bias alteration
- prompt-conditioned behavior suppression
- modality-specific behavior suppression
- cross-modal coupling attenuation

### Supported edit method classes
Examples include:

- runtime masking
- activation suppression
- tensor delta patching
- reversible patch sets
- routing adjustments
- adapter-level interventions
- cross-attention interventions

### Required fields
At minimum:

- `target`
- `protected_capabilities`
- `edit_method`
- `target_loci`
- `required_capture_types`
- `required_validation_suites`
- `reversible`
- `delta_expected`
- `notes`

### Rules
- Edit strategy is a semantic decision, not a hardware placement decision.
- Hardware assignment belongs later.
- Protected capabilities must be explicit, not implied.
- For multimodal models, modality-local and cross-modal targets must be distinguishable.
- For MoE models, routing effects must be considered during planning, not just validation.

### Existing repo anchors
- `src/aegis_lab/editing/planner.py`

### Recommended repo changes
- create `src/aegis_lab/editing/contracts.py`
- keep hardware logic out of edit planning

---

## Stage 4 — Capture, Probe, and Analysis Artifact Generation

### Objective
Collect the internal signals required to identify the intervention locus and quantify pre-edit behavior.

### Inputs
- model profile
- edit plan
- selected datasets
- stage assignment from execution plan

### Outputs
- `capture_manifest.json`
- capture artifacts
- telemetry artifacts

### Supported capture classes
Examples include:

- hidden-state capture
- activation statistics
- routing telemetry
- cross-attention capture
- differential prompt traces
- modality-conditioned response traces

### Rules
- Capture must be resumable.
- Capture artifacts must be manifest-addressed and hash-checked.
- Capture outputs must be decoupled from promotion logic.
- Capture stage is allowed to be expensive if it materially improves edit quality.

### Existing repo anchors
- `src/aegis_lab/probing/capture.py`

### Recommended repo changes
- create `src/aegis_lab/probing/contracts.py`
- expand capture beyond text-only assumptions

---

## Stage 5 — Hardware-Aware Execution Planning

### Objective
Assign each pipeline stage to the best available hardware path.

### Inputs
- `model_profile.json`
- `hardware_capability_matrix.json`
- `edit_plan.json`
- scheduler policy

### Outputs
- `execution_plan.json`

### Responsibilities
The execution planner must assign:

- stage sequence
- stage device
- stage precision
- stage batch strategy
- stage memory budget
- stage fallback chain
- stage execution mode declaration
- promotion target precision

### Device preference guidance
General default priorities:

- **CPU**
  - control flow
  - serialization
  - graph surgery
  - universal fallback
- **iGPU / dGPU**
  - dense tensor-heavy execution
  - large verification sweeps
  - high-throughput inference
- **NPU**
  - compact repeated inference
  - sentinel loops
  - efficient quantized validation where supported
- **VPU**
  - low-power lightweight validation
  - continuous guard/sentinel paths where integration exists

### Required fields
At minimum:

- `job_id`
- `stage_assignments`
- `promotion_target_precision`
- `fallback_chain`
- `execution_mode_summary`
- `scheduler_notes`

### Rules
- Scheduler decisions must be capability-driven, not brand-driven.
- Degradation must be explicit.
- Unsupported precision on a device must trigger plan rewrite, not wishful execution.
- The scheduler is the one true owner of placement logic.

### Existing repo anchors
- `src/aegis_lab/scheduler/engine.py`

### Recommended repo changes
- create `src/aegis_lab/scheduler/contracts.py`
- create `src/aegis_lab/scheduler/policies.py`
- make execution planning canonical and explicit

---

## Stage 6 — Edit Execution

### Objective
Apply the planned intervention in a staged and resumable way.

### Inputs
- `model_profile.json`
- `edit_plan.json`
- `execution_plan.json`
- capture artifacts

### Outputs
- edit artifacts
- delta artifacts
- rollback metadata
- stage manifest

### Responsibilities
The edit execution layer must:

- apply runtime or persistent interventions
- persist intermediate outputs
- support retries
- support deterministic resume
- record touched tensors or loci where applicable
- record runtime mode used
- record device and precision used

### Persistent edit classes
When producing persistent artifacts, the pipeline should emit:

- tensor delta bundle
- touched layer/locus manifest
- precision used
- reversibility metadata
- lineage hash

### Rules
- Reversible edits must ship rollback data.
- Runtime edits must ship replay configuration.
- Multimodal edits must record targeted modalities or cross-modal loci.
- MoE edits must record routing-sensitive metadata where relevant.

### Existing repo anchors
- `src/aegis_lab/editing/pipeline.py`
- `src/aegis_lab/editing/runtime.py`
- `src/aegis_lab/editing/delta_builder.py`
- `src/aegis_lab/editing/rollback.py`

### Recommended repo changes
- break monolithic pipeline into explicit stage transitions
- require per-stage manifests

---

## Stage 7 — Quantization and Packaging

### Objective
Turn the edited artifact into a deployable artifact, favoring INT8 promotion where appropriate.

### Inputs
- edited model artifact
- execution metadata
- calibration bundle
- quantization policy

### Outputs
- quantized artifact
- `quantization_report.json`
- calibration manifest
- packaging manifest

### Responsibilities
The quantization layer must decide:

- whether quantization is allowed
- which backend is appropriate
- which calibration set is valid
- whether partial quantization is necessary
- what precision fallback occurred

### Rules
- INT8 is the primary promotion target where supported.
- INT8 is not required for every internal stage.
- Multimodal calibration must not be text-only.
- Already quantized models must disclose whether they were re-quantized, wrapped, preserved, or rejected.

### Existing repo anchors
- `src/aegis_lab/quantization/exporter.py`
- `src/aegis_lab/quantization/calibration.py`
- `src/aegis_lab/quantization/validators.py`

### Recommended repo changes
- create `src/aegis_lab/quantization/pipeline.py`
- create `src/aegis_lab/quantization/contracts.py`
- treat exporter backends as implementations, not policy owners

---

## Stage 8 — Verification and Promotion Gating

### Objective
Decide whether the result is actually good enough to keep.

### Inputs
- edited artifact
- quantized artifact if present
- baseline captures
- validation suites
- execution disclosures

### Outputs
- `validation_report.json`
- `promotion_gate.json`

### Required validation dimensions
At minimum:

1. **target suppression efficacy**
2. **protected capability retention**
3. **semantic drift**
4. **adversarial robustness**
5. **hallucination regression**
6. **hardware execution disclosure**
7. **MoE routing stability** where applicable
8. **multimodal cross-modal consistency** where applicable

### Required report fields
At minimum:

- `job_id`
- `stage`
- `execution_mode`
- `metrics`
- `gate_results`
- `passed`
- `blocking_reasons`

### Rules
- Deterministic mock/fallback validators may exist, but must never masquerade as authoritative semantic proof.
- Validation must disclose if any part ran under fallback or simulation.
- A failed or incomplete validation must block promotion.
- Validation policy must know the difference between:
  - unsupported
  - untested
  - failed
  - passed under fallback
  - passed under native accelerated execution

### Existing repo anchors
- `src/aegis_lab/verification/authority.py`
- `src/aegis_lab/verification/proofs.py`
- `src/aegis_lab/evaluation/*`

### Recommended repo changes
- create `src/aegis_lab/verification/contracts.py`
- create `src/aegis_lab/verification/suites.py`
- have verification orchestrate metric providers from `evaluation/*`

---

## Stage 9 — Promotion, Registry, and Audit Trail

### Objective
Promote only validated artifacts and preserve lineage.

### Inputs
- final artifacts
- validation report
- quantization report
- execution plan
- model profile
- integrity metadata

### Outputs
- `promotion_manifest.json`
- durable registry entry
- promoted artifact bundle

### Responsibilities
The promotion layer must enforce:

- atomic promotion
- integrity checks
- lineage checks
- validation gate completion
- execution mode disclosure
- quantization metadata presence if promoted artifact is quantized

### Rules
No artifact may be promoted unless all of the following are true:

- model profile exists
- hardware sitrep exists
- edit plan exists
- execution plan exists
- validation report exists
- integrity hashes verify
- promotion gating passes
- quantization report exists if quantized artifact is being promoted

### Existing repo anchors
- `src/aegis_lab/orchestrator/promotion.py`
- `src/aegis_lab/artifacts/hashing.py`
- `src/aegis_lab/artifacts/manifests.py`
- `src/aegis_lab/artifacts/store.py`
- `src/aegis_lab/state/db.py`

### Recommended repo changes
- keep manifests as source of truth for stage outputs
- keep DB as index and lifecycle state only

---

## Stage 10 — Operator Surfaces

### Objective
Expose the pipeline consistently through CLI, API, and GUI.

### CLI requirements
The CLI must support:

```bash
aegis intake --model /path/to/model
aegis hardware sitrep
aegis hardware matrix
aegis plan-edit --model /path/to/model --target refusal
aegis plan-exec --job-id JOB_ID
aegis run --job-spec /path/to/spec.json
aegis resume --job-id JOB_ID
aegis validate --job-id JOB_ID
aegis promote --job-id JOB_ID
```

### API requirements

The API should expose at minimum:

* `POST /jobs`
* `GET /jobs/{id}`
* `GET /jobs/{id}/stages`
* `GET /jobs/{id}/artifacts`
* `GET /jobs/{id}/validation`
* `GET /hardware/sitrep`
* `GET /hardware/capability-matrix`
* `POST /jobs/{id}/resume`
* `POST /jobs/{id}/promote`

### GUI requirements

The GUI must show:

* normalized model profile
* hardware sitrep
* stage progress timeline
* execution plan
* stage-specific device assignments
* fallback notices
* quantization report
* validation gates
* artifact lineage
* promotion readiness

### Rules

* GUI must display the truth, not the aspiration.
* If execution is in fallback or simulation, that must be impossible to miss.
* CLI remains first-class. GUI convenience must not become an architectural dependency.

### Existing repo anchors

* `src/aegis_lab/cli/main.py`
* `src/aegis_lab/api/server.py`
* `src/aegis_lab/gui/main_window.py`

---

## Canonical Runtime Modes

Every stage must declare one of these runtime modes:

### `native_accelerated`

The assigned accelerator backend was available and actually used.

### `native_cpu`

The stage ran natively on CPU without pretending to use an unavailable accelerator.

### `fallback`

The preferred backend was unavailable or unsuitable, and execution downgraded to another real path.

### `simulation`

The stage did not execute its real semantics and instead produced mock, emulated, or placeholder results.

### Rules

* Simulation is allowed for development and smoke testing.
* Simulation is not allowed to silently satisfy production promotion gates.
* Fallback is valid but must be disclosed.

---

## Canonical Artifacts

Each job should generate the following core artifacts where applicable:

* `model_profile.json`
* `hardware_sitrep.json`
* `hardware_capability_matrix.json`
* `edit_plan.json`
* `capture_manifest.json`
* `execution_plan.json`
* edit artifact bundle
* rollback bundle if reversible
* `quantization_report.json`
* `validation_report.json`
* `promotion_manifest.json`

---

## Artifact Layout

Recommended layout:

```text
/jobs/<job_id>/
  manifests/
    model_profile.json
    hardware_sitrep.json
    hardware_capability_matrix.json
    edit_plan.json
    execution_plan.json
    quantization_report.json
    validation_report.json
    promotion_manifest.json
  stages/
    01-intake/
      manifest.json
      artifacts/
      logs/
      metrics/
    02-hardware/
      manifest.json
      artifacts/
      logs/
      metrics/
    03-edit-plan/
      manifest.json
      artifacts/
      logs/
      metrics/
    04-capture/
      manifest.json
      artifacts/
      logs/
      metrics/
    05-exec-plan/
      manifest.json
      artifacts/
      logs/
      metrics/
    06-edit/
      manifest.json
      artifacts/
      logs/
      metrics/
    07-quantization/
      manifest.json
      artifacts/
      logs/
      metrics/
    08-verification/
      manifest.json
      artifacts/
      logs/
      metrics/
    09-promotion/
      manifest.json
      artifacts/
      logs/
      metrics/
```

---

## Required Contracts

The following internal contracts should become authoritative across the codebase.

### `NormalizedModelProfile`

Fields:

* `model_id`
* `source_path`
* `source_format`
* `family`
* `topology`
* `modalities`
* `parameter_estimate`
* `context_length`
* `quantization_state`
* `expert_topology`
* `runtime_support`
* `estimated_memory_by_precision`
* `supports_runtime_hooks`
* `supports_delta_edit`
* `supports_activation_capture`
* `notes`

### `HardwareCapabilityMatrix`

Fields:

* `devices`
* `supported_stage_types`
* `precision_support`
* `runtime_health`
* `thermal_state`
* `preferred_assignments`
* `fallback_paths`

### `EditPlan`

Fields:

* `target`
* `protected_capabilities`
* `edit_method`
* `target_loci`
* `required_capture_types`
* `required_validation_suites`
* `reversible`
* `delta_expected`
* `notes`

### `ExecutionPlan`

Fields:

* `job_id`
* `stage_assignments`
* `promotion_target_precision`
* `fallback_chain`
* `execution_mode_summary`
* `scheduler_notes`

### `ValidationReport`

Fields:

* `job_id`
* `stage`
* `execution_mode`
* `metrics`
* `gate_results`
* `passed`
* `blocking_reasons`

### `PromotionManifest`

Fields:

* `artifact_hash`
* `lineage`
* `model_profile_hash`
* `execution_plan_hash`
* `validation_report_hash`
* `quantization_report_hash`
* `promoted_at`

---

## Policy Rules

### Rule 1 — Separation of concerns

* intake decides what the model is
* edit planning decides what semantic change is desired
* scheduling decides where stages run
* verification decides whether results are acceptable
* promotion decides whether results are retainable

### Rule 2 — No silent fallback

Any stage that deviates from preferred execution must record why.

### Rule 3 — No fake acceleration claims

A configured backend does not count as active acceleration unless successfully used.

### Rule 4 — No promotion without proof

Promotion requires validation, integrity, lineage, and execution disclosure.

### Rule 5 — Resumability is mandatory

Every expensive stage must be restartable from persisted state.

### Rule 6 — Multimodal and MoE are first-class

They are not special-case afterthoughts. Their requirements must appear in profile, plan, validation, and reporting.

### Rule 7 — Quantization is policy-governed

Quantization is not just “run exporter and hope.”

---

## Recommended Repository Mapping

### Intake

* `src/aegis_lab/intake/fingerprint.py`
* add `src/aegis_lab/intake/contracts.py`
* add `src/aegis_lab/intake/loaders.py`

### Hardware

* `src/aegis_lab/hardware/discovery.py`
* `src/aegis_lab/hardware/telemetry.py`
* `src/aegis_lab/hardware/thermal.py`
* add `src/aegis_lab/hardware/contracts.py`

### Edit planning and execution

* `src/aegis_lab/editing/planner.py`
* `src/aegis_lab/editing/pipeline.py`
* `src/aegis_lab/editing/runtime.py`
* `src/aegis_lab/editing/delta_builder.py`
* `src/aegis_lab/editing/rollback.py`
* add `src/aegis_lab/editing/contracts.py`

### Probing

* `src/aegis_lab/probing/capture.py`
* add `src/aegis_lab/probing/contracts.py`

### Scheduling

* `src/aegis_lab/scheduler/engine.py`
* add `src/aegis_lab/scheduler/contracts.py`
* add `src/aegis_lab/scheduler/policies.py`

### Quantization

* `src/aegis_lab/quantization/exporter.py`
* `src/aegis_lab/quantization/calibration.py`
* `src/aegis_lab/quantization/validators.py`
* add `src/aegis_lab/quantization/pipeline.py`
* add `src/aegis_lab/quantization/contracts.py`

### Verification

* `src/aegis_lab/verification/authority.py`
* `src/aegis_lab/verification/proofs.py`
* `src/aegis_lab/evaluation/*`
* add `src/aegis_lab/verification/contracts.py`
* add `src/aegis_lab/verification/suites.py`

### Orchestration and promotion

* `src/aegis_lab/orchestrator/service.py`
* `src/aegis_lab/orchestrator/promotion.py`
* `src/aegis_lab/orchestrator/optimizer.py`
* add `src/aegis_lab/orchestrator/contracts.py`

### Artifact and state

* `src/aegis_lab/artifacts/hashing.py`
* `src/aegis_lab/artifacts/manifests.py`
* `src/aegis_lab/artifacts/store.py`
* `src/aegis_lab/artifacts/layout.py`
* `src/aegis_lab/state/db.py`

### Surfaces

* `src/aegis_lab/cli/main.py`
* `src/aegis_lab/api/server.py`
* `src/aegis_lab/gui/main_window.py`

---

## Minimum Test Matrix

The repo should treat these as mandatory pipeline validation scenarios.

### Unit tests

* normalized dense model profile
* normalized MoE model profile
* normalized multimodal model profile
* hardware capability matrix generation
* edit plan contract generation
* execution plan generation with fallback chain
* quantization report generation
* validation gate blocking logic
* promotion manifest integrity logic

### Integration tests

* dense model full pipeline
* MoE model full pipeline
* multimodal model full pipeline
* resume from capture stage
* resume from quantization stage
* promotion refusal when verification incomplete
* promotion refusal when simulation mode used as final proof
* promotion success when all gates pass

---

## Success Criteria

A job is successful only if it produces:

* correct normalized model profile
* truthful hardware capability matrix
* explicit edit plan
* explicit execution plan
* persisted stage artifacts
* verification report with meaningful gates
* promotion decision backed by integrity and lineage

A job is not successful merely because it “ran.”

---

## Immediate Implementation Priorities

Implement in this order:

1. contracts for model, hardware, edit, execution, validation, promotion
2. intake normalization hardening
3. hardware capability matrix hardening
4. scheduler canonicalization
5. editing pipeline stage split
6. quantization stage formalization
7. verification gate hardening
8. promotion gate hardening
9. API/CLI/GUI truth surfacing
10. end-to-end tests

This order matters. Doing GUI polish before locking contracts would be lipstick on a compiler error.

---

## Final Doctrine

AEGIS-LAB is not a single-model patcher and not a hardware vanity project. It is a **truthful, staged, resumable, hardware-adaptive ablation platform**.

Its job is to take an arbitrary supported model, determine what it is, determine what compute is available, choose the correct intervention path, execute the intervention safely, package the result toward deployable precision, verify that the semantic target was achieved without unacceptable collateral damage, and promote only artifacts that can defend their own lineage.

Anything less is demo theater.
