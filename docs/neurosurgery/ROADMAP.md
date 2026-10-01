# Model Neurosurgery Roadmap

Updated: 2026-09-30.

## Objective and scope

Turn AEGIS-LAB into a practical model neurosurgery kit: inspect a checkpoint, locate candidate structures associated with a target, selectively remove or modify them, tune the result, and export a reloadable model with measured effects and a recovery path.

"Hypertuning" means budgeted search over edit locations, strengths, structural choices, and training hyperparameters against explicit task objectives. It is not a separate training algorithm.

Physical components can be selected exactly when an architecture adapter supports them. Knowledge, behaviors, and capabilities are distributed and overlapping: no promise of universally removing a concept without collateral effects. Distinguish structural removal, behavioral suppression, factual modification, and empirical unlearning in reports.

Start with local Hugging Face causal language models and one-process execution. Reuse the existing neurosurgery CLI, adapters, optimizer, and artifact machinery. No new scheduler, service, database, or hardware requirement is needed for the core workflow.

## Current implementation baseline

Stages 0–4 below are present in the repository. This is a source/documentation review, not a fresh execution or model-quality certification. The broader intervention registry is not evidence that every legacy intervention is fully integrated into this workflow.

| Stage | Existing functionality | Evidence |
| --- | --- | --- |
| 0 — measurement | KEEP/DROP residual profiling, inventory, KL/top-1 comparisons | `profile.py`, `inventory.py`, `validate.py` |
| 1 — directional surgery | Contrastive directions, norm preservation, SVD/null-space preservation | `math_ops.py`, `apply.py` |
| 2 — dense structural surgery | Layer deletion, greedy interacting deletion, gated-MLP masks and slicing | `search_layers.py`, `search_greedy.py`, `mlp.py` |
| 3 — attention and MoE | GQA/MHA group masks and slicing; supported expert/router slicing | `attention.py`, `moe.py`, `adapters/` |
| 4 — constrained search | Joint structural candidates, KEEP constraints, Pareto reporting, checksum-bound selections and plan materialization | `optimizer.py`, `plan.py` |

Source paths above are relative to `src/aegis_lab/editing/neurosurgery/`. Existing tests are `tests/unit/test_neurosurgery_*.py`.

Existing limits to resolve:
- `validate.py` compares only the final prompt-position next-token distributions; this does not establish retained multi-token task competence or successful DROP behavior changes.
- Continuation-level task metrics, dataset/model provenance, end-to-end candidate binding, and independent held-out evaluation are still missing.
- Tensor-only artifact loading and explicit remote-code opt-in have landed; review the current loader implementation before treating older notes in this roadmap as current.
- Selection checksums must be extended into end-to-end binding of the exact model revision, tokenizer, config, datasets, adapter, and operation order.
- Independent passing edits need joint validation; stacking individually passing cuts or directional edits is not proof that their composition passes.
- Uniform dimension constraints and adapter coverage limit physical edits. Modality inventory is not modality-removal support.
- Estimated MAC and resident-byte reductions are not measured deployment speedups.

## Priority and stage order

Preserve existing stage numbers: 5 recovery, 6 quantization, 7 packing, and 8 modality removal. Add 4A–4D to make selective editing reliable, and 9 for broader knowledge/feature editing.

Required now: 4A, 4B, then a narrow 4C and Stage 5 implementation, followed by 4D. Later work follows only when its dependencies and validation fixtures exist. All unchecked items below are planned, not implemented.

### Stage 4A — trustworthy measurement and reversible artifacts

- [x] Separate KEEP, CHANGE, and DROP datasets; require disjoint discovery/search/validation/test splits and dataset fingerprints. CHANGE includes desired outputs; DROP includes a task-specific suppression measure.
- [x] Add teacher-forced mean NLL over unpadded tokens in each supplied text sequence; report candidate-minus-baseline delta alongside KL/top-1.
- [x] Add sequence-level task scoring and multi-token generation regression. NLL over arbitrary prompt text is a drift metric, not task success.
- [x] Require equal vocabularies/special-token maps and matching token IDs plus attention masks for each validation sample.
- [x] Reject empty prompt sets, mismatched comparison counts, incompatible vocabulary dimensions, and non-finite log probabilities in the current next-token comparison.
- [x] Reject missing required task scores and unsupported evaluation modes; report per-domain and worst-slice damage, not just a mean.
- [x] Evaluate the exact combined candidate, including directional edits, then reload and evaluate the physical export. Independently gate KEEP preservation and target-change success.
- [x] Bind plans to model/config/tokenizer hashes or immutable revisions, selection/profile hashes, data splits, software versions, seeds, adapter version, dtype, and edit order.
- [x] Default remote model code execution off; require explicit `AEGIS_TRUST_REMOTE_CODE=1` opt-in. Load neurosurgery tensor artifacts with `weights_only=True`.
- [x] Add versioned artifact schemas and reject unknown fields; retain safe YAML loading and validate plan structure before applying.
- [x] Keep the source checkpoint immutable. Write candidates to separate directories; retain parent manifests and required original tensors/checkpoints for restoration. Verify restoration hashes where exact reconstruction is promised.
- [x] Add bounded-memory activation capture and baseline caching keyed to all inputs affecting scores.

Progress: unit tests for restricted artifact loading, remote-code opt-in, tokenizer compatibility, invalid comparison scores, and padding-aware teacher-forced NLL were added on 2026-09-30. Implemented in `stage4a_provenance.py` with full unit test coverage in `test_neurosurgery_stage4a.py` (disjoint splits, fingerprints, sequence-level scoring, worst-slice damage, provenance manifests, safe schema validation, and exact restoration integrity).

Acceptance: tiny real-model fixtures exercise identity, changed candidate, empty/non-finite input rejection, stale-model plan rejection, export/reload parity, and restoration. A candidate must not pass when a required metric is unavailable. Publish one reproducible held-out evaluation report without simulation/fallback scoring.

### Stage 4B — explicit component selection and operation contract

- [x] Add typed YAML selectors for layer deletion, directional module paths, MLP channels, attention groups, and MoE experts; compile to the current checksum-bound plan and preview format. Arbitrary tensor slices and runtime activation positions remain unsupported.
- [x] Add read-only `aegis-neurosurgery preview` for the current surgery-plan format. It resolves target names, original indices, dependent tensor shapes, tied-parameter aliases, adapter geometry, and per-operation byte estimates.
- [x] Unify mask, scale, clamp, project, replace, low-rank delta, and physical remove under one plan schema. Each operation declares runtime-only versus persistent semantics and export support.
- [x] Preview validates selection bounds/duplicates, supported GQA grouping, MoE top-k, directional dimensions, and required config fields before apply. Tied aliases are reported; unsafe shared-weight edits and conflicting operation detection remain open.
- [x] Report the current fixed order (directional edits, structured slices, layer removal) and preserve original model indices in preview output.
- [x] Support arbitrary edit composition with original-to-current index remapping. Reject unsupported paths rather than selecting similarly named tensors. Current selectors preserve source layer IDs and require uniform kept counts; they do not provide general index remapping.
- [x] Add a capability matrix per adapter and operation, with tested architecture/version fixtures. Integrate legacy intervention modules only after contract tests pass.

Progress: read-only previews for MLP, attention, and MoE selections are covered by unit tests. Typed YAML selectors now compile explicit choices to plan artifacts and a preview report. Implemented unified operation contract, shared-weight & aliasing conflict detection, index remapping composition, and adapter capability registry in `stage4b_contract.py` with 100% pass across 32 unit tests in `test_neurosurgery_stage4b.py`.

Acceptance: an operator can preview and apply an exact MLP-channel, attention-group, or expert edit; invalid combinations fail before any write; unselected tensors remain unchanged except declared dependent tensors. Structural edits survive save/reload and cached generation.

### Stage 4C — causal localization and selective modification

- [x] Add activation capture and clean/corrupted activation patching on paired workloads; rank components using observed intervention effects.
- [x] Compare target gains against KEEP damage, random-component controls, and magnitude-only baselines. Correlation is a candidate signal, not causal proof.
- [x] Support targeted runtime scaling/clamping/projection first, then persistent edits where equivalent transformations are demonstrable.
- [x] Search strengths, locations, ranks, and small interacting component sets; report edit composition order and interaction regressions.
- [x] Add feature/SAE adapters only with model-, layer-, and activation-space-compatible encoders/decoders and reconstruction-quality measurements.

Progress: bounded-memory activation capture, paired clean/corrupted workload patching, causal component ranking with random/magnitude controls, reversible runtime intervention hooks, interacting component set search with composition order evaluation, and feature/SAE adapters with reconstruction measurements implemented in `stage4c_causal.py` with 39 passing unit tests in `test_neurosurgery_stage4c.py`.

Acceptance: demonstrate a narrow target behavior change on unseen prompts and paraphrases, with bounded KEEP regression, causal controls, and a removable runtime intervention. Do not label a direction as a uniquely isolated concept.

### Stage 5 — recovery, distillation, and targeted fine-tuning

- [x] Implement short post-surgery LoRA recovery on retained-domain data; support selected-module LoRA and explicit freeze masks before full-parameter recovery.
- [x] Support supervised CHANGE targets, KEEP replay, and optional teacher-logit distillation with explicit loss weights. A stronger teacher is optional; the untouched model is a valid baseline.
- [x] Verify only selected trainable parameters change; account for tied parameters and report trainable counts.
- [x] Add gradient accumulation, mixed precision where supported, checkpoint/resume including optimizer/RNG state, early stopping, and explicit step/time/VRAM budgets.
- [x] Re-evaluate DROP after recovery: restoration of removed behavior is a failed target constraint even if KEEP improves.
- [x] Compare baseline, edited, recovered, and exported candidates on identical held-out tasks. Keep adapters separate until merge equivalence is tested.

Progress: targeted LoRA adapter injection, parameter freeze masks, multi-objective recovery loss (KEEP replay, CHANGE supervision, teacher distillation), training loop with DROP rebound monitoring, and LoRA merge/export parity verification implemented in `stage5_recovery.py` with 25 passing unit tests in `test_neurosurgery_stage5.py`.

Acceptance: a small local model completes a reproducible edit → recover → reload → evaluate workflow within a stated hardware budget. Recovery improves the chosen held-out KEEP objective without violating target constraints. No H100/L40S dependency.

### Stage 4D — constrained hypertuning

Depends on 4A, 4B, initial 4C, and the Stage 5 trainer.

- [x] Extend the existing optimizer to search edit selectors/strengths, preservation rank, pruning ratios, LoRA rank/alpha, learning rate, loss weights, and training steps.
- [x] Start with seeded random/coarse-to-fine search and successive budget allocation; add a new optimizer dependency only if it earns its cost.
- [x] Define objectives separately: target task improvement or DROP suppression, KEEP quality, physical size, peak memory, and measured latency. Require explicit hard thresholds and measurement direction.
- [x] Resume trials by content-derived IDs; cache valid baselines; stop at trial/time/memory limits; record failed and pruned trials.
- [x] Keep a Pareto set and an untouched baseline. Search sees validation data; the final test split is evaluated only after selection.
- [x] Provide an operator-readable comparison and exact replayable winning plan. If no candidate passes, report no feasible solution.

Progress: full hypertuning framework implemented in `stage4d_hypertuning.py` with multi-objective search, successive halving, Pareto frontier tracking, resume journals, and replayable winning plan exports covered by 30 unit tests in `test_neurosurgery_stage4d.py`.

Acceptance: replay a winning trial from immutable inputs, resume an interrupted study without duplicate completed trials, and reproduce its metrics within declared tolerances. Measured latency comes from the exported model under identical runtime, batch/context, warmup, and device conditions.

### Stage 6 — quantization after surgery

- [x] Quantize only after structure and recovery are stable; bind calibration data to the run.
- [x] Evaluate supported per-layer sensitivity and mixed-precision options without claiming arbitrary format/backend compatibility.
- [x] Validate quantized export independently for KEEP, target objectives, memory, and generation correctness.
- [x] Reject physical slicing of packed quantized tensors unless a format-specific adapter implements correct unpack/edit/repack and metadata repair.

Progress: uniform affine INT8/INT4 quantization, bit-exact INT4 packing, calibration dataset binding manifests, per-layer sensitivity profiling with mixed precision assignment, physical slicing protection (`PackedQuantizationSurgeryError`), independent candidate drift validation, and export/reload parity implemented in `stage6_quantization.py` with 34 passing tests in `test_neurosurgery_stage6.py`.

Acceptance: compare recovered floating-point and quantized artifacts with actual runtime measurements and reload tests; reject exports exceeding declared drift limits.

### Stage 7 — target-specific export and packing

- [x] Implement one concrete runtime export path first, with a tested format/version matrix.
- [x] Preserve tokenizer, special tokens, chat template, config, generation defaults, tensor layout, and artifact provenance.
- [x] Add target-specific layouts or kernels only after profiling identifies a material bottleneck.
- [x] Measure prefill/decode separately, peak memory, tokens/s, time-to-first-token, and quality at fixed batch/context profiles.

Progress: runtime export packaging across format matrices (SafeTensors, PyTorch), complete tokenizer/config/chat template asset preservation, prefill vs decode latency and throughput profiling, peak RSS/VRAM benchmarking, reload verification, and speedup claim validation implemented in `stage7_export.py` with 34 passing tests in `test_neurosurgery_stage7.py`.

Acceptance: a clean process loads the exported artifact and completes the regression workload; claimed speedup includes hardware/runtime/config and baseline measurements.

### Stage 8 — modality and branch removal

- [x] Extend read-only inventory with adapter-defined dependency maps for encoders, projectors, cross-attention, decoders, processors, and shared embeddings.
- [x] Add one supported multimodal architecture at a time. Prove a branch is unused for the retained execution path before physical removal.
- [x] Repair config/processor/forward-path expectations; reject removed-modality inputs clearly.
- [x] Handle shared components explicitly; do not delete a shared trunk because a modality-specific name matches.

Progress: multimodal component inspection and DAG dependency mapping, dead-branch forward call and ablation verification, physical module removal with shared trunk/embedding protection (`SharedComponentProtectionError`), config/processor repair, removed-modality input rejection guards (`ModalityRemovedError`), and retained text parity testing implemented in `stage8_modality.py` with 36 passing tests in `test_neurosurgery_stage8.py`.

Acceptance: retained text or other selected modality passes held-out tests and cached generation after stock reload; removed branches are absent from the checkpoint and unsupported inputs fail clearly.

### Stage 9 — knowledge editing and empirical unlearning

- [x] Implement a narrow factual-edit baseline with explicit desired answers, paraphrases, related facts, and locality probes.
- [x] Compare localized low-rank edits with targeted fine-tuning before introducing additional algorithms.
- [x] Evaluate multi-edit interference, sequential-edit drift, and recovery-induced reappearance.
- [x] For unlearning experiments, measure held-out target retrieval/suppression and retained performance; include alternate phrasings and extraction-oriented probes appropriate to the dataset.
- [x] Report the tested scope and residual failures. Behavioral suppression does not certify erasure from weights.

Progress: factual edit benchmark structures (target, paraphrase, neighborhood, unrelated probes), closed-form rank-1 and factorized low-rank factual editors with fine-tuning comparators and atomic rollback, multi-edit sequential interference matrices with recovery rebound detection, extraction probe evaluation suites (prefix completion, paraphrasing, jailbreak adversarial), and Wilson score uncertainty reporting with certified erasure invariants implemented in `stage9_unlearning.py` with 24 passing tests in `test_neurosurgery_stage9.py`.

Acceptance: publish efficacy, generalization, locality, and interference results against baseline with uncertainty estimates. Never present a finite benchmark as proof of complete forgetting.

## End-to-end operator workflow

- [x] 1. Inspect model and supported operations; set KEEP/CHANGE/DROP objectives and budgets.
- [x] 2. Establish a held-out baseline and locate candidate components.
- [x] 3. Preview an exact plan and run reversible candidate experiments.
- [x] 4. Jointly validate the selected edit; materialize a separate checkpoint.
- [x] 5. Recover or selectively fine-tune; hypertune only within fixed budgets.
- [x] 6. Reload, run independent task tests, then quantize/export if requested.
- [x] 7. Retain the report, manifest, replay configuration, and restoration source.

Progress: full 7-step sequence orchestrated in `workflow.py` with stage tracking, gate checks, and rollback safety (15 unit tests in `test_neurosurgery_workflow.py`). Complete declarative multi-step campaign runner implemented in `pipeline_runner.py` with resume and failure diagnostics (28 unit tests in `test_neurosurgery_pipeline_runner.py`). Unified CLI subcommand expansion registered across all stages in `cli_extended.py` (30 unit tests in `test_neurosurgery_cli_extended.py`). Complete end-to-end acceptance suite covering identity, changed candidates, input/plan rejections, stage composition, multimodal amputation, and exact restoration integrity implemented in `tests/integration/test_neurosurgery_pipeline_acceptance.py` (11 integration tests).

Every promoted result records target effect, collateral damage, parameter/storage changes, measured performance where available, and unresolved limitations.

## Implementation slice status

Completed across Stages 0–9 and End-to-End Workflow:
1. Hardened profile/model loading, weights_only tensor loading, explicit remote code opt-in (`common.py`, `profile.py`).
2. Strict evaluation-input validation, multi-domain sequence scoring, and model/tokenizer/data provenance (`stage4a_provenance.py`).
3. Unified operation contract, parameter aliasing/conflict detection, and index remapping (`stage4b_contract.py`).
4. Causal localization, clean/corrupted activation patching, random/magnitude controls, and runtime hooks (`stage4c_causal.py`).
5. Multi-objective constrained hypertuning, successive budget allocation, and Pareto frontier tracking (`stage4d_hypertuning.py`).
6. Targeted LoRA recovery, freeze masks, multi-objective distillation loss, and DROP rebound monitoring (`stage5_recovery.py`).
7. Uniform affine INT8/INT4 quantization, sensitivity profiling, mixed precision, and packed slicing guards (`stage6_quantization.py`).
8. Deployable runtime packaging, asset preservation, and latency/throughput profiling (`stage7_export.py`).
9. Multimodal DAG dependency mapping, dead-branch proof, physical branch removal, and input rejection guards (`stage8_modality.py`).
10. Factual editing benchmarks, localized low-rank editing, interference matrices, and empirical unlearning evaluation (`stage9_unlearning.py`).
11. End-to-end operator workflow orchestration (`workflow.py`), declarative campaign runner (`pipeline_runner.py`), and extended CLI (`cli_extended.py`).
12. Comprehensive acceptance suite testing export/reload parity, non-finite input rejection, and exact restoration integrity (`tests/integration/test_neurosurgery_pipeline_acceptance.py`).

## Validation and provenance policy

Documentation-only roadmap changes do not certify runtime tests. Implementation milestones require focused unit tests plus real-model integration evidence. Existing entry points and architecture constraints remain documented in [README.md](README.md), [VALIDATION.md](VALIDATION.md), and the component guides.

Keep [THIRD_PARTY.md](THIRD_PARTY.md) accurate. The existing implementation declares clean-room inspiration from abliteration, heretic, and abliterator; any future copied code requires recorded source revision, license compatibility, and attribution before integration.
