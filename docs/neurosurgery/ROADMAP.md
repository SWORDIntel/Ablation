# All-in-one model brain surgery kit roadmap

Reviewed: 2026-10-04. This roadmap separates available instruments from integration and real-model acceptance. Checked items indicate repository functionality or completed documentation work, not universal model support.

## Objective

Bring inspection, selective editing, structural cuts, recovery, tuning and export into one reproducible model-engineering kit. Keep the direct one-process path usable without service or specialist-hardware requirements. Physical removal, behavioral suppression, factual modification and empirical unlearning require different reports and acceptance measures.

Stage numbering remains: 0–4 core measurement/surgery/search, 4A provenance, 4B contracts, 4C causal localization, 4D hypertuning, 5 recovery, 6 quantization, 7 export, 8 modality removal, 9 knowledge editing/unlearning.

## Available instruments

- [x] Core residual profiling, inventory and KL/top-1/text-NLL comparisons.
- [x] Contrastive directional edits with norm and preservation-subspace options.
- [x] Layer-deletion search, interacting greedy search and materialization.
- [x] Gated-MLP channel masks and physical gate/up/down slicing.
- [x] Compatible GQA/MHA group masks and physical Q/K/V/O slicing.
- [x] Supported MoE router/expert masks and slicing.
- [x] Joint constrained optimizer with baseline, Pareto reports and checksummed selections.
- [x] Typed selector compilation and read-only geometry preview.
- [x] Stage 4A dataset/task/provenance/restoration helpers.
- [x] Stage 4B operation contracts, alias/conflict helpers and index-remapping library.
- [x] Stage 4C activation patching, controls and feature-adapter library.
- [x] Stage 4D constrained hypertuning and resume/Pareto library.
- [x] Stage 5 LoRA, freeze-mask, distillation and merge-parity helpers.
- [x] Stage 6 affine INT8/packed INT4, sensitivity and slicing guards.
- [x] Stage 7 checkpoint packaging, asset and benchmark helpers.
- [x] Stage 8 modality dependency and branch-removal helpers.
- [x] Stage 9 factual-edit/interference/unlearning evaluation helpers.
- [x] Workflow state engine and campaign runner with replaceable step handlers.

Modules live under `src/aegis_lab/editing/neurosurgery/`. Tests live under `tests/unit/test_neurosurgery_*.py` and `tests/integration/test_neurosurgery_pipeline_acceptance.py`. Read [capabilities](CAPABILITIES.md) for which functions are wired to each interface.

## Integration and acceptance still required

- [x] Fix the missing tensor-loader import in standalone MLP search and support direct optimizer version-4 previews, with producer/consumer, checksum, sparse-MoE and offline HF reload regression coverage.

- [ ] Wire advanced commands into the installed entry point, or provide a dedicated installed advanced entry point with consistent dispatch.
- [ ] Replace advanced workflow placeholder localization/apply and demonstration recovery data with measured, plan-driven implementations.
- [ ] Replace campaign random profiling and constant recovery/rebound metrics with real model/data handlers; reject unavailable required measurements.
- [ ] Bind the exact model, tokenizer, datasets, adapter and edit order end-to-end through every operator path, rather than relying on optional provenance helpers.
- [ ] Integrate the Stage 4B general operation/remapping contract into the core selector/apply path with composition and alias rejection tests.
- [ ] Publish a reproducible real-model edit → recover → reload → evaluate campaign on independent KEEP/CHANGE/DROP tasks, with exact restoration evidence.
- [ ] Qualify each claimed architecture/version on stock reload and cached generation; distinguish fixtures from supported deployments.
- [ ] Qualify quantized formats on actual target kernels/runtimes and publish quality, size, peak memory, prefill/decode latency and baseline conditions.
- [ ] Demonstrate retained-modality execution after branch removal for each supported real multimodal architecture.
- [ ] Publish empirical factual-edit/unlearning results with locality, interference, extraction probes and uncertainty; make no erasure certification claim.

The [previous roadmap](../archive/NEUROSURGERY_ROADMAP_20261004.md) preserves implementation milestone history. Its all-complete checkboxes and test counts do not establish the integration acceptance above.

## Documentation refresh

- [x] Reframe README and docs around the all-in-one model brain surgery kit.
- [x] Document core versus advanced dispatch, placeholder/simulation limits and adapter boundaries.
- [x] Replace stale setup/architecture/schema claims and fix current examples to relative paths.
- [x] Verify command parsing, relative links and focused committed-source tests; record results in the verification guide.

## Acceptance policy

A promoted result must retain its untouched parent, inputs and replay plan; state target effect and collateral damage; pass independent held-out and reload tests; and identify unresolved limits. Report physical parameter/storage changes separately from measured runtime gains. Workflow completion and synthetic metrics cannot satisfy quality gates.
