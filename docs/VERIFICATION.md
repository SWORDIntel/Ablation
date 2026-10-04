# Documentation verification

Review date: 2026-10-04. Committed implementation reviewed at `551992e` before the documentation refresh. Existing RX470 preparation changes and notes were excluded from the docs commit and from these test runs.

## Executed checks

- Neurosurgery unit discovery: **409 tests passed**.
- Pipeline acceptance module: **11 tests passed**.
- Parsed 40 current documentation CLI invocations against the committed parsers (including help commands).
- Checked 20 current Markdown files, 77 local links and balanced code fences; `git diff --check` passed.
- Executed inventory → MLP profile → optimizer → checked selector conversion → select/preview → apply → reload/validate on a locally initialized tiny HF Llama checkpoint, without downloads. The cut removed 12,288 parameters; held-out mean KL was approximately `1.11e-5`, below the example `0.02` gate. Held-out top-1 agreement was `0.5`; this untrained fixture does not establish retained task quality, and the validate CLI gates only the requested KL threshold.
- Tests executed in a separate checkout of committed code with CPU thread limits; no model downloads or hardware setup were required.

These tests exercise small fixtures, library contracts and rejection/rollback cases. They do not establish production model quality, complete knowledge erasure, native VPU operation or a measured deployment speedup. The local PyTorch build reported an incompatible CUDA driver; validation used CPU rather than qualifying CUDA.

## Source findings reflected in the docs

The installed entry point dispatches core operations. Advanced commands use `cli_extended`; inherited core parsers there lack dispatch handlers. Workflow apply/localization and campaign profiling/recovery include placeholder or synthetic behavior. At revision `551992e`, standalone `search-mlp` lacked a tensor-loader import and optimizer version-4 plans were rejected by preview. The initial docs refresh used a selector-conversion workaround; the core fixes below supersede that workaround. The roadmap now records the remaining integration acceptance separately from available stage modules.

Current guidance replaces old platform-first descriptions and unsupported hardware/performance claims. Historical documents remain under `docs/archive/`; commands and links in historical material are preserved context, not current instructions.

## Core workflow fixes (2026-10-04)

Standalone MLP search now imports `load_tensor_artifact`. Preview accepts the optimizer's version-4 plans directly, preserving checksum and adapter geometry validation. The two source changes are one line each; unrelated local RX470 changes remain excluded.

New regression coverage uses actual optimizer plan materialization for MLP, attention and MoE, including a MoE block at transformer layer 1 rather than block position 0. It checks read-only preview, tampered selection rejection, directional profile requirements and unknown-version rejection.

An offline tiny HF Llama workflow exercises MLP profiling/search, direct version-4 optimizer preview/apply, saved-checkpoint reload, held-out validation, cached generation and unchanged source weights. Its deliberately loose fixture gates force a structural cut; it is not a model-quality benchmark.

Verification of the scoped fixes in an isolated checkout: **412 neurosurgery unit tests and 12 integration tests passed**, including the new core workflow and the existing pipeline acceptance module. CPU thread limits were used; no downloads were needed. Current documentation links and code fences were checked, and `git diff --check` passed.

## Measured integration and operator audits (2026-10-04)

The dedicated installed advanced entry point now dispatches core commands consistently. Advanced workflows and campaigns use real model activations, tokenized training data, checked plans, recovery measurements and reload validation. Missing or nonfinite required gate measurements are errors. General persistent operation composition and index remapping are integrated into core preview/apply; incompatible aliases and runtime-only exports are rejected before changing the source. Middle-layer removal also updates attention cache indices.

Installed commands retain mandatory byte-level input, implementation, option and producer audits. Model-free plan generation carries forward source-model lineage. Workflow/campaign manifests also bind tokenizer processing semantics, edit order and inspectable custom callback sources; resume rejects changed inputs. Legacy artifacts without producer audits are explicitly marked. Lower-level numerical helpers are not independently audited operator runs.

Verification in an isolated checkout, excluding existing local RX470 changes: **412 neurosurgery unit tests and 29 integration tests passed**. The integration suite exercises actual offline HF save/reload, recovery, held-out validation and cached generation, along with source/producer tampering, inherited model mismatch, missing metrics, supervised CHANGE targets, structural rollback, alias rejection, composition and remapping. Fixture quality gates do not qualify trained deployments.

A separate [trained Qwen diagnostic](neurosurgery/QUALIFICATION.md) records real CPU recovery, stock reload, cached generation, disjoint held-out text checks, unchanged source hashes and host-specific benchmark measurements. Its optimizer rejected the proposed cut under the top-1 gate and selected the unchanged structural baseline: **zero parameters were removed**. The diagnostic predates the final audit integration and is not a released acceptance certificate. Five task/runtime/modality qualification items remain open in the [roadmap](neurosurgery/ROADMAP.md).

Documentation links, code fences, entry-point metadata and CLI help were checked, and `git diff --check` passed.
