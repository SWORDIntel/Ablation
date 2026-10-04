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

The installed entry point dispatches core operations. Advanced commands use `cli_extended`; inherited core parsers there lack dispatch handlers. Workflow apply/localization and campaign profiling/recovery include placeholder or synthetic behavior. The committed standalone `search-mlp` lacks a tensor-loader import, and optimizer version-4 plans are rejected by preview. The quick start uses joint optimization and recompiles exact checksummed selections through the supported selector interface. The roadmap now records the remaining integration acceptance separately from available stage modules.

Current guidance replaces old platform-first descriptions and unsupported hardware/performance claims. Historical documents remain under `docs/archive/`; commands and links in historical material are preserved context, not current instructions.
