# All-in-one model brain surgery kit roadmap

Reviewed: 2026-10-31 (neuron steering expansion; other stage status last audited 2026-10-04). This roadmap separates available instruments from integration and real-model acceptance. Checked items indicate repository functionality or completed documentation work, not universal model support.

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

- [x] Wire advanced commands into the installed entry point, or provide a dedicated installed advanced entry point with consistent dispatch.
- [x] Replace advanced workflow placeholder localization/apply and demonstration recovery data with measured, plan-driven implementations.
- [x] Replace campaign random profiling and constant recovery/rebound metrics with real model/data handlers; reject unavailable required measurements.
- [x] Bind exact local model/tokenizer assets, datasets, operator/adapter implementation and edit order across installed commands and measured workflow/campaign paths; verify producer audits and resume inputs, and explicitly mark incomplete legacy lineage.
- [x] Integrate the Stage 4B general operation/remapping contract into the core selector/apply path with composition and alias rejection tests.
- [ ] Publish a reproducible real-model edit → recover → reload → evaluate campaign on independent KEEP/CHANGE/DROP tasks, with exact restoration evidence.
- [ ] Qualify each claimed architecture/version on stock reload and cached generation; distinguish fixtures from supported deployments.
- [ ] Qualify quantized formats on actual target kernels/runtimes and publish quality, size, peak memory, prefill/decode latency and baseline conditions.
- [ ] Demonstrate retained-modality execution after branch removal for each supported real multimodal architecture.
- [ ] Publish empirical factual-edit/unlearning results with locality, interference, extraction probes and uncertainty; make no erasure certification claim.

The [archived multi-model pipeline proposal](../archive/MULTI_MODEL_ABLATION_PIPELINE_FROM_CODEX_BRANCH.md) preserves the divergent historical `codex/define-multi-model-ablation-pipeline` design without overriding the current contracts.

The [previous roadmap](../archive/NEUROSURGERY_ROADMAP_20261004.md) preserves implementation milestone history. Its all-complete checkboxes and test counts do not establish the integration acceptance above.

The [trained Qwen diagnostics](QUALIFICATION.md) now record recovery/reload and packed INT8/INT4 measurements with host-specific CPU timings. Structural pruning retained the unchanged baseline under its strict gate; INT4 had severe held-out drift. These small generic-text runs establish neither full task acceptance nor accelerated target-kernel qualification.

## Neuron steering and causal behaviour engineering roadmap (2026-10-31)

**Goal:** Extend the committed inference-time MLP steering feature into a reproducible, model-bound causal-discovery, optimisation, evaluation and visualisation system. Preserve the current single-process neurosurgery path. Reuse Stage 4A provenance, 4B operation contracts, 4C causal experiments, 4D optimisation, and the existing GUI/API/platform; do not build competing frameworks. Progress labels refer only to merged, demonstrable behaviour. Persona labels in third-party recipes are **reported claims**, not evidence of local reproduction.

### P0 — Establish trusted runtime and evaluation semantics

- [x] Merge the initial reversible dense MLP pre-\`down_proj\` steering runtime, signed doses, built-in reported preset registry, baseline-vs-steered text/token comparisons and regression fixtures (PR #10). **No real checkpoint qualification claimed.**
- [x] P0 increment: record exact checkpoint, weight, config and tokenizer asset SHA-256 bindings; reuse the mandatory operator audit, expose an optional independently trusted expected tree digest and retain an explicitly unauthenticated model-ID label where no trusted hash is supplied. Runtime config/tokenizer semantics are also bound. **Known upstream checkpoint identities and real-model replay qualification remain pending.**
- [x] P0 increment: scoped RNG state for paired conditions, after-sweep baseline replay, per-prompt hashes, and Wilson intervals for descriptive response-change/regex rates. Independent semantic evaluations, matched controls and multi-seed real-model studies remain pending.
- [ ] Implement a machine-verifiable **model binding** contract: hash the model config, tokenizer vocab/templates/special tokens, and all weight shards; record revision/commit, framework and adapter versions, device/dtype/quantization, run identifier and provenance. Treat user-supplied \`--model-id\` as a label only. Fail closed on missing or mismatched bindings, including edited checkpoints with unchanged model IDs.
- [ ] Qualify Qwen3.5's language decoder location, \`qwen3_5\`/text-config distinction, generation hooks and cached decode on **real** Qwen3.5-4B and Qwen3-8B checkpoints before publishing compatible presets; verify reported Qwen3.8-27B only on a matching real checkpoint. Version-pin the minimum verified \`transformers\` runtime. Preserve a single synthetic toy-model CI test without confusing it with real-model acceptance.
- [ ] Precisely define and test all intervention sites: gated MLP intermediate, attention head, residual stream, SAE feature and MoE expert. Validate rank/shape/stride/dtype/device, sharded module placement, tied components and supported quantized access; reject silently bypassed hooks, unsupported architecture paths, non-finite changes and fused kernels lacking equivalent intervention sites.
- [ ] Prove baseline parity before and after each steered run, even on exception; validate prompt vs generated-token phase gates with KV cache, \`generate()\`, early EOS, beam/batch cases and model-specific chat/thinking templates. Detect active intervention per *token and layer*, not just hook counts.
- [ ] Make paired experiments statistically reproducible: separate RNG generators/states per arm, fixed prompt/tokenization, controlled sampling and warmup, repeated runs/seeds, randomized arm order, fixed context/truncation and documented nondeterminism. Include CPU/CUDA/MPS parity checks where supported.
- [ ] Build a **measured evaluator** with distinct target-effect, off-target drift, logit KL/top-1, text NLL, task success, factuality, coding correctness, coherence and safety/refusal-regression gates. Keep text-change rate and regex matches as descriptive diagnostics only. Persist per-prompt and aggregate results with effect sizes, confidence intervals and corrected multiple-comparison statistics when many candidates are screened.
- [ ] Implement discovery/search/validation/test splits with exact prompt hashes, overlap detection and no benchmark contamination; hold out distinct prompt sets and unseen topics for final claims. Test score calibration against independently annotated examples.
- [ ] Add signed dose-response and **counterfactual controls**: zero-dose identity; reversed sign; randomized channels matched for layer/activation/norm; same-cardinality controls; null effects and noisy-feature controls. Quantify causal specificity and off-target damage.
- [ ] Bound runtime intervention by calibrated activation distributions, per-feature and whole-layer delta norms, NaN/Inf checks and saturation tests; stop or reject incoherent/unstable outputs. Explicitly separate steering *effect* from quality *retention*.
- [ ] Add deterministic, device-qualified real-checkpoint smoke/acceptance runs and immutable replay artifacts. CI may use small public fixtures; full-model acceptance must be recorded as a separate measured hardware result, not inferred from synthetic tests.

**P0 exit gate:** Repeatable signed dose-response on at least one verified real model; actual hook delivery and exact checkpoint identity proven; targeted effect exceeds same-budget controls on an independent test set without unacceptable KEEP regressions.

### P1 — Automated neuron/feature discovery and causal optimisation

- [x] **P1 screening increment:** `aegis-neurosurgery steer-rank` consumes the existing measured `profile-mlp` artifact using restricted tensor loading, per-layer-normalized KEEP/DROP mean-absolute activation contrasts, bounded top-k selection and negative-control candidates, with source hash and explicit correlational-only interpretation (PR #13). **Causal confirmation, effect-score validation and held-out replication are not completed.**
- [ ] Accept a structured target specification (\`KEEP\`, \`CHANGE\`, \`DROP\`) with positive/negative example sets, expected response properties, behavioural scoring rubric and protected capability thresholds.
- [ ] Implement contrastive activation capture (dense MLP intermediate, residual channels, router/top-k expert activity where supported) with bounded memory, chunking, streaming statistics, CPU offload, sampled token positions and resumable manifests.
- [ ] Rank candidates by robust activation separation, effect-size normalisation and stability across prompts. Distinguish correlation-based ranking from causal effect; provide a calibrated \`rank-neurons\` command using the Stage 4C \`rank_mlp_channels()\` and activation patching primitives.
- [ ] Add a causal evaluation pass for each top candidate: intervention vs counterfactual controls at matched dose, per-task KEEP metrics, confidence intervals, evidence attribution, and search-budget accounting. Reject candidates whose measured benefit is explained by general output degradation.
- [ ] Search compatible signed dose values via bounded grid/coordinate search first; add Optuna/TPE or equivalent budgeted optimisation optionally. Record trials, seeds, objective weights, search constraints and Pareto front: target gain vs KEEP loss vs incoherence vs runtime cost.
- [ ] Search pairs and small interacting sets with ablation interactions, synergy/antagonism metrics, beam or greedy expansion, hard cardinality budgets and matched random-group controls. Integrate Stage 4C \`evaluate_interacting_components()\` and Stage 4D optimisation rather than duplicating them.
- [ ] Confirm discovered recipes on independent held-out prompts and repeat seeds. Export provenance-labelled, immutable, model-bound JSON recipes only after acceptance; unqualified recipes remain explicitly experimental.
- [ ] Add a repeatable campaign CLI: \`discover -> rank -> causal-test -> optimise -> validate -> publish\`, with crash/resume, quota/VRAM limits, progress, structured result files and regression-gated promotion.

**P1 exit gate:** On a real checkpoint, automatic discovery yields a statistically supported behaviour-specific neuron/combination with measurable uplift over same-compute random candidates, and preserved independent KEEP performance.

### P2 — Adaptive steering, adapter coverage and deployment

- [ ] Generalise the existing hook/runtime operation contract (not a parallel steering engine) to **residual-vector steering, CAA directions, SAE latent interventions, attention features and verified MoE-expert steering**. Each adapter must report the exact mathematical edit and validate topology/tensor semantics.
- [ ] Implement conditional steering gates by token position, prefill/decode phase, activation threshold, output uncertainty and bounded deterministic predicate. Document gate decision traces, use-case-specific preconditions, propagation under caching and rollback on failure.
- [ ] Support stateful schedules only with explicitly defined state lifetime and deterministic resets; add compositional ordering/alias checks for simultaneous interventions and test effects on model safety/capability gates.
- [ ] Integrate SAE feature dictionaries and independently sourced feature descriptions as **candidate annotations**, never authoritative causal labels. Support optional Neuronpedia feature lookup and graph context with a pinned API schema, cache/provenance, offline mode and explicit opt-in to remote requests (avoid leaking private prompts or models).
- [ ] Add efficient device mapping, optional CPU/GPU offload, bounded selective tensor copying, correct quantized intervention sites, model-parallel execution and per-device memory/latency accounting. Require real kernel tests for compiled/fused, GGUF or OpenVINO paths; unsupported paths fail closed.
- [ ] Measure native vs offloaded vs compiled prefill/decode latency, throughput, peak RAM/VRAM and effect quality. Separate execution-mode changes from the proposed speedups and tie deployment claims to actual hardware.
- [ ] Add a runtime session API for **scoped** steering that cannot leak hooks, feature state or RNG across users/jobs; provide cancellation and reset invariants. Run adversarial tests for cross-run contamination and incompatible model adapters.

**P2 exit gate:** Each claimed intervention type is semantically equivalent to its reference implementation on verified supported models, with measured isolation, memory and latency; unsupported runtimes are clearly identified.

### P3 — Interactive operator console and experiment lifecycle

- [ ] Build a lightweight UI within the **existing** \`src/aegis_lab/gui/\` and API surfaces, backed by the same audited CLI/library. Expose model+checkpoint identity, preset provenance, neuron coordinates and signed strengths.
- [ ] Show an interactive model/neuron browser with searchable layer/channel indexes, activation distributions, feature descriptions and data-supported uncertainty; do not label third-party interpretations as verified facts.
- [ ] Support adjustable strength sliders, prompt/generated-token gates, single/pair/composite recipes, and immutable baseline-vs-steered response panels including token-level divergence, effect scores and confidence bands.
- [ ] Add discovery campaign tracking, candidate ranking, Pareto views, per-prompt trace drilldown, comparison history and signed recipe import/export with schema validation.
- [ ] Enforce local-only execution by default and explicit operator control for remote model/prompt/feature API access. Preserve access control, encrypted or private artifact paths, audit log and redaction of sensitive prompts.
- [ ] Provide reproducible CLI/API/UI parity tests, screenshot-based GUI tests only where suitable, cancellation/resume tests, and minimal portable export formats.

**P3 exit gate:** An operator can load a verified checkpoint, perform measured A/B tests, auto-discover/optimise candidates, validate independently and export a trustworthy preset through the UI without writing custom scripts.

### Cross-cutting acceptance and maintenance

- [ ] Reuse \`editing/inference_steering.py\` and \`editing/neurosurgery/stage4c_causal.py\` responsibilities rather than maintaining two incompatible hook lifecycles; document the selected canonical API.
- [ ] Merge only validated work into \`main\`. Protect existing cryptographic/storage and model-editing guarantees when reconciling stale historical branches. Archive superseded proposals or transplant discrete features rather than overwriting newer implementations.
- [ ] Report exact command, git revision, model/weight/tokenizer hashes, dataset hashes, baseline/control definitions, confidence intervals, runtime mode and test artifacts for every promoted research claim. Never equate unit-test passing with a reproduced persona.


## Documentation refresh

- [x] Reframe README and docs around the all-in-one model brain surgery kit.
- [x] Document core versus advanced dispatch, placeholder/simulation limits and adapter boundaries.
- [x] Replace stale setup/architecture/schema claims and fix current examples to relative paths.
- [x] Verify command parsing, relative links and focused committed-source tests; record results in the verification guide.

## Acceptance policy

A promoted result must retain its untouched parent, inputs and replay plan; state target effect and collateral damage; pass independent held-out and reload tests; and identify unresolved limits. Report physical parameter/storage changes separately from measured runtime gains. Workflow completion and synthetic metrics cannot satisfy quality gates.
