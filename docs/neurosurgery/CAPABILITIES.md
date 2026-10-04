# Capability and interface boundaries

The all-in-one model brain surgery kit has multiple entry points. Choose one explicitly rather than assuming a single launcher wires every stage together.

## Core CLI

`aegis-neurosurgery` and `python3 -m aegis_lab.editing.neurosurgery` dispatch to `cli.py`.

Commands: `inventory`, `profile`, `profile-mlp`, `search-mlp`, `profile-attention`, `search-attention`, `profile-moe`, `search-moe`, `search-layers`, `search-layers-greedy`, `optimize`, `plan`, `select`, `preview`, `apply`, `validate`.

This is the explicit structural/directional path documented in the [manual](README.md). Preview is read-only. Physical edits require supported geometry and floating-point projections. Unsupported architectures need adapters and reload tests.

### Known committed-source issues

At implementation revision `551992e`, `search-mlp` calls `load_tensor_artifact` without importing it and fails with `NameError`. The quick start uses `profile-mlp` followed by `optimize`, which has the loader import and performs constrained MLP candidate search. The extended manual's separate `search-mlp` command requires that import fix; an existing local patch is outside this docs-only change.

The optimizer writes plan version 4, while preview accepts only versions 1–3. The [example converter](../examples/optimizer_to_selectors.py) extracts exact checksummed structural indices into typed selector YAML. Run `select` to produce a supported version-3 plan and geometry preview, then apply that plan. Do not simply relabel a plan version or bypass preview. Directional and MoE edits are rejected by this small dense-selection example. MoE selectors need adapter-specific source-layer IDs; prepare them explicitly rather than assuming block positions equal transformer-layer IDs.

## Advanced module CLI and libraries

```bash
python3 -m aegis_lab.editing.neurosurgery.cli_extended --help
```

This dispatcher handles `patch`, `recover`, `hypertune`, `quantize`, `export-runtime`, `ampute`, `unlearn` and `workflow`. Its help includes inherited core parsers, but core commands have no advanced dispatch handler. Invoke them with the core CLI.

Stage modules provide separate contracts for provenance/task scoring, operation composition, causal patching, hypertuning, LoRA recovery, quantization, export, modality dependencies and knowledge-editing experiments. A passing fixture is evidence for that tested function and geometry, not universal model support. Advanced handlers need workload-specific qualification; see [advanced instruments](ADVANCED.md).

## Workflow integration limits

The `workflow.py` library, advanced `workflow` handler and `pipeline_runner.py` campaign runner are distinct interfaces.

- `cli_extended.handle_workflow` currently chooses layer 0 in its default localization step; its apply step records an `applied` status without calling the core plan applicator. Its recovery path builds data from a demonstration prompt. The dry-run branch emits a report without loading or validating the model's geometry or checking task acceptance.
- `CampaignPipelineRunner._step_profile` generates seeded random directions/bases and fixed-form metrics instead of collecting real model activations. `_step_recover` reports constant losses and rebound values; its synthetic loss is not connected to model parameters.
- Campaign handler registration enables real implementations to replace defaults, but the operator must supply and verify those handlers. Campaign completion, resume journals and report files do not make default metrics empirical measurements.

Use the explicit core commands for structural surgery. Call advanced instruments with real data and independent gates. Do not present default wrapper reports as evidence that an end-to-end edit/recovery campaign succeeded.

## Architecture and result limits

Dense adapters expect separate gated-MLP and Q/K/V/O projections with compatible tensor geometry. Attention cuts preserve Q-to-KV group mapping; MoE requires supported `ModuleList` experts and a linear gate/router. Packed, fused, latent-attention and custom modality implementations may be unsupported.

The core selector format retains global uniform-dimension restrictions and fixed edit ordering. General Stage 4B remapping and operation contracts are library functionality, not automatically core-selector support.

INT4 packing and INT8 quantization helpers do not imply acceleration or compatibility with GGUF/AWQ/GPTQ or a particular kernel. SafeTensors storage alone does not prove a stock runtime can execute the exported checkpoint. Reload and benchmark the actual target.

Structural byte estimates describe parameter storage in the loaded dtype. Task success, quality, memory and latency require separate measurement. Suppressed outputs do not prove knowledge erasure.
