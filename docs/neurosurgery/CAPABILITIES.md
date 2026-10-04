# Capability and interface boundaries

The all-in-one model brain surgery kit has a core entry point and an installed advanced entry point. Both dispatch core commands; the advanced entry also dispatches the stage instruments.

## Core CLI

`aegis-neurosurgery` and `python3 -m aegis_lab.editing.neurosurgery` dispatch to `cli.py`.

Commands: `inventory`, `profile`, `profile-mlp`, `search-mlp`, `profile-attention`, `search-attention`, `profile-moe`, `search-moe`, `search-layers`, `search-layers-greedy`, `optimize`, `plan`, `select`, `preview`, `apply`, `validate`.

This is the explicit structural/directional path documented in the [manual](README.md). Preview is read-only. Physical edits require supported geometry and floating-point projections. Unsupported architectures need adapters and reload tests.

### Supported plan versions

Standalone `search-mlp` imports the restricted tensor-artifact loader. Preview accepts surgery-plan versions 1–4, including the optimizer's unchanged version-4 output and its version-2 selection artifacts. Pass `optimized_plan.yaml` directly to `preview` and `apply`; no selector conversion or version relabeling is required.

Version-4 support keeps the same selection checksum, bounds, uniform geometry, router top-k and required-profile checks. Preview remains read-only. Supported MoE selections retain adapter block ordering and original transformer-layer IDs, including models with non-MoE layers between expert blocks. Directional edits still require their residual profile. Unknown plan versions fail.

The older [dense selector conversion example](../examples/optimizer_to_selectors.py) remains optional for callers needing typed selector YAML. Its dense-only limitations do not apply to direct plan preview/apply.

## Advanced module CLI and libraries

```bash
aegis-neurosurgery-advanced --help
# Equivalent module entry:
python3 -m aegis_lab.editing.neurosurgery.cli_extended --help
```

This dispatcher handles `patch`, `recover`, `hypertune`, `quantize`, `export-runtime`, `ampute`, `unlearn` and `workflow`. It also dispatches inherited core commands through the same core handler. Reinstall the package after updating to create `aegis-neurosurgery-advanced`.

Stage modules provide separate contracts for provenance/task scoring, operation composition, causal patching, hypertuning, LoRA recovery, quantization, export, modality dependencies and knowledge-editing experiments. A passing fixture is evidence for that tested function and geometry, not universal model support. Advanced handlers need workload-specific qualification; see [advanced instruments](ADVANCED.md).

## Workflow integration limits

The `workflow.py` library, advanced `workflow` handler and `pipeline_runner.py` campaign runner are distinct interfaces.

- The advanced `workflow` command requires a local stock-HF checkpoint. It measures localization from KEEP/DROP activations, previews the actual surgery plan, calls core apply, trains LoRA on real tokenized KEEP/CHANGE data, saves merged checkpoints and independently reloads for held-out KEEP validation. `--validation-keep` must be disjoint from KEEP training prompts. Dry-run loads and previews actual geometry but does not apply, recover or claim task acceptance.
- Workflow configuration can provide `profile`, `max_mean_kl` and `max_drop_rebound`. Explicit plans require their linked selections and residual profile when directional edits are present. CHANGE data must contain JSON/JSONL `prompt`/`target` records; recovery masks prompt labels and supervises target tokens.
- Campaign defaults now collect real activations, use the checked core applicator, train real LoRA recovery, quantize actual weights and benchmark the actual runtime. Calibration data/tokenizer are required. Missing or non-finite configured measurements fail; no random profiles or constant quality/performance values substitute for measurements. Resume binds local inputs and rejects changed existing edit order/parameters.
- The older workflow library requires datasets or an explicit evaluator and an explicit plan/editor. It no longer fabricates baseline/candidate/test reports or demo recovery tokens. Its convenience hypertuning mode rejects execution without a measured evaluator; the dedicated instrument remains available. Topology-changing rollback restores the baseline module structure and verifies exact original weights.

Installed commands now require local input paths and always write operator audits: exact input/output hashes, code/adapter hashes, options, package versions and producer links. Output paths cannot overlap inputs. Audited upstream artifacts must match their producer checksums and model bindings. Legacy artifacts remain readable but their missing producer lineage is explicit. Workflow/campaign APIs also bind model/tokenizer state, datasets, implementations, callback source and edit order; resume checks these bindings. These records do not qualify task behavior or hidden external dependencies in custom callbacks. `qualified` stays false for the convenience workflow; `heldout_validation_passed` records its requested KL check. See [qualification evidence](QUALIFICATION.md) for the trained Qwen diagnostic and unresolved acceptance.

## Architecture and result limits

Dense adapters expect separate gated-MLP and Q/K/V/O projections with compatible tensor geometry. Attention cuts preserve Q-to-KV group mapping; MoE requires supported `ModuleList` experts and a linear gate/router. Packed, fused, latent-attention and custom modality implementations may be unsupported.

The core selector format retains global uniform-dimension restrictions and fixed edit ordering. Preview/apply validate resolved core operations through Stage 4B alias and composition checks. They also accept Stage 4B version-4 `operations` plans with arbitrary original-index remapping. Unified plans cannot mix core sections or export runtime-only hooks. Their preview executes on an isolated model copy, so allow RAM for that copy.

INT4 packing and INT8 quantization helpers do not imply acceleration or compatibility with GGUF/AWQ/GPTQ or a particular kernel. SafeTensors storage alone does not prove a stock runtime can execute the exported checkpoint. Reload and benchmark the actual target.

Structural byte estimates describe parameter storage in the loaded dtype. Task success, quality, memory and latency require separate measurement. Suppressed outputs do not prove knowledge erasure.

Operator audits are saved beside `--out` (directory `operator_audit.json`, or file `.operator_audit.json` suffix). Commands without output paths use `AEGIS_AUDIT_DIR`, defaulting to the user state directory. Download remote checkpoints to a local directory before invoking an installed operator. Numerical stage APIs alone are not independently audited operator runs.
