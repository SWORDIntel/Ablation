# AEGIS-LAB Model Neurosurgery

Model neurosurgery is the structural counterpart to AEGIS-LAB's existing behavioral ablation tools. Instead of only editing directions in weight space, it profiles a model against **KEEP** and **DROP** workloads, searches reversible masks, measures damage, and only then rewrites tensor shapes.

The design goal is simple: **remove parameters, memory traffic, and capabilities the deployment does not need while preserving measured behavior on the retained workload.**

## Implemented workflow

The `aegis-neurosurgery` CLI currently supports:

1. residual KEEP/DROP profiling and contrastive directional editing;
2. whole-transformer-layer deletion search, including greedy interacting deletion;
3. gated-MLP intermediate-channel profiling, masked search, and physical tensor slicing;
4. GQA/MHA attention-group profiling, masked search, and physical Q/K/V/O slicing;
5. HF-style MoE router profiling, masked search, and physical expert/router slicing;
6. checksum-bound YAML surgery plans;
7. post-surgery KL/top-1 validation against the untouched model;
8. read-only inventory of likely modality-specific branches;
9. Stage-4 joint constrained search across layers, MLP width, attention groups, and MoE experts with Pareto reporting and automatic plan materialization.

The physical operations are deliberately behind architecture adapters. Search can be generic; changing shapes cannot.

Use `select` to compile explicit typed selectors into a checksum-bound surgery plan and read-only preview. Use `preview` before `apply` to inspect the resolved targets, original and resulting tensor shapes, kept indices, tied-parameter aliases, selection checksums, and approximate parameter bytes removed. Preview is read-only; it does not rewrite or reload-test the candidate checkpoint.

## Basic sequence

```bash
# 1. Residual behavior map
aegis-neurosurgery profile \
  --model /models/Qwen3-8B \
  --keep keep.txt \
  --drop drop.txt \
  --out runs/residual

# 2. Dense MLP search
aegis-neurosurgery profile-mlp \
  --model /models/Qwen3-8B --keep keep.txt --drop drop.txt --out runs/mlp

aegis-neurosurgery search-mlp \
  --model /models/Qwen3-8B --keep keep.txt \
  --profile runs/mlp/mlp_profile.pt --out runs/mlp-search \
  --ratios 0.95,0.90,0.85,0.80

# 3. Attention search
aegis-neurosurgery profile-attention \
  --model /models/Qwen3-8B --keep keep.txt --drop drop.txt --out runs/attn

aegis-neurosurgery search-attention \
  --model /models/Qwen3-8B --keep keep.txt \
  --profile runs/attn/attention_profile.pt --out runs/attn-search \
  --ratios 0.875,0.75,0.625

# 4. Stage-4 joint optimizer
aegis-neurosurgery optimize \
  --model /models/Qwen3-8B \
  --keep keep.txt \
  --out runs/opt \
  --mlp-profile runs/mlp/mlp_profile.pt \
  --attention-profile runs/attn/attention_profile.pt \
  --layer-search runs/layers/greedy_layer_search.json \
  --mlp-ratios 1,0.95,0.90,0.85,0.80 \
  --attention-ratios 1,0.875,0.75,0.625 \
  --max-mean-kl 0.02 \
  --min-top1-agreement 0.95 \
  --max-trials 64

# 5. Preview the exact edits without modifying the model
aegis-neurosurgery preview \
  --model /models/Qwen3-8B \
  --plan runs/opt/optimized_plan.yaml

# If the plan includes directional edits, also pass:
#   --profile runs/residual/profile.pt

# 6. Physical surgery from the reviewed selections
aegis-neurosurgery apply \
  --model /models/Qwen3-8B \
  --plan runs/opt/optimized_plan.yaml \
  --out /models/Qwen3-8B-surgery

# 7. Independent KEEP validation
aegis-neurosurgery validate \
  --base /models/Qwen3-8B \
  --candidate /models/Qwen3-8B-surgery \
  --keep keep-validation.txt \
  --max-mean-kl 0.02
```

For MoE models, generate `moe_profile.pt` with `profile-moe` and pass it to `optimize --moe-profile`. The optimizer produces `optimization.json`, a Pareto front, checksum-bound selection artifacts, and `optimized_plan.yaml`.

### Stage-4 search behavior

The default `frontier` strategy starts from the unmodified model and expands only candidates that still satisfy the KEEP constraints. This makes the search practical when the full Cartesian product would be expensive. `--strategy exhaustive` evaluates the highest-value states from the complete grid and always includes the identity baseline.

The optimizer deliberately does **not** report mask-search wall-clock time as expected deployment speedup. Reversible masks leave the original tensor shapes in place, so that timing would be misleading. Instead it reports exact resident parameter bytes removed for the supported physical cuts and an estimated dense linear-MAC reduction per token. Real latency is measured after the selected structure is materialized.

## Invariants

- Search first, cut second.
- Candidate search uses reversible masks rather than writing a checkpoint per trial.
- Structured selection artifacts are SHA-256 bound into the YAML plan.
- Stock Hugging Face reload is preserved by enforcing globally representable dimensions: one `intermediate_size`, one attention head geometry, and one expert count where the model config requires them.
- Quantized tensors are not physically shape-edited. Dequantize or operate on an unquantized checkpoint, then quantize after surgery.
- Layer deletion is applied after per-layer structured surgery so recorded layer indices remain stable.

## Capability preservation

Residual KEEP activations are used to construct an SVD preservation basis. Directional edits can be projected into the null space of that basis and row norms restored. Structural operations use a stronger rule: a candidate must survive measured KEEP validation before its exact indices are materialized into a plan.

## Current architecture coverage

Dense MLP and attention surgery cover the common Llama/Qwen/Mistral/Gemma-style `gate_proj/up_proj/down_proj` and `q_proj/k_proj/v_proj/o_proj` layouts when the tensor geometry matches the declared config.

MoE surgery currently covers HF-style blocks exposing a `ModuleList` named `experts` and a linear `gate` or `router`, including Mixtral-like layouts and compatible derivatives. Shared experts are intentionally left untouched.

Unsupported shapes fail closed rather than guessing.

## Explicit selector files

A selector file uses schema version 1. It accepts layer deletion, directional module paths, and the current structural adapters. Every structural map must name every supported layer and retain the same number of channels/groups/experts per layer so the result remains representable by the model config.

```yaml
version: 1
drop_layers: [5]
directional:
  layers: [2, 3]
  targets: [self_attn.o_proj]
  strength: 0.5
  norm_preserve: true
  preserve_subspace: true
mlp:
  keep_indices:
    "0": [0, 2, 4, 6]
    "1": [1, 3, 5, 7]
```

Compile and inspect it before applying:

```bash
aegis-neurosurgery select --model /models/model --selectors selectors.yaml \
  --profile runs/residual/profile.pt --out runs/compiled
aegis-neurosurgery preview --model /models/model \
  --plan runs/compiled/surgery_plan.yaml --profile runs/residual/profile.pt
# Review surgery_plan.yaml and preview.json, then apply to a separate output path.
```

Structural selector sections are `mlp.keep_indices`, `attention.keep_groups`, and `moe.keep_experts`; each maps layer IDs to explicit retained indices. Omit sections for untouched structures. Directional `targets` are module paths relative to each selected transformer layer and must resolve to 2D weights. The compiler rejects unknown fields, invalid indices, incomplete maps, and an existing output directory. Compilation does not edit model weights. The compiled plan still requires operator review and independent KEEP/target validation; selectors do not establish quality or safe composition.
