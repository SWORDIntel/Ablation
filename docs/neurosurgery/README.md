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
8. read-only inventory of likely modality-specific branches.

The physical operations are deliberately behind architecture adapters. Search can be generic; changing shapes cannot.

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

# 4. Build one reproducible plan
aegis-neurosurgery plan \
  --profile runs/residual/profile.json \
  --mlp-search runs/mlp-search/mlp_search.json \
  --attention-search runs/attn-search/attention_search.json \
  --out runs/plan.yaml \
  --max-mean-kl 0.02

# 5. Physical surgery
aegis-neurosurgery apply \
  --model /models/Qwen3-8B \
  --profile runs/residual/profile.pt \
  --plan runs/plan.yaml \
  --out /models/Qwen3-8B-surgery

# 6. Independent KEEP validation
aegis-neurosurgery validate \
  --base /models/Qwen3-8B \
  --candidate /models/Qwen3-8B-surgery \
  --keep keep-validation.txt \
  --max-mean-kl 0.02
```

For MoE models, add `profile-moe`, `search-moe`, and `--moe-search` to the plan command.

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
