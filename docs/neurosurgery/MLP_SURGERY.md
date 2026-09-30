# Gated-MLP Channel Surgery

For Llama/Qwen/Gemma/Mistral-style gated MLPs:

```text
gate = gate_proj(x)
up   = up_proj(x)
h    = activation(gate) * up
y    = down_proj(h)
```

An intermediate channel `j` exists in three coupled tensor locations:

```text
gate_proj.weight[j, :]
up_proj.weight[j, :]
down_proj.weight[:, j]
```

Physical channel deletion therefore slices all three together. Deleting only one of them produces either an invalid shape or a different operation.

## Profiling and ranking

For each channel AEGIS measures mean absolute intermediate activation on KEEP and DROP corpora. Let `K` and `D` be the per-channel vectors. Each is normalized by its layer mean, then ranked by:

```text
score = K_norm - lambda * relu(D_norm - K_norm)
```

High KEEP activity protects a channel. Activity disproportionately associated with DROP work makes a channel more expendable.

The ranking is only a proposal generator. Search masks the candidate channels immediately before `down_proj` and measures KEEP KL before any tensor shape is changed.

## Physical operation

For a passing selection:

```text
gate_proj rows -> selected channels
up_proj rows   -> selected channels
down_proj cols -> selected channels
config.intermediate_size -> selected count
```

Stock HF configs expose one `intermediate_size`, so v0.3 requires the same surviving channel count in every edited layer. The actual channel identities may differ by layer.
