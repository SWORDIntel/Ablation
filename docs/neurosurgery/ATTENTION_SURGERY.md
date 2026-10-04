# Attention Surgery

## Why prune GQA groups rather than arbitrary heads

For standard multi-head attention, each query head has its own key/value head. For grouped-query attention (GQA), several query heads share one key/value head. Stock Transformers represents this with:

```text
num_attention_heads = Hq
num_key_value_heads = Hkv
head_dim = D
query group size = Hq / Hkv
```

Deleting arbitrary Q heads while retaining arbitrary K/V heads can change the original Q→KV mapping. AEGIS therefore treats one **KV head plus its contiguous group of Q heads** as the physical pruning unit.

For selected groups `G`:

```text
q_proj: keep rows for every Q head belonging to G
k_proj: keep rows for KV heads G
v_proj: keep rows for KV heads G
o_proj: keep columns for every retained Q head
```

`head_dim` and `hidden_size` are unchanged. `num_attention_heads` and `num_key_value_heads` are reduced together, preserving the original group size.

## Search equivalence

During search, AEGIS does not reshape tensors. It masks the selected head-group contribution immediately before `o_proj`. Remaining heads are unaffected and the candidate can be compared with the untouched model cheaply.

After a candidate passes the KL gate, the exact group indices are stored in the plan artifact and the Q/K/V/O tensors are physically sliced.

## Limitations

The adapter intentionally rejects attention implementations whose Q/K/V/O tensor geometry does not match `heads × head_dim`. Architectures with fused QKV matrices, latent attention, head-specific nonlinearities, or non-contiguous GQA mappings need dedicated adapters.

## Operator sequence

Use the matching profile/search commands from the [manual](README.md), then the [joint optimizer](OPTIMIZER.md) or typed selectors. Preview the exact plan, apply to a separate checkpoint, and run [independent validation](VALIDATION.md) after reload. Channel/group/expert ranking proposes cuts; it does not prove target behavior is localized there.
