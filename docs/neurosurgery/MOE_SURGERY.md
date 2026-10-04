# Mixture-of-Experts Surgery

MoE models often contain far more total parameters than are active for one token. That makes unused experts one of the highest-value structural targets.

## Profiling

For every supported router, AEGIS records two KEEP/DROP signals per expert:

- mean softmax routing probability;
- frequency of selection in router top-k.

The two are normalized and combined into an importance signal. Candidate selection favors KEEP-active experts and penalizes experts whose activity is disproportionately DROP-specific.

## Search

Search is non-destructive. The router output for removed experts is replaced with `-inf` before downstream softmax/top-k routing. This makes the candidate behave as if those experts did not exist while leaving tensors unchanged.

## Physical cut

For a passing candidate:

1. router weight rows are sliced to the retained experts;
2. router bias is sliced when present;
3. the expert `ModuleList` is rebuilt in retained-router order;
4. block expert-count attributes are repaired where present;
5. model config `num_local_experts` or `num_experts` is repaired;
6. `num_experts_per_tok` is preserved and the surgery is rejected if fewer experts would remain than the routing top-k requires.

Shared experts are not removed by this adapter.

## Supported shape

The current adapter requires an HF-style MoE block with:

```text
block.experts -> torch.nn.ModuleList
block.gate or block.router -> linear module with [num_experts, hidden] weight
```

Mixtral-like blocks fit this contract. Compatible Qwen MoE derivatives may also fit through feature detection. Architectures with packed expert tensors or custom router semantics require a dedicated adapter.

## Operator sequence

Use the matching profile/search commands from the [manual](README.md), then the [joint optimizer](OPTIMIZER.md) or typed selectors. Preview the exact plan, apply to a separate checkpoint, and run [independent validation](VALIDATION.md) after reload. Channel/group/expert ranking proposes cuts; it does not prove target behavior is localized there.
