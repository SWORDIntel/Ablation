# Stage 4: Constrained Structural Optimizer

Stage 4 searches **combined** structural edits instead of treating layer, MLP, attention, and MoE decisions as independent.

## Search state

A candidate is represented by four discrete levels:

```text
(layer_drop_level, mlp_keep_level, attention_keep_level, moe_keep_level)
```

The level values index ordered search grids. Ratio grids always include `1.0`, which is the exact identity/no-cut state.

Whole-layer deletion uses the ordered candidates supplied by `search-layers-greedy` or `search-layers`. A layer level of `N` drops the first `N` entries from that order.

## Candidate evaluation

The untouched model's next-token distribution on the KEEP corpus is captured once.

For each candidate:

1. MLP channel masks are attached before `down_proj`.
2. Attention masks are attached before `o_proj`.
3. MoE router outputs are masked before routing.
4. Candidate transformer blocks are removed from the live module list.
5. The candidate next-token distribution is compared with the untouched baseline.
6. The original layer list and hooks are restored.

No candidate checkpoint is written during search.

## Constraints

A candidate is feasible when:

```text
mean_KL <= max_mean_kl
top1_agreement >= min_top1_agreement
```

The final physical checkpoint should still be validated on a held-out KEEP set after materialization. One-token KL is a search gate, not a complete behavioral proof.

## Objectives

Among feasible candidates the optimizer tracks a Pareto front over:

```text
maximize resident parameter bytes removed
maximize estimated dense linear MACs removed per token
minimize KEEP mean KL
```

The default selected candidate is lexicographic:

1. most resident bytes removed;
2. most estimated dense MACs removed;
3. lowest KL;
4. highest top-1 agreement.

This matches the deployment goal: fit the strongest surviving model into a fixed memory target without pretending that every parameter contributes equally to runtime.

## Why masked latency is not an objective

Reversible masks preserve the original tensor shapes. Timing a masked candidate would therefore mostly measure the original matrix multiplications plus hook overhead.

Stage 4 reports:

- exact loaded-dtype parameter bytes that the corresponding physical cut will remove;
- an overlap-corrected dense linear-work proxy.

Real latency belongs **after** `optimized_plan.yaml` has been applied and the shortened checkpoint has been reloaded.

MoE expert deletion is treated primarily as a resident-memory saving. Active top-k expert compute is rerouted rather than necessarily reduced, so the compute proxy only credits the narrower router unless whole layers are also removed.

## Frontier strategy

`--strategy frontier` is the default.

It starts from the identity model and expands one structural dimension at a time. A candidate is expanded only when it satisfies the quality constraints. Among the current frontier, higher estimated byte/MAC savings are evaluated first.

This uses the usually monotonic relationship between more deletion and more model drift to avoid evaluating the full Cartesian product.

## Exhaustive strategy

`--strategy exhaustive` ranks the full discrete grid by estimated structural savings and evaluates up to `--max-trials` states. The identity baseline is always evaluated first.

Use exhaustive search when the space is small or when interactions appear non-monotonic.

## Example

```bash
aegis-neurosurgery optimize \
  --model /models/Qwen3-8B \
  --keep data/keep-calibration.txt \
  --out runs/qwen3-stage4 \
  --mlp-profile runs/mlp/mlp_profile.pt \
  --attention-profile runs/attention/attention_profile.pt \
  --layer-search runs/layers/greedy_layer_search.json \
  --mlp-ratios 1,0.95,0.90,0.85,0.80 \
  --attention-ratios 1,0.875,0.75,0.625 \
  --max-drop-layers 3 \
  --max-mean-kl 0.02 \
  --min-top1-agreement 0.95 \
  --max-trials 64
```

Outputs:

```text
runs/qwen3-stage4/
├── optimization.json
├── optimized_plan.yaml
├── stage4.mlp.pt          # only when MLP surgery selected
├── stage4.attention.pt    # only when attention surgery selected
└── stage4.moe.pt          # only when MoE surgery selected
```

The selection artifacts are SHA-256 bound into the plan. `apply` verifies them before altering tensor shapes.

## Materialization

Stage-4 plans contain no directional edit by default, so a residual profile is not required:

```bash
aegis-neurosurgery apply \
  --model /models/Qwen3-8B \
  --plan runs/qwen3-stage4/optimized_plan.yaml \
  --out /models/Qwen3-8B-surgery
```

Then validate the physical result on a separate KEEP set:

```bash
aegis-neurosurgery validate \
  --base /models/Qwen3-8B \
  --candidate /models/Qwen3-8B-surgery \
  --keep data/keep-validation.txt \
  --max-mean-kl 0.02
```

## Design lineage

The optimizer follows the same general idea that made Heretic useful: expensive model edits are proposed in a search loop and judged against a preservation objective rather than chosen by hand. AEGIS applies that philosophy to physical tensor-structure deletion and keeps the implementation independent and dependency-light.
