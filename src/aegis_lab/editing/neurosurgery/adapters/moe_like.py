from __future__ import annotations

import torch

from .base import ArchitectureAdapter, MoEBlock
from ..common import get_layers


def _replace_parameter(module, name: str, value: torch.Tensor) -> None:
    old = getattr(module, name)
    requires_grad = bool(getattr(old, "requires_grad", False))
    setattr(module, name, torch.nn.Parameter(value.contiguous(), requires_grad=requires_grad))


def _find_moe_block(layer):
    candidates = []
    for name in ("block_sparse_moe", "mlp", "feed_forward"):
        block = getattr(layer, name, None)
        if block is not None:
            candidates.append(block)
    for block in candidates:
        experts = getattr(block, "experts", None)
        router = getattr(block, "gate", None)
        if router is None:
            router = getattr(block, "router", None)
        if isinstance(experts, torch.nn.ModuleList) and router is not None and hasattr(router, "weight"):
            return block, router, experts
    return None


class MoELikeAdapter(ArchitectureAdapter):
    """Physical expert surgery for HF-style ModuleList + linear-router MoE blocks."""

    name = "moe-like"

    def supports_moe(self, model) -> bool:
        try:
            _, layers = get_layers(model)
            return any(_find_moe_block(layer) is not None for layer in layers)
        except Exception:
            return False

    def moe_blocks(self, model) -> list[MoEBlock]:
        _, layers = get_layers(model)
        out: list[MoEBlock] = []
        for i, layer in enumerate(layers):
            found = _find_moe_block(layer)
            if found is None:
                continue
            block, router, experts = found
            if router.weight.ndim != 2 or not router.weight.dtype.is_floating_point:
                raise ValueError(f"layer {i}: router must be an unquantized 2D floating-point linear weight")
            if int(router.weight.shape[0]) != len(experts):
                raise ValueError(
                    f"layer {i}: router outputs {router.weight.shape[0]} experts but ModuleList has {len(experts)}"
                )
            out.append(MoEBlock(i, block, router, experts))
        if not out:
            raise ValueError("no supported MoE blocks found")
        return out

    def apply_moe_selection(self, model, keep_experts: list[torch.Tensor]) -> dict:
        blocks = self.moe_blocks(model)
        if len(keep_experts) != len(blocks):
            raise ValueError(f"selection has {len(keep_experts)} MoE layers, model has {len(blocks)}")
        keep_counts = {int(x.numel()) for x in keep_experts}
        if len(keep_counts) != 1:
            raise ValueError(
                "stock Transformers reload requires one global expert count; "
                f"got kept expert counts {sorted(keep_counts)}"
            )
        new_experts = next(iter(keep_counts))
        if new_experts <= 0:
            raise ValueError("cannot remove every MoE expert")

        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        top_k = None
        for candidate in (text_cfg, cfg):
            if candidate is not None and hasattr(candidate, "num_experts_per_tok"):
                top_k = int(candidate.num_experts_per_tok)
                break
        if top_k is not None and new_experts < top_k:
            raise ValueError(f"cannot keep {new_experts} experts when num_experts_per_tok={top_k}")

        params_before = 0
        params_after = 0
        old_counts = []
        for block_info, raw_idx in zip(blocks, keep_experts):
            old_count = len(block_info.experts)
            old_counts.append(old_count)
            idx = torch.sort(raw_idx.to(dtype=torch.long)).values
            if idx.ndim != 1 or idx.numel() != new_experts:
                raise ValueError(f"layer {block_info.layer_index}: malformed expert selection")
            if idx.min().item() < 0 or idx.max().item() >= old_count:
                raise IndexError(f"layer {block_info.layer_index}: expert index outside valid range")
            if torch.unique(idx).numel() != idx.numel():
                raise ValueError(f"layer {block_info.layer_index}: duplicate expert index")

            params_before += sum(p.numel() for p in block_info.router.parameters(recurse=False))
            params_before += sum(p.numel() for expert in block_info.experts for p in expert.parameters())

            router_idx = idx.to(block_info.router.weight.device)
            _replace_parameter(
                block_info.router,
                "weight",
                block_info.router.weight.data.index_select(0, router_idx),
            )
            bias = getattr(block_info.router, "bias", None)
            if bias is not None:
                _replace_parameter(block_info.router, "bias", bias.data.index_select(0, router_idx.to(bias.device)))
            if hasattr(block_info.router, "out_features"):
                block_info.router.out_features = new_experts

            selected = [block_info.experts[int(i)] for i in idx.tolist()]
            block_info.block.experts = torch.nn.ModuleList(selected)
            for attr in ("num_experts", "num_local_experts"):
                if hasattr(block_info.block, attr):
                    setattr(block_info.block, attr, new_experts)

            params_after += sum(p.numel() for p in block_info.router.parameters(recurse=False))
            params_after += sum(p.numel() for expert in block_info.block.experts for p in expert.parameters())

        touched = 0
        for candidate in (text_cfg, cfg):
            if candidate is None:
                continue
            for attr in ("num_local_experts", "num_experts"):
                if hasattr(candidate, attr):
                    setattr(candidate, attr, new_experts)
                    touched += 1
        if touched == 0:
            raise ValueError("model config has no num_local_experts/num_experts field to repair")

        return {
            "adapter": self.name,
            "moe_layers": [b.layer_index for b in blocks],
            "old_expert_counts": old_counts,
            "new_expert_count": new_experts,
            "num_experts_per_tok": top_k,
            "parameters_before": params_before,
            "parameters_after": params_after,
            "parameters_removed": params_before - params_after,
        }
