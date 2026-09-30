from __future__ import annotations

import torch

from .base import ArchitectureAdapter, AttentionPack, MLPTriplet
from ..common import get_layers


LLAMA_LIKE_MODEL_TYPES = {
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "gemma",
    "gemma2",
    "gemma3_text",
}


def _linear_shape(module) -> tuple[int, int]:
    weight = module.weight
    if not weight.dtype.is_floating_point:
        raise ValueError("physical surgery requires an unquantized floating-point checkpoint")
    if weight.ndim != 2:
        raise ValueError(f"expected 2D linear weight, got {tuple(weight.shape)}")
    return int(weight.shape[0]), int(weight.shape[1])


def _replace_parameter(module, name: str, value: torch.Tensor) -> None:
    old = getattr(module, name)
    requires_grad = bool(getattr(old, "requires_grad", False))
    setattr(module, name, torch.nn.Parameter(value.contiguous(), requires_grad=requires_grad))


def _slice_bias(module, index: torch.Tensor) -> None:
    bias = getattr(module, "bias", None)
    if bias is None:
        return
    idx = index.to(bias.device)
    _replace_parameter(module, "bias", bias.data.index_select(0, idx))


def _config_candidates(model):
    cfg = getattr(model, "config", None)
    return tuple(x for x in (cfg, getattr(cfg, "text_config", None) if cfg is not None else None) if x is not None)


class LlamaLikeAdapter(ArchitectureAdapter):
    name = "llama-like"

    def _model_type_allowed(self, model) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        return model_type in LLAMA_LIKE_MODEL_TYPES or model_type is None

    def supports_mlp(self, model) -> bool:
        if not self._model_type_allowed(model):
            return False
        try:
            _, layers = get_layers(model)
            if not layers:
                return False
            mlp = getattr(layers[0], "mlp", None)
            return mlp is not None and all(
                hasattr(mlp, name) and hasattr(getattr(mlp, name), "weight")
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        except Exception:
            return False

    def supports_attention(self, model) -> bool:
        if not self._model_type_allowed(model):
            return False
        try:
            _, layers = get_layers(model)
            if not layers:
                return False
            attn = getattr(layers[0], "self_attn", None)
            return attn is not None and all(
                hasattr(attn, name) and hasattr(getattr(attn, name), "weight")
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        except Exception:
            return False

    def mlp_triplets(self, model) -> list[MLPTriplet]:
        _, layers = get_layers(model)
        triplets: list[MLPTriplet] = []
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not all(hasattr(mlp, x) for x in ("gate_proj", "up_proj", "down_proj")):
                raise ValueError(f"layer {i} is not a supported gated MLP block")
            gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            gate_out, gate_in = _linear_shape(gate)
            up_out, up_in = _linear_shape(up)
            down_out, down_in = _linear_shape(down)
            if gate_out != up_out or gate_in != up_in or down_in != gate_out:
                raise ValueError(
                    f"layer {i} incompatible MLP shapes: gate={tuple(gate.weight.shape)} "
                    f"up={tuple(up.weight.shape)} down={tuple(down.weight.shape)}"
                )
            triplets.append(MLPTriplet(gate=gate, up=up, down=down))
        if not triplets:
            raise ValueError("no supported MLP triplets found")
        return triplets

    @staticmethod
    def _attention_geometry(model, attn, layer_index: int) -> tuple[int, int, int]:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        num_heads = getattr(attn, "num_heads", None)
        num_kv_heads = getattr(attn, "num_key_value_heads", None)
        head_dim = getattr(attn, "head_dim", None)
        for candidate in (text_cfg, cfg):
            if candidate is None:
                continue
            if num_heads is None:
                num_heads = getattr(candidate, "num_attention_heads", None)
            if num_kv_heads is None:
                num_kv_heads = getattr(candidate, "num_key_value_heads", None)
            if head_dim is None:
                head_dim = getattr(candidate, "head_dim", None)
        if num_heads is None:
            raise ValueError(f"layer {layer_index}: cannot determine num_attention_heads")
        num_heads = int(num_heads)
        num_kv_heads = int(num_kv_heads if num_kv_heads is not None else num_heads)
        if head_dim is None:
            q_out = int(attn.q_proj.weight.shape[0])
            if q_out % num_heads:
                raise ValueError(f"layer {layer_index}: q_proj rows not divisible by num_heads")
            head_dim = q_out // num_heads
        head_dim = int(head_dim)
        if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0 or num_heads % num_kv_heads:
            raise ValueError(
                f"layer {layer_index}: unsupported attention geometry heads={num_heads}, "
                f"kv_heads={num_kv_heads}, head_dim={head_dim}"
            )
        return num_heads, num_kv_heads, head_dim

    def attention_packs(self, model) -> list[AttentionPack]:
        _, layers = get_layers(model)
        packs: list[AttentionPack] = []
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not all(hasattr(attn, x) for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
                raise ValueError(f"layer {i} is not a supported q/k/v/o attention block")
            q, k, v, o = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
            q_out, q_in = _linear_shape(q)
            k_out, k_in = _linear_shape(k)
            v_out, v_in = _linear_shape(v)
            o_out, o_in = _linear_shape(o)
            heads, kv_heads, head_dim = self._attention_geometry(model, attn, i)
            if q_out != heads * head_dim:
                raise ValueError(f"layer {i}: q_proj rows {q_out} != heads*head_dim {heads * head_dim}")
            if k_out != kv_heads * head_dim or v_out != kv_heads * head_dim:
                raise ValueError(f"layer {i}: k/v rows do not match kv_heads*head_dim")
            if o_in != heads * head_dim:
                raise ValueError(f"layer {i}: o_proj input {o_in} != heads*head_dim")
            if len({q_in, k_in, v_in, o_out}) != 1:
                raise ValueError(f"layer {i}: attention hidden dimensions are inconsistent")
            packs.append(
                AttentionPack(
                    q=q,
                    k=k,
                    v=v,
                    o=o,
                    num_heads=heads,
                    num_key_value_heads=kv_heads,
                    head_dim=head_dim,
                )
            )
        if not packs:
            raise ValueError("no supported attention blocks found")
        return packs

    def apply_mlp_selection(self, model, keep_indices: list[torch.Tensor]) -> dict:
        triplets = self.mlp_triplets(model)
        if len(keep_indices) != len(triplets):
            raise ValueError(f"selection has {len(keep_indices)} layers, model has {len(triplets)}")
        keep_counts = {int(x.numel()) for x in keep_indices}
        if len(keep_counts) != 1:
            raise ValueError(
                "stock Transformers reload requires a uniform intermediate_size across layers; "
                f"got keep counts {sorted(keep_counts)}"
            )
        new_intermediate = next(iter(keep_counts))
        if new_intermediate <= 0:
            raise ValueError("cannot remove every MLP channel")

        old_sizes = []
        params_before = 0
        params_after = 0
        for i, (triplet, raw_idx) in enumerate(zip(triplets, keep_indices)):
            gate, up, down = triplet.gate, triplet.up, triplet.down
            old_intermediate = int(gate.weight.shape[0])
            old_sizes.append(old_intermediate)
            idx = raw_idx.to(dtype=torch.long)
            if idx.ndim != 1 or idx.numel() != new_intermediate:
                raise ValueError(f"layer {i} malformed keep index")
            if idx.min().item() < 0 or idx.max().item() >= old_intermediate:
                raise IndexError(f"layer {i} keep index outside [0,{old_intermediate})")
            if torch.unique(idx).numel() != idx.numel():
                raise ValueError(f"layer {i} keep index contains duplicates")
            idx = torch.sort(idx).values

            for module in (gate, up, down):
                params_before += sum(p.numel() for p in module.parameters(recurse=False))
            gi = idx.to(gate.weight.device)
            ui = idx.to(up.weight.device)
            di = idx.to(down.weight.device)
            _replace_parameter(gate, "weight", gate.weight.data.index_select(0, gi))
            _replace_parameter(up, "weight", up.weight.data.index_select(0, ui))
            _replace_parameter(down, "weight", down.weight.data.index_select(1, di))
            _slice_bias(gate, gi)
            _slice_bias(up, ui)
            if hasattr(gate, "out_features"):
                gate.out_features = new_intermediate
            if hasattr(up, "out_features"):
                up.out_features = new_intermediate
            if hasattr(down, "in_features"):
                down.in_features = new_intermediate
            for module in (gate, up, down):
                params_after += sum(p.numel() for p in module.parameters(recurse=False))

        touched = 0
        for candidate in _config_candidates(model):
            if hasattr(candidate, "intermediate_size"):
                candidate.intermediate_size = new_intermediate
                touched += 1
        if touched == 0:
            raise ValueError("model config has no intermediate_size field to repair")
        return {
            "adapter": self.name,
            "layers": len(triplets),
            "old_intermediate_sizes": old_sizes,
            "new_intermediate_size": new_intermediate,
            "parameters_before": params_before,
            "parameters_after": params_after,
            "parameters_removed": params_before - params_after,
        }

    def apply_attention_selection(self, model, keep_groups: list[torch.Tensor]) -> dict:
        packs = self.attention_packs(model)
        if len(keep_groups) != len(packs):
            raise ValueError(f"selection has {len(keep_groups)} layers, model has {len(packs)}")
        keep_counts = {int(x.numel()) for x in keep_groups}
        if len(keep_counts) != 1:
            raise ValueError(
                "stock Transformers reload requires uniform attention geometry across layers; "
                f"got kept KV-group counts {sorted(keep_counts)}"
            )
        new_kv_heads = next(iter(keep_counts))
        if new_kv_heads <= 0:
            raise ValueError("cannot remove every attention group")

        group_sizes = {pack.group_size for pack in packs}
        head_dims = {pack.head_dim for pack in packs}
        if len(group_sizes) != 1 or len(head_dims) != 1:
            raise ValueError("mixed attention geometry is not supported by the stock-config adapter")
        group_size = next(iter(group_sizes))
        head_dim = next(iter(head_dims))
        new_heads = new_kv_heads * group_size

        _, layers = get_layers(model)
        params_before = 0
        params_after = 0
        old_geometry = []
        for i, (pack, raw_groups) in enumerate(zip(packs, keep_groups)):
            old_geometry.append(
                {"num_heads": pack.num_heads, "num_key_value_heads": pack.num_key_value_heads, "head_dim": pack.head_dim}
            )
            groups = torch.sort(raw_groups.to(dtype=torch.long)).values
            if groups.ndim != 1 or groups.numel() != new_kv_heads:
                raise ValueError(f"layer {i} malformed attention group selection")
            if groups.min().item() < 0 or groups.max().item() >= pack.num_key_value_heads:
                raise IndexError(f"layer {i} attention group outside valid range")
            if torch.unique(groups).numel() != groups.numel():
                raise ValueError(f"layer {i} attention group selection contains duplicates")

            q_heads = torch.cat(
                [torch.arange(int(g) * group_size, (int(g) + 1) * group_size, dtype=torch.long) for g in groups]
            )
            q_rows = torch.cat(
                [torch.arange(int(h) * head_dim, (int(h) + 1) * head_dim, dtype=torch.long) for h in q_heads]
            )
            kv_rows = torch.cat(
                [torch.arange(int(g) * head_dim, (int(g) + 1) * head_dim, dtype=torch.long) for g in groups]
            )

            for module in (pack.q, pack.k, pack.v, pack.o):
                params_before += sum(p.numel() for p in module.parameters(recurse=False))

            q_idx = q_rows.to(pack.q.weight.device)
            k_idx = kv_rows.to(pack.k.weight.device)
            v_idx = kv_rows.to(pack.v.weight.device)
            o_idx = q_rows.to(pack.o.weight.device)
            _replace_parameter(pack.q, "weight", pack.q.weight.data.index_select(0, q_idx))
            _replace_parameter(pack.k, "weight", pack.k.weight.data.index_select(0, k_idx))
            _replace_parameter(pack.v, "weight", pack.v.weight.data.index_select(0, v_idx))
            _replace_parameter(pack.o, "weight", pack.o.weight.data.index_select(1, o_idx))
            _slice_bias(pack.q, q_idx)
            _slice_bias(pack.k, k_idx)
            _slice_bias(pack.v, v_idx)

            if hasattr(pack.q, "out_features"):
                pack.q.out_features = new_heads * head_dim
            if hasattr(pack.k, "out_features"):
                pack.k.out_features = new_kv_heads * head_dim
            if hasattr(pack.v, "out_features"):
                pack.v.out_features = new_kv_heads * head_dim
            if hasattr(pack.o, "in_features"):
                pack.o.in_features = new_heads * head_dim

            attn = getattr(layers[i], "self_attn", None)
            if attn is not None:
                for name, value in (
                    ("num_heads", new_heads),
                    ("num_key_value_heads", new_kv_heads),
                    ("num_key_value_groups", group_size),
                ):
                    if hasattr(attn, name):
                        setattr(attn, name, value)

            for module in (pack.q, pack.k, pack.v, pack.o):
                params_after += sum(p.numel() for p in module.parameters(recurse=False))

        touched_heads = 0
        touched_kv = 0
        for candidate in _config_candidates(model):
            if hasattr(candidate, "num_attention_heads"):
                candidate.num_attention_heads = new_heads
                touched_heads += 1
            if hasattr(candidate, "num_key_value_heads"):
                candidate.num_key_value_heads = new_kv_heads
                touched_kv += 1
        if touched_heads == 0:
            raise ValueError("model config has no num_attention_heads field to repair")
        if touched_kv == 0 and new_kv_heads != new_heads:
            raise ValueError("GQA model config has no num_key_value_heads field to repair")

        return {
            "adapter": self.name,
            "layers": len(packs),
            "old_geometry": old_geometry,
            "new_num_attention_heads": new_heads,
            "new_num_key_value_heads": new_kv_heads,
            "head_dim": head_dim,
            "group_size": group_size,
            "parameters_before": params_before,
            "parameters_after": params_after,
            "parameters_removed": params_before - params_after,
        }
