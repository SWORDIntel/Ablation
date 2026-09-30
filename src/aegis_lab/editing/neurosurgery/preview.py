from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .adapters import get_attention_adapter, get_mlp_adapter, get_moe_adapter
from .common import file_sha256, get_layers, load_tensor_artifact, nested_getattr, parameter_bytes, resolve_device
from .validate import _load_hf


def _module_names(model) -> dict[int, str]:
    try:
        pairs = model.named_modules(remove_duplicate=False)
    except TypeError:
        pairs = model.named_modules()
    out = {}
    for name, module in pairs:
        out.setdefault(id(module), name)
    return out


def _parameter_aliases(model) -> dict[int, list[str]]:
    try:
        pairs = model.named_parameters(remove_duplicate=False)
    except TypeError:
        pairs = model.named_parameters()
    aliases: dict[int, list[str]] = {}
    for name, parameter in pairs:
        aliases.setdefault(id(parameter), []).append(name)
    return aliases


def _indices(value: Any, count: int, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        raise ValueError(f"{label} must be an integer tensor")
    idx = value.detach().to(device="cpu", dtype=torch.long)
    if idx.ndim != 1 or idx.numel() == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional tensor")
    if int(idx.min()) < 0 or int(idx.max()) >= count:
        raise IndexError(f"{label} contains an index outside [0,{count})")
    if torch.unique(idx).numel() != idx.numel():
        raise ValueError(f"{label} contains duplicate indices")
    return torch.sort(idx).values


def _selection(plan_file: Path, config: dict, key: str) -> dict:
    if not isinstance(config, dict) or not config.get("enabled"):
        raise ValueError(f"{key} selection must be an enabled mapping")
    if not config.get("selection") or not config.get("selection_sha256"):
        raise ValueError(f"{key} selection path and checksum are required")
    raw = Path(str(config["selection"]))
    path = raw if raw.is_absolute() else plan_file.parent / raw
    if not path.is_file():
        raise FileNotFoundError(f"{key} selection not found: {path}")
    if file_sha256(path) != config["selection_sha256"]:
        raise ValueError(f"{key} selection checksum mismatch")
    payload = load_tensor_artifact(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{key} selection payload must be a mapping")
    return payload


def _module_path(module, names: dict[int, str]) -> str:
    path = names.get(id(module))
    if path is None:
        raise ValueError(f"selected module {type(module).__name__} is not registered in model.named_modules()")
    return path or "<root>"


def _parameter_op(kind: str, layer: int, module, parameter_name: str, new_shape: list[int],
                  indices: list[int], axis: str, names: dict[int, str],
                  aliases: dict[int, list[str]]) -> dict:
    parameter = getattr(module, parameter_name, None)
    if parameter is None:
        raise ValueError(f"{_module_path(module, names)} has no {parameter_name}")
    before = list(parameter.shape)
    after = list(new_shape)
    if len(before) != len(after) or any(x < 0 for x in after):
        raise ValueError(f"invalid planned shape for {_module_path(module, names)}.{parameter_name}")
    removed = (parameter.numel() - int(torch.tensor(after).prod().item())) * parameter.element_size()
    return {
        "kind": kind,
        "layer": layer,
        "parameter": f"{_module_path(module, names)}.{parameter_name}",
        "shape_before": before,
        "shape_after": after,
        "selection_axis": axis,
        "kept_indices": indices,
        "estimated_bytes_removed": removed,
        "parameter_aliases": aliases.get(id(parameter), []),
    }


def build_preview(model, plan_file: str, profile_path: str | None = None) -> dict:
    plan_path = Path(plan_file)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("surgery plan must be a YAML mapping")
    if plan.get("version") not in (1, 2, 3):
        raise ValueError(f"unsupported surgery plan version: {plan.get('version')}")

    layer_path, layers_obj = get_layers(model)
    layers = list(layers_obj)
    names = _module_names(model)
    aliases = _parameter_aliases(model)
    operations: list[dict] = []
    original_bytes = parameter_bytes(model)
    structured = plan.get("structured") or {}
    if not isinstance(structured, dict):
        raise ValueError("plan structured section must be a mapping")
    removed_by_layer: dict[int, int] = {}

    def add(operation: dict) -> None:
        operations.append(operation)
        layer = operation.get("layer")
        if layer is not None:
            removed_by_layer[layer] = removed_by_layer.get(layer, 0) + operation.get("estimated_bytes_removed", 0)

    ablation = plan.get("ablation") or {}
    if not isinstance(ablation, dict):
        raise ValueError("plan ablation section must be a mapping")
    directional_layers = ablation.get("layers", [])
    if not isinstance(directional_layers, list):
        raise ValueError("ablation layers must be a list")
    if len(set(int(x) for x in directional_layers)) != len(directional_layers):
        raise ValueError("ablation layers contain duplicates")
    targets = ablation.get("targets", ["self_attn.o_proj", "mlp.down_proj"])
    if not isinstance(targets, list) or len(set(targets)) != len(targets):
        raise ValueError("ablation targets must be a unique list")
    profile = None
    if directional_layers:
        if not profile_path:
            raise ValueError("--profile is required to preview directional edits")
        profile = load_tensor_artifact(profile_path)
        if not isinstance(profile, dict) or not isinstance(profile.get("directions"), list):
            raise ValueError("directional profile is malformed")
        if len(profile["directions"]) != len(layers):
            raise ValueError("directional profile layer count does not match model")
    for raw_layer in directional_layers:
        layer = int(raw_layer)
        if layer < 0 or layer >= len(layers):
            raise IndexError(f"ablation layer {layer} outside [0,{len(layers)})")
        for target in targets:
            module = nested_getattr(layers[layer], str(target))
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                raise ValueError(f"layer {layer} target {target} has no 2D weight")
            direction = profile["directions"][layer]
            if not isinstance(direction, torch.Tensor) or direction.numel() not in weight.shape:
                raise ValueError(f"direction shape is incompatible with layer {layer} target {target}")
            add({
                "kind": "directional_weight_edit",
                "layer": layer,
                "module": _module_path(module, names),
                "parameter": f"{_module_path(module, names)}.weight",
                "shape_before": list(weight.shape),
                "shape_after": list(weight.shape),
                "direction_length": int(direction.numel()),
                "strength": float(ablation.get("strength", 0.5)),
                "preserve_subspace": bool(ablation.get("preserve_subspace", True)),
                "parameter_aliases": aliases.get(id(weight), []),
            })

    mlp_cfg = structured.get("mlp")
    if mlp_cfg and mlp_cfg.get("enabled"):
        payload = _selection(plan_path, mlp_cfg, "MLP")
        selections = payload.get("keep_indices")
        adapter = get_mlp_adapter(model)
        triplets = adapter.mlp_triplets(model)
        if not isinstance(selections, list) or len(selections) != len(triplets):
            raise ValueError("MLP selection count does not match model layers")
        kept = [_indices(x, int(t.gate.weight.shape[0]), f"MLP layer {i}") for i, (x, t) in enumerate(zip(selections, triplets))]
        counts = {int(x.numel()) for x in kept}
        if len(counts) != 1:
            raise ValueError(f"MLP selection counts are not reload-compatible: {sorted(counts)}")
        for i, (triplet, idx) in enumerate(zip(triplets, kept)):
            k = int(idx.numel())
            for module, parameter_name, shape, axis in (
                (triplet.gate, "weight", [k, triplet.gate.weight.shape[1]], "rows"),
                (triplet.up, "weight", [k, triplet.up.weight.shape[1]], "rows"),
                (triplet.down, "weight", [triplet.down.weight.shape[0], k], "columns"),
            ):
                add(_parameter_op("mlp_channel_slice", i, module, parameter_name, list(shape), idx.tolist(), axis, names, aliases))
            for module in (triplet.gate, triplet.up):
                if getattr(module, "bias", None) is not None:
                    add(_parameter_op("mlp_channel_slice", i, module, "bias", [k], idx.tolist(), "entries", names, aliases))

    attention_cfg = structured.get("attention")
    if attention_cfg and attention_cfg.get("enabled"):
        payload = _selection(plan_path, attention_cfg, "attention")
        selections = payload.get("keep_groups")
        adapter = get_attention_adapter(model)
        packs = adapter.attention_packs(model)
        if not isinstance(selections, list) or len(selections) != len(packs):
            raise ValueError("attention selection count does not match model layers")
        kept = [_indices(x, p.num_key_value_heads, f"attention layer {i}") for i, (x, p) in enumerate(zip(selections, packs))]
        counts = {int(x.numel()) for x in kept}
        if len(counts) != 1 or len({p.group_size for p in packs}) != 1 or len({p.head_dim for p in packs}) != 1:
            raise ValueError("attention selections do not have reload-compatible geometry")
        for i, (pack, groups) in enumerate(zip(packs, kept)):
            group_ids = groups.tolist()
            q_heads = [g * pack.group_size + offset for g in group_ids for offset in range(pack.group_size)]
            q_rows = [h * pack.head_dim + offset for h in q_heads for offset in range(pack.head_dim)]
            kv_rows = [g * pack.head_dim + offset for g in group_ids for offset in range(pack.head_dim)]
            for module, rows, axis in (
                (pack.q, q_rows, "rows"),
                (pack.k, kv_rows, "rows"),
                (pack.v, kv_rows, "rows"),
                (pack.o, q_rows, "columns"),
            ):
                shape = list(module.weight.shape)
                shape[0 if axis == "rows" else 1] = len(rows)
                add(_parameter_op("attention_group_slice", i, module, "weight", shape, rows, axis, names, aliases))
                if axis == "rows" and getattr(module, "bias", None) is not None:
                    add(_parameter_op("attention_group_slice", i, module, "bias", [len(rows)], rows, "entries", names, aliases))

    moe_cfg = structured.get("moe")
    if moe_cfg and moe_cfg.get("enabled"):
        payload = _selection(plan_path, moe_cfg, "MoE")
        selections = payload.get("keep_experts")
        adapter = get_moe_adapter(model)
        blocks = adapter.moe_blocks(model)
        if not isinstance(selections, list) or len(selections) != len(blocks):
            raise ValueError("MoE selection count does not match model")
        kept = [_indices(x, len(b.experts), f"MoE layer {b.layer_index}") for x, b in zip(selections, blocks)]
        counts = {int(x.numel()) for x in kept}
        if len(counts) != 1:
            raise ValueError(f"MoE selection counts are not reload-compatible: {sorted(counts)}")
        for block, idx in zip(blocks, kept):
            k = int(idx.numel())
            weight = block.router.weight
            shape = list(weight.shape)
            shape[0] = k
            add(_parameter_op("moe_router_slice", block.layer_index, block.router, "weight", shape, idx.tolist(), "rows", names, aliases))
            if getattr(block.router, "bias", None) is not None:
                add(_parameter_op("moe_router_slice", block.layer_index, block.router, "bias", [k], idx.tolist(), "entries", names, aliases))
            keep_set = set(idx.tolist())
            for expert_id, expert in enumerate(block.experts):
                if expert_id in keep_set:
                    continue
                expert_name = _module_path(expert, names)
                params = [
                    {"name": name, "shape": list(param.shape), "bytes": param.numel() * param.element_size()}
                    for name, param in expert.named_parameters()
                ]
                add({
                    "kind": "moe_expert_remove",
                    "layer": block.layer_index,
                    "expert": expert_id,
                    "module": expert_name,
                    "parameters": params,
                    "estimated_bytes_removed": sum(x["bytes"] for x in params),
                })

    drop_layers = plan.get("drop_layers", [])
    if not isinstance(drop_layers, list) or len(set(int(x) for x in drop_layers)) != len(drop_layers):
        raise ValueError("drop_layers must be a unique list")
    drop_set = set(int(x) for x in drop_layers)
    for layer in sorted(drop_set):
        if layer < 0 or layer >= len(layers):
            raise IndexError(f"dropped layer {layer} outside [0,{len(layers)})")
        module = layers[layer]
        base_bytes = parameter_bytes(module)
        structured_bytes = removed_by_layer.get(layer, 0)
        add({
            "kind": "transformer_layer_remove",
            "layer": layer,
            "module": f"{layer_path}.{layer}",
            "parameter_bytes_before": base_bytes,
            "prior_structural_bytes_removed": structured_bytes,
            "estimated_bytes_removed": max(0, base_bytes - structured_bytes),
        })

    return {
        "version": 1,
        "model_type": getattr(getattr(model, "config", None), "model_type", type(model).__name__),
        "source_model_bytes": original_bytes,
        "plan": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
        "operation_order": ["directional edits", "MLP/attention/MoE slicing", "layer removal"],
        "operations": operations,
        "summary": {
            "operation_count": len(operations),
            "layers_removed": sorted(drop_set),
            "note": "Byte savings are per-operation estimates; tied parameter aliases are listed and can affect net checkpoint size.",
        },
    }


def run_preview(model_path: str, plan_path: str, profile_path: str | None = None, device: str = "auto") -> dict:
    device = resolve_device(device)
    model, _ = _load_hf(model_path, device)
    return build_preview(model, plan_path, profile_path)
