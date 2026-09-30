from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from .adapters import get_attention_adapter, get_mlp_adapter, get_moe_adapter
from .common import file_sha256, get_layers, nested_getattr
from .preview import build_preview


_ALLOWED_ROOT = {"version", "drop_layers", "directional", "mlp", "attention", "moe"}


def _require_mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _layer_ids(values: Any, count: int, label: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list of layer IDs")
    ids = [_integer(value, label) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate layer IDs")
    if any(value < 0 or value >= count for value in ids):
        raise IndexError(f"{label} contains an ID outside [0,{count})")
    return sorted(ids)


def _layer_index_map(value: Any, expected_ids: list[int], label: str) -> list[torch.Tensor]:
    mapping = _require_mapping(value, label)
    normalized: dict[int, Any] = {}
    for raw_id, indices in mapping.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
            raise ValueError(f"{label} keys must be layer IDs")
        try:
            layer_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} key {raw_id!r} is not an integer layer ID") from exc
        if str(layer_id) != str(raw_id) and not isinstance(raw_id, int):
            raise ValueError(f"{label} key {raw_id!r} is not a canonical integer layer ID")
        if layer_id in normalized:
            raise ValueError(f"{label} repeats layer ID {layer_id}")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"{label}[{layer_id}] must be a non-empty list of indices")
        clean = [_integer(index, f"{label}[{layer_id}]") for index in indices]
        if len(clean) != len(set(clean)):
            raise ValueError(f"{label}[{layer_id}] contains duplicate indices")
        normalized[layer_id] = torch.tensor(sorted(clean), dtype=torch.long)
    actual_ids = sorted(normalized)
    if actual_ids != sorted(expected_ids):
        raise ValueError(
            f"{label} must specify every supported layer exactly once; "
            f"expected {sorted(expected_ids)}, got {actual_ids}"
        )
    return [normalized[layer_id] for layer_id in expected_ids]


def _same_keep_count(indices: list[torch.Tensor], label: str) -> None:
    counts = {int(value.numel()) for value in indices}
    if len(counts) != 1:
        raise ValueError(f"{label} must keep the same count in each layer, got {sorted(counts)}")


def _validate_indices(indices: list[torch.Tensor], sizes: list[int], label: str) -> None:
    if len(indices) != len(sizes):
        raise ValueError(f"{label} layer count mismatch")
    for index, size in zip(indices, sizes):
        if int(index.min()) < 0 or int(index.max()) >= size:
            raise IndexError(f"{label} has an index outside [0,{size})")


def _write_selector_outputs(
    model,
    selector_file: Path,
    output_dir: str,
    model_path: str,
    profile_path: Optional[str],
) -> dict:
    spec = yaml.safe_load(selector_file.read_text(encoding="utf-8"))
    spec = _require_mapping(spec, "selector document")
    unknown = sorted(set(spec) - _ALLOWED_ROOT)
    if unknown:
        raise ValueError(f"unsupported selector fields: {unknown}")
    if spec.get("version") != 1:
        raise ValueError("selector version must be 1")

    layer_path, layers_obj = get_layers(model)
    layers = list(layers_obj)
    operations: dict[str, Any] = {}
    drop_layers = _layer_ids(spec.get("drop_layers", []), len(layers), "drop_layers")

    directional = spec.get("directional")
    if directional is not None:
        directional = _require_mapping(directional, "directional")
        allowed = {"layers", "targets", "strength", "norm_preserve", "preserve_subspace"}
        extra = sorted(set(directional) - allowed)
        if extra:
            raise ValueError(f"unsupported directional fields: {extra}")
        selected_layers = _layer_ids(directional.get("layers"), len(layers), "directional.layers")
        targets = directional.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("directional.targets must be a non-empty list of module paths")
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("directional.targets must contain non-empty module paths")
        if len(targets) != len(set(targets)):
            raise ValueError("directional.targets contains duplicates")
        if profile_path is None:
            raise ValueError("--profile is required when selectors include directional edits")
        for layer_id in selected_layers:
            for target in targets:
                module = nested_getattr(layers[layer_id], target)
                weight = getattr(module, "weight", None)
                if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                    raise ValueError(f"layer {layer_id} target {target} must expose a 2D weight")
        operations["ablation"] = {
            "layers": selected_layers,
            "targets": targets,
            "strength": float(directional.get("strength", 0.5)),
            "norm_preserve": bool(directional.get("norm_preserve", True)),
            "preserve_subspace": bool(directional.get("preserve_subspace", True)),
        }
    else:
        operations["ablation"] = {"layers": []}

    selection_payloads: dict[str, dict] = {}

    if "mlp" in spec:
        cfg = _require_mapping(spec["mlp"], "mlp")
        if set(cfg) != {"keep_indices"}:
            raise ValueError("mlp selector requires only keep_indices")
        adapter = get_mlp_adapter(model)
        triplets = adapter.mlp_triplets(model)
        indices = _layer_index_map(cfg["keep_indices"], list(range(len(triplets))), "mlp.keep_indices")
        sizes = [int(item.gate.weight.shape[0]) for item in triplets]
        _validate_indices(indices, sizes, "mlp.keep_indices")
        _same_keep_count(indices, "mlp.keep_indices")
        selection_payloads["mlp"] = {"version": 1, "keep_indices": indices}

    if "attention" in spec:
        cfg = _require_mapping(spec["attention"], "attention")
        if set(cfg) != {"keep_groups"}:
            raise ValueError("attention selector requires only keep_groups")
        adapter = get_attention_adapter(model)
        packs = adapter.attention_packs(model)
        layer_ids = [int(pack.layer_index) if hasattr(pack, "layer_index") else index for index, pack in enumerate(packs)]
        indices = _layer_index_map(cfg["keep_groups"], layer_ids, "attention.keep_groups")
        sizes = [int(pack.num_key_value_heads) for pack in packs]
        _validate_indices(indices, sizes, "attention.keep_groups")
        _same_keep_count(indices, "attention.keep_groups")
        selection_payloads["attention"] = {"version": 1, "keep_groups": indices}

    if "moe" in spec:
        cfg = _require_mapping(spec["moe"], "moe")
        if set(cfg) != {"keep_experts"}:
            raise ValueError("moe selector requires only keep_experts")
        adapter = get_moe_adapter(model)
        blocks = adapter.moe_blocks(model)
        layer_ids = [int(block.layer_index) for block in blocks]
        indices = _layer_index_map(cfg["keep_experts"], layer_ids, "moe.keep_experts")
        sizes = [len(block.experts) for block in blocks]
        _validate_indices(indices, sizes, "moe.keep_experts")
        _same_keep_count(indices, "moe.keep_experts")
        selection_payloads["moe"] = {"version": 1, "keep_experts": indices}

    if not drop_layers and not directional and not selection_payloads:
        raise ValueError("selector document contains no edits")

    out_dir = Path(output_dir).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")

    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}.", dir=str(out_dir.parent)) as temp:
        stage = Path(temp)
        structured = {}
        for name, payload in selection_payloads.items():
            selection_path = stage / f"{name}_selection.pt"
            torch.save(payload, selection_path)
            structured[name] = {
                "enabled": True,
                "selection": selection_path.name,
                "selection_sha256": file_sha256(selection_path),
            }

        plan = {
            "version": 3,
            "source_model": model_path,
            "selector_spec": selector_file.name,
            "selector_spec_sha256": file_sha256(selector_file),
            "source_profile": str(Path(profile_path).resolve()) if profile_path else None,
            "drop_layers": drop_layers,
            "ablation": operations["ablation"],
            "structured": structured,
        }
        plan_path = stage / "surgery_plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

        preview = build_preview(model, str(plan_path), profile_path)
        (stage / "preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
        os.replace(stage, out_dir)

    return {
        "output_dir": str(out_dir),
        "plan": str(out_dir / "surgery_plan.yaml"),
        "preview": str(out_dir / "preview.json"),
        "operations": preview["summary"]["operation_count"],
        "plan_sha256": file_sha256(out_dir / "surgery_plan.yaml"),
        "selector_spec_sha256": file_sha256(selector_file),
    }


def run_select(
    model_path: str,
    selector_path: str,
    output_dir: str,
    profile_path: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    from .validate import _load_hf

    model, _ = _load_hf(model_path, device)
    return _write_selector_outputs(model, Path(selector_path), output_dir, model_path, profile_path)
