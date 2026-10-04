from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import yaml

from .adapters import get_attention_adapter, get_mlp_adapter, get_moe_adapter
from .common import LOG, file_sha256, get_layers, load_tensor_artifact, nested_getattr, parameter_bytes, resolve_device, set_layers
from .math_ops import apply_constrained_directional_surgery
from .validate import _load_hf


def _apply_directional(model, layers, profile: dict, plan: dict) -> list[dict]:
    ablation = plan.get("ablation") or {}
    ablate_layers = [int(x) for x in ablation.get("layers", [])]
    targets = list(ablation.get("targets", ["self_attn.o_proj", "mlp.down_proj"]))
    strength = float(ablation.get("strength", 0.5))
    norm_preserve = bool(ablation.get("norm_preserve", True))
    use_basis = bool(ablation.get("preserve_subspace", True))
    applied = []

    with torch.no_grad():
        for idx in ablate_layers:
            if idx < 0 or idx >= len(layers):
                raise IndexError(f"ablation layer {idx} out of range")
            direction = profile["directions"][idx]
            basis = profile["preserve_bases"][idx] if use_basis else None
            layer = layers[idx]
            for path in targets:
                try:
                    module = nested_getattr(layer, path)
                except AttributeError:
                    LOG.warning("layer %d missing target %s", idx, path)
                    continue
                if not hasattr(module, "weight"):
                    LOG.warning("layer %d target %s has no weight", idx, path)
                    continue
                old = module.weight.data
                try:
                    new = apply_constrained_directional_surgery(
                        old,
                        direction,
                        strength=strength,
                        preserve_basis=basis,
                        norm_preserve=norm_preserve,
                    )
                except ValueError as exc:
                    LOG.warning("skip layer %d %s: %s", idx, path, exc)
                    continue
                module.weight.data.copy_(new.to(device=old.device, dtype=old.dtype))
                applied.append({"layer": idx, "target": path, "strength": strength})
    return applied


def _resolve_selection(plan_path: Path, cfg: dict) -> tuple[Path, dict]:
    raw = Path(str(cfg["selection"]))
    selection = raw if raw.is_absolute() else plan_path.parent / raw
    if not selection.exists():
        raise FileNotFoundError(selection)
    expected = cfg.get("selection_sha256")
    actual = file_sha256(selection)
    if expected and actual != expected:
        raise ValueError(f"selection checksum mismatch: {selection}")
    payload = load_tensor_artifact(selection)
    return selection, payload


def _apply_structured(model, plan_file: Path, plan: dict) -> dict:
    structured = plan.get("structured") or {}
    log: dict[str, object] = {}

    mlp_cfg = structured.get("mlp") or {}
    if mlp_cfg.get("enabled"):
        path, payload = _resolve_selection(plan_file, mlp_cfg)
        adapter = get_mlp_adapter(model)
        summary = adapter.apply_mlp_selection(model, payload["keep_indices"])
        summary.update({"selection": str(path), "selection_sha256": file_sha256(path)})
        log["mlp"] = summary
        LOG.info("MLP surgery removed %d parameters", summary["parameters_removed"])

    attention_cfg = structured.get("attention") or {}
    if attention_cfg.get("enabled"):
        path, payload = _resolve_selection(plan_file, attention_cfg)
        adapter = get_attention_adapter(model)
        summary = adapter.apply_attention_selection(model, payload["keep_groups"])
        summary.update({"selection": str(path), "selection_sha256": file_sha256(path)})
        log["attention"] = summary
        LOG.info("attention surgery removed %d parameters", summary["parameters_removed"])

    moe_cfg = structured.get("moe") or {}
    if moe_cfg.get("enabled"):
        path, payload = _resolve_selection(plan_file, moe_cfg)
        adapter = get_moe_adapter(model)
        summary = adapter.apply_moe_selection(model, payload["keep_experts"])
        summary.update({"selection": str(path), "selection_sha256": file_sha256(path)})
        log["moe"] = summary
        LOG.info("MoE surgery removed %d parameters", summary["parameters_removed"])

    return log


def apply_to_model(model, profile_path, plan_path):
    """Preflight every operation before applying the fixed core edit order."""
    from .preview import build_preview
    build_preview(model, plan_path, profile_path)
    plan_file = Path(plan_path)
    plan = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
    if "operations" in plan:
        from .stage4b_contract import UnifiedPlan, apply_plan
        result = apply_plan(model, UnifiedPlan.from_dict(plan))
        return dict(unified=[op.to_dict() for op in result.applied_operations],
                    dropped_layers=result.removed_components.get("layers", []),
                    details=result.details)
    profile = load_tensor_artifact(profile_path) if (plan.get("ablation") or {}).get("layers") else None
    layer_path, layers_obj = get_layers(model)
    layers = list(layers_obj)

    operation_log: dict[str, object] = {
        "directional": _apply_directional(model, layers, profile, plan) if profile is not None else [],
        "structured": _apply_structured(model, plan_file, plan),
    }

    layer_path, layers_obj = get_layers(model)
    layers = list(layers_obj)
    drop = sorted(set(int(x) for x in plan.get("drop_layers", [])))
    if drop:
        if drop[0] < 0 or drop[-1] >= len(layers):
            raise IndexError(f"drop_layers out of range: {drop}")
        drop_set = set(drop)
        set_layers(model, layer_path, [layer for i, layer in enumerate(layers) if i not in drop_set])
        LOG.info("dropped layers: %s", drop)
    operation_log["dropped_layers"] = drop

    return operation_log


def run_apply(model_path: str, profile_path: Optional[str], plan_path: str, out_dir: str, device: str = "auto") -> None:
    from .measured import bind_plan_inputs, verify_binding
    source = Path(model_path).resolve()
    destination = Path(out_dir).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Candidate output must be separate from the source checkpoint")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Candidate output directory must be empty")
    binding = bind_plan_inputs(model_path, plan_path, profile_path)
    device = resolve_device(device)
    plan_file = Path(plan_path)
    plan = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
    ablation = plan.get("ablation") or {}
    directional_layers = [int(x) for x in ablation.get("layers", [])]
    if directional_layers:
        if not profile_path:
            raise ValueError("--profile is required when plan contains directional ablation layers")
        profile = load_tensor_artifact(profile_path)
    else:
        profile = None
    model, tokenizer = _load_hf(model_path, device)
    params_before = sum(p.numel() for p in model.parameters())
    bytes_before = parameter_bytes(model)
    layer_path, _ = get_layers(model)
    operation_log = apply_to_model(model, profile_path, plan_path)

    params_after = sum(p.numel() for p in model.parameters())
    bytes_after = parameter_bytes(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    verify_binding(binding)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    manifest = {
        "version": int(plan.get("version", 3)),
        "source_model": model_path,
        "binding": binding,
        "edit_order": ([{"kind": op["kind"], "target_path": op["target_path"], "order": op.get("order")}
                        for op in operation_log["unified"]] if "unified" in operation_log else
                       ["directional", "mlp", "attention", "moe", "layer_removal"]),
        "adapter": {"architecture": getattr(model.config, "model_type", type(model).__name__),
                    "implementation": file_sha256(Path(__file__).with_name("apply.py"))},
        "profile": str(profile_path) if profile_path else None,
        "plan": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "layer_path": layer_path,
        "operations": operation_log,
        "remaining_layers": len(get_layers(model)[1]),
        "parameters_before": params_before,
        "parameters_after": params_after,
        "parameters_removed": params_before - params_after,
        "loaded_parameter_bytes_before": bytes_before,
        "loaded_parameter_bytes_after": bytes_after,
        "loaded_parameter_bytes_removed": bytes_before - bytes_after,
    }
    (out / "neurosurgery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
