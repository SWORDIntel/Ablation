from __future__ import annotations

from typing import Optional, Union

import json
from pathlib import Path

import torch
import yaml

from .attention import select_attention_groups
from .common import file_sha256
from .mlp import select_mlp_channels
from .moe import select_moe_experts


def _select_drop_layers(
    search_path: Optional[str],
    greedy_search_path: Optional[str],
    max_mean_kl: float,
    max_drop_layers: int,
) -> list[int]:
    if max_drop_layers <= 0:
        return []
    if greedy_search_path:
        greedy = json.loads(Path(greedy_search_path).read_text(encoding="utf-8"))
        return [int(x) for x in greedy.get("selected_layers", [])[:max_drop_layers]]
    if not search_path:
        return []
    search = json.loads(Path(search_path).read_text(encoding="utf-8"))
    passing = [r for r in search["results"] if float(r["mean_kl"]) <= max_mean_kl]
    passing.sort(key=lambda r: (-int(r["params_saved"]), float(r["mean_kl"])))
    return [int(r["layer"]) for r in passing[:max_drop_layers]]


def _choose_structured_candidate(search: dict, max_mean_kl: float) -> Optional[dict]:
    passing = [r for r in search.get("results", []) if float(r["mean_kl"]) <= max_mean_kl]
    if not passing:
        return None
    passing.sort(key=lambda r: (-int(r["bytes_saved_in_loaded_dtype"]), float(r["mean_kl"])))
    return passing[0]


def _load_checked_profile(search: dict, profile_path: Optional[str], label: str) -> tuple[Path, dict]:
    path = Path(profile_path or search.get("profile", ""))
    if not path.exists():
        raise FileNotFoundError(f"{label} profile not found: {path}")
    expected = search.get("profile_sha256")
    if expected and file_sha256(path) != expected:
        raise ValueError(f"{label} profile checksum does not match search artifact")
    return path, torch.load(path, map_location="cpu", weights_only=False)


def _write_selection(out_path: Path, suffix: str, payload: dict) -> tuple[Path, str]:
    selection_path = out_path.with_suffix(suffix)
    torch.save(payload, selection_path)
    return selection_path, file_sha256(selection_path)


def _build_mlp_plan(search_path: Optional[str], profile_path: Optional[str], out: Path, max_mean_kl: float):
    if not search_path:
        return None
    search = json.loads(Path(search_path).read_text(encoding="utf-8"))
    chosen = _choose_structured_candidate(search, max_mean_kl)
    if chosen is None:
        return None
    profile_path_obj, profile = _load_checked_profile(search, profile_path, "MLP")
    contrast_weight = float(search.get("contrast_weight", 0.25))
    align_to = int(search.get("align_to", 64))
    requested = float(chosen["keep_ratio_requested"])
    keep_indices = select_mlp_channels(
        profile["keep_importance"],
        profile["drop_importance"],
        requested,
        contrast_weight=contrast_weight,
        align_to=align_to,
    )
    selection, checksum = _write_selection(
        out,
        ".mlp.pt",
        {
            "version": 1,
            "source_profile": str(profile_path_obj.resolve()),
            "source_profile_sha256": file_sha256(profile_path_obj),
            "keep_ratio_requested": requested,
            "contrast_weight": contrast_weight,
            "align_to": align_to,
            "keep_indices": keep_indices,
        },
    )
    return {
        "enabled": True,
        "selection": selection.name,
        "selection_sha256": checksum,
        "keep_ratio_requested": requested,
        "effective_keep_ratio": float(chosen["effective_keep_ratio"]),
        "estimated_bytes_saved_in_loaded_dtype": int(chosen["bytes_saved_in_loaded_dtype"]),
        "search_mean_kl": float(chosen["mean_kl"]),
        "search_top1_agreement": float(chosen["top1_agreement"]),
    }


def _build_attention_plan(search_path: Optional[str], profile_path: Optional[str], out: Path, max_mean_kl: float):
    if not search_path:
        return None
    search = json.loads(Path(search_path).read_text(encoding="utf-8"))
    chosen = _choose_structured_candidate(search, max_mean_kl)
    if chosen is None:
        return None
    profile_path_obj, profile = _load_checked_profile(search, profile_path, "attention")
    contrast_weight = float(search.get("contrast_weight", 0.25))
    requested = float(chosen["keep_ratio_requested"])
    keep_groups = select_attention_groups(
        profile["keep_importance"],
        profile["drop_importance"],
        requested,
        contrast_weight=contrast_weight,
    )
    selection, checksum = _write_selection(
        out,
        ".attention.pt",
        {
            "version": 1,
            "source_profile": str(profile_path_obj.resolve()),
            "source_profile_sha256": file_sha256(profile_path_obj),
            "keep_ratio_requested": requested,
            "contrast_weight": contrast_weight,
            "keep_groups": keep_groups,
        },
    )
    return {
        "enabled": True,
        "selection": selection.name,
        "selection_sha256": checksum,
        "keep_ratio_requested": requested,
        "effective_keep_ratio": float(chosen["effective_keep_ratio"]),
        "estimated_bytes_saved_in_loaded_dtype": int(chosen["bytes_saved_in_loaded_dtype"]),
        "search_mean_kl": float(chosen["mean_kl"]),
        "search_top1_agreement": float(chosen["top1_agreement"]),
    }


def _build_moe_plan(search_path: Optional[str], profile_path: Optional[str], out: Path, max_mean_kl: float):
    if not search_path:
        return None
    search = json.loads(Path(search_path).read_text(encoding="utf-8"))
    chosen = _choose_structured_candidate(search, max_mean_kl)
    if chosen is None:
        return None
    profile_path_obj, profile = _load_checked_profile(search, profile_path, "MoE")
    contrast_weight = float(search.get("contrast_weight", 0.25))
    requested = float(chosen["keep_ratio_requested"])
    min_keep = int(chosen.get("experts_kept", 1))
    keep_experts = select_moe_experts(
        profile["keep_importance"],
        profile["drop_importance"],
        requested,
        contrast_weight=contrast_weight,
        min_keep=min_keep,
    )
    selection, checksum = _write_selection(
        out,
        ".moe.pt",
        {
            "version": 1,
            "source_profile": str(profile_path_obj.resolve()),
            "source_profile_sha256": file_sha256(profile_path_obj),
            "keep_ratio_requested": requested,
            "contrast_weight": contrast_weight,
            "keep_experts": keep_experts,
        },
    )
    return {
        "enabled": True,
        "selection": selection.name,
        "selection_sha256": checksum,
        "keep_ratio_requested": requested,
        "effective_keep_ratio": float(chosen["effective_keep_ratio"]),
        "estimated_bytes_saved_in_loaded_dtype": int(chosen["bytes_saved_in_loaded_dtype"]),
        "search_mean_kl": float(chosen["mean_kl"]),
        "search_top1_agreement": float(chosen["top1_agreement"]),
    }


def generate_plan(
    profile_path: str,
    search_path: Optional[str],
    out_path: str,
    max_mean_kl: float = 0.02,
    max_drop_layers: int = 1,
    ablation_layers: int = 3,
    ablation_strength: float = 0.5,
    greedy_search_path: Optional[str] = None,
    mlp_search_path: Optional[str] = None,
    mlp_profile_path: Optional[str] = None,
    attention_search_path: Optional[str] = None,
    attention_profile_path: Optional[str] = None,
    moe_search_path: Optional[str] = None,
    moe_profile_path: Optional[str] = None,
) -> dict:
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    drop_layers = _select_drop_layers(search_path, greedy_search_path, max_mean_kl, max_drop_layers)

    ranked = []
    for row in profile["layers"]:
        if int(row["layer"]) in drop_layers:
            continue
        keep = max(float(row["keep_delta_ratio"]), 1e-8)
        drop = float(row["drop_delta_ratio"])
        separation = float(row["contrast_separation"])
        score = max(0.0, (drop - keep) / keep) * separation
        ranked.append((score, int(row["layer"])))
    ranked.sort(reverse=True)
    ablate = [layer for _, layer in ranked[:ablation_layers]]

    structured = {}
    mlp_plan = _build_mlp_plan(mlp_search_path, mlp_profile_path, out, max_mean_kl)
    attention_plan = _build_attention_plan(attention_search_path, attention_profile_path, out, max_mean_kl)
    moe_plan = _build_moe_plan(moe_search_path, moe_profile_path, out, max_mean_kl)
    if mlp_plan:
        structured["mlp"] = mlp_plan
    if attention_plan:
        structured["attention"] = attention_plan
    if moe_plan:
        structured["moe"] = moe_plan

    plan = {
        "version": 3,
        "source_profile": str(Path(profile_path).resolve()),
        "drop_layers": drop_layers,
        "ablation": {
            "layers": ablate,
            "strength": float(ablation_strength),
            "targets": ["self_attn.o_proj", "mlp.down_proj"],
            "norm_preserve": True,
            "preserve_subspace": True,
        },
        "structured": structured,
        "validation_gate": {"max_mean_kl": float(max_mean_kl)},
    }
    out.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return plan
