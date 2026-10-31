"""Correlational neuron screening from existing measured MLP profiles.

This is an inexpensive *candidate generator*, not a causal or behavioural
score. It reuses the gated-MLP KEEP/DROP measurements from mlp.profile-mlp.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import torch

from .common import file_sha256, load_tensor_artifact

_EPS = 1e-8


def _as_vector(x: Any, layer: int, label: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.ndim != 1 or not x.numel():
        raise ValueError(f"Layer {layer}: {label} must be a nonempty 1D activation tensor")
    vec = x.detach().float().cpu()
    if not torch.isfinite(vec).all() or torch.any(vec < 0):
        raise ValueError(f"Layer {layer}: {label} contains invalid nonnegative activation statistics")
    return vec


def rank_steering_candidates(profile: dict[str, Any], *, top_k: int = 25,
                             source_sha256: Optional[str] = None) -> dict[str, Any]:
    """Screen neurons on relative mean absolute activation, per layer.

    score_target = DROP / mean(DROP) - KEEP / mean(KEEP)
    score_protected = -score_target

    The signed output is a contrast statistic, *not* a steering dose.
    Cross-layer magnitudes are heuristic; do not infer causality from ranks.
    """
    if isinstance(top_k, bool) or top_k < 1 or top_k > 1000:
        raise ValueError("top_k must be within [1,1000]")
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError("Unsupported MLP profile schema")
    keep = profile.get("keep_importance")
    drop = profile.get("drop_importance")
    if not isinstance(keep, (list, tuple)) or not isinstance(drop, (list, tuple)) or not keep or len(keep) != len(drop):
        raise ValueError("MLP profile requires matched KEEP and DROP vectors")
    target: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    geometry = []
    for layer, (raw_k, raw_d) in enumerate(zip(keep, drop)):
        k = _as_vector(raw_k, layer, "keep_importance")
        d = _as_vector(raw_d, layer, "drop_importance")
        if k.shape != d.shape:
            raise ValueError(f"Layer {layer}: KEEP/DROP widths disagree")
        if k.mean().item() <= _EPS or d.mean().item() <= _EPS:
            raise ValueError(f"Layer {layer}: uninformative zero-activation statistics")
        kn = k / k.mean()
        dn = d / d.mean()
        delta = dn - kn
        geometry.append({"layer": layer, "channels": int(k.numel())})
        # Avoid converting large layer-wide tensors to Python objects.
        for sign, output in ((1, target), (-1, protected)):
            scores = delta * sign
            indices = torch.topk(scores, k=min(top_k, int(scores.numel())), largest=True).indices
            for index in indices.tolist():
                score = float(scores[index])
                if not math.isfinite(score) or score <= 0:
                    continue
                i = int(index)
                output.append({
                    "layer": layer,
                    "neuron": i,
                    "contrast_score": round(score, 9),
                    "relative_keep_activity": round(float(kn[i]), 9),
                    "relative_target_activity": round(float(dn[i]), 9),
                    "activity_ratio_target_to_keep": round(float((dn[i] + _EPS) / (kn[i] + _EPS)), 9),
                    "direction": "target_enriched" if sign == 1 else "keep_enriched",
                })
    def top(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda x: (
            -x["contrast_score"], x["layer"], x["neuron"],
        ))[:top_k]
    return {
        "schema_version": 1,
        "screening_only": True,
        "model_asserted_by_profile": profile.get("model"),
        "adapter": profile.get("adapter"),
        "source_profile_sha256": source_sha256,
        "profile_keep_sha256": profile.get("profile_keep_sha256"),
        "profile_drop_sha256": profile.get("profile_drop_sha256"),
        "metric": "per-layer DROP-relative-mean-absolute-activation minus KEEP-relative-mean-absolute-activation",
        "interpretation": (
            "Correlational screening only; no causal effect, dose direction, "
            "persona specificity, statistical significance or capability retention is established."
        ),
        "requires_causal_confirmation": True,
        "geometry": geometry,
        "target_enriched": top(target),
        "keep_enriched_controls": top(protected),
    }


def run_steer_rank_command(args) -> dict[str, Any]:
    source = Path(args.profile)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate ranking: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("Use an explicit .json output path")
    data = load_tensor_artifact(source)  # torch.load(..., weights_only=True)
    report = rank_steering_candidates(
        data, top_k=args.top_k, source_sha256=file_sha256(source),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {"out": str(output), "target_candidates": len(report["target_enriched"]),
            "keep_controls": len(report["keep_enriched_controls"])}
