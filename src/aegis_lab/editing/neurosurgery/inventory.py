from __future__ import annotations

import json
from pathlib import Path

from .common import parameter_bytes, resolve_device
from .validate import _load_hf


SUSPICIOUS_MODALITY_TOKENS = (
    "vision",
    "visual",
    "image",
    "audio",
    "speech",
    "projector",
    "mm_projector",
    "multimodal",
    "vae",
)


def run_inventory(model_path: str, out_path: str | None = None, device: str = "cpu") -> dict:
    """Inventory large named modules and modality-looking branches.

    This is intentionally read-only. A module appearing here is not proof that it
    can be deleted while retaining stock Transformers reload semantics.
    """
    device = resolve_device(device)
    model, _ = _load_hf(model_path, device)
    rows = []
    for name, module in model.named_children():
        rows.append(
            {
                "name": name,
                "type": type(module).__name__,
                "parameter_bytes": parameter_bytes(module),
                "looks_modality_specific": any(t in name.lower() for t in SUSPICIOUS_MODALITY_TOKENS),
            }
        )
    rows.sort(key=lambda x: x["parameter_bytes"], reverse=True)
    payload = {
        "model": model_path,
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
        "top_level_modules": rows,
        "note": "Read-only inventory; shape-changing modality amputation requires an architecture adapter.",
    }
    if out_path:
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
