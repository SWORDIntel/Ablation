from __future__ import annotations

from typing import Optional, Union

import json
from pathlib import Path

import torch

from .common import LOG, get_layers, load_prompts, parameter_bytes, resolve_device, set_layers
from .validate import _load_hf, compare_logprobs, next_token_logprobs


def run_layer_search(
    model_path: str,
    keep_path: str,
    out_dir: str,
    batch_size: int = 4,
    max_prompts: Optional[int] = 64,
    device: str = "auto",
) -> None:
    device = resolve_device(device)
    prompts = load_prompts(keep_path, max_prompts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = _load_hf(model_path, device)
    layer_path, module_list = get_layers(model)
    original = list(module_list)
    baseline = next_token_logprobs(model, tok, prompts, batch_size)

    results = []
    try:
        for idx, layer in enumerate(original):
            candidate = original[:idx] + original[idx + 1 :]
            set_layers(model, layer_path, candidate)
            try:
                candidate_lp = next_token_logprobs(model, tok, prompts, batch_size)
                metrics = compare_logprobs(baseline, candidate_lp)
                results.append(
                    {
                        "layer": idx,
                        "params_saved": sum(p.numel() for p in layer.parameters()),
                        "bytes_saved_in_loaded_dtype": parameter_bytes(layer),
                        **metrics,
                    }
                )
                LOG.info(
                    "layer %d: mean_kl=%.6f top1=%.3f params_saved=%d",
                    idx,
                    metrics["mean_kl"],
                    metrics["top1_agreement"],
                    results[-1]["params_saved"],
                )
            finally:
                set_layers(model, layer_path, original)
    finally:
        set_layers(model, layer_path, original)

    results.sort(key=lambda x: (x["mean_kl"], -x["params_saved"]))
    payload = {
        "model": model_path,
        "layer_path": layer_path,
        "prompts": len(prompts),
        "results": results,
    }
    (out / "layer_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
