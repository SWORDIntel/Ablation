from __future__ import annotations

import json
from pathlib import Path

from .common import LOG, get_layers, load_prompts, parameter_bytes, resolve_device, set_layers
from .validate import _load_hf, compare_logprobs, next_token_logprobs


def run_greedy_layer_search(
    model_path: str,
    keep_path: str,
    out_dir: str,
    max_mean_kl: float = 0.02,
    max_layers: int = 4,
    batch_size: int = 4,
    max_prompts: int | None = 64,
    device: str = "auto",
) -> None:
    """Greedy interacting layer deletion search.

    Every candidate is measured against the original model distribution, not the
    previous reduced model. After a layer is accepted the remaining candidates are
    re-evaluated, so interactions are observed rather than assumed independent.
    """
    device = resolve_device(device)
    prompts = load_prompts(keep_path, max_prompts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = _load_hf(model_path, device)
    layer_path, module_list = get_layers(model)
    original_layers = list(module_list)
    baseline = next_token_logprobs(model, tok, prompts, batch_size)

    active = list(range(len(original_layers)))
    selected: list[int] = []
    steps = []
    try:
        for step in range(max_layers):
            trials = []
            for original_idx in list(active):
                candidate_ids = [i for i in active if i != original_idx]
                candidate_layers = [original_layers[i] for i in candidate_ids]
                set_layers(model, layer_path, candidate_layers)
                candidate_lp = next_token_logprobs(model, tok, prompts, batch_size)
                metrics = compare_logprobs(baseline, candidate_lp)
                layer = original_layers[original_idx]
                trials.append(
                    {
                        "layer": original_idx,
                        "mean_kl": metrics["mean_kl"],
                        "max_kl": metrics["max_kl"],
                        "top1_agreement": metrics["top1_agreement"],
                        "params_incremental": sum(p.numel() for p in layer.parameters()),
                        "bytes_incremental": parameter_bytes(layer),
                    }
                )
            passing = [x for x in trials if x["mean_kl"] <= max_mean_kl]
            if not passing:
                LOG.info("greedy layer search stopped at step %d: no candidate passes KL gate", step + 1)
                steps.append({"step": step + 1, "accepted": None, "trials": trials})
                break
            passing.sort(key=lambda x: (-x["bytes_incremental"], x["mean_kl"]))
            chosen = passing[0]
            selected.append(int(chosen["layer"]))
            active.remove(int(chosen["layer"]))
            set_layers(model, layer_path, [original_layers[i] for i in active])
            steps.append({"step": step + 1, "accepted": chosen, "trials": trials})
            LOG.info(
                "greedy step %d accepted layer %d; cumulative removed=%s mean_kl=%.6f",
                step + 1,
                chosen["layer"],
                selected,
                chosen["mean_kl"],
            )
    finally:
        set_layers(model, layer_path, original_layers)

    payload = {
        "version": 1,
        "model": model_path,
        "layer_path": layer_path,
        "max_mean_kl": max_mean_kl,
        "selected_layers": selected,
        "steps": steps,
    }
    (out / "greedy_layer_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
