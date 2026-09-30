from __future__ import annotations

from typing import Optional, Union

import json
from pathlib import Path

import torch
from tqdm import tqdm

from .common import LOG, batches, get_layers, load_prompts, resolve_device, trust_remote_code_enabled
from .math_ops import contrast_direction, preservation_basis


def _load_hf(model_path: str, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers is required: pip install -e .") from e
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code_enabled())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = {"trust_remote_code": trust_remote_code_enabled()}
    if device != "cpu":
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, tokenizer


def _collect(model, tokenizer, prompts: list[str], batch_size: int, device: str, basis_samples: int):
    layer_path, layers = get_layers(model)
    n_layers = len(layers)
    sum_hidden = [None] * (n_layers + 1)
    sum_delta_ratio = torch.zeros(n_layers, dtype=torch.float64)
    count = 0
    keep_samples: list[list[torch.Tensor]] = [[] for _ in range(n_layers + 1)]

    with torch.inference_mode():
        for batch in tqdm(list(batches(prompts, batch_size)), desc="profile", unit="batch"):
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            hs = out.hidden_states
            if hs is None or len(hs) != n_layers + 1:
                raise RuntimeError(f"expected {n_layers + 1} hidden-state tensors, got {0 if hs is None else len(hs)}")
            bsz = enc["input_ids"].shape[0]
            for i, h in enumerate(hs):
                last = h[:, -1, :].detach().float().cpu()
                part = last.sum(dim=0, dtype=torch.float64)
                sum_hidden[i] = part if sum_hidden[i] is None else sum_hidden[i] + part
                remaining = max(0, basis_samples - len(keep_samples[i]))
                if remaining:
                    keep_samples[i].extend([x.clone() for x in last[:remaining]])
            for i in range(n_layers):
                pre = hs[i][:, -1, :].float()
                post = hs[i + 1][:, -1, :].float()
                ratio = (post - pre).norm(dim=1) / pre.norm(dim=1).clamp_min(1e-8)
                sum_delta_ratio[i] += ratio.double().sum().cpu()
            count += bsz
            del out, hs

    means = [x.float() / count for x in sum_hidden]
    delta = (sum_delta_ratio / count).float()
    samples = [torch.stack(x) if x else torch.empty((0, means[i].numel())) for i, x in enumerate(keep_samples)]
    return layer_path, means, delta, samples


def run_profile(
    model_path: str,
    keep_path: str,
    drop_path: str,
    out_dir: str,
    batch_size: int = 4,
    basis_rank: int = 32,
    basis_samples: int = 64,
    max_prompts: Optional[int] = None,
    device: str = "auto",
) -> None:
    device = resolve_device(device)
    keep = load_prompts(keep_path, max_prompts)
    drop = load_prompts(drop_path, max_prompts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    LOG.info("loading %s on %s", model_path, device)
    model, tokenizer = _load_hf(model_path, device)
    layer_path, keep_means, keep_delta, keep_samples = _collect(
        model, tokenizer, keep, batch_size, device, basis_samples
    )
    _, drop_means, drop_delta, _ = _collect(model, tokenizer, drop, batch_size, device, 0)

    directions = []
    bases = []
    metrics = []
    for i in range(len(keep_delta)):
        direction, separation = contrast_direction(keep_means[i + 1], drop_means[i + 1])
        basis = preservation_basis(keep_samples[i + 1], basis_rank)
        directions.append(direction.cpu())
        bases.append(basis.cpu())
        k = float(keep_delta[i])
        d = float(drop_delta[i])
        metrics.append(
            {
                "layer": i,
                "keep_delta_ratio": k,
                "drop_delta_ratio": d,
                "drop_to_keep_ratio": d / max(k, 1e-8),
                "contrast_separation": separation,
            }
        )

    meta = {
        "model": model_path,
        "layer_path": layer_path,
        "num_layers": len(metrics),
        "keep_prompts": len(keep),
        "drop_prompts": len(drop),
        "basis_rank": basis_rank,
        "layers": metrics,
    }
    (out / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    torch.save(
        {
            "model": model_path,
            "layer_path": layer_path,
            "directions": directions,
            "preserve_bases": bases,
            "metrics": metrics,
        },
        out / "profile.pt",
    )
    LOG.info("wrote %s", out)
