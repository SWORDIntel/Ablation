from __future__ import annotations

from typing import Optional, Union

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .adapters import get_adapter
from .common import LOG, batches, load_prompts, resolve_device
from .validate import _load_hf, compare_logprobs, next_token_logprobs

EPS = 1e-8


@dataclass
class _ActivationAccumulator:
    sums: list[torch.Tensor]
    counts: list[int]
    attention_mask: Optional[torch.Tensor] = None


def _sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _accumulate_down_input(acc: _ActivationAccumulator, layer_idx: int, args) -> None:
    x = args[0].detach().float()
    if x.ndim < 2:
        raise RuntimeError(f"unexpected MLP activation shape {tuple(x.shape)}")
    hidden = x.shape[-1]
    flat = x.reshape(-1, hidden)
    mask = acc.attention_mask
    if mask is not None and x.ndim >= 3 and mask.numel() == flat.shape[0]:
        m = mask.reshape(-1).to(device=x.device, dtype=torch.bool)
        flat = flat[m]
    if flat.numel() == 0:
        return
    part = flat.abs().sum(dim=0, dtype=torch.float64).cpu()
    acc.sums[layer_idx] += part
    acc.counts[layer_idx] += int(flat.shape[0])


def _collect_mlp_importance(model, tokenizer, prompts: list[str], batch_size: int, adapter) -> list[torch.Tensor]:
    triplets = adapter.mlp_triplets(model)
    sums = [torch.zeros(int(t.down.weight.shape[1]), dtype=torch.float64) for t in triplets]
    acc = _ActivationAccumulator(sums=sums, counts=[0] * len(triplets))
    handles = []
    for idx, t in enumerate(triplets):
        def hook(_module, args, idx=idx):
            _accumulate_down_input(acc, idx, args)
            return None
        handles.append(t.down.register_forward_pre_hook(hook))

    try:
        with torch.inference_mode():
            for batch in tqdm(list(batches(prompts, batch_size)), desc="mlp-profile", unit="batch"):
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                acc.attention_mask = enc.get("attention_mask")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                acc.attention_mask = enc.get("attention_mask")
                model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
        acc.attention_mask = None

    out = []
    for i, (s, c) in enumerate(zip(acc.sums, acc.counts)):
        if c <= 0:
            raise RuntimeError(f"no activations collected for layer {i}")
        out.append((s / c).float())
    return out


def run_mlp_profile(
    model_path: str,
    keep_path: str,
    drop_path: str,
    out_dir: str,
    batch_size: int = 2,
    max_prompts: Optional[int] = 64,
    device: str = "auto",
) -> None:
    device = resolve_device(device)
    keep = load_prompts(keep_path, max_prompts)
    drop = load_prompts(drop_path, max_prompts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = _load_hf(model_path, device)
    adapter = get_adapter(model)
    keep_imp = _collect_mlp_importance(model, tok, keep, batch_size, adapter)
    drop_imp = _collect_mlp_importance(model, tok, drop, batch_size, adapter)
    if len(keep_imp) != len(drop_imp):
        raise RuntimeError("KEEP/DROP layer mismatch")

    layer_meta = []
    for i, (k, d) in enumerate(zip(keep_imp, drop_imp)):
        if k.shape != d.shape:
            raise RuntimeError(f"layer {i} importance shape mismatch")
        layer_meta.append(
            {
                "layer": i,
                "channels": int(k.numel()),
                "keep_mean_abs_activation": float(k.mean()),
                "drop_mean_abs_activation": float(d.mean()),
                "keep_p95": float(torch.quantile(k, 0.95)),
                "drop_p95": float(torch.quantile(d, 0.95)),
            }
        )

    pt = out / "mlp_profile.pt"
    torch.save(
        {
            "version": 1,
            "model": model_path,
            "adapter": adapter.name,
            "keep_importance": keep_imp,
            "drop_importance": drop_imp,
        },
        pt,
    )
    meta = {
        "version": 1,
        "model": model_path,
        "adapter": adapter.name,
        "keep_prompts": len(keep),
        "drop_prompts": len(drop),
        "artifact": pt.name,
        "artifact_sha256": _sha256(pt),
        "layers": layer_meta,
    }
    (out / "mlp_profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("wrote MLP profile to %s", out)


def select_mlp_channels(
    keep_importance: list[torch.Tensor],
    drop_importance: list[torch.Tensor],
    keep_ratio: float,
    contrast_weight: float = 0.25,
    align_to: int = 64,
) -> list[torch.Tensor]:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0,1]")
    if len(keep_importance) != len(drop_importance):
        raise ValueError("importance layer count mismatch")
    if align_to < 1:
        raise ValueError("align_to must be >= 1")

    selections: list[torch.Tensor] = []
    expected_k = None
    for i, (k_raw, d_raw) in enumerate(zip(keep_importance, drop_importance)):
        k = k_raw.float().clamp_min(0)
        d = d_raw.float().clamp_min(0)
        if k.ndim != 1 or d.shape != k.shape:
            raise ValueError(f"layer {i} malformed importance vectors")
        n = int(k.numel())
        if keep_ratio >= 1.0:
            target = n
        else:
            target = max(1, int(n * keep_ratio))
            if align_to > 1 and target >= align_to:
                target = max(align_to, (target // align_to) * align_to)
            target = min(n, target)
        if expected_k is None:
            expected_k = target
        elif target != expected_k:
            raise ValueError(f"layer {i} target {target} differs from earlier {expected_k}")

        kn = k / k.mean().clamp_min(EPS)
        dn = d / d.mean().clamp_min(EPS)
        keep_score = kn - float(contrast_weight) * torch.relu(dn - kn)
        idx = torch.topk(keep_score, k=target, largest=True, sorted=False).indices
        selections.append(torch.sort(idx.cpu()).values)
    return selections


def _register_mlp_masks(model, adapter, keep_indices: list[torch.Tensor]):
    triplets = adapter.mlp_triplets(model)
    if len(triplets) != len(keep_indices):
        raise ValueError("mask/model layer mismatch")
    handles = []
    for i, (triplet, idx) in enumerate(zip(triplets, keep_indices)):
        n = int(triplet.down.weight.shape[1])
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx.long()] = True

        def hook(_module, args, mask=mask, i=i):
            x = args[0]
            if x.shape[-1] != mask.numel():
                raise RuntimeError(f"layer {i} mask {mask.numel()} != activation {x.shape[-1]}")
            m = mask.to(device=x.device)
            y = x * m.to(dtype=x.dtype)
            return (y, *args[1:])

        handles.append(triplet.down.register_forward_pre_hook(hook))
    return handles


def _estimate_saved(model, adapter, selections: list[torch.Tensor]) -> tuple[int, int]:
    params = 0
    bytes_ = 0
    for t, idx in zip(adapter.mlp_triplets(model), selections):
        old = int(t.gate.weight.shape[0])
        removed = old - int(idx.numel())
        if removed <= 0:
            continue
        hidden = int(t.gate.weight.shape[1])
        p = removed * (hidden + hidden + int(t.down.weight.shape[0]))
        if getattr(t.gate, "bias", None) is not None:
            p += removed
        if getattr(t.up, "bias", None) is not None:
            p += removed
        params += p
        bytes_ += removed * hidden * t.gate.weight.element_size()
        bytes_ += removed * hidden * t.up.weight.element_size()
        bytes_ += removed * int(t.down.weight.shape[0]) * t.down.weight.element_size()
        if getattr(t.gate, "bias", None) is not None:
            bytes_ += removed * t.gate.bias.element_size()
        if getattr(t.up, "bias", None) is not None:
            bytes_ += removed * t.up.bias.element_size()
    return params, bytes_


def run_mlp_search(
    model_path: str,
    keep_path: str,
    profile_path: str,
    out_dir: str,
    keep_ratios: list[float],
    batch_size: int = 2,
    max_prompts: Optional[int] = 64,
    contrast_weight: float = 0.25,
    align_to: int = 64,
    device: str = "auto",
) -> None:
    device = resolve_device(device)
    prompts = load_prompts(keep_path, max_prompts)
    profile = torch.load(profile_path, map_location="cpu", weights_only=False)
    keep_imp = profile["keep_importance"]
    drop_imp = profile["drop_importance"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = _load_hf(model_path, device)
    adapter = get_adapter(model)
    baseline = next_token_logprobs(model, tok, prompts, batch_size)

    results = []
    for ratio in keep_ratios:
        selections = select_mlp_channels(keep_imp, drop_imp, ratio, contrast_weight, align_to)
        handles = _register_mlp_masks(model, adapter, selections)
        try:
            candidate = next_token_logprobs(model, tok, prompts, batch_size)
        finally:
            for h in handles:
                h.remove()
        metrics = compare_logprobs(baseline, candidate)
        params_saved, bytes_saved = _estimate_saved(model, adapter, selections)
        row = {
            "keep_ratio_requested": float(ratio),
            "channels_kept": int(selections[0].numel()),
            "channels_original": int(keep_imp[0].numel()),
            "effective_keep_ratio": float(selections[0].numel() / keep_imp[0].numel()),
            "parameters_saved": params_saved,
            "bytes_saved_in_loaded_dtype": bytes_saved,
            **metrics,
        }
        results.append(row)
        LOG.info(
            "MLP keep=%.3f effective=%.3f KL=%.6f top1=%.3f saved=%.2f MiB",
            ratio,
            row["effective_keep_ratio"],
            row["mean_kl"],
            row["top1_agreement"],
            bytes_saved / (1024 * 1024),
        )

    payload = {
        "version": 1,
        "model": model_path,
        "adapter": adapter.name,
        "profile": str(Path(profile_path).resolve()),
        "profile_sha256": _sha256(profile_path),
        "contrast_weight": float(contrast_weight),
        "align_to": int(align_to),
        "prompts": len(prompts),
        "results": results,
    }
    (out / "mlp_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
