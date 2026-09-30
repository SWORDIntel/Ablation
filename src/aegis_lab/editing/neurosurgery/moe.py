from __future__ import annotations

from typing import Optional, Union

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .adapters import get_moe_adapter
from .common import LOG, batches, file_sha256, load_tensor_artifact, load_prompts, resolve_device
from .validate import _load_hf, compare_logprobs, next_token_logprobs

EPS = 1e-8


@dataclass
class _RouterAccumulator:
    probability_sums: list[torch.Tensor]
    topk_counts: list[torch.Tensor]
    token_counts: list[int]
    attention_mask: Optional[torch.Tensor] = None


def _configured_top_k(model, expert_count: int) -> int:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    for candidate in (text_cfg, cfg):
        value = getattr(candidate, "num_experts_per_tok", None) if candidate is not None else None
        if value is not None:
            return max(1, min(int(value), expert_count))
    return min(2, expert_count)


def _accumulate_router(acc: _RouterAccumulator, index: int, top_k: int, output) -> None:
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("router output is not a tensor")
    logits = logits.detach().float()
    experts = logits.shape[-1]
    flat = logits.reshape(-1, experts)
    mask = acc.attention_mask
    if mask is not None and mask.numel() == flat.shape[0]:
        m = mask.reshape(-1).to(device=flat.device, dtype=torch.bool)
        flat = flat[m]
    if flat.numel() == 0:
        return
    probs = torch.softmax(flat, dim=-1)
    acc.probability_sums[index] += probs.sum(dim=0, dtype=torch.float64).cpu()
    top = torch.topk(probs, k=top_k, dim=-1).indices.reshape(-1).cpu()
    acc.topk_counts[index] += torch.bincount(top, minlength=experts).to(torch.float64)
    acc.token_counts[index] += int(flat.shape[0])


def _collect_moe_usage(model, tokenizer, prompts: list[str], batch_size: int, adapter):
    blocks = adapter.moe_blocks(model)
    prob = [torch.zeros(len(block.experts), dtype=torch.float64) for block in blocks]
    counts = [torch.zeros(len(block.experts), dtype=torch.float64) for block in blocks]
    acc = _RouterAccumulator(prob, counts, [0] * len(blocks))
    handles = []
    for i, block in enumerate(blocks):
        top_k = _configured_top_k(model, len(block.experts))

        def hook(_module, _args, output, i=i, top_k=top_k):
            _accumulate_router(acc, i, top_k, output)
            return None

        handles.append(block.router.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            for batch in tqdm(list(batches(prompts, batch_size)), desc="moe-profile", unit="batch"):
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                acc.attention_mask = enc.get("attention_mask")
                model(**enc, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
        acc.attention_mask = None

    probability = []
    frequency = []
    for i, block in enumerate(blocks):
        tokens = acc.token_counts[i]
        if tokens <= 0:
            raise RuntimeError(f"no router activations collected for MoE layer {block.layer_index}")
        top_k = _configured_top_k(model, len(block.experts))
        probability.append((acc.probability_sums[i] / tokens).float())
        frequency.append((acc.topk_counts[i] / (tokens * top_k)).float())
    return probability, frequency, blocks


def _combine_importance(probability: list[torch.Tensor], frequency: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for p, f in zip(probability, frequency):
        pn = p / p.mean().clamp_min(EPS)
        fn = f / f.mean().clamp_min(EPS)
        out.append((0.5 * pn + 0.5 * fn).float())
    return out


def run_moe_profile(
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
    adapter = get_moe_adapter(model)
    keep_prob, keep_freq, blocks = _collect_moe_usage(model, tok, keep, batch_size, adapter)
    drop_prob, drop_freq, _ = _collect_moe_usage(model, tok, drop, batch_size, adapter)
    keep_imp = _combine_importance(keep_prob, keep_freq)
    drop_imp = _combine_importance(drop_prob, drop_freq)

    artifact = out / "moe_profile.pt"
    torch.save(
        {
            "version": 1,
            "model": model_path,
            "adapter": adapter.name,
            "layer_indices": [b.layer_index for b in blocks],
            "keep_probability": keep_prob,
            "drop_probability": drop_prob,
            "keep_frequency": keep_freq,
            "drop_frequency": drop_freq,
            "keep_importance": keep_imp,
            "drop_importance": drop_imp,
        },
        artifact,
    )
    meta = {
        "version": 1,
        "model": model_path,
        "adapter": adapter.name,
        "keep_prompts": len(keep),
        "drop_prompts": len(drop),
        "artifact": artifact.name,
        "artifact_sha256": file_sha256(artifact),
        "layers": [
            {
                "layer": block.layer_index,
                "experts": len(block.experts),
                "keep_probability_mean": float(kp.mean()),
                "drop_probability_mean": float(dp.mean()),
                "keep_topk_frequency_mean": float(kf.mean()),
                "drop_topk_frequency_mean": float(df.mean()),
            }
            for block, kp, dp, kf, df in zip(blocks, keep_prob, drop_prob, keep_freq, drop_freq)
        ],
    }
    (out / "moe_profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("wrote MoE profile to %s", out)


def select_moe_experts(
    keep_importance: list[torch.Tensor],
    drop_importance: list[torch.Tensor],
    keep_ratio: float,
    contrast_weight: float = 0.25,
    min_keep: int = 1,
) -> list[torch.Tensor]:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0,1]")
    if len(keep_importance) != len(drop_importance):
        raise ValueError("importance layer count mismatch")
    selections = []
    expected = None
    for i, (k_raw, d_raw) in enumerate(zip(keep_importance, drop_importance)):
        k = k_raw.float().clamp_min(0)
        d = d_raw.float().clamp_min(0)
        if k.ndim != 1 or d.shape != k.shape:
            raise ValueError(f"MoE layer {i} malformed importance vectors")
        n = int(k.numel())
        target = min(n, max(int(min_keep), max(1, int(n * keep_ratio))))
        if expected is None:
            expected = target
        elif target != expected:
            raise ValueError(f"MoE layer {i} target {target} differs from earlier {expected}")
        kn = k / k.mean().clamp_min(EPS)
        dn = d / d.mean().clamp_min(EPS)
        score = kn - float(contrast_weight) * torch.relu(dn - kn)
        idx = torch.topk(score, k=target, largest=True, sorted=False).indices
        selections.append(torch.sort(idx.cpu()).values)
    return selections


def _register_router_masks(model, adapter, keep_experts: list[torch.Tensor]):
    blocks = adapter.moe_blocks(model)
    if len(blocks) != len(keep_experts):
        raise ValueError("mask/model MoE layer mismatch")
    handles = []
    for block, kept in zip(blocks, keep_experts):
        expert_count = len(block.experts)
        mask = torch.zeros(expert_count, dtype=torch.bool)
        mask[kept.long()] = True

        def hook(_module, _args, output, mask=mask):
            logits = output[0] if isinstance(output, tuple) else output
            if logits.shape[-1] != mask.numel():
                raise RuntimeError("router mask shape mismatch")
            y = logits.clone()
            removed = ~mask.to(y.device)
            y[..., removed] = -torch.inf
            if isinstance(output, tuple):
                return (y, *output[1:])
            return y

        handles.append(block.router.register_forward_hook(hook))
    return handles


def _estimate_saved(adapter, model, selections: list[torch.Tensor]) -> tuple[int, int]:
    params = 0
    bytes_ = 0
    for block, kept in zip(adapter.moe_blocks(model), selections):
        keep_set = set(int(x) for x in kept.tolist())
        for i, expert in enumerate(block.experts):
            if i in keep_set:
                continue
            for p in expert.parameters():
                params += p.numel()
                bytes_ += p.numel() * p.element_size()
        removed = len(block.experts) - len(keep_set)
        if removed:
            row = block.router.weight.shape[1]
            params += removed * row
            bytes_ += removed * row * block.router.weight.element_size()
            bias = getattr(block.router, "bias", None)
            if bias is not None:
                params += removed
                bytes_ += removed * bias.element_size()
    return params, bytes_


def run_moe_search(
    model_path: str,
    keep_path: str,
    profile_path: str,
    out_dir: str,
    keep_ratios: list[float],
    batch_size: int = 2,
    max_prompts: Optional[int] = 64,
    contrast_weight: float = 0.25,
    device: str = "auto",
) -> None:
    device = resolve_device(device)
    prompts = load_prompts(keep_path, max_prompts)
    profile = load_tensor_artifact(profile_path)
    keep_imp = profile["keep_importance"]
    drop_imp = profile["drop_importance"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = _load_hf(model_path, device)
    adapter = get_moe_adapter(model)
    blocks = adapter.moe_blocks(model)
    min_keep = _configured_top_k(model, len(blocks[0].experts))
    baseline = next_token_logprobs(model, tok, prompts, batch_size)

    results = []
    for ratio in keep_ratios:
        selections = select_moe_experts(keep_imp, drop_imp, ratio, contrast_weight, min_keep=min_keep)
        handles = _register_router_masks(model, adapter, selections)
        try:
            candidate = next_token_logprobs(model, tok, prompts, batch_size)
        finally:
            for handle in handles:
                handle.remove()
        metrics = compare_logprobs(baseline, candidate)
        params_saved, bytes_saved = _estimate_saved(adapter, model, selections)
        original = len(blocks[0].experts)
        kept = int(selections[0].numel())
        row = {
            "keep_ratio_requested": float(ratio),
            "experts_kept": kept,
            "experts_original": original,
            "effective_keep_ratio": kept / original,
            "parameters_saved": params_saved,
            "bytes_saved_in_loaded_dtype": bytes_saved,
            **metrics,
        }
        results.append(row)
        LOG.info(
            "MoE keep=%.3f effective=%.3f KL=%.6f top1=%.3f saved=%.2f MiB",
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
        "profile_sha256": file_sha256(profile_path),
        "contrast_weight": float(contrast_weight),
        "prompts": len(prompts),
        "results": results,
    }
    (out / "moe_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
