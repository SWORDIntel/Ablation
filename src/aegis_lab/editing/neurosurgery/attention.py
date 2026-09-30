from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .adapters import get_attention_adapter
from .common import LOG, batches, file_sha256, load_prompts, resolve_device
from .validate import _load_hf, compare_logprobs, next_token_logprobs

EPS = 1e-8


@dataclass
class _AttentionAccumulator:
    sums: list[torch.Tensor]
    counts: list[int]
    attention_mask: torch.Tensor | None = None


def _accumulate_o_input(acc: _AttentionAccumulator, layer_idx: int, pack, args) -> None:
    x = args[0].detach().float()
    expected = pack.num_heads * pack.head_dim
    if x.shape[-1] != expected:
        raise RuntimeError(
            f"layer {layer_idx}: o_proj input {x.shape[-1]} does not match heads*head_dim {expected}"
        )
    flat = x.reshape(-1, pack.num_heads, pack.head_dim)
    mask = acc.attention_mask
    if mask is not None and mask.numel() == flat.shape[0]:
        m = mask.reshape(-1).to(device=x.device, dtype=torch.bool)
        flat = flat[m]
    if flat.numel() == 0:
        return
    per_head = flat.abs().sum(dim=(0, 2), dtype=torch.float64).cpu()
    group_size = pack.group_size
    grouped = per_head.reshape(pack.num_key_value_heads, group_size).sum(dim=1)
    acc.sums[layer_idx] += grouped
    acc.counts[layer_idx] += int(flat.shape[0]) * pack.head_dim * group_size


def _collect_attention_importance(model, tokenizer, prompts: list[str], batch_size: int, adapter):
    packs = adapter.attention_packs(model)
    sums = [torch.zeros(pack.num_key_value_heads, dtype=torch.float64) for pack in packs]
    acc = _AttentionAccumulator(sums=sums, counts=[0] * len(packs))
    handles = []
    for idx, pack in enumerate(packs):
        def hook(_module, args, idx=idx, pack=pack):
            _accumulate_o_input(acc, idx, pack, args)
            return None
        handles.append(pack.o.register_forward_pre_hook(hook))

    try:
        with torch.inference_mode():
            for batch in tqdm(list(batches(prompts, batch_size)), desc="attention-profile", unit="batch"):
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                acc.attention_mask = enc.get("attention_mask")
                model(**enc, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
        acc.attention_mask = None

    out = []
    for i, (s, c) in enumerate(zip(acc.sums, acc.counts)):
        if c <= 0:
            raise RuntimeError(f"no attention activations collected for layer {i}")
        out.append((s / c).float())
    return out, packs


def run_attention_profile(
    model_path: str,
    keep_path: str,
    drop_path: str,
    out_dir: str,
    batch_size: int = 2,
    max_prompts: int | None = 64,
    device: str = "auto",
) -> None:
    device = resolve_device(device)
    keep = load_prompts(keep_path, max_prompts)
    drop = load_prompts(drop_path, max_prompts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = _load_hf(model_path, device)
    adapter = get_attention_adapter(model)
    keep_imp, packs = _collect_attention_importance(model, tok, keep, batch_size, adapter)
    drop_imp, _ = _collect_attention_importance(model, tok, drop, batch_size, adapter)

    rows = []
    for i, (k, d, pack) in enumerate(zip(keep_imp, drop_imp, packs)):
        rows.append(
            {
                "layer": i,
                "kv_groups": pack.num_key_value_heads,
                "query_heads": pack.num_heads,
                "group_size": pack.group_size,
                "head_dim": pack.head_dim,
                "keep_mean_abs_output": float(k.mean()),
                "drop_mean_abs_output": float(d.mean()),
            }
        )

    artifact = out / "attention_profile.pt"
    torch.save(
        {
            "version": 1,
            "model": model_path,
            "adapter": adapter.name,
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
        "layers": rows,
    }
    (out / "attention_profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("wrote attention profile to %s", out)


def select_attention_groups(
    keep_importance: list[torch.Tensor],
    drop_importance: list[torch.Tensor],
    keep_ratio: float,
    contrast_weight: float = 0.25,
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
            raise ValueError(f"layer {i} malformed attention importance vectors")
        n = int(k.numel())
        target = min(n, max(1, int(n * keep_ratio)))
        if expected is None:
            expected = target
        elif target != expected:
            raise ValueError(f"layer {i} target {target} differs from earlier {expected}")
        kn = k / k.mean().clamp_min(EPS)
        dn = d / d.mean().clamp_min(EPS)
        score = kn - float(contrast_weight) * torch.relu(dn - kn)
        idx = torch.topk(score, k=target, largest=True, sorted=False).indices
        selections.append(torch.sort(idx.cpu()).values)
    return selections


def _register_attention_masks(model, adapter, keep_groups: list[torch.Tensor]):
    packs = adapter.attention_packs(model)
    if len(packs) != len(keep_groups):
        raise ValueError("mask/model layer mismatch")
    handles = []
    for i, (pack, groups) in enumerate(zip(packs, keep_groups)):
        head_keep = torch.zeros(pack.num_heads, dtype=torch.bool)
        for group in groups.tolist():
            start = int(group) * pack.group_size
            head_keep[start : start + pack.group_size] = True
        vector = head_keep[:, None].expand(pack.num_heads, pack.head_dim).reshape(-1)

        def hook(_module, args, vector=vector, i=i):
            x = args[0]
            if x.shape[-1] != vector.numel():
                raise RuntimeError(f"layer {i} attention mask mismatch")
            mask = vector.to(device=x.device, dtype=x.dtype)
            return (x * mask, *args[1:])

        handles.append(pack.o.register_forward_pre_hook(hook))
    return handles


def _estimate_saved(model, adapter, selections: list[torch.Tensor]) -> tuple[int, int]:
    params = 0
    bytes_ = 0
    for pack, groups in zip(adapter.attention_packs(model), selections):
        removed_kv = pack.num_key_value_heads - int(groups.numel())
        if removed_kv <= 0:
            continue
        removed_q = removed_kv * pack.group_size
        q_rows = removed_q * pack.head_dim
        kv_rows = removed_kv * pack.head_dim
        hidden = int(pack.q.weight.shape[1])
        out_hidden = int(pack.o.weight.shape[0])
        params += q_rows * hidden + 2 * kv_rows * hidden + q_rows * out_hidden
        bytes_ += q_rows * hidden * pack.q.weight.element_size()
        bytes_ += kv_rows * hidden * pack.k.weight.element_size()
        bytes_ += kv_rows * hidden * pack.v.weight.element_size()
        bytes_ += q_rows * out_hidden * pack.o.weight.element_size()
        for module, removed_rows in ((pack.q, q_rows), (pack.k, kv_rows), (pack.v, kv_rows)):
            bias = getattr(module, "bias", None)
            if bias is not None:
                params += removed_rows
                bytes_ += removed_rows * bias.element_size()
    return params, bytes_


def run_attention_search(
    model_path: str,
    keep_path: str,
    profile_path: str,
    out_dir: str,
    keep_ratios: list[float],
    batch_size: int = 2,
    max_prompts: int | None = 64,
    contrast_weight: float = 0.25,
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
    adapter = get_attention_adapter(model)
    packs = adapter.attention_packs(model)
    baseline = next_token_logprobs(model, tok, prompts, batch_size)

    results = []
    for ratio in keep_ratios:
        selections = select_attention_groups(keep_imp, drop_imp, ratio, contrast_weight)
        handles = _register_attention_masks(model, adapter, selections)
        try:
            candidate = next_token_logprobs(model, tok, prompts, batch_size)
        finally:
            for handle in handles:
                handle.remove()
        metrics = compare_logprobs(baseline, candidate)
        params_saved, bytes_saved = _estimate_saved(model, adapter, selections)
        original_groups = packs[0].num_key_value_heads
        kept_groups = int(selections[0].numel())
        row = {
            "keep_ratio_requested": float(ratio),
            "kv_groups_kept": kept_groups,
            "kv_groups_original": original_groups,
            "effective_keep_ratio": kept_groups / original_groups,
            "parameters_saved": params_saved,
            "bytes_saved_in_loaded_dtype": bytes_saved,
            **metrics,
        }
        results.append(row)
        LOG.info(
            "attention keep=%.3f effective=%.3f KL=%.6f top1=%.3f saved=%.2f MiB",
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
    (out / "attention_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
