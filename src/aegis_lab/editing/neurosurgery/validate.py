from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import LOG, batches, load_prompts, resolve_device


def _load_hf(model_path: str, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers is required: pip install -e .") from e
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = {"trust_remote_code": True}
    if device != "cpu":
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, tok


def next_token_logprobs(model, tokenizer, prompts: list[str], batch_size: int) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(list(batches(prompts, batch_size)), desc="logits", unit="batch"):
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, use_cache=False, return_dict=True)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()
            result.extend([x.contiguous() for x in lp])
    return result


def compare_logprobs(base: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict:
    if len(base) != len(candidate):
        raise ValueError("baseline/candidate sample count mismatch")
    kls = []
    agree = 0
    for p_log, q_log in zip(base, candidate):
        p = p_log.exp()
        kl = torch.sum(p * (p_log - q_log)).item()
        kls.append(kl)
        agree += int(int(torch.argmax(p_log)) == int(torch.argmax(q_log)))
    return {
        "samples": len(kls),
        "mean_kl": sum(kls) / max(len(kls), 1),
        "max_kl": max(kls) if kls else 0.0,
        "top1_agreement": agree / max(len(kls), 1),
    }


def run_validate(
    base_path: str,
    candidate_path: str,
    keep_path: str,
    batch_size: int,
    max_prompts: int | None,
    max_mean_kl: float | None,
    out_path: str | None,
    device: str,
) -> dict:
    device = resolve_device(device)
    prompts = load_prompts(keep_path, max_prompts)
    LOG.info("baseline logits: %s", base_path)
    base_model, tok = _load_hf(base_path, device)
    base = next_token_logprobs(base_model, tok, prompts, batch_size)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOG.info("candidate logits: %s", candidate_path)
    cand_model, cand_tok = _load_hf(candidate_path, device)
    cand = next_token_logprobs(cand_model, cand_tok, prompts, batch_size)
    metrics = compare_logprobs(base, cand)
    metrics["passed"] = max_mean_kl is None or metrics["mean_kl"] <= max_mean_kl
    if out_path:
        Path(out_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
