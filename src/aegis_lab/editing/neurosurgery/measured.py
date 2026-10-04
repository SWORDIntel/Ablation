"""Measured HF workflow primitives; never substitute synthetic task metrics."""
from pathlib import Path
import json
import math

import torch

from .common import file_sha256, load_prompts
from .math_ops import contrast_direction, preservation_basis
from .profile import _collect
from .validate import mean_teacher_forced_nll


def measured_profile(model, tokenizer, keep, drop, out, batch_size=2, basis_rank=4):
    if tokenizer is None or not keep or not drop:
        raise ValueError('Measured profiling requires a tokenizer and nonempty KEEP/DROP data')
    model.eval()
    layer_path, km, kd, samples = _collect(model, tokenizer, keep, batch_size,
                                         next(model.parameters()).device, 64)
    _, dm, dd, _ = _collect(model, tokenizer, drop, batch_size,
                           next(model.parameters()).device, 0)
    directions, bases, metrics = [], [], []
    for i in range(len(kd)):
        direction, separation = contrast_direction(km[i + 1], dm[i + 1])
        directions.append(direction.cpu())
        bases.append(preservation_basis(samples[i + 1], basis_rank).cpu())
        metrics.append(dict(layer=i, keep_delta_ratio=float(kd[i]), drop_delta_ratio=float(dd[i]),
                            drop_to_keep_ratio=float(dd[i]) / max(float(kd[i]), 1e-8),
                            contrast_separation=float(separation)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(layer_path=layer_path, num_layers=len(metrics), layers=metrics,
                    keep_prompts=len(keep), drop_prompts=len(drop), measurement='model_activations')
    (out / 'profile.json').write_text(json.dumps(metadata, indent=2))
    torch.save(dict(metadata, directions=directions, preserve_bases=bases, metrics=metrics),
               out / 'profile.pt')
    return metadata


def text_nll(model, tokenizer, prompts):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        return mean_teacher_forced_nll(model, tokenizer, prompts, 2)['mean_nll']
    finally:
        for module, training in modes.items():
            module.training = training


def load_change_examples(path):
    path = Path(path)
    rows = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == '.jsonl' else json.loads(path.read_text()))
    if not isinstance(rows, list) or not rows:
        raise ValueError('CHANGE data must be a nonempty JSON/JSONL list of prompt/target records')
    if any(not isinstance(row, dict) or not isinstance(row.get('prompt'), str)
           or not row['prompt'].strip() or not isinstance(row.get('target'), str)
           or not row['target'].strip() for row in rows):
        raise ValueError('Each CHANGE record requires nonempty prompt and target strings')
    return rows


def training_batches(tokenizer, model, examples, batch_size=2):
    if not examples:
        return None
    result = []
    device = next(model.parameters()).device
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        if isinstance(chunk[0], dict):
            encoded = []
            for row in chunk:
                prefix = tokenizer.encode(row['prompt'], add_special_tokens=False)
                target = tokenizer.encode(row['target'], add_special_tokens=False)
                if not prefix or not target:
                    raise ValueError('CHANGE prompt/target tokenization must be nonempty')
                encoded.append((prefix + target, [-100] * len(prefix) + target))
            width = max(len(ids) for ids, _ in encoded)
            enc = dict(input_ids=torch.tensor([ids + [tokenizer.pad_token_id] * (width - len(ids)) for ids, _ in encoded]),
                       labels=torch.tensor([labels + [-100] * (width - len(labels)) for _, labels in encoded]),
                       attention_mask=torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in encoded]))
        else:
            enc = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
            enc.pop('token_type_ids', None)
        result.append({k: v.to(device) for k, v in enc.items()})
    return result


def recover_model(model, tokenizer, keep, drop=None, change=None, steps=5, lr=1e-4,
                  rank=8, max_rebound=None, seed=42):
    from .stage5_recovery import (inject_lora, apply_freeze_mask, run_recovery_training,
                                 RecoveryConfig, merge_lora)
    if tokenizer is None or not keep:
        raise ValueError('Recovery requires a tokenizer and nonempty KEEP data')
    if steps <= 0 or rank <= 0 or lr <= 0 or not math.isfinite(lr):
        raise ValueError('Recovery steps, rank and finite learning rate must be positive')
    if max_rebound is not None and not drop:
        raise ValueError('DROP rebound gate requires DROP evaluation data')
    initial_loss = text_nll(model, tokenizer, keep)
    def score(current):
        return math.exp(-text_nll(current, tokenizer, drop))
    initial_drop = score(model) if drop else None
    inject_lora(model, target_modules=['q_proj', 'v_proj', 'down_proj'], r=rank)
    apply_freeze_mask(model, allow_lora_only=True)
    result = run_recovery_training(model, keep_data=training_batches(tokenizer, model, keep),
                                  change_data=training_batches(tokenizer, model, change),
                                  drop_evaluator=score if drop else None,
                                  config=RecoveryConfig(max_steps=steps, lr=lr, seed=seed,
                                                        max_drop_rebound=max_rebound))
    model = merge_lora(model)
    model.eval()
    final_loss = text_nll(model, tokenizer, keep)
    final_drop = score(model) if drop else None
    rebound = final_drop - initial_drop if drop else None
    report = dict(training_steps=result.steps_completed, success=result.success, status=result.status,
                  initial_loss=initial_loss, final_loss=final_loss, loss_delta=final_loss - initial_loss,
                  drop_rebound_score=rebound, initial_drop_score=initial_drop,
                  final_drop_score=final_drop, drop_score_definition='exp(-DROP text token NLL)',
                  rebound_passed=(max_rebound is None or rebound <= max_rebound))
    if not report['success'] or not report['rebound_passed']:
        raise ValueError(f'Recovery failed its measured gates: {report}')
    return model, report


def bind_files(paths):
    """Bind exact local bytes, including weights, config and tokenizer assets."""
    result = {}
    for role, name in paths.items():
        if name is None:
            continue
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        if not files:
            raise ValueError(f'No files to bind for {role}: {path}')
        result[role] = dict(path=str(path), files={str(p.relative_to(path) if path.is_dir() else p.name):
                                                 file_sha256(p) for p in files})
    return result


def verify_binding(binding):
    current = bind_files({role: item['path'] for role, item in binding.items()})
    if current != binding:
        raise ValueError('Workflow input bytes changed after provenance binding')


def bind_plan_inputs(model_path, plan_path, profile_path=None):
    import yaml
    plan_path = Path(plan_path)
    plan = yaml.safe_load(plan_path.read_text())
    paths = dict(model=model_path, plan=plan_path, profile=profile_path)
    for kind, section in (plan.get('structured') or {}).items():
        if section.get('enabled'):
            raw = Path(section['selection'])
            paths[f'selection_{kind}'] = raw if raw.is_absolute() else plan_path.parent / raw
    return bind_files(paths)
