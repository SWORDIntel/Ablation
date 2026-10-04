"""Plan-driven local HF workflow with measured gates and reloadable outputs."""
import json
from pathlib import Path

import yaml

from .common import load_prompts, file_sha256
from .measured import bind_files, verify_binding, measured_profile, recover_model, load_change_examples
from .preview import build_preview
from .validate import _load_hf, run_validate
from .apply import run_apply


def run_operator(args, config):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Workflow output directory must be empty; use a new run directory')
    out.mkdir(parents=True, exist_ok=True)
    model_path = args.model or config.get('model')
    stages = [s.strip() for s in (args.stages or config.get('stages',
              'inspect,localize,plan,apply,recover,validate,export')).split(',') if s.strip()]
    canonical = ['inspect', 'localize', 'plan', 'apply', 'recover', 'validate', 'export']
    if not stages or any(s not in canonical for s in stages) or len(set(stages)) != len(stages):
        raise ValueError('Unknown, empty or duplicate workflow stages')
    if stages != sorted(stages, key=canonical.index):
        raise ValueError('Workflow stages must follow inspect/localize/plan/apply/recover/validate/export order')
    keep = args.keep or config.get('keep')
    drop = args.drop or config.get('drop')
    change = args.change or config.get('change')
    heldout = getattr(args, 'validation_keep', None) or config.get('validation_keep')
    plan_path = args.plan or config.get('plan')
    profile = config.get('profile')
    if not isinstance(model_path, (str, Path)):
        raise ValueError('Measured workflows require a local reloadable HF checkpoint path')
    if 'localize' in stages and (not keep or not drop):
        raise ValueError('Localization requires KEEP and DROP datasets')
    if 'recover' in stages and not keep:
        raise ValueError('Recovery requires KEEP training data')
    if 'validate' in stages and not heldout:
        raise ValueError('Validation requires independent --validation-keep data')
    if heldout and keep and set(load_prompts(heldout)) & set(load_prompts(keep)):
        raise ValueError('Validation KEEP overlaps training KEEP prompts')
    if ('apply' in stages or 'plan' in stages) and not plan_path and 'localize' not in stages:
        raise ValueError('Plan/apply requires --plan or measured localization')
    binding = bind_files(dict(model=model_path, keep=keep, drop=drop, change=change,
                              validation_keep=heldout, plan=plan_path, profile=profile))
    manifest = dict(version=1, binding=binding, edit_order=stages)
    (out / 'workflow_manifest.json').write_text(json.dumps(manifest, indent=2))
    model, tokenizer = _load_hf(str(model_path), args.device)
    steps = {}
    if 'inspect' in stages:
        steps['inspect'] = dict(total_parameters=sum(p.numel() for p in model.parameters()),
                                architecture=model.config.model_type)
    if 'localize' in stages:
        meta = measured_profile(model, tokenizer, load_prompts(keep), load_prompts(drop), out / 'profile')
        winner = max(meta['layers'], key=lambda m: m['drop_to_keep_ratio'])
        steps['localize'] = dict(selected_candidate_layers=[winner['layer']], metrics=meta['layers'])
        profile = str(out / 'profile' / 'profile.pt')
        plan_path = str(out / 'surgery_plan.yaml')
        Path(plan_path).write_text(yaml.safe_dump(dict(
            version=3, drop_layers=[], structured={}, ablation=dict(
                layers=[winner['layer']], targets=['mlp.down_proj'], strength=0.1,
                norm_preserve=True, preserve_subspace=True))))
    if plan_path:
        from .measured import bind_plan_inputs
        manifest["plan_binding"] = bind_plan_inputs(model_path, plan_path, profile)
        (out / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2))
        preview = build_preview(model, str(plan_path), profile)
        (out / 'preview.json').write_text(json.dumps(preview, indent=2))
        steps['plan'] = dict(plan=str(plan_path), plan_sha256=file_sha256(plan_path), preview=preview)
    if args.dry_run:
        report = dict(command='workflow', mode='dry_run', status='previewed', stages=stages,
                      completed_steps=steps, qualified=False)
    else:
        del model
        candidate = str(model_path)
        if 'apply' in stages:
            verify_binding(binding)
            candidate = str(out / 'candidate')
            run_apply(str(model_path), profile, str(plan_path), candidate, args.device)
            steps['apply'] = json.loads((Path(candidate) / 'neurosurgery_manifest.json').read_text())
        if 'recover' in stages:
            model, tokenizer = _load_hf(candidate, args.device)
            model, recovery = recover_model(model, tokenizer, load_prompts(keep),
                load_prompts(drop) if drop else None, load_change_examples(change) if change else None,
                steps=args.max_recovery_steps, rank=args.lora_r,
                max_rebound=config.get('max_drop_rebound'))
            candidate = str(out / 'recovered')
            model.save_pretrained(candidate, safe_serialization=True)
            tokenizer.save_pretrained(candidate)
            del model
            steps['recover'] = recovery
        if 'validate' in stages:
            result = run_validate(str(model_path), candidate, str(heldout), 2, None,
                                  float(config.get('max_mean_kl', 0.02)), str(out / 'validation.json'), args.device)
            steps['validate'] = result
            if not result['passed']:
                raise ValueError('Held-out reload validation failed')
        if 'export' in stages:
            model, tokenizer = _load_hf(candidate, args.device)
            model.save_pretrained(out / 'exported_runtime', safe_serialization=args.export_format == 'safetensors')
            tokenizer.save_pretrained(out / 'exported_runtime')
            steps['export'] = dict(export_dir=str(out / 'exported_runtime'), runtime='transformers')
            del model
        verify_binding(binding)
        if manifest.get("plan_binding"):
            verify_binding(manifest["plan_binding"])
        report = dict(command='workflow', mode='measured', status='success', stages=stages,
                      completed_steps=steps, candidate=candidate,
                      qualified=False, heldout_validation_passed=('validate' in stages), restoration='Untouched bound source checkpoint')
    (out / 'workflow_report.json').write_text(json.dumps(report, indent=2))
    return report
