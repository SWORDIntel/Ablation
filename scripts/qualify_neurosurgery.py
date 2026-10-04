"""Run and retain a trained-model structural/recovery/reload experiment.

Use a local stock HF checkpoint. Results qualify only the stated model/runtime,
not other adapters or task quality. No downloads are performed by this script.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import torch
import transformers

from aegis_lab.editing.neurosurgery.apply import run_apply
from aegis_lab.editing.neurosurgery.mlp import run_mlp_profile
from aegis_lab.editing.neurosurgery.optimizer import run_optimizer
from aegis_lab.editing.neurosurgery.preview import build_preview
from aegis_lab.editing.neurosurgery.measured import bind_files, verify_binding, recover_model
from aegis_lab.editing.neurosurgery.validate import _load_hf, run_validate
from aegis_lab.editing.neurosurgery.stage7_export import benchmark_runtime_profile, BenchmarkProfile


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    start = time.time()
    data = root / 'inputs'
    data.mkdir()
    keep = ['Python lists preserve insertion order.', 'The sum of two and three is five.',
            'A triangle has three sides.', 'Water freezes at zero degrees Celsius.']
    drop = ['In this fictional story, the secret passphrase is violet comet.',
            'The fictional vault opens with the code silver willow.']
    heldout = ['A square has four equal sides.', 'The product of three and four is twelve.',
               'Python dictionaries map keys to values.', 'An hour contains sixty minutes.']
    for name, prompts in [('keep', keep), ('drop', drop), ('heldout', heldout)]:
        (data / f'{name}.txt').write_text('\n'.join(prompts) + '\n')
    binding = bind_files(dict(model=args.model, keep=data / 'keep.txt', drop=data / 'drop.txt',
                              heldout=data / 'heldout.txt'))
    (root / 'binding.json').write_text(json.dumps(binding, indent=2))
    run_mlp_profile(args.model, str(data / 'keep.txt'), str(data / 'drop.txt'),
                    str(root / 'profile'), batch_size=1, device='cpu')
    run_optimizer(args.model, str(data / 'keep.txt'), str(root / 'opt'),
                  mlp_profile_path=str(root / 'profile' / 'mlp_profile.pt'),
                  mlp_ratios=[1.0, 0.99], max_mean_kl=0.02, min_top1_agreement=0.95,
                  max_trials=4, batch_size=1, device='cpu')
    plan = root / 'opt' / 'optimized_plan.yaml'
    model, tokenizer = _load_hf(args.model, 'cpu')
    preview = build_preview(model, str(plan))
    (root / 'preview.json').write_text(json.dumps(preview, indent=2))
    base_benchmark = benchmark_runtime_profile(model, BenchmarkProfile(prompt_length=16, decode_steps=4),
                                               device='cpu', num_warmup=1, num_repeats=3).to_dict()
    del model
    run_apply(args.model, None, str(plan), str(root / 'candidate'), device='cpu')
    pre_recovery = run_validate(args.model, str(root / 'candidate'), str(data / 'heldout.txt'),
                               1, None, 0.02, str(root / 'before_recovery.json'), 'cpu')
    model, tokenizer = _load_hf(str(root / 'candidate'), 'cpu')
    model, recovery = recover_model(model, tokenizer, keep, drop, steps=3, rank=4, max_rebound=0.02)
    model.save_pretrained(root / 'recovered', safe_serialization=True)
    tokenizer.save_pretrained(root / 'recovered')
    del model
    validation = run_validate(args.model, str(root / 'recovered'), str(data / 'heldout.txt'),
                              1, None, 0.02, str(root / 'validation.json'), 'cpu')
    model, tokenizer = _load_hf(str(root / 'recovered'), 'cpu')
    with torch.no_grad():
        inputs = tokenizer('Python is a programming language', return_tensors='pt')
        output = model.generate(**inputs, max_new_tokens=4, use_cache=True, do_sample=False)
    candidate_benchmark = benchmark_runtime_profile(model, BenchmarkProfile(prompt_length=16, decode_steps=4),
                                                    device='cpu', num_warmup=1, num_repeats=3).to_dict()
    verify_binding(binding)
    manifest = json.loads((root / 'candidate' / 'neurosurgery_manifest.json').read_text())
    report = dict(model=args.model, architecture=model.config.model_type,
                  torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                  cpu_threads=torch.get_num_threads(), runtime='stock transformers CPU', dtype=str(next(model.parameters()).dtype),
                  parameters_removed=manifest['parameters_removed'], recovery=recovery,
                  before_recovery=pre_recovery, heldout_validation=validation,
                  cached_generation=tokenizer.decode(output[0]), source_restoration_hashes_unchanged=True,
                  source_restoration='untouched byte-bound source checkpoint; candidate discarded for restoration',
                  baseline_benchmark=base_benchmark, candidate_benchmark=candidate_benchmark,
                  duration_seconds=time.time() - start,
                  limitations=['Small text-only diagnostic set, not a task-quality qualification.',
                    'DROP score is text likelihood, not knowledge erasure or extraction resistance.',
                    'No CHANGE objective or multimodal qualification in this experiment.',
                    'Single host and runtime; no GPU or target quantized-kernel qualification.'])
    (root / 'qualification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
