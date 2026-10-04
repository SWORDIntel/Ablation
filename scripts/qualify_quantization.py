"""Measure this repository's packed quantization path on a local stock HF model.

This reports diagnostic CPU behavior for Qwen2.5-0.5B-Instruct or another
compatible local checkpoint. The QuantizedLinear implementation dequantizes
weights during forward and calls floating-point F.linear; this does not qualify
an accelerated integer kernel.
"""
import argparse
import json
from pathlib import Path
import platform
import resource
import time

import torch
import transformers

from aegis_lab.editing.neurosurgery.stage6_quantization import (
    QuantizationConfig, QuantizationValidationThresholds, bind_calibration_dataset,
    calculate_model_memory_bytes, export_quantized_checkpoint, load_quantized_checkpoint,
    quantize_model, validate_quantized_candidate, verify_export_reload_parity,
)
from aegis_lab.editing.neurosurgery.stage7_export import BenchmarkProfile, benchmark_runtime_profile
from aegis_lab.editing.neurosurgery.measured import bind_files
from aegis_lab.editing.neurosurgery.validate import _load_hf

CALIBRATION = [
    "Python lists preserve insertion order.",
    "The sum of two and three is five.",
    "A triangle has three sides.",
    "Water freezes at zero degrees Celsius.",
]
HELDOUT = [
    "A square has four equal sides.",
    "The product of three and four is twelve.",
    "Python dictionaries map keys to values.",
    "An hour contains sixty minutes.",
]


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def run(model_path: Path, out: Path, bits: list[int], threads: int):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    start = time.time()
    model, tokenizer = _load_hf(str(model_path), 'cpu')
    if tokenizer is None:
        raise ValueError('A local tokenizer is required for text quantization validation')
    model.eval()
    baseline_memory = calculate_model_memory_bytes(model)
    prompt = tokenizer('Water freezes at', return_tensors='pt')['input_ids']
    if prompt.shape[1] < 8:
        prompt = torch.cat([prompt, prompt[:, -1:].expand(1, 8-prompt.shape[1])], dim=1)
    prompt = prompt[:, :8].contiguous()
    profile = BenchmarkProfile(name='short_cpu_diagnostic', batch_size=1, prompt_length=8, decode_steps=1)
    baseline_bench = benchmark_runtime_profile(model, profile, 'cpu', num_warmup=0, num_repeats=2,
                                               sample_inputs=prompt).to_dict()
    calibration = bind_calibration_dataset(CALIBRATION, tokenizer=tokenizer,
                                           metadata={'split': 'calibration', 'source': 'script'})
    heldout_binding = bind_calibration_dataset(HELDOUT, tokenizer=tokenizer,
                                              metadata={'split': 'heldout', 'source': 'script'})
    model_binding = bind_files({'model': model_path})
    (out / 'calibration_binding.json').write_text(json.dumps(calibration.to_dict(), indent=2))
    (out / 'heldout_binding.json').write_text(json.dumps(heldout_binding.to_dict(), indent=2))
    results = []
    for bit in bits:
        target = out / f'int{bit}'
        target.mkdir()
        config = QuantizationConfig(bits=bit, symmetric=True, granularity='per_channel', pack=True)
        candidate = quantize_model(model, config)
        thresholds = QuantizationValidationThresholds(
            max_kl_drift=1e9, max_mse_drift=1e9, min_keep_retention=0.0,
            max_drop_rebound=1e9, min_memory_reduction_ratio=0.0,
            max_generation_nll_drift=1e9,
        )
        quality = validate_quantized_candidate(model, candidate, HELDOUT, tokenizer=tokenizer,
                                               thresholds=thresholds, device='cpu', raise_on_failure=False)
        files = export_quantized_checkpoint(candidate, target, config, calibration,
            metadata={'qualification': 'diagnostic', 'validation_split': 'heldout'})
        reloaded, _metadata = load_quantized_checkpoint(target, base_model=__import__('copy').deepcopy(model))
        parity = verify_export_reload_parity(candidate, reloaded, HELDOUT, tokenizer=tokenizer)
        with torch.no_grad():
            generated = reloaded.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                      max_new_tokens=2, use_cache=True, do_sample=False)
        (target / 'generation.json').write_text(json.dumps({
            'prompt': tokenizer.decode(prompt[0]), 'output': tokenizer.decode(generated[0]),
            'prompt_tokens': int(prompt.shape[1]), 'generated_tokens': int(generated.shape[1]-prompt.shape[1]),
            'cache_enabled': True,
        }, indent=2))
        bench = benchmark_runtime_profile(reloaded, profile, 'cpu', num_warmup=0, num_repeats=2,
                                          sample_inputs=prompt).to_dict()
        modules = [m for m in candidate.modules() if m.__class__.__name__ == 'QuantizedLinear']
        report = {
            'bits': bit, 'config': config.to_dict(), 'module_count': len(modules),
            'packed_module_count': sum(bool(m.is_packed) for m in modules),
            'baseline_tensor_bytes': baseline_memory,
            'quantized_tensor_bytes': calculate_model_memory_bytes(candidate),
            'tensor_memory_reduction_ratio': quality.memory_reduction_ratio,
            'exported_file_bytes': tree_bytes(target),
            'quality_metrics': {**{key: quality.to_dict()[key] for key in
                                ('keep_retention', 'kl_drift', 'mse_drift',
                                 'generation_nll_drift', 'memory_reduction_ratio',
                                 'baseline_memory_bytes', 'quantized_memory_bytes')},
                                'metrics': {key: value for key, value in quality.metrics.items()
                                            if key != 'drop_leak'}},
            'library_validator_thresholds': thresholds.to_dict(),
            'quality_gate_status': 'not assessed; acceptance limits were not supplied',
            'export_reload': 'custom loader re-materialized quantized modules; internal forward parity checked',
            'reload_parity': parity,
            'cached_generation_executed': True,
            'runtime_benchmark': bench,
            'kernel_path': 'QuantizedLinear dequantizes weight then calls torch.nn.functional.linear',
            'kernel_acceleration_claimed': False,
            'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'runtime_memory_note': 'Process RSS includes baseline, candidate and loader copies during this diagnostic.',
            'checkpoint_files': {k: Path(v).name for k, v in files.items()},
        }
        (target / 'diagnostic.json').write_text(json.dumps(report, indent=2, sort_keys=True))
        results.append(report)
        del candidate, reloaded
    result = {
        'model_path_label': model_path.name,
        'model_config_architecture': model.config.model_type,
        'model_revision': model_path.name,
        'model_binding': model_binding,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'python': platform.python_version(), 'device': 'CPU',
        'cpu_threads': torch.get_num_threads(), 'dtype': str(next(model.parameters()).dtype),
        'prompt_tokens': int(prompt.shape[1]), 'decode_steps': profile.decode_steps,
        'baseline_memory_bytes': baseline_memory,
        'baseline_checkpoint_file_bytes': tree_bytes(model_path),
        'calibration_sha256': calibration.sha256, 'calibration_samples': calibration.sample_count,
        'heldout_sha256': heldout_binding.sha256, 'heldout_samples': heldout_binding.sample_count,
        'baseline_runtime_benchmark': baseline_bench,
        'quantization_results': results,
        'duration_seconds': time.time()-start,
        'limitations': [
            'Generic trivia prompts are a diagnostic, not a user task-quality evaluation.',
            'The measured CPU path dequantizes each linear weight during forward and uses floating-point F.linear.',
            'No accelerated target integer kernel, native stock Transformers loading, or deployment speedup was qualified.',
            'The custom quantized checkpoint requires the repository loader and a matching model architecture.',
            'Peak RSS includes resident source, candidate and loader copies; cross-format values are process high-water marks.',
            'No DROP examples were provided; DROP leakage is not evaluated.',
        ],
    }
    (out / 'qualification.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Existing local stock HF checkpoint')
    parser.add_argument('--out', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--bits', type=int, nargs='+', default=[8, 4], choices=[4, 8])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    run(args.model, args.out, args.bits, args.threads)
