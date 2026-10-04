# Qualification evidence

Repository instruments, offline stock-HF fixtures and trained-model experiments are distinct evidence. Only the exact model, package versions, dtype, runtime and workload in a report were exercised. No adapter family is universally qualified.

## Trained Qwen diagnostic (2026-10-04)

- Model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Runtime: stock Transformers `5.5.4`, PyTorch `2.13.0+cu130`, CPU, BF16, four CPU threads. CUDA was unavailable with this installed driver/build combination.
- Workload: four KEEP training texts, two fictional DROP texts and four independent held-out KEEP texts. The reproducible script contains the exact prompts. There was no CHANGE objective, behavioral extraction audit or multimodal task.
- The optimizer rejected its 0.99 MLP candidate: KEEP top-1 agreement was `0.75`, below the configured `0.95` gate, although mean KL was `0.0103792`. It correctly chose the unchanged baseline. **Zero structural parameters were removed.**
- Three real LoRA recovery steps changed the checkpoint. KEEP training NLL moved from `3.64115` to `3.02263`. The DROP text-likelihood rebound was `0.000104324`, below the `0.02` diagnostic gate; that score does not measure knowledge erasure.
- Stock reload and cached generation executed. Held-out mean KL was `0.00357068`, top-1 agreement was `1.0`, and text-token NLL changed by `-0.253453` on 25 scored tokens.
- Source model, tokenizer and input hashes remained identical. Restoration means returning to that untouched checkpoint; this is not evidence for arbitrary inverse surgery or recovery rollback.
- Prefill means were approximately 3209 ms before and 2965 ms after recovery, on this small host-specific sample. These noisy timings do not establish acceleration. The raw benchmark includes repeats, context/decode conditions, hardware and process peak RSS.

[Measured report](results/qwen2_diagnostic_20261004.json) and [exact input hashes](results/qwen2_diagnostic_inputs_20261004.json) retain the observations. This experiment ran during integration development and is a diagnostic, not a released-model acceptance certificate. Full local checkpoints and logs are retained outside the repository in the qualification run directory.

Reproduce using a local checkpoint at the pinned revision:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src \
  python3 scripts/qualify_neurosurgery.py --model "$MODEL_ROOT" --out runs/qwen2-diagnostic
```

Use a fresh output directory. The script downloads nothing, preserves its parent checkpoint, verifies source/input hashes and retains profiles, the optimizer plan, preview, saved candidates, held-out validation and raw benchmarks. A feasible optimizer result can retain everything; inspect `parameters_removed` rather than assuming a physical cut occurred.

## Remaining real-model acceptance

The Qwen diagnostic does not complete these roadmap items:

- Independent KEEP/CHANGE/DROP task acceptance and full edit/recovery/restoration evidence.
- Every supported architecture/version on stock reload and cached generation.
- Actual quantized target kernels with quality, size, peak memory, prefill/decode latency and baseline conditions. Current `QuantizedLinear` dequantizes weights for PyTorch `F.linear`; custom checkpoint loading is required. Packed storage does not imply an integer kernel or stock runtime support.
- Retained-modality execution after branch removal on real multimodal checkpoints.
- Factual-edit/unlearning locality, interference, extraction probes and uncertainty.
Installed commands and workflow/campaign execution now retain mandatory provenance audits. Legacy upstream producer lineage and hidden external dependencies of custom callbacks remain explicit limits; the development diagnostic predates the final operator audit integration.

Supply concrete task datasets, acceptance gates, architecture revisions and runtime targets for those experiments. Their checkboxes remain open until the actual evidence exists.
