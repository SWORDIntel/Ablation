# Neuron steering and controlled A/B experiments

The core `aegis-neurosurgery` CLI now supports **reversible runtime MLP-neuron edits** with signed doses, scale sweeps, paired generations and a small, provenance-labelled library of published persona effects.

These experiments **do not change checkpoint weights**. The intervention is a forward pre-hook on the *input to* a transformer's gated MLP `down_proj`; it edits one intermediate activation per token, not a raw residual coordinate. The experimental feature is independent of structural pruning, recovery and deployment export.

## Show model/effect presets

Run from the repository root after `pip install -e .`:

```bash
aegis-neurosurgery steer-presets
aegis-neurosurgery steer-presets --model-id Qwen/Qwen3.5-4B
```

The curated registry currently includes:

- **Qwen/Qwen3.5-4B:** Lyra's hostile/evil pair (`L14N8949+1.0 L13N7956-1.0`), the single-neuron control and additional reported tones (flatterer, contempt, poet, upbeat, suspicious, condescending, wistful, oracle, bleak, dismissive, ornate, curt) from independent experiments.
- **Qwen/Qwen3.8-27B:** reported hostile (`L34N3838+0.75`) and extra-hostile (`L34N3838+0.75 L38N17180+0.75`) combinations.
- **Qwen/Qwen3-8B:** three single-prompt self-description examples from the Neuron Persona Library.

All presets include a source URL and carry the evidence label **reported; not locally reproduced**. They are hypotheses for your own A/B runs, not verified universal personality controls. Qwen3-8B library examples were originally calibrated on the prompt being tested, while the default runner uses one fixed calibration text; outcomes may differ. Do not copy these neuron indices to other models or architecture revisions.

Sources: [Lyra](https://x.com/_lyraaaa_/status/2113833330429083662), [independent Qwen3.5-4B evaluation](https://www.hedwigai.com/blog/qwen-personality-neurons), [Qwen3.8-27B experiment](https://x.com/chiefofautism/status/2115274388425900291), [Neuron Persona Library](https://neuronpersonas.com/), [Neuronary reference implementation](https://github.com/lyramakesmusic/neuronary/blob/main/scripts/run.py).

## Paired trials

Download a supported Hugging Face checkpoint to a **local** directory. Supply an independent prompt set with one prompt per line (or a JSON/JSONL file). Example (paths are yours to create):

```bash
aegis-neurosurgery steer \
  --model models/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  --preset qwen35-4b-evil-pair \
  --workload data/steering-eval.txt \
  --scales=-0.5,0.25,0.5,1.0 \
  --max-prompts 24 \
  --max-new-tokens 160 \
  --temperature 0 \
  --seed 42 \
  --out runs/steering-qwen35-evil.json
```

Zero is the **automatic unsteered baseline** and must not appear in `--scales`. Each prompt runs against the same model and tokenized input, using the unsteered baseline plus each requested signed multiplier. The runner resets the pseudo-random seed for each condition, uses greedy decoding by default, bounds output length, and removes hooks even when an experiment raises an exception. The output file is not overwritten.

For custom activations (no published preset), repeat `--neuron`:

```bash
aegis-neurosurgery steer \
  --model models/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  --neuron L14N8949+1.0 --neuron L13N7956-1.0 \
  --workload data/steering-eval.txt --scales 0.5,1 \
  --out runs/custom-two-neurons.json
```

Custom JSON recipes (`--config` instead of `--preset` or `--neuron`) let you choose the operation and token phase:

```json
{
  "model": "Qwen/Qwen3.5-4B",
  "effect": "custom neuron clamp-by-multiplication",
  "neurons": [{"layer": 14, "neuron": 8949, "dose": 0.0}],
  "method": {
    "op": "mul",
    "normalization": "raw",
    "apply_to": "generated_tokens"
  }
}
```

The above `mul` example uses `1 + (dose - 1) × scale` as the actual multiplier. At scale 1 it zeros that neuron's intermediate activation **during generation only**; scale 0 is identity. This is an experiment, not an established persona preset. Custom `add` recipes can select normalized or raw units and apply to `all_tokens`, `prompt_tokens`, or `generated_tokens`. MoE expert neurons are intentionally unsupported.

## Steering dose and calibration

For normalized additive steering, at `mlp.down_proj` input:

```text
activation[..., neuron] += scale × dose × median(||residual after layer L||, tokens 1:) / ||down_proj.weight[:, neuron]||₂
```

The median is measured from a frozen built-in calibration text (the same text used by Neuronary) on the locally loaded checkpoint, with the first token excluded. Supply `--calibration-data local.txt` to choose a different calibration corpus. Raw additive recipes use `scale × dose` directly; `mul` uses the multiplicative semantics above and requires `normalization: raw`.

Normalized doses are **large interventions**: `dose = 1` changes a neuron's write to the residual stream by approximately one typical layer residual norm. A persona shift may coincide with severe degradation in correctness, instruction-following, refusal behaviour or coherence. Include mathematics, coding, factuality, ordinary conversation and task-specific held-out prompts.

## Reports, provenance and limits

The report contains the exact recipe and its source, asserted model ID, checkpoint path, architecture, calibration hash and layer norms, resulting per-neuron amplitudes, generation settings, full token IDs and decoded text for both arms, first divergent token, timings, and the changed-response rate for each scale.

Optionally, `--target-regex 'pattern'` adds matched/unmatched counts on both arms. Regex matching and text difference are **observations**, not semantic success or proof of specificity. Scores should be validated with independent labelled evaluations and random-neuron/negative controls.

`operator_audit` binds the local checkpoint, workload, optional calibration input, local recipe and implementation hashes before/after execution. The `--model-id` value is an explicit **operator assertion**, not a cryptographic proof of the model's original identity. Known presets also check the model family (`qwen3` or `qwen3_5`) and fail closed on missing/invalid neuron geometry, non-finite doses, unsupported quantized layers or bypassed hooks. Use a separately verified model manifest if the exact weights matter.

Limitations:

- Requires an ordinary hookable PyTorch model; fused, compiled, GGUF, OpenVINO and MoE paths are not supported by this runner.
- Loads a local model via Transformers. Qwen3.5/Qwen3.8 need a sufficiently recent Transformers release; remote code stays disabled unless `AEGIS_TRUST_REMOTE_CODE=1`.
- The suite handles a single prompt per generation, so results are not a throughput benchmark. Timing fields are diagnostic only.
- Common seeds do not eliminate stochastic or device-related variability. Repeat trials and keep disjoint discovery/validation sets.
- The CLI does not assert any target persona was reproduced. Verify with independent evaluation before using a steered model for downstream decisions.

## Validation

```bash
PYTHONPATH=src python3 -m unittest tests.unit.test_neurosurgery_neuron_steering -v
PYTHONPATH=src python3 -m unittest tests.unit.test_neurosurgery_cli -v
aegis-neurosurgery steer-presets
```

Real-model runs additionally require the checkpoint and a representative held-out workload. No live model measurements are committed as regression fixtures.
