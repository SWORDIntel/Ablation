# Real-checkpoint steering qualification

This harness is implemented; Qwen3.5-4B behaviour and hardware acceptance are **not qualified** by its fixture tests. It runs existing audited operators, not a new inference framework. No trained checkpoint was executed during harness development.

## Target and prerequisites

Initial target: **Qwen/Qwen3.5-4B**, post-trained stock Hugging Face checkpoint, not the Base variant, a fine-tune, GGUF, AWQ/GPTQ, MoE, compiled or serving-engine export. Pin a full upstream commit and preserve the complete local snapshot. The supplied revision is an operator assertion, not an authenticity check. Compare snapshot provenance against that commit independently before recording the trusted canonical tree digest.

The published [config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) and [model card](https://huggingface.co/Qwen/Qwen3.5-4B) describe `Qwen3_5ForConditionalGeneration`, outer model type `qwen3_5`, nested text type `qwen3_5_text`, 32 layers, hidden width 2560, intermediate width 9216 and vocabulary 248320. The loader uses `AutoModelForImageTextToText`, with text-only prompts; this workflow does not qualify vision behaviour. Preflight checks these fields, rejects declared quantization and resolves/imports the installed stock model class without allocating weights. Unsupported runtimes fail before an experiment. Geometry alone does not attest upstream identity.

Use Python 3.12, PyTorch and a Transformers release containing this actual adapter. The repository's generic `transformers>=4.46` floor is **not** a Qwen3.5 support guarantee. Development tests ran with Python 3.12, torch 2.15.0+cu132 and transformers 5.22.1; adapter resolution succeeded without loading weights. Those versions are a tested fixture environment, not a proven full-checkpoint combination. Reproduce with a clean venv and lock all resolved dependencies (`pip freeze`), Python, driver and OS versions alongside the result. No remote code is allowed by the harness; unset the project's remote-code opt-in.

For a real CUDA attempt, provision a BF16-capable GPU and sufficient memory for the entire floating-point multimodal model, KV/recurrent state and logits. A 16–24 GiB GPU and at least 32 GiB host RAM are planning allowances for bounded prompts, **not measured minimums**. CPU float32 needs substantially more parameter memory (roughly 16 GB for 4B parameters before overhead) and may be very slow; provision at least 32 GiB RAM and verify actual peak RSS. No GPU access or GPU success is established by these tests. The harness loads a fresh model per run via `steer`, so memory must be reclaimed between operators by the Python runtime. Use eager stock modules: fused bypasses, rounded-to-zero doses, nonfinite activations, restoration failures and OOM are failures, not accepted observations.

## Freeze independent inputs

Copy `data/ablation-smoke/steering-qualification/plan.example.json` to your experiment directory and edit it. Paths in the plan resolve relative to that file. The checked-in prompt files and rubric are deliberately tiny development fixtures; they are insufficient for behavioural acceptance. Replace them before the real run.

Required plan fields:

| Field | Meaning |
|---|---|
| `model` | Complete local snapshot directory |
| `revision` | Independently verified upstream 40-hex commit |
| `checkpoint_sha256` | Trusted `sha256:` canonical tree digest from `checkpoint_binding`, not the HF commit or a shard digest |
| `recipe` | Frozen validated recipe explicitly naming Qwen/Qwen3.5-4B |
| `rubric` | Independently authored target/KEEP scoring instructions |
| `keep` | Held-out preservation prompts |
| `excluded_workloads` | Nonempty list of all original discovery/search/custom calibration corpora |
| `runs` | 2–8 disjoint held-out behavioural workload files |
| `repeat_seeds` | 2–8 distinct seeds; each workload is repeated with every seed |
| `scale` | One preselected nonzero dose multiplier, locked before evaluation |

Record the tree digest during trusted snapshot preparation, using the existing provenance function, then independently compare it on the run host:

```bash
PYTHONPATH=src python -c 'from aegis_lab.editing.neurosurgery.steering_provenance import checkpoint_binding; print(checkpoint_binding("/absolute/path/to/pinned-Qwen3.5-4B")["checkpoint_sha256"])'
```

Do not treat computing and trusting a digest on the same potentially modified host as upstream authentication. The manifest also binds individual weights, config and tokenizer files; each A/B report adds runtime config/tokenizer semantic hashes. Snapshot metadata changes alter the complete tree digest too.

Prompt overlap is checked after whitespace/case normalization across declared exclusion corpora, KEEP and all evaluation shards. The built-in normalization calibration text is excluded automatically (do not repeat it in exclusion files). Undeclared training data, paraphrases and semantic leakage cannot be detected by this check. Keep original selection records with the experiment; a declaration cannot prove reviewer independence. The fixed built-in calibration is reused across runs and its hash and measured amplitudes are recorded. Lock the recipe/dose before observing held-out outputs.

Each workload is bounded to 128 prompts and generation to 2–256 new tokens. Seeds are offset by 128 per disjoint workload, then paired identically between baseline and intervention for each prompt. Repeated seeds on the **same** prompts are sensitivity measurements; only the first repeat of each disjoint workload enters the replication command. Never pass the entire repeat matrix to `steer-replicate` as independent evidence. The matrix and actual seeds are retained in `qualification.json`.

## Execute

From the repository root, after installing this checkout into the chosen environment:

```bash
python -m pip install -e .
python -m pip freeze > /absolute/path/to/experiment/requirements.freeze.txt
python -m aegis_lab.editing.neurosurgery.steering_qualification --plan /absolute/path/to/experiment/plan.json --out /absolute/path/to/experiment/preflight --preflight-only
python -m aegis_lab.editing.neurosurgery.steering_qualification --plan /absolute/path/to/experiment/plan.json --out /absolute/path/to/experiment/run --device cuda:0 --dtype bfloat16 --max-new-tokens 64
```

For CPU use `--device cpu --dtype float32`. Output directories must be new and outside inputs; interrupted/failed results are retained with failure status and cannot be resumed or overwritten. Start a new directory after correcting the cause. The harness never downloads weights. Preflight-only verifies identity/config/runtime availability, not generation, tokenization, memory sufficiency or hardware execution.

The run uses fixed temperature 0.7, top-p 1.0 and thinking disabled by the existing prompt builder. Every `steer` report includes generation parameters, actual seeds, hook hits, **changed activation-value counts per neuron**, baseline replay inside the context and another replay after all intervention hooks are removed. Qualification additionally compares exact unhooked last-token logits on KEEP prompts, checks hook-registry restoration, and measures baseline-to-intervention next-token KL and top-1 agreement for each KEEP prompt. Floating-point saturation/no-op edits fail delivery even when a hook executes.

Default preservation gates are mean next-token KL <= 0.05 and top-1 agreement >= 0.95. These are explicit provisional engineering thresholds, not validated persona thresholds. The underlying `steer --qualification-keep` interface exposes `--max-mean-kl` and `--min-top1-agreement` for separately preregistered experiments; this initial harness uses the defaults. KEEP metrics cover only the **last prompt token**, not full-sequence perplexity or independent task performance. Generation/token preservation alone does not establish behavioural success. Preservation failures are recorded as failed gates, and no report is automatically promoted to `qualified: true`.

## Independent labels and replication

Each run has an audited `ab.json` and blank hash-bound `labels.jsonl`. Freeze outputs first. Give reviewers blinded prompt/response material and the frozen rubric; keep arm, scale and selection metadata away from reviewers. The built-in template is hash-bound but **not blinded**; its scale column must be hidden in reviewer copies and correctly restored by a separate coordinator. Fill target_score and keep_score from actual independent review, preserving both response and prompt hashes. Do not generate labels from intervention metadata or regexes.

`qualification.json` stores `evaluate_argv` for every run and `replicate_argv` for disjoint first-repeat runs. These are argument arrays for the existing CLI, not shell strings. Execute them after filling labels, for example:

```bash
python -c 'import json,sys; from aegis_lab.editing.neurosurgery.cli import main; m=json.load(open(sys.argv[1])); [main(r["evaluate_argv"]) for r in m["runs"]]; main(m["replicate_argv"])' /absolute/path/to/experiment/run/qualification.json
```

The existing evaluation defaults require 10 prompts per run. Replication defaults require at least 2 runs, 20 total disjoint prompts and 5 per run; use larger preregistered independent samples for meaningful inference. The command deliberately retains these defaults instead of weakening them to fit fixtures. Existing replication verifies audited report/ratings lineage, matching model/recipe/calibration/rubric, disjoint prompts and seeds, and computes corrected statistical gates. It does not rerun models. Retain independently completed repeat-seed evaluations too; examine instability rather than choosing the favourable seed.

Acceptance requires reviewing all delivery/restoration and preservation gates, independent behavioural evaluation, replication statistics and observed hardware measurements. `qualification.json` remains `qualified: false` and `awaiting_independent_labels` after generation; later reports are separate immutable evidence. Hardware acceptance is manual and must include actual device/driver, peak memory, completion/latency, finite results, and repeated successful runs on the intended deployment machine. No automatic persona or universal hardware-support assertion is made.

## Executable development tests

```bash
PYTHONPATH=src python -m unittest discover -s tests/unit -p 'test_neurosurgery_steering_qualification.py' -v
PYTHONPATH=src python -m unittest discover -s tests/unit -p 'test_neurosurgery_neuron*.py'
PYTHONPATH=src python -m unittest discover -s tests/unit -p 'test_neurosurgery_steering*.py'
```

CPU toy fixtures test delivered edits, rejected no-ops, exception cleanup, restoration failure, normalized split isolation, seed collisions, stock runtime adapter resolution without weights, output collision and the audited generation → label → evaluation → replication path. Synthetic fixture ratings exercise that path only; they are not human-labelled model evidence. No CI workflow is added.
