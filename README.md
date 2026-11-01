# Ablation / AEGIS-LAB

## The all-in-one model brain surgery kit

**Inspect a model. Find what matters. Edit or cut selected components. Recover, measure and export the result.**

Ablation brings structural pruning, behavioral interventions, causal localization, targeted recovery and deployment experiments into one Python toolkit. The package is named `aegis-lab`; its core surgery command is `aegis-neurosurgery`.

Define what the patient must **KEEP**, what should **CHANGE**, and what to **DROP**. Explore reversible candidates, preview the exact tensor edits, then save a separate checkpoint and measure the outcome. “Brain surgery” describes model engineering: components can be selected precisely, but learned concepts overlap and their effects need experiments.

## What's in the kit?

| Instrument | What it does | Interface |
| --- | --- | --- |
| Inspection and profiling | Inventory layers and branches; compare KEEP/DROP activations | Core CLI |
| Structural surgery | Delete transformer blocks; slice gated-MLP channels, GQA/MHA groups and supported MoE experts | Core CLI and adapters |
| Directional editing | Contrastive weight projections with optional norm and preservation-subspace constraints | Core CLI |
| Candidate search | Reversible masks, interacting layer deletion, constrained joint search and Pareto reports | Core CLI |
| Explicit targeting | Typed selectors, checksummed selection artifacts and read-only previews | Core CLI |
| Causal experiments | Clean/corrupted activation patching, runtime interventions and feature adapters | Advanced module CLI / Python |
| **Neuron steering and A/B testing** | 19 published model-specific presets, reversible signed doses, paired A/B output, measured candidate ranking, causal tests and bounded pair interactions | Core CLI |
| Recovery and tuning | Targeted LoRA, freeze masks, distillation helpers and budgeted hypertuning | Advanced module CLI / Python |
| Compression and export | Affine INT8/packed INT4 helpers, sensitivity analysis and SafeTensors/PyTorch packaging | Advanced module CLI / Python |
| Branch and knowledge experiments | Dependency-aware modality removal, low-rank factual edits and empirical unlearning metrics | Advanced module CLI / Python |
| Optional lab services | Workers, scheduling, QIHSE state, REST API and desktop GUI | Platform launchers |

All-in-one means the instruments live together. Architecture support and integration differ across instruments; measured wrappers require real data and explicit acceptance gates. See the [capability and interface guide](docs/neurosurgery/CAPABILITIES.md).

## Install

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
aegis-neurosurgery --help
```

Python 3.9+ is declared in `pyproject.toml`. Dependencies include PyTorch, Transformers, PyYAML and pyzmq. Install a PyTorch build appropriate to your device. The core workflow can run on CPU and does not need a server, VPU or QIHSE library. Model size determines RAM/VRAM requirements. Installed operators require local checkpoints and input files so their mandatory audits can bind exact bytes; download remote checkpoints first.

Optional extras are `.[dev]`, `.[api]`, `.[gui]` and `.[hardware]`. See [setup and operations](docs/OPERATIONS.md) for services and device qualification.

## First operation: measured MLP compression

Use a supported, floating-point local Hugging Face checkpoint at `models/base`. Prepare representative `data/keep-search.txt` and `data/drop-search.txt`, plus an independent `data/keep-validation.txt`, one prompt per line. DROP profiling is required by the committed CLI; it supplies a contrast workload for channel ranking. Paths below are relative to the repository root; model and data files are inputs you supply.

```bash
# Inspect the model without changing its weights.
mkdir -p runs
aegis-neurosurgery inventory --model models/base --out runs/inventory.json

# Measure channels and search reversible candidates.
aegis-neurosurgery profile-mlp --model models/base \
  --keep data/keep-search.txt --drop data/drop-search.txt --out runs/mlp-profile

# Build a constrained plan; ratios include the unchanged baseline.
aegis-neurosurgery optimize --model models/base \
  --keep data/keep-search.txt --mlp-profile runs/mlp-profile/mlp_profile.pt \
  --mlp-ratios 1,0.95,0.90 --max-mean-kl 0.02 \
  --min-top1-agreement 0.95 --max-trials 16 --out runs/mlp-opt

aegis-neurosurgery preview --model models/base \
  --plan runs/mlp-opt/optimized_plan.yaml

# Review the preview, then write a separate checkpoint.
aegis-neurosurgery apply --model models/base \
  --plan runs/mlp-opt/optimized_plan.yaml --out models/candidate

aegis-neurosurgery validate --base models/base --candidate models/candidate \
  --keep data/keep-validation.txt --max-mean-kl 0.02 \
  --out runs/validation.json
```

Optimizer version-4 plans can be previewed and applied directly, with selection checksums and adapter geometry verified by preview. The numbers are example tolerances, not universal quality thresholds. A feasible plan can retain every component. CLI drift scores need task and generation tests before deployment. The independent validation reloads both checkpoints; do not overwrite the source or reuse an existing candidate directory.

For attention, MoE, layer deletion, directional edits and typed selectors, continue with the [surgery manual](docs/neurosurgery/README.md).

## Advanced instruments

The core entry point exposes core commands. The installed advanced entry point also dispatches the stage instruments:

```bash
aegis-neurosurgery-advanced --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended recover --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended quantize --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended export-runtime --help
```

The advanced entry point also dispatches core commands through the same handler. The module entry remains available. The advanced command `ampute` is spelled that way in the source.

## Neuron steering, presets and A/B experiments

Ablation supports reversible inference-time activation steering of **gated-MLP intermediate neurons** (input to `mlp.down_proj`). No model weights are changed. Inspect the registry with:

```bash
aegis-neurosurgery steer-presets
aegis-neurosurgery steer-presets --model-id Qwen/Qwen3.5-4B
```

### Published steering presets (19)

These are **externally reported effects, not behaviours independently reproduced or qualified by Ablation**. The coordinates use **zero-based layer/neuron indices**, and their doses use calibrated residual-norm units, not raw activation units. They apply only to the listed model/checkpoint family; do not transfer indices to another model or to a structurally edited checkpoint. The CLI binds local checkpoint bytes and accepts an optional independently trusted checkpoint digest.

**Qwen/Qwen3.5-4B** — original [Evil Qwen experiment](https://x.com/_lyraaaa_/status/2113833330429083662); additional [reported persona controls](https://www.hedwigai.com/blog/qwen-personality-neurons):

| Preset ID (`--preset`) | Reported effect | Neurons and doses |
| --- | --- | --- |
| `qwen35-4b-evil-pair` | Hostile / cartoon-villain | `L14N8949+1.0 L13N7956-1.0` |
| `qwen35-4b-evil-single` | Hostile single-neuron comparison | `L14N8949+1.0` |
| `qwen35-4b-flatterer` | Excessive praise | `L20N7450-1.0` |
| `qwen35-4b-contempt` | Contemptuous tone | `L17N8623+1.0` |
| `qwen35-4b-poet` | Poetic voice | `L17N7426+1.0` |
| `qwen35-4b-upbeat` | Relentlessly upbeat | `L5N8348-1.0` |
| `qwen35-4b-suspicious` | Suspicious tone | `L5N6573+1.0` |
| `qwen35-4b-condescending` | Condescending tone | `L11N6819+1.0` |
| `qwen35-4b-wistful` | Wistful voice | `L5N8940-1.0` |
| `qwen35-4b-oracle` | Cryptic oracle | `L5N4530-1.0` |
| `qwen35-4b-bleak` | Bleak tone | `L11N2930+1.0` |
| `qwen35-4b-dismissive` | Dismissive tone | `L6N6424-1.0` |
| `qwen35-4b-ornate` | Ornate language | `L15N9125+1.0` |
| `qwen35-4b-curt` | Curt replies | `L5N7656-1.0` |

**Qwen/Qwen3.8-27B** — [reported Qwen3.8 experiment](https://x.com/chiefofautism/status/2115274388425900291):

| Preset ID | Reported effect | Neurons and doses |
| --- | --- | --- |
| `qwen38-27b-hostile` | Hostile | `L34N3838+0.75` |
| `qwen38-27b-extra-hostile` | Stronger hostile combination | `L34N3838+0.75 L38N17180+0.75` |

**Qwen/Qwen3-8B** — one-prompt observations from the [Neuron Persona Library](https://neuronpersonas.com/):

| Preset ID | Reported effect | Neurons and doses |
| --- | --- | --- |
| `qwen3-8b-coffee-bean` | Coffee-bean self-description | `L9N166+1.0` |
| `qwen3-8b-ghost` | Ghost self-description | `L12N545-1.0` |
| `qwen3-8b-toilet` | Toilet self-description | `L29N1807+1.25` |

For the last three, the source observation used per-prompt normalization; Ablation's default fixed calibration may produce different results. All presets carry source links and an explicit *reported; not locally reproduced* provenance label in the CLI.

### Run an A/B preset

Use a locally downloaded, compatible Hugging Face checkpoint and your own prompt file (one line per prompt). Example paths below are illustrative inputs, not files shipped with the repository:

```bash
aegis-neurosurgery steer --model models/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B --preset qwen35-4b-evil-pair \
  --workload data/steering-eval.txt --scales=0.25,0.5,1.0 \
  --max-new-tokens 160 --seed 42 --out runs/evil-qwen-ab.json
```

The unsteered zero-dose baseline runs automatically. For an unsigned or reversed experimental recipe, provide explicit signed `--neuron` targets or a local JSON recipe instead of a published preset. `--expected-checkpoint-sha256` optionally verifies the canonical checkpoint tree digest from an independent trusted reference. The report records exact asset hashes, token-level outputs, seed-matched A/B results and baseline-replay integrity.

### Automatically screen and test neuron interactions

The existing `profile-mlp` and `steer-rank` commands generate correlational candidate neurons; `steer-causal` tests actual **next-token target-vs-contrast logit changes**, protected KEEP KL/top-1 drift, and matched single-neuron controls. The optional `--max-pairs` searches combinations of positively screened single neurons at their selected **signed doses**. It compares each pair against its monomers, reports additive/super-additive/sub-additive effects, and runs matched same-layer control pairs. Only the search split selects a winner; the held-out test is reported separately.

```bash
aegis-neurosurgery profile-mlp --model models/base \
  --keep data/discovery-keep.txt --drop data/discovery-target.txt --out runs/mlp-discovery
aegis-neurosurgery steer-rank --profile runs/mlp-discovery/mlp_profile.pt \
  --top-k 20 --out runs/neurons-ranked.json
aegis-neurosurgery steer-causal --model models/base \
  --profile runs/mlp-discovery/mlp_profile.pt --ranking runs/neurons-ranked.json \
  --discovery-keep data/discovery-keep.txt --discovery-drop data/discovery-target.txt \
  --search-target data/search-target.jsonl --search-keep data/search-keep.txt \
  --test-target data/test-target.jsonl --test-keep data/test-keep.txt \
  --scales=-0.5,-0.25,0.25,0.5 --max-candidates 4 --max-pairs 3 \
  --max-forwards 1024 --out runs/causal-pairs.json
```

Target files require `prompt` and either single-token `target`/`contrast` strings or their tokenizer-specific `target_token_id`/`contrast_token_id` integers. The discovery, search and test prompt sets must be disjoint. Producer audits and checkpoint binding are required; older unaudited discovery artifacts must be regenerated. Pair search is opt-in (`--max-pairs 0` by default) and limited by the explicit forward budget.

**A positive logit-margin change is not proof of a persona effect, semantic correctness or statistical significance.** No candidate or combination is automatically promoted into a verified preset. See the [full steering guide](docs/neurosurgery/NEURON_STEERING.md) for calibration, data schemas, interaction measurements, audit details and model/runtime restrictions.

### Bounded 3–5 neuron optimisation

For candidates that already passed `steer-causal` single-neuron and pairwise KEEP gates, the optional beam search expands selected pairs into larger groups and refines each signed neuron dose with one coordinate-descent pass. It uses the **existing** intervention engine, a hard preflight forward budget and search-only selection:

```bash
aegis-neurosurgery steer-causal --model models/base \
  --profile runs/mlp-discovery/mlp_profile.pt --ranking runs/neurons-ranked.json \
  --discovery-keep data/discovery-keep.txt --discovery-drop data/discovery-target.txt \
  --search-target data/search-target.jsonl --search-keep data/search-keep.txt \
  --test-target data/test-target.jsonl --test-keep data/test-keep.txt \
  --max-candidates 5 --max-pairs 8 --max-groups 12 \
  --max-group-size 4 --beam-width 2 --scales=-0.5,-0.25,0.25,0.5 \
  --keep-penalty 1.0 --neuron-penalty 0.02 --max-forwards 8192 \
  --out runs/causal-groups.json
```

The optimiser keeps Pareto-ranked experimental groups and compares the search-selected group to its parent, all constituent monomers and a same-layer control group. Only one group undergoes the final held-out evaluation. `--max-groups 0` (default) disables it and preserves previous pair/single behaviour. The objective is a **next-token logit margin**, not semantic correctness or verified personality control; this feature never auto-publishes presets.


### Independently labelled response evaluation

After an A/B generation run, use the **offline** `steer-label-template` command to create hash-bound annotation rows. Have an evaluator rate target behaviour and KEEP retention against a predefined rubric, then aggregate the completed ratings using `steer-evaluate`:

```bash
aegis-neurosurgery steer-label-template --ab-report runs/evil-qwen-ab.json \
  --rubric docs/neurosurgery/STEERING_LABEL_RUBRIC_EXAMPLE.md \
  --out runs/ratings-template.jsonl
# Independently score responses and save completed values to ratings-completed.jsonl.
aegis-neurosurgery steer-evaluate --ab-report runs/evil-qwen-ab.json \
  --ratings runs/ratings-completed.jsonl \
  --rubric docs/neurosurgery/STEERING_LABEL_RUBRIC_EXAMPLE.md \
  --min-prompts 10 --out runs/labelled-evaluation.json
```

The tool rejects stale or missing annotations by matching the exact prompt/response SHA-256 hashes and reports paired target gains, KEEP losses and descriptive confidence intervals per dose. **It does not perform semantic judgement automatically** or certify independence/quality of externally supplied labels. See the [example scoring rubric](docs/neurosurgery/STEERING_LABEL_RUBRIC_EXAMPLE.md).


### Replicate a locked steering recipe across independent runs

Use `steer-replicate` to combine two or more **separately generated and labelled A/B reports** that share the same checkpoint, recipe, calibration and scoring rubric but use distinct seeds and prompts:

```bash
aegis-neurosurgery steer-replicate \
  --ab-report runs/rep1/steering.json --evaluation runs/rep1/evaluation.json \
  --ab-report runs/rep2/steering.json --evaluation runs/rep2/evaluation.json \
  --locked-scale 0.5 --min-prompts 40 \
  --out runs/replication.json
```

The command checks report audits, source and checkpoint hashes, non-overlapping prompts, per-run target/KEEP retention, cluster-bootstrap intervals and Holm-corrected significance across all tested strengths. Pass `--causal-report` to verify that the locked neuron/dose and replication prompts are disjoint from earlier causal-search data. Outputs remain **exploratory**, not automatically verified persona presets. See the [full steering guide](docs/neurosurgery/NEURON_STEERING.md).

The report's `candidate_recipe` contains the locked **effective** dose, and `steer --config` enforces its embedded checkpoint SHA-256 before loading an incompatible checkpoint.

## What results mean

- Physical cuts can reduce parameter storage; masked-search timings are not deployment speedups.
- Dense surgery requires compatible separate projection tensors and representable config dimensions. Packed quantized tensors require format-specific handling.
- KEEP KL/top-1 agreement and text NLL measure drift. They do not establish target success or complete retained competence.
- Empirical unlearning reports tested suppression and residual failures; it cannot certify erasure of knowledge from weights.
- Workflow and campaign defaults use real measurements and reject missing configured gates. Qualify task, architecture and runtime behavior against your own data; see the [trained-model diagnostic](docs/neurosurgery/QUALIFICATION.md).

See [validation](docs/neurosurgery/VALIDATION.md), [capabilities](docs/neurosurgery/CAPABILITIES.md) and the [roadmap](docs/neurosurgery/ROADMAP.md) for the current boundaries.

## Documentation and development

Start at the [docs index](docs/README.md). It links setup, architecture, artifact schemas, component manuals, research provenance and historical platform notes.

```bash
python3 -m pip install -e '.[dev]'
PYTHONPATH=src python3 -m unittest discover -s tests/unit -p 'test_neurosurgery_*.py'
PYTHONPATH=src python3 -m unittest \
  tests.integration.test_neurosurgery_core_workflow \
  tests.integration.test_neurosurgery_measured_workflow \
  tests.integration.test_neurosurgery_pipeline_acceptance
```

Test fixtures are implementation checks, not a model-quality or hardware-performance certificate. See [verification notes](docs/VERIFICATION.md) for the latest documentation review.
