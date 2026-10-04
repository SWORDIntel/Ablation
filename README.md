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
