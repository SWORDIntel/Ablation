# Setup and operations

## Core kit

Use the virtual-environment install in the [root README](../README.md). Run commands from the repository root. Without an editable install, prefix module invocations with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m aegis_lab.editing.neurosurgery --help
PYTHONPATH=src python3 -m aegis_lab.editing.neurosurgery.cli_extended recover --help
```

Core and advanced commands use different dispatchers. See [capabilities](neurosurgery/CAPABILITIES.md). Local HF model directories must contain compatible config, tokenizer and checkpoint assets. A GGUF file is not a Hugging Face structural-surgery checkpoint.

Start on a small model with `--device cpu` where offered. Reserve enough disk for the untouched source, selection/profile artifacts, candidate and exports. Recovery adds optimizer/adapter memory; loading baseline and candidate for validation can be expensive. Inspect each subcommand's help for device defaults and batch limits.

Use representative KEEP discovery/search prompts and separate validation/test prompts. CHANGE datasets need desired outputs; DROP needs a task-specific measure. Dataset/provenance helpers must be invoked explicitly where the chosen command does not bind them automatically.

## Optional services

```bash
python3 -m pip install -e '.[api]'
python3 -m pip install -e '.[gui]'
python3 aegis.py orchestrator
python3 aegis.py api --api-port 18000
python3 aegis.py worker
python3 aegis.py gui
```

Run long-lived services in separate terminals. `aegis` is the job-management console entry point, while `aegis.py` is the repository launcher. Their arguments differ; inspect `--help`. Jobs use the platform orchestrator and do not automatically execute every neurosurgery stage.

`bootstrap.sh` performs additional environment/native setup; inspect it before running because it can download and install dependencies. `launch.sh` starts platform processes. Neither is required for direct local surgery.

## Hardware and state

The hardware extra declares OpenVINO >=2024.0. The legacy MYRIAD setup in `scripts/vpu_env.sh` instead targets OpenVINO 2022.3; keep this qualification environment separate and do not assume the extra makes Movidius devices usable.

```bash
./scripts/vpu_probe.sh
./scripts/vpu_env.sh --shell
```

A VPU simulation result is a worker-contract check, not device inference. See [VPU integration](VPU_INTEGRATION.md). CPU/CUDA and other device availability depend on the installed PyTorch/runtime and driver, not the presence of hardware alone.

Platform state uses `state/db.py` and `state/qihse_wrapper.py`. Check the selected backend and persistence behavior; an in-memory fallback does not provide durable recovery. `AEGIS_QIHSE_LIB_PATH` selects a native library. Keep authentication tokens in the environment, not tracked files.

## Candidate handling

Retain the source checkpoint and all profile/selection artifacts. Use fresh output directories. Read-only preview resolves geometry and selection checksums; it cannot establish model quality. Apply once, reload, evaluate independent KEEP and target tasks, then recover/quantize/export as needed and re-evaluate each final artifact.

Direct commands do not constitute an automatically gated deployment pipeline. Record software versions, model/tokenizer revision, data fingerprints, seed, dtype, operation order, runtime/device, and evaluation results with every accepted candidate.
