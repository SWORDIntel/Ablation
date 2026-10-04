# Architecture of the model surgery kit

AEGIS-LAB has a direct, one-process surgery toolkit and an optional service platform. The direct path is the operator starting point; it does not depend on a scheduler or vector database.

## Surgery path

```text
checkpoint + KEEP/CHANGE/DROP workloads
  -> inventory / profiles / reversible candidate search
  -> selector or constrained plan + selection hashes
  -> read-only preview
  -> apply to separate checkpoint
  -> reload + independent validation
  -> optional recovery / quantization / runtime export
  -> final task tests + provenance report
```

| Layer | Source under src/aegis_lab/ | Responsibility |
| --- | --- | --- |
| Core entry | `editing/neurosurgery/cli.py`, `__main__.py` | Installed `aegis-neurosurgery` and module command |
| Advanced entry | `editing/neurosurgery/cli_extended.py` | Separate dispatcher for patch/recover/hypertune/quantize/export-runtime/ampute/unlearn/workflow |
| Model access | `editing/neurosurgery/common.py`, `adapters/` | Loading, transformer paths, device selection and supported geometry |
| Measurement | `profile.py`, `mlp.py`, `attention.py`, `moe.py`, `validate.py` within neurosurgery | Activations, rankings, masks and drift scores |
| Planning | `plan.py`, `selectors.py`, `preview.py`, `optimizer.py` within neurosurgery | Typed targets, selection hashes, constrained search and geometry inspection |
| Materialization | `editing/neurosurgery/apply.py` and component adapters | Directional edits, structured cuts, then layer removal |
| Advanced instruments | `editing/neurosurgery/stage4a_provenance.py` through `stage9_unlearning.py` | Evaluation/provenance, contracts, causal experiments, recovery, compression, export and knowledge edits |
| Workflow research | `editing/neurosurgery/workflow.py`, `pipeline_runner.py` | State tracking, campaign configuration and handler composition |

The Stage 4B operation-contract library is distinct from the core selector/plan format. Having a Python operation implemented does not make it available through `select` or `apply`. Workflow/campaign defaults contain demonstration behavior; consult [capabilities](../neurosurgery/CAPABILITIES.md) before treating their reports as measurements.

## Optional platform

`orchestrator/` manages jobs and worker dispatch. `workers/` contains CPU/NPU/VPU implementations; `scheduler/` handles placement. `state/` wraps QIHSE and fallback state. `artifacts/` provides hashing and content-addressed file storage. `api/` and `gui/` provide operator surfaces. The older `editing/pipeline.py` and intervention registry are adjacent workflows, not automatically unified surgery adapters.

Hardware discovery or a simulated worker success does not establish native acceleration. Persistence and recovery depend on the actual backend, artifacts and executed gates. Earlier architectural proposals are preserved in the [archive](../archive/ARCHITECTURE_20261004.md).
