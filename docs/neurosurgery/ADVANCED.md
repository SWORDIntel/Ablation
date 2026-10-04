# Advanced instruments

These tools expand the all-in-one kit beyond structural cuts. They use a separate module CLI and Python stage libraries. Start with a validated candidate and preserve its parent checkpoint.

```bash
python3 -m aegis_lab.editing.neurosurgery.cli_extended patch --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended recover --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended hypertune --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended quantize --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended export-runtime --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended ampute --help
python3 -m aegis_lab.editing.neurosurgery.cli_extended unlearn --help
```

## Choose an instrument

| Objective | Source module within editing/neurosurgery/ | Evidence to retain |
| --- | --- | --- |
| Locate causal effects | `stage4c_causal.py` / `patch` | Paired clean/corrupted workload, component effects, KEEP damage and controls |
| Repair retained behavior | `stage5_recovery.py` / `recover` | Actual training data/losses, freeze mask, trainable counts and held-out recovery results |
| Explore constrained settings | `stage4d_hypertuning.py` / `hypertune` | Search space, budgets, failed trials, measured objectives and replayable winning trial |
| Compress weights | `stage6_quantization.py` / `quantize` | Calibration binding, bit widths, drift, reload parity and runtime qualification |
| Package and benchmark | `stage7_export.py` / `export-runtime` | Assets, checkpoint format, actual runtime load and fixed-profile latency/memory |
| Remove a modality branch | `stage8_modality.py` / `ampute` | Dependency map, shared-component checks and retained execution tests |
| Edit facts or test forgetting | `stage9_unlearning.py` / `unlearn` | Desired answers, paraphrases, locality, interference, extraction probes and residual failures |

Command help defines each handler's inputs. Use the stage APIs when you need explicit evaluators, datasets, adapters or operation composition beyond the handler's defaults. Confirm that a metric is computed from the actual model and workload before using it as an acceptance gate.

## Recovery and compression order

Apply the reviewed edit to a separate checkpoint, establish its regressions, recover against real retained/desired-output data, then compare baseline, edited and recovered candidates on the same held-out tasks. Check DROP rebound after recovery. Merge adapters only after parity testing. Quantize after structure and recovery are stable; validate the quantized artifact independently before runtime packaging.

## Workflow research

```bash
python3 -m aegis_lab.editing.neurosurgery.cli_extended workflow --help
```

This is an experimental convenience handler with placeholder stages. Its `--dry-run` checks neither geometry nor quality. It is not the execution route for the core plan format.

`pipeline_runner.run_campaign` is a Python campaign API with step hooks, journals and reports. Its default profiling and recovery are synthetic; replace them with qualified handlers before a measured campaign. Its configuration schema differs from the module CLI's workflow config. See [capabilities](CAPABILITIES.md) and [schemas](../schemas/schemas.md).
