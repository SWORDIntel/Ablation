# Legacy refusal-ablation experiments

Refusal ablation is one behavior-editing instrument in Ablation's broader [model brain surgery kit](../README.md). This is a separate workflow from HF structural pruning; it does not establish universal removal of a learned behavior.

## Interfaces

```bash
bash ablate_model_refusal.sh --help
# No arguments launches the interactive TUI.
bash ablate_model_refusal.sh
```

The wrapper accepts positional input, output, method (`zero`, `prune`, `clamp`), strategy (`ablation` or `heretic`) and optional Heretic config. Although its usage mentions model directories, its current input check requires a regular file. Use the underlying Python interface for directory-based experiments and inspect its loader support.

```bash
PYTHONPATH=src python3 src/aegis_lab/editing/model_refusal_ablation.py --help
```

Heretic mode is report-only unless `--apply-heretic-edits` is supplied. Its config and imported datasets affect the experiment; keep these and the untouched baseline with every report. The repository has config files in multiple locations: choose an existing file after reviewing its contents rather than copying an obsolete path.

## GGUF path

`src/aegis_lab/editing/gguf_ablation.py` provides `GGUFRefusalAblator`. GGUF modifications are format-specific and separate from the neurosurgery adapters' floating-point tensor cuts. Qualify the exact tensor encoding and load the output in the intended runtime before accepting results.

`run_gguf_batch_ablation.py` is an Ollama-oriented batch script with a built-in model/blob catalog and storage assumptions. It is not a generic `--input/--output` converter. Inspect the script and its help before a batch run; model registration and export are side effects.

## Evaluate the experiment

Compare actual generated outputs and retained task performance against baseline on independent data. Heuristic layer/neuron identification is a candidate proposal, not causal proof that a component exclusively encodes refusal. Non-refusal alone is not answer quality, target correctness or erasure evidence. Keep source weights and validate saved artifacts independently.

For structural compression, causal experiments, recovery and export, use the [surgery manual](neurosurgery/README.md) and [capability guide](neurosurgery/CAPABILITIES.md). The [previous refusal guide](archive/REFUSAL_GUIDE_20261004.md) remains as historical context; its paths and broad behavior guarantees are not current instructions.
