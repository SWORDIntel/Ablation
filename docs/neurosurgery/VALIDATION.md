# Neurosurgery Validation

The structural search gate compares next-token KL divergence from the untouched model on a KEEP workload, plus top-1 token agreement. The validate command also reports teacher-forced mean NLL across each supplied text sequence. These are useful drift measurements, not proof of retained task competence.

## Current checks

- Baseline and candidate tokenizers must have identical vocabularies and special-token maps. Validation also compares each sample's token IDs and attention mask before comparing scores.
- Empty prompt sets, sample-count mismatches, incompatible vocabulary dimensions, and non-finite log probabilities fail closed.
- Model repository code is disabled by default. To run a model that requires custom code, set `AEGIS_TRUST_REMOTE_CODE=1` only after reviewing and trusting that repository. This executes its code in the current Python process.
- Neurosurgery profile and selection `.pt` files load with PyTorch `weights_only=True`. Treat older artifacts that cannot be loaded this way as untrusted; regenerate them rather than enabling unrestricted pickle loading.
- Provenance manifests (`ProvenanceManifest`, `SurgeryManifest`) bind validation and plans to immutable model/tokenizer hashes, dataset fingerprints, seeds, adapter versions, and operation ordering.
- Disjoint split validation enforces zero prompt overlap across `discovery`, `search`, `validation`, and `test` partitions.
- Exact restoration integrity checks verify SHA-256 hashes against original parent checkpoint manifests before and after restoration.

## Recommended acceptance sequence

1. Inspect model architecture, tied weights, and adapter capability matrices.
2. Search candidates on a calibration KEEP set or rank with clean/corrupted causal patching.
3. Generate a checksum-bound surgery plan and preview conflicts/indices.
4. Apply structural edits once to a separate output directory; record parent restoration manifest.
5. Reload the saved checkpoint from disk and verify export/reload parity.
6. Run targeted LoRA recovery or hypertuning with multi-objective distillation and DROP rebound monitoring.
7. Run `aegis-neurosurgery validate` and sequence-level task evaluations on independent held-out splits.
8. Quantize (INT8/INT4 with mixed precision) and package runtime deployment exports after quality acceptance.
9. Verify exact restoration integrity against the original baseline checkpoint.

Teacher-forced NLL scores every unpadded next-token target in the provided text, skipping padding and each sequence's first token. Its delta indicates how the candidate's fit to that text changed; it is not a desired-answer score. For aggressive surgery, add long-generation evaluations, perplexity, task benchmarks, and domain-specific regression suites. A low one-token KL or NLL can miss failures that appear in generation. Validation does not yet measure whether the candidate successfully changes a target behavior.

Do not reuse the exact search corpus as the only final validation corpus.