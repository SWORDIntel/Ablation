# Neurosurgery Validation

The structural search gate compares next-token KL divergence from the untouched model on a KEEP workload, plus top-1 token agreement. The validate command also reports teacher-forced mean NLL across each supplied text sequence. These are useful drift measurements, not proof of retained task competence.

## Current checks

- Baseline and candidate tokenizers must have identical vocabularies and special-token maps.
- Empty prompt sets, sample-count mismatches, incompatible vocabulary dimensions, and non-finite log probabilities fail closed.
- Model repository code is disabled by default. To run a model that requires custom code, set `AEGIS_TRUST_REMOTE_CODE=1` only after reviewing and trusting that repository. This executes its code in the current Python process.
- Neurosurgery profile and selection `.pt` files load with PyTorch `weights_only=True`. Treat older artifacts that cannot be loaded this way as untrusted; regenerate them rather than enabling unrestricted pickle loading.
- These checks do not yet bind validation to immutable model/tokenizer revisions or dataset fingerprints.

## Recommended acceptance sequence

1. Search candidates on a calibration KEEP set.
2. Generate a checksum-bound surgery plan.
3. Apply structural edits once to a separate output directory.
4. Reload the saved checkpoint from disk.
5. Run `aegis-neurosurgery validate` on a separate KEEP validation set.
6. Run task-specific benchmarks before promotion.
7. Quantize or produce hardware-specific packed artifacts only after quality acceptance.

Teacher-forced NLL scores every unpadded next-token target in the provided text, skipping padding and each sequence's first token. Its delta indicates how the candidate's fit to that text changed; it is not a desired-answer score. For aggressive surgery, add long-generation evaluations, perplexity, task benchmarks, and domain-specific regression suites. A low one-token KL or NLL can miss failures that appear in generation. Validation does not yet measure whether the candidate successfully changes a target behavior.

Do not reuse the exact search corpus as the only final validation corpus.