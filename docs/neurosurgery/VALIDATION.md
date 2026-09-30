# Neurosurgery Validation

The search metric is next-token KL divergence from the untouched model on a held-out KEEP workload, plus top-1 token agreement.

This metric is intentionally cheap enough to run for many structural candidates. It is a **gate**, not a claim that the models are behaviorally identical.

Recommended acceptance sequence:

1. search candidates on a calibration KEEP set;
2. generate a checksum-bound surgery plan;
3. apply structural edits once;
4. reload the saved checkpoint from disk;
5. run `aegis-neurosurgery validate` on a separate KEEP validation set;
6. run task-specific benchmarks before promotion;
7. only then quantize or produce hardware-specific packed artifacts.

For aggressive surgery, add long-generation evaluations, perplexity, task benchmarks, and domain-specific regression suites. A low one-token KL can miss failures that appear later in generation.

Do not reuse the exact search corpus as the only final validation corpus.
