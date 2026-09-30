# Model Neurosurgery Roadmap

## Implemented

### Stage 0 — measurement substrate
- KEEP/DROP residual profiling
- KL and top-1 comparison
- reproducible artifacts
- model/module inventory

### Stage 1 — directional surgery
- generalized DROP−KEEP residual directions
- norm-preserving projected edits
- KEEP preservation basis via SVD/null-space constraint

### Stage 2 — dense structural surgery
- whole-layer deletion search
- greedy interacting layer deletion
- gated-MLP channel profiling/search
- physical gate/up/down tensor slicing
- checksum-bound surgery plans

### Stage 3 — attention and MoE structural surgery
- GQA/MHA output-group profiling
- reversible attention-group masking
- physical Q/K/V/O slicing with config repair
- MoE routing probability/top-k profiling
- reversible router masking
- physical expert/router slicing with config repair

## Next

### Stage 4 — search optimizer
Replace hand-supplied ratio sweeps with constrained multi-objective search. Candidate objective:

```text
maximize bytes_removed / latency_improvement
subject to KEEP mean_KL <= threshold
           retained task scores >= floors
```

The search engine should operate on masks and only materialize the final candidate.

### Stage 5 — recovery/distillation
Use the untouched model or a stronger teacher on H100/L40S-class hardware to recover quality after a structural cut. First target: short post-surgery LoRA/full-parameter recovery on retained-domain data.

### Stage 6 — quantization after surgery
Run sensitivity-driven mixed quantization only after structural dimensions are stable. Avoid spending calibration effort on weights that will later be removed.

### Stage 7 — target-specific packing
Produce deployment artifacts already blocked/reordered for the target runtime: Vulkan/OpenCL/other accelerator kernels, fixed context profiles, packed low-bit weights, and preallocated KV layouts.

### Stage 8 — modality amputation
Add architecture adapters for physically removing independently parameterized vision/audio encoders, projectors, image decoders, and other unreachable modality branches while preserving reload semantics.
