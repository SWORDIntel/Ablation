# Ablation documentation

Ablation / AEGIS-LAB is an all-in-one model brain surgery kit. Start with a local checkpoint and explicit KEEP / CHANGE / DROP objectives, then select the instrument appropriate to the experiment.

## Operator path

1. [Project overview and quick start](../README.md)
2. [Setup, devices and optional services](OPERATIONS.md)
3. [Surgery manual](neurosurgery/README.md)
4. [Capability and interface boundaries](neurosurgery/CAPABILITIES.md)
5. [Validation and acceptance](neurosurgery/VALIDATION.md)
6. [Roadmap](neurosurgery/ROADMAP.md) and [verification record](VERIFICATION.md)

## Instruments

| Guide | Scope |
| --- | --- |
| [MLP surgery](neurosurgery/MLP_SURGERY.md) | Coupled gate/up/down channel slicing |
| [Attention surgery](neurosurgery/ATTENTION_SURGERY.md) | Coupled Q/K/V/O group slicing |
| [MoE surgery](neurosurgery/MOE_SURGERY.md) | Expert and router removal |
| [Joint optimizer](neurosurgery/OPTIMIZER.md) | Constraints, candidate search and materialization |
| [Advanced instruments](neurosurgery/ADVANCED.md) | Causal patching, recovery, tuning, quantization, export and knowledge experiments |
| [Refusal ablation](MODEL_REFUSAL_ABLATION_GUIDE.md) | Separate legacy behavior-editing workflow |
| [VPU integration](VPU_INTEGRATION.md) | Optional worker qualification and simulation limits |

## Developer references

- [Architecture](architecture/architecture.md)
- [Artifact schemas](schemas/schemas.md)
- [Research provenance](neurosurgery/THIRD_PARTY.md)
- [Contributor guidelines](../AGENTS.md)

## Historical material

The [old platform overview](archive/README_PLATFORM_20261004.md), [hardware design](archive/VPU_INTEGRATION_20261004.md), [architecture](archive/ARCHITECTURE_20261004.md) and [schema notes](archive/SCHEMAS_20261004.md) preserve earlier descriptions. Their feature and performance claims require current verification.

The [Khoj study](KHOJ_ABLATION_STUDY.md), [hardware notebook](../hardware_docs/THE_TRUE_GOLDEN_BIBLE.md) and [UI benchmark report](../not_stisla/not_stisla/benchmark_report.md) are separate historical/reference material, not acceptance evidence for model surgery.
