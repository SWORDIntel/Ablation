# Plans, selectors and artifacts

AEGIS-LAB contains several schemas. They are not interchangeable, even where their version labels match. Source dataclasses and loaders are authoritative.

## Core surgery inputs

| Artifact | Producer / consumer | Meaning |
| --- | --- | --- |
| Prompt text | Profile/search/validate core commands | One prompt per line; supply separate search and held-out files |
| `profile.json`, `profile.pt` | Residual `profile`; `plan` / directional `apply` | Layer scores, directions and preservation bases |
| `mlp_profile.pt`, `attention_profile.pt`, `moe_profile.pt` | Component profiling; matching search/optimizer | Component importance and geometry |
| Search JSON | Component/layer search; `plan` / optimizer | Candidate drift and estimated removals |
| Selector YAML | `select` | Version 1; explicit layer IDs and retained channel/group/expert indices |
| Surgery-plan YAML | `plan`, `select`, `optimize`; `preview` / `apply` | Directional edits and structural selections, with linked selection artifacts |
| Selection `.pt` files | Plan builders; `preview` / `apply` | Exact retained indices bound by SHA-256 |
| `preview.json` | `select` | Resolved targets, shapes, aliases and byte estimates |
| Validation JSON | `validate --out` | Drift comparisons and teacher-forced NLL, not task acceptance by itself |

Optimizer plans use version 4; selector/plan-builder outputs use version 3. Preview accepts versions 1–4. Pass optimizer plans directly to preview/apply with their linked selection files; no conversion is required. MoE artifacts use adapter block ordering while previews report source transformer-layer IDs. Explicit MoE selector YAML still needs adapter-specific source-layer maps.

Keep linked plan and selection files together. The core loaders check selection hashes; this is not a blanket guarantee that every command binds the complete model, tokenizer and data provenance. See [validation](../neurosurgery/VALIDATION.md).

Structural selector sections are `mlp.keep_indices`, `attention.keep_groups` and `moe.keep_experts`. Maps use source-layer IDs. Supported global configs require uniform retained dimensions. Directional targets are module paths relative to selected transformer layers. The [manual](../neurosurgery/README.md#explicit-selector-files) shows a schema illustration; adapt it to actual inventory and geometry.

## Advanced library formats

`stage4a_provenance.py` provides dataset fingerprints and provenance/restoration manifests. `stage4b_contract.py` defines an operation contract for mask, scale, clamp, project, replace, low-rank delta and physical remove. These are Python library contracts, not arbitrary extra fields accepted by the core plan loader.

`pipeline_runner.py` defines `CampaignConfig`, `ModelConfig`, dataset specs, steps, thresholds and objectives. Its campaign schema version is `1.0` (also accepts `1`). `cli_extended workflow --config` reads a different, simpler configuration: its `model` is a string, whereas a campaign uses a model mapping with `path`. Do not feed one format to the other. Default campaign profile/recovery metrics include synthetic values.

## Platform storage

`state/db.py` defines jobs, stages and other metadata stores. `artifacts/store.py` stores file content at `<first-two-hash-chars>/<next-two>/<sha256>` beneath its configured root; metadata registration is optional. This file store does not imply complete model provenance or native database durability.

Earlier job/manifest descriptions remain in the [schema archive](../archive/SCHEMAS_20261004.md); verify fields against the current producer before writing an integration.
