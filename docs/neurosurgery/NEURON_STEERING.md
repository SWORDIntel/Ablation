# Neuron steering and controlled A/B experiments

The core `aegis-neurosurgery` CLI now supports **reversible runtime MLP-neuron edits** with signed doses, scale sweeps, paired generations and a small, provenance-labelled library of published persona effects.

These experiments **do not change checkpoint weights**. The intervention is a forward pre-hook on the *input to* a transformer's gated MLP `down_proj`; it edits one intermediate activation per token, not a raw residual coordinate. The experimental feature is independent of structural pruning, recovery and deployment export.

## Show model/effect presets

Run from the repository root after `pip install -e .`:

```bash
aegis-neurosurgery steer-presets
aegis-neurosurgery steer-presets --model-id Qwen/Qwen3.5-4B
```

The curated registry currently includes:

- **Qwen/Qwen3.5-4B:** Lyra's hostile/evil pair (`L14N8949+1.0 L13N7956-1.0`), the single-neuron control and additional reported tones (flatterer, contempt, poet, upbeat, suspicious, condescending, wistful, oracle, bleak, dismissive, ornate, curt) from independent experiments.
- **Qwen/Qwen3.8-27B:** reported hostile (`L34N3838+0.75`) and extra-hostile (`L34N3838+0.75 L38N17180+0.75`) combinations.
- **Qwen/Qwen3-8B:** three single-prompt self-description examples from the Neuron Persona Library.

All presets include a source URL and carry the evidence label **reported; not locally reproduced**. They are hypotheses for your own A/B runs, not verified universal personality controls. Qwen3-8B library examples were originally calibrated on the prompt being tested, while the default runner uses one fixed calibration text; outcomes may differ. Do not copy these neuron indices to other models or architecture revisions.

Sources: [Lyra](https://x.com/_lyraaaa_/status/2113833330429083662), [independent Qwen3.5-4B evaluation](https://www.hedwigai.com/blog/qwen-personality-neurons), [Qwen3.8-27B experiment](https://x.com/chiefofautism/status/2115274388425900291), [Neuron Persona Library](https://neuronpersonas.com/), [Neuronary reference implementation](https://github.com/lyramakesmusic/neuronary/blob/main/scripts/run.py).

## P1 first step — screen neuron candidates from an existing MLP profile

The `steer-rank` command consumes the existing Stage 3 `profile-mlp` artifact instead of duplicating activation capture. Prepare **independent** KEEP (behaviour to preserve) and DROP (target behaviour or contrasting corpus) prompt files. Then:

```bash
aegis-neurosurgery profile-mlp \
  --model models/Qwen3.5-4B \
  --keep data/keep-search.txt --drop data/target-search.txt \
  --out runs/target-mlp-profile

aegis-neurosurgery steer-rank \
  --profile runs/target-mlp-profile/mlp_profile.pt \
  --top-k 30 --out runs/target-neuron-candidates.json
```

The generated JSON includes target-enriched candidate channels, KEEP-enriched negative controls, per-layer channel counts and source-profile SHA-256. It uses the *difference of per-layer-normalized mean absolute activations*, `DROP/mean(DROP) - KEEP/mean(KEEP)`, to rank channels. The loader uses PyTorch `weights_only=True`, and the existing operator audit verifies a profile producer's checksum when an audit is available.

**Interpretation:** This is **correlational screening**, not a proof that altering a channel changes the behaviour or that the sign of the activation contrast predicts the correct additive steering direction. Follow candidates with single-neuron A/B dose sweeps, matched-random controls, target-specific scoring and independent held-out tests. Later P1 milestones add causal testing, pair synergy and constrained optimisation.

## P1 causal screen — dose trials, matched controls and held-out objectives

`steer-causal` is a **measured intervention experiment** built on `steer-rank`, the existing gated-MLP `profile-mlp` artifact and `NeuronSteerer`. It tests whether pushing a candidate MLP channel causally changes a *specified next-token logit margin*, not whether a complex persona or quality objective has been achieved.

Generate a **fresh** MLP profile on discovery corpora with the current code, then rank candidates. New profiles carry exact discovery-file SHA-256 hashes. The installed command also requires successful producer audits for both the MLP profile and ranking, each bound to the exact current checkpoint bytes. Rebuild unaudited legacy artifacts rather than assuming the source model is identical. Older profiles without these bindings must be regenerated before causal testing; this prevents silently asserting that an arbitrary rank list came from a particular discovery corpus.

Provide six **disjoint** prompt sets: the two original discovery inputs used by `profile-mlp`; independent search-target and search-KEEP inputs; and unseen test-target and test-KEEP inputs. The target files are JSONL/JSON arrays with rows such as:

```json
{"prompt":"The capital of France is","target_token_id":123,"contrast_token_id":456}
```

Use **actual token IDs for the selected tokenizer**. Alternatively, supply single-token strings under `target` and `contrast` and the runner will require that each encodes to exactly one token. These numbers are illustrative, *not* real tokenizer IDs. This probes next-token preference only; longer completions and behavioural labelling require an independent semantic evaluator.

```bash
aegis-neurosurgery profile-mlp \
  --model models/local-model --keep data/discovery-keep.txt \
  --drop data/discovery-target.txt --out runs/mlp-discovery

aegis-neurosurgery steer-rank \
  --profile runs/mlp-discovery/mlp_profile.pt --top-k 20 \
  --out runs/neurons-ranked.json

aegis-neurosurgery steer-causal \
  --model models/local-model \
  --profile runs/mlp-discovery/mlp_profile.pt \
  --ranking runs/neurons-ranked.json \
  --discovery-keep data/discovery-keep.txt \
  --discovery-drop data/discovery-target.txt \
  --search-target data/search-target.jsonl \
  --search-keep data/search-keep.txt \
  --test-target data/test-target.jsonl \
  --test-keep data/test-keep.txt \
  --scales=-0.5,-0.25,0.25,0.5 \
  --max-candidates 2 --controls-per-candidate 1 \
  --max-forwards 512 \
  --max-mean-kl 0.05 --min-top1-agreement 0.95 \
  --out runs/causal-screen.json
```

For development fixtures, `--normalization raw` uses absolute input activations instead of median-residual normalized doses. The default uses Neuronary-style per-layer calibration. The experiment applies each dose at the pre-`down_proj` intermediate-channel input, evaluates the identical input against the unsteered model, and reports per-prompt target-minus-contrast logit-margin changes. On KEEP prompts, it measures next-token KL divergence and top-1 agreement against baseline.

Each candidate is tested with equal-magnitude positive and negative doses and approximately **activity-/projection-norm-matched same-layer random control neurons**, sampled from a nearest-magnitude pool. This is better than arbitrary unrelated-layer controls but not perfect feature matching or a randomized clinical study. All trials are bounded by a precomputed forward-pass budget; errors reject a run before writing its report. A best search candidate is chosen only if its positive margin effect exceeds matched controls while KEEP gates pass. **Only the search split decides the winner.** The winner and its matched controls are then evaluated on the held-out test split, which is never used to choose the dose.

The report includes per-prompt margin effects, 300-resample paired-bootstrap descriptive intervals, KEEP KL and top-1 metrics, same-layer control matching distances, discovery and split provenance, checkpoint byte hashes, budget, selected dose, and held-out observation. It verifies that the unmodified model's test logits are **exactly restored** after intervention hooks are removed.

Even when the observed held-out gate passes, the report always sets `promotable_as_verified_persona: false`. Binary target-token margin is **not** a semantic behavioural metric, and a small held-out sample or selected candidate is not proof of statistical significance. Add rubric-labelled task evaluation and independent replication before promoting presets.


## P1 pair interactions and signed-dose combination search

The existing `steer-causal` command now accepts `--max-pairs N` (default `0` for backward compatibility). Pair trials reuse the same checkpoint, discovery ranking, **search-target** logit objective, search-KEEP gates, neuron calibration and audited inputs as single-neuron trials; there is no second steering implementation.

Run the original documented `steer-causal` example with `--max-candidates 4 --max-pairs 3 --max-forwards 1024`. The extra budget accounts for pair and matched-control-pair search runs, plus a possible held-out pair and its two individual monomers. The command rejects an insufficient budget before executing any model forward. If fewer than two screened single-neuron/dose conditions satisfy positive target/KEEP/control gates, it reports no pairs rather than manufacturing results.

Pair construction:
- Take each candidate's **best passing signed single-neuron dose**, chosen only on the search set, then try a bounded number of distinct pairs.
- Apply both doses simultaneously in one `NeuronSteerer` recipe, including across different layers if selected.
- Choose distinct same-layer, activation-/projection-norm-matched controls for each member, excluding candidate coordinates and duplicate controls; use the same signed doses on the control pair.
- Measure joint target-vs-contrast logit-margin change and KEEP KL/top-1 damage. Record gain over the best monomer, gain over the control pair, and **prompt-aligned synergy**:

```text
synergy = (margin_AB − margin_baseline)
          − (margin_A − margin_baseline)
          − (margin_B − margin_baseline)
```

Positive synergy is super-additive and negative synergy is sub-additive under this **specific next-token objective**. A useful additive combination need not have positive synergy. Descriptive bootstrap intervals quantify prompt variation but are not corrected for searching multiple candidate pairs.

A pair is selected on the search split only if it improves on the best monomer, exceeds its matched-control pair and satisfies KEEP gates. The selected pair is independently measured on the held-out set along with its two monomers and its matched control pair. The report provides `pair_search.trials`, `pair_search.selected`, `heldout_pair` and `search_budgets.pair_budget_count`, plus unchanged single-neuron fields.

**Boundaries:** Pair search is a bounded first increment, not a full combinatorial beam/TPE optimizer or proof of any persona. A joint next-token margin can rise while generation quality degrades; `promotable_as_verified_persona` remains false. Actual multi-turn semantic evaluation, calibration on verified models, repeat seeds and multiple-comparison correction remain later roadmap work.


## P1 beam/coordinate optimisation for larger neuron groups

`steer-causal` now supports optional `--max-groups N` (default `0`) with `--max-group-size 3..5`, `--beam-width 1..8`, `--keep-penalty` and `--neuron-penalty`. Group search requires `--max-pairs > 0` and at least three ranked candidates, because it grows only search-qualified pairs. It uses the same audited discovery/search/test splits, calibration, reversible `NeuronSteerer` hooks and logit-margin/KEEP metrics as the existing causal experiment.

At each depth, the optimiser extends a bounded beam of passing groups with the best search-qualified remaining monomers, deduplicates group coordinates, and performs **one bounded coordinate-descent pass** over the existing signed dose grid. A candidate dose is retained only if its KEEP gate passes and its objective improves:

```text
J = mean target-minus-contrast logit-margin gain
    − keep_penalty × mean KEEP KL
    − neuron_penalty × active neuron count
```

After tuning, each group is measured against a distinct activity- and down-projection-norm-matched control group. An expansion qualifies only if KEEP limits pass and target margin improves on its parent and its matched controls. A bounded Pareto report preserves nondominated qualified groups on target margin, KEEP KL and neuron count. `--max-groups` limits *attempted expansions*, not the number of winners.

The number of search and potential held-out forward calls is **conservatively calculated before model execution**. The selected group (if any) is chosen strictly from search-set observations. On held-out prompts it is measured once alongside its control group, all individual component neurons at their selected doses and its parent subgroup. Reports add `group_search` and `heldout_group` without changing existing single/pair report fields. The exact post-intervention test baseline is replayed.

Example flags for the documented `steer-causal` command:

```bash
--max-candidates 5 --max-pairs 8 --max-groups 12 \
  --max-group-size 4 --beam-width 2 \
  --keep-penalty 1.0 --neuron-penalty 0.02 --max-forwards 8192
```

**Limits:** The process is bounded beam expansion, not exhaustive combinatorial search or global TPE/gradient optimisation. Bootstrap intervals are descriptive and uncorrected for candidate search. Model effect, task-level behavioural quality, independent real-model replication and authentic upstream checkpoint identity still require their own qualification. Reports do **not** certify a persona.


## Offline independently labelled response scoring

`steer-evaluate` reads a completed `steer` generation report and a separate file of human- or otherwise externally assigned target/KEEP scores. It performs **no model inference or remote API calls** and does not pretend that regex differences or logit-margin improvements prove a semantic effect.

The operator must define a rubric before examining generated responses. A shipped [example rubric](STEERING_LABEL_RUBRIC_EXAMPLE.md) demonstrates the two required ratings (`target_score`, `keep_score`, each in `[0,1]`) and the blank-annotation workflow:

```bash
aegis-neurosurgery steer-label-template --ab-report runs/steering-ab.json \
  --rubric docs/neurosurgery/STEERING_LABEL_RUBRIC_EXAMPLE.md \
  --out runs/labels.jsonl
# Complete every rating score on all baseline and steered arms.
aegis-neurosurgery steer-evaluate --ab-report runs/steering-ab.json \
  --ratings runs/completed-labels.jsonl \
  --rubric docs/neurosurgery/STEERING_LABEL_RUBRIC_EXAMPLE.md \
  --min-prompts 10 --min-target-gain 0.1 --max-keep-loss 0.05 \
  --out runs/labelled-evaluation.json
```

The annotations bind to both a prompt hash and the **exact decoded response SHA-256**, plus numeric steering scale. Ratings must cover every A/B arm once. The completed source report must have a successful `steer` producer audit and baseline replay. Reports bind the rubric, ratings and source-report bytes, aggregate prompt-paired target-gain / KEEP-loss deltas with 300-resample descriptive bootstrap intervals, and distinguish observed threshold flags from conservative interval threshold flags.

No positive threshold is automatically a validated persona: the ratings are externally supplied and software cannot establish annotator independence, study power, data-split independence or generalisability. The report always sets `promotable_as_verified_persona: false`.


## P1 independent labelled replication — locked recipe, multiple runs

Use the existing `steer` + `steer-label-template` + `steer-evaluate` sequence to produce **two or more** independently labelled A/B runs. Different runs must use the **same checkpoint bytes, neuron recipe, calibration and rubric**, but independent non-overlapping prompt sets and different generation seeds. Every `steer-evaluate` run must cover **all** baseline and steered arms.

Then aggregate them **offline**, with no new model inference:

```bash
aegis-neurosurgery steer-replicate \
  --ab-report runs/repl-01/steering.json --evaluation runs/repl-01/evaluation.json \
  --ab-report runs/repl-02/steering.json --evaluation runs/repl-02/evaluation.json \
  --locked-scale 0.5 \
  --min-runs 2 --min-prompts 40 --min-per-run 15 \
  --min-target-gain 0.05 --max-keep-loss 0.10 --alpha 0.05 \
  --out runs/steering-replication.json
```

These are example paths and acceptance thresholds, **not** claims of a completed real-model experiment. An optional `--causal-report runs/causal-screen.json --causal-selection single` binds the locked dose/neuron to the **search-selected** single, pair, or group from the previous `steer-causal` report. It checks that the replication corpus does not overlap any previous discovery, search, or held-out causal-test split and that normalization and calibration are identical. Without a causal source, it is an independent replication of an operator-supplied recipe, not proof of the original candidate's causal lineage.

The report's `candidate_recipe` rewrites neuron doses to their **effective values at the locked scale** so it can be replayed at scale 1.0 without changing the intervention. It includes `checkpoint_sha256`, which `steer --config candidate.json` now checks **before model loading**. To reuse it, extract the JSON `candidate_recipe` object to a standalone file; it is **not** a verified model-independent preset.

The command:

- Verifies every source `steer` and `steer-evaluate` producer audit and corresponding output hashes; requires evaluator input bindings for the exact A/B report, rubric and ratings.
- Rejects mismatched model byte hashes, tokenizer/config semantic hashes, recipes, calibration, rubrics, tested dose sets, duplicated prompt hashes, repeated generation seeds, incomplete ratings and unexecuted steering arms.
- Holds the **operator-selected** `--locked-scale` fixed. It does not select the most impressive tested dose from replication outcomes.
- Reports paired target-score gains and KEEP-score losses, per-run effects, two-level run/prompt bootstrap intervals, a one-sided paired sign-flip target test, and Holm-adjusted p-values over **every** evaluated scale.
- Requires a positive effect in every run plus the configured gain, KEEP-loss and corrected-significance gates for an **exploratory replication pass**. It includes a model-bound, explicitly unverified `candidate_recipe` in the report, without making it an accepted or deployable preset.

The statistics test annotated ratings, **not** a general notion of alignment or persona reliability. Replication prompt diversity and annotator independence cannot be established from hashes alone. In particular, the corrected p-value controls comparisons **within the tested scale family**, not previously searched neurons, edited thresholds, unpublished experiments, or unobserved near-duplicate prompts. If the dose or thresholds were chosen after seeing these results, treat the conclusion as exploratory. A full verified-preset promotion step remains on the roadmap.

## Paired trials

Download a supported Hugging Face checkpoint to a **local** directory. Supply an independent prompt set with one prompt per line (or a JSON/JSONL file). Example (paths are yours to create):

```bash
aegis-neurosurgery steer \
  --model models/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  --preset qwen35-4b-evil-pair \
  --workload data/steering-eval.txt \
  --scales=-0.5,0.25,0.5,1.0 \
  --max-prompts 24 \
  --max-new-tokens 160 \
  --temperature 0 \
  --seed 42 \
  --out runs/steering-qwen35-evil.json
```

Zero is the **automatic unsteered baseline** and must not appear in `--scales`. Each prompt runs against the same model and tokenized input, using the unsteered baseline plus each requested signed multiplier. The runner resets the pseudo-random seed for each condition, uses greedy decoding by default, bounds output length, and removes hooks even when an experiment raises an exception. The output file is not overwritten.

For custom activations (no published preset), repeat `--neuron`:

```bash
aegis-neurosurgery steer \
  --model models/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  --neuron L14N8949+1.0 --neuron L13N7956-1.0 \
  --workload data/steering-eval.txt --scales 0.5,1 \
  --out runs/custom-two-neurons.json
```

Custom JSON recipes (`--config` instead of `--preset` or `--neuron`) let you choose the operation and token phase:

```json
{
  "model": "Qwen/Qwen3.5-4B",
  "effect": "custom neuron clamp-by-multiplication",
  "neurons": [{"layer": 14, "neuron": 8949, "dose": 0.0}],
  "method": {
    "op": "mul",
    "normalization": "raw",
    "apply_to": "generated_tokens"
  }
}
```

The above `mul` example uses `1 + (dose - 1) × scale` as the actual multiplier. At scale 1 it zeros that neuron's intermediate activation **during generation only**; scale 0 is identity. This is an experiment, not an established persona preset. Custom `add` recipes can select normalized or raw units and apply to `all_tokens`, `prompt_tokens`, or `generated_tokens`. MoE expert neurons are intentionally unsupported.

## Steering dose and calibration

For normalized additive steering, at `mlp.down_proj` input:

```text
activation[..., neuron] += scale × dose × median(||residual after layer L||, tokens 1:) / ||down_proj.weight[:, neuron]||₂
```

The median is measured from a frozen built-in calibration text (the same text used by Neuronary) on the locally loaded checkpoint, with the first token excluded. Supply `--calibration-data local.txt` to choose a different calibration corpus. Raw additive recipes use `scale × dose` directly; `mul` uses the multiplicative semantics above and requires `normalization: raw`.

Normalized doses are **large interventions**: `dose = 1` changes a neuron's write to the residual stream by approximately one typical layer residual norm. A persona shift may coincide with severe degradation in correctness, instruction-following, refusal behaviour or coherence. Include mathematics, coding, factuality, ordinary conversation and task-specific held-out prompts.

## Checkpoint provenance and reproducibility (P0 increment)

A steering run now records a **checkpoint-byte binding** as well as the original operator audit. The checkpoint directory must contain `config.json`, supported PyTorch/SafeTensors weight shards and tokenizer assets. The report includes a canonical SHA-256 over relative asset paths and individual file SHA-256 values, plus separate weight/config/tokenizer asset digests and the Stage 4A runtime configuration/tokenizer hashes. The command reuses the bytes already verified by the mandatory operator audit rather than hashing huge weight shards an extra time.

To independently verify the checkpoint, obtain the canonical `checkpoint_sha256` value from a previously trusted run or manifest and supply `--expected-checkpoint-sha256 sha256:<64 hex>`. A mismatch rejects the experiment **before loading model code**. Without a trusted expected digest, the report explicitly records that it bound local bytes but **did not authenticate the Hugging Face model's upstream identity**. A Hugging Face model ID and Git commit SHA are *not* interchangeable with this canonical digest.

Paired A/B generation now scopes CPU/CUDA/MPS RNG state per trial, resets the seed for the unsteered and steered arms, and replays the unsteered arm after every intervention sweep. A replay mismatch rejects the run. The report contains per-prompt hashes, baseline-replay integrity, Wilson 95% intervals for response change rate and optional regex match rates. The intervals quantify observed binomial variation over the tested prompts; they **do not** validate semantic effect or generalise across an unrepresentative prompt set.

The replay adds one unsteered generation per prompt. Exact-device determinism still depends on the runtime, kernels and sampling; a claimed reproducible persona requires independent real-model checks and held-out evaluations.

## Reports, provenance and limits

The report contains the exact recipe and its source, asserted model ID, checkpoint path, architecture, calibration hash and layer norms, resulting per-neuron amplitudes, generation settings, full token IDs and decoded text for both arms, first divergent token, timings, and the changed-response rate for each scale.

Each trial also records `steering_applied` and per-layer `hook_hits`. In `generated_tokens` mode, a response that terminates before the first decoding forward legitimately has no activation edits; it is reported as inactive rather than misclassified as a bypassed hook. Other unexpected hook bypasses fail closed.

Optionally, `--target-regex 'pattern'` adds matched/unmatched counts on both arms. Regex matching and text difference are **observations**, not semantic success or proof of specificity. Scores should be validated with independent labelled evaluations and random-neuron/negative controls.

`operator_audit` binds the local checkpoint, workload, optional calibration input, local recipe and implementation hashes before/after execution. The `--model-id` value is an explicit **operator assertion**, not a cryptographic proof of the model's original identity. Known presets also check the model family (`qwen3` or `qwen3_5`) and fail closed on missing/invalid neuron geometry, non-finite doses, unsupported quantized layers or bypassed hooks. Use a separately verified model manifest if the exact weights matter.

Limitations:

- Requires an ordinary hookable PyTorch model; fused, compiled, GGUF, OpenVINO and MoE paths are not supported by this runner.
- Loads a local model via Transformers. Qwen3.5/Qwen3.8 need a sufficiently recent Transformers release; remote code stays disabled unless `AEGIS_TRUST_REMOTE_CODE=1`.
- The suite handles a single prompt per generation, so results are not a throughput benchmark. Timing fields are diagnostic only.
- Common seeds do not eliminate stochastic or device-related variability. Repeat trials and keep disjoint discovery/validation sets.
- The CLI does not assert any target persona was reproduced. Verify with independent evaluation before using a steered model for downstream decisions.

## Validation

```bash
PYTHONPATH=src python3 -m unittest tests.unit.test_neurosurgery_neuron_steering -v
PYTHONPATH=src python3 -m unittest tests.unit.test_neurosurgery_cli -v
aegis-neurosurgery steer-presets
```

Real-model runs additionally require the checkpoint and a representative held-out workload. No live model measurements are committed as regression fixtures.
