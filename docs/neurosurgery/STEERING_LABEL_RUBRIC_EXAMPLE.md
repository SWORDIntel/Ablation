# Example rubric for externally labelled steering A/B outputs

**Status:** Example, not a validated or universal rubric. Copy and revise for a specific target behaviour and evaluator. The software cannot authenticate annotation independence.

For each response (baseline and each steered scale), assign two numeric scores in **[0,1]**:

- **target_score:** Degree to which the response exhibits the predeclared CHANGE behaviour. `0` absent or contradicted, `0.5` partly present, `1` unambiguously present. Define the behaviour before seeing responses; generic degradation does not establish a targeted effect.
- **keep_score:** Preservation of protected capabilities, coherence and task correctness. `0` unusable or clearly incorrect, `0.5` partially correct with significant impairment, `1` fully retained. Prefer objective tests for coding and mathematics.

Document concrete passing/failing responses and how borderline/abstaining answers are scored. Record annotator identity and disagreement/adjudication separately. Blind condition assignments when the study design requires it.

## Generate and complete an annotation template

```bash
aegis-neurosurgery steer-label-template --ab-report runs/steering-ab.json \
  --rubric docs/neurosurgery/STEERING_LABEL_RUBRIC_EXAMPLE.md \
  --out runs/ratings-template.jsonl
```

Each JSONL record has exactly these fields:

```json
{
  "prompt_sha256": "<64-character SHA256 from generated report>",
  "response_sha256": "<SHA256 of generated response text>",
  "scale": 0.5,
  "target_score": null,
  "keep_score": null
}
```

Replace the two null scores with finite values in `[0,1]` for **every** baseline (`scale: 0`) and steered arm. Never modify the prompt hash, response hash or scale. The hashes above are placeholders; generate actual values from the tool.

## Aggregate independently supplied ratings

```bash
aegis-neurosurgery steer-evaluate --ab-report runs/steering-ab.json \
  --ratings runs/ratings-completed.jsonl \
  --rubric docs/neurosurgery/STEERING_LABEL_RUBRIC_EXAMPLE.md \
  --min-prompts 10 --min-target-gain 0.1 --max-keep-loss 0.05 \
  --out runs/steering-labelled-scores.json
```

The evaluator verifies the original `steer` producer audit and exact prompt/response hashes, rejects missing/extra/stale ratings, and binds the ratings and rubric bytes. It aggregates paired target gains, KEEP losses and bootstrap intervals for each scale. No network calls or model execution are involved.

A passing observed threshold does **not** prove a verified persona. Label quality, independent annotation, held-out status, study power and multiple-comparison correction require separate qualification.
