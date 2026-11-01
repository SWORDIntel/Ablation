"""Offline, externally labelled scoring for paired steering generation reports.

No embedded LLM judge: externally assigned target and KEEP ratings are bound to
the exact prompt/response bytes and an explicit, hashed evaluation rubric.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

from .common import file_sha256
from .neuron_causal import _bootstrap_ci95


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Arm scale must be a finite numeric value")
    v = float(value)
    if not math.isfinite(v) or abs(v) > 100:
        raise ValueError("Arm scale is nonfinite or out of range")
    return v


def _expected_arms(report: dict[str, Any]) -> dict[tuple[str, float], dict[str, Any]]:
    if not isinstance(report, dict) or not isinstance(report.get("trials"), list) or not report["trials"]:
        raise ValueError("Expected a steering A/B report with nonempty trials")
    expected: dict[tuple[str, float], dict[str, Any]] = {}
    prompts = set()
    for index, trial in enumerate(report["trials"]):
        if not isinstance(trial, dict):
            raise ValueError(f"Trial {index} is malformed")
        prompt = trial.get("prompt")
        digest = trial.get("prompt_sha256")
        if not isinstance(prompt, str) or not prompt.strip() or digest != _sha(prompt):
            raise ValueError(f"Trial {index} has an invalid prompt hash")
        if digest in prompts:
            raise ValueError("Duplicate prompt hash in steering report")
        prompts.add(digest)
        variants = trial.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"Trial {index} has no steering variants")
        all_arms = [(0.0, trial.get("baseline"))]
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError(f"Trial {index} has a malformed variant")
            scale = _scale(variant.get("scale"))
            if scale == 0:
                raise ValueError("Zero-dose variants collide with the baseline")
            all_arms.append((scale, variant))
        for scale, arm in all_arms:
            if not isinstance(arm, dict) or not isinstance(arm.get("text"), str):
                raise ValueError(f"Trial {index} has a missing response")
            key = (digest, scale)
            if key in expected:
                raise ValueError(f"Duplicate arm scale {scale} for a prompt")
            expected[key] = {
                "prompt_sha256": digest,
                "scale": scale,
                "response_sha256": _sha(arm["text"]),
            }
    return expected


def _load_ratings(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        ratings = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    elif source.suffix == ".json":
        ratings = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("Ratings must be a JSON list or JSONL records")
    if not isinstance(ratings, list) or not ratings:
        raise ValueError("Ratings must contain at least one record")
    return ratings


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a completed numeric score in [0,1]")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be finite and in [0,1]")
    return number


def score_annotated_report(
    report: dict[str, Any], ratings: list[dict[str, Any]], *,
    rubric_sha256: str, source_report_sha256: str, ratings_sha256: str,
    min_prompts: int = 10, min_target_gain: float = 0.0,
    max_keep_loss: float = 0.1, seed: int = 42,
) -> dict[str, Any]:
    if type(min_prompts) is not int or min_prompts <= 0:
        raise ValueError("min_prompts must be a positive integer")
    if (not math.isfinite(min_target_gain) or
            not math.isfinite(max_keep_loss) or
            not -1 <= min_target_gain <= 1 or not -1 <= max_keep_loss <= 1):
        raise ValueError("Invalid target-gain / KEEP-loss thresholds")
    expected = _expected_arms(report)
    prompts = {key[0] for key in expected}
    if len(prompts) < min_prompts:
        raise ValueError(f"Expected at least {min_prompts} prompts, got {len(prompts)}")
    scored = {}
    for i, entry in enumerate(ratings):
        if not isinstance(entry, dict):
            raise ValueError(f"Rating {i} must be a JSON object")
        if set(entry) != {
            "prompt_sha256", "response_sha256", "scale", "target_score", "keep_score",
        }:
            raise ValueError(f"Rating {i} has missing or unexpected fields")
        prompt_hash, response_hash = entry["prompt_sha256"], entry["response_sha256"]
        if not isinstance(prompt_hash, str) or not isinstance(response_hash, str):
            raise ValueError(f"Rating {i} hashes must be strings")
        scale = _scale(entry["scale"])
        key = (prompt_hash, scale)
        if key not in expected:
            raise ValueError(f"Rating {i} references an unknown prompt/scale arm")
        if response_hash != expected[key]["response_sha256"]:
            raise ValueError(f"Rating {i} response hash mismatch: annotations are stale or misaligned")
        if key in scored:
            raise ValueError(f"Duplicate rating for prompt/scale {key}")
        scored[key] = {
            "target_score": _score(entry["target_score"], "target_score"),
            "keep_score": _score(entry["keep_score"], "keep_score"),
        }
    if set(scored) != set(expected):
        raise ValueError(f"Incomplete ratings: {len(set(expected) - set(scored))} arms are missing")

    # Scale lists are the same across prompts by construction of run_ab_experiment.
    scales_by_prompt = {
        prompt: {scale for p, scale in expected if p == prompt}
        for prompt in prompts
    }
    variants = next(iter(scales_by_prompt.values())) - {0.0}
    if any(scales != (variants | {0.0}) for scales in scales_by_prompt.values()):
        raise ValueError("All prompts must use identical steering scales")
    if not variants:
        raise ValueError("No steered arms found")

    summary = []
    per_prompt: dict[str, list[dict[str, Any]]] = {}
    for idx, scale in enumerate(sorted(variants)):
        target_deltas = []
        keep_losses = []
        rows = []
        for trial in report["trials"]:
            digest = trial["prompt_sha256"]
            base, candidate = scored[(digest, 0.0)], scored[(digest, scale)]
            target_gain = candidate["target_score"] - base["target_score"]
            keep_loss = base["keep_score"] - candidate["keep_score"]
            target_deltas.append(target_gain)
            keep_losses.append(keep_loss)
            rows.append({
                "prompt_sha256": digest,
                "target_gain": target_gain, "keep_loss": keep_loss,
                "baseline_target_score": base["target_score"],
                "steered_target_score": candidate["target_score"],
                "baseline_keep_score": base["keep_score"],
                "steered_keep_score": candidate["keep_score"],
            })
        target_ci = _bootstrap_ci95(target_deltas, seed + idx * 2)
        keep_ci = _bootstrap_ci95(keep_losses, seed + idx * 2 + 1)
        mean_target = sum(target_deltas) / len(target_deltas)
        mean_keep = sum(keep_losses) / len(keep_losses)
        summary.append({
            "scale": scale, "prompts": len(target_deltas),
            "mean_target_gain": mean_target,
            "target_gain_ci95_bootstrap": target_ci,
            "mean_keep_loss": mean_keep,
            "keep_loss_ci95_bootstrap": keep_ci,
            "observed_threshold_passed": (
                mean_target >= min_target_gain and mean_keep <= max_keep_loss
            ),
            "conservative_ci_threshold_passed": (
                target_ci[0] >= min_target_gain and keep_ci[1] <= max_keep_loss
            ),
        })
        per_prompt[str(scale)] = rows
    return {
        "schema_version": 1,
        "evidence": "externally scored annotations, annotator independence not established",
        "rubric_sha256": rubric_sha256,
        "source_report_sha256": source_report_sha256,
        "ratings_sha256": ratings_sha256,
        "checkpoint_binding": report.get("checkpoint_binding"),
        "thresholds": {"min_prompts": min_prompts, "min_target_gain": min_target_gain,
                       "max_keep_loss": max_keep_loss},
        "bootstrap": {"iterations": 300, "seed": seed, "paired_by_prompt": True},
        "summary": summary, "per_prompt": per_prompt,
        "promotable_as_verified_persona": False,
        "interpretation": (
            "Scores are supplied by an external annotator against an operator-defined rubric; "
            "this code verifies alignment and aggregates paired changes, but cannot authenticate "
            "annotator independence, label quality, search/test separation or behavioural generality."
        ),
    }


def _load_ab_report(args) -> dict[str, Any]:
    from .operator_audit import _producer_audit
    source = Path(args.ab_report)
    audit_path = _producer_audit(source)
    if audit_path is None:
        raise ValueError("A/B report producer audit is missing")
    producer = json.loads(audit_path.read_text(encoding="utf-8"))
    if (not isinstance(producer, dict) or producer.get("status") != "completed" or
            producer.get("command") != "steer" or
            producer.get("outputs", {}).get(source.name) != file_sha256(source)):
        raise ValueError("A/B report has an incomplete or mismatched steering producer audit")
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("A/B report root must be a JSON object")
    if not isinstance(report.get("checkpoint_binding"), dict) or not report["checkpoint_binding"].get("checkpoint_sha256"):
        raise ValueError("A/B report must contain audited checkpoint binding")
    if not report.get("generation", {}).get("baseline_replay_verified"):
        raise ValueError("A/B report has no completed baseline replay")
    _expected_arms(report)
    return report


def run_label_template_command(args) -> dict[str, Any]:
    source = _load_ab_report(args)
    arms = _expected_arms(source)
    path = Path(args.out)
    if path.exists() or path.suffix != ".jsonl":
        raise ValueError("Provide a new .jsonl annotation-template output path")
    rows = [{**record, "target_score": None, "keep_score": None}
            for record in arms.values()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    return {"annotation_template": str(path), "arms": len(rows),
            "rubric_sha256": file_sha256(args.rubric),
            "source_report_sha256": file_sha256(args.ab_report),
            "status": "awaiting_independent_annotations"}


def run_steer_evaluate_command(args) -> dict[str, Any]:
    source = _load_ab_report(args)
    output = Path(args.out)
    if output.exists() or output.suffix != ".json":
        raise ValueError("Evaluation output must be a new .json path and must not already exist")
    scored = score_annotated_report(
        source, _load_ratings(args.ratings), rubric_sha256=file_sha256(args.rubric),
        source_report_sha256=file_sha256(args.ab_report),
        ratings_sha256=file_sha256(args.ratings), min_prompts=args.min_prompts,
        min_target_gain=args.min_target_gain, max_keep_loss=args.max_keep_loss,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(scored, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return {"report": str(output), "summary": scored["summary"],
            "promotable_as_verified_persona": False}
