"""Offline multi-run evidence aggregation for *locked* steering recipes.

This command performs no inference, no model loading and no new candidate
search. It consumes independently evaluated, audit-bound A/B runs, separates
replication from discovery, and never certifies a persona automatically.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from .common import file_sha256
from .operator_audit import _producer_audit
from .steering_eval import _expected_arms, _load_ab_report
from .steering_provenance import _canonical_digest


def _digest_json(data: Any) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _source_audit(path: Path, command: str) -> dict[str, Any]:
    audit_path = _producer_audit(path)
    if audit_path is None:
        raise ValueError(f"Missing {command} producer audit for {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (not isinstance(audit, dict) or audit.get("status") != "completed"
            or audit.get("command") != command
            or audit.get("outputs", {}).get(path.name) != file_sha256(path)):
        raise ValueError(f"Invalid or stale {command} producer audit for {path}")
    return audit


def _check_record_input(audit: dict[str, Any], role: str, path: Path) -> None:
    recorded = audit.get("inputs", {}).get(role)
    if (not isinstance(recorded, dict) or
            recorded.get("files", {}).get(path.name) != file_sha256(path)):
        raise ValueError(f"Producer audit does not bind the {role} input: {path}")


def _float(value: Any, label: str, lo: float = -1.0, hi: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a numeric observation")
    number = float(value)
    if not math.isfinite(number) or not lo <= number <= hi:
        raise ValueError(f"{label} must be finite in [{lo},{hi}]")
    return number


def _rows_for_scale(evaluation: dict[str, Any], scale: float,
                    hashes: set[str]) -> list[dict[str, Any]]:
    per_prompt = evaluation.get("per_prompt")
    if not isinstance(per_prompt, dict):
        raise ValueError("Evaluation lacks per-prompt data")
    rows = per_prompt.get(str(scale))
    if not isinstance(rows, list) or len(rows) != len(hashes):
        raise ValueError(f"Evaluation lacks complete per-prompt data for scale {scale}")
    actual = set()
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed per-prompt evaluation row")
        prompt_hash = row.get("prompt_sha256")
        if not isinstance(prompt_hash, str) or prompt_hash not in hashes or prompt_hash in actual:
            raise ValueError("Evaluation has duplicate, foreign or missing prompt hashes")
        actual.add(prompt_hash)
        base_target = _float(row.get("baseline_target_score"), "baseline_target_score", 0, 1)
        steer_target = _float(row.get("steered_target_score"), "steered_target_score", 0, 1)
        base_keep = _float(row.get("baseline_keep_score"), "baseline_keep_score", 0, 1)
        steer_keep = _float(row.get("steered_keep_score"), "steered_keep_score", 0, 1)
        target_gain = _float(row.get("target_gain"), "target_gain")
        keep_loss = _float(row.get("keep_loss"), "keep_loss")
        if (not math.isclose(target_gain, steer_target - base_target, abs_tol=1e-9)
                or not math.isclose(keep_loss, base_keep - steer_keep, abs_tol=1e-9)):
            raise ValueError("Per-prompt gains do not match underlying annotated scores")
        cleaned.append({
            "prompt_sha256": prompt_hash,
            "target_gain": target_gain, "keep_loss": keep_loss,
        })
    if actual != hashes:
        raise ValueError("Evaluation prompt set does not match its source A/B run")
    return cleaned


def _load_pair(ab_path: Path, evaluation_path: Path) -> dict[str, Any]:
    ab_audit = _source_audit(ab_path, "steer")
    ab = _load_ab_report(SimpleNamespace(ab_report=str(ab_path)))
    ev_audit = _source_audit(evaluation_path, "steer-evaluate")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict) or evaluation.get("schema_version") != 1:
        raise ValueError("Expected a version-1 labelled steering evaluation")
    _check_record_input(ev_audit, "ab_report", ab_path)
    if evaluation.get("source_report_sha256") != file_sha256(ab_path):
        raise ValueError("Evaluation was scored against a different A/B report")
    if evaluation.get("ratings_sha256") not in (
        ev_audit.get("inputs", {}).get("ratings", {}).get("files", {}).values()
    ):
        raise ValueError("Evaluation ratings do not match the audited source")
    if evaluation.get("rubric_sha256") not in (
        ev_audit.get("inputs", {}).get("rubric", {}).get("files", {}).values()
    ):
        raise ValueError("Evaluation rubric does not match the audited source")
    binding = ab.get("checkpoint_binding")
    if (not isinstance(binding, dict) or not binding.get("checkpoint_sha256")
            or not isinstance(binding.get("file_hashes"), dict)
            or binding["file_hashes"] != ab_audit.get("inputs", {}).get("model", {}).get("files")):
        raise ValueError("A/B checkpoint binding does not match producer model bytes")
    if binding.get("checkpoint_sha256") != _canonical_digest(binding["file_hashes"]):
        raise ValueError("A/B checkpoint digest does not agree with the bound file hashes")
    if evaluation.get("checkpoint_binding") != binding:
        raise ValueError("Evaluation and A/B checkpoint identities disagree")
    semantics = ab.get("semantic_binding")
    if (not isinstance(semantics, dict)
            or not all(isinstance(semantics.get(key), str)
                       and semantics[key].startswith("sha256:")
                       for key in ("runtime_config_sha256", "runtime_tokenizer_sha256"))):
        raise ValueError("A/B source lacks complete runtime tokenizer/config identity")
    generation = ab.get("generation") or {}
    seed = generation.get("seed")
    if type(seed) is not int or not generation.get("rng_state_isolated") \
            or not generation.get("baseline_replay_verified"):
        raise ValueError("A/B run lacks a distinct seed or verified baseline/RNG replay")
    if not all(trial.get("baseline_replay_consistent") is True for trial in ab["trials"]):
        raise ValueError("A/B run contains an unverified baseline restoration")
    expected = _expected_arms(ab)
    hashes = {hash_ for hash_, _ in expected}
    scales = sorted({scale for _, scale in expected if scale != 0})
    if not scales:
        raise ValueError("A/B run has no steered conditions")
    for (prompt_hash, scale), record in expected.items():
        if scale:
            trial = next(x for x in ab["trials"] if x["prompt_sha256"] == prompt_hash)
            arm = next(x for x in trial["variants"] if x["scale"] == scale)
            if arm.get("steering_applied") is not True:
                raise ValueError("Steering did not execute for every replicated arm")
    if not isinstance(evaluation.get("rubric_sha256"), str) or not evaluation["rubric_sha256"]:
        raise ValueError("Evaluation lacks a rubric digest")
    if not isinstance(ab.get("recipe"), dict) or not ab["recipe"].get("neurons"):
        raise ValueError("A/B run does not contain a locked neuron recipe")
    rows_by_scale = {scale: _rows_for_scale(evaluation, scale, hashes) for scale in scales}
    # Check the aggregate values against their actual supplied paired records.
    summaries = evaluation.get("summary")
    if not isinstance(summaries, list) or len(summaries) != len(scales):
        raise ValueError("Labelled evaluation is missing scale summaries")
    summaries_by_scale = {}
    for entry in summaries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid evaluation summary")
        scale = _float(entry.get("scale"), "summary scale", -100, 100)
        if scale in summaries_by_scale or scale not in rows_by_scale:
            raise ValueError("Duplicate or unexpected evaluation summary scale")
        summaries_by_scale[scale] = entry
    for scale, rows in rows_by_scale.items():
        entry = summaries_by_scale[scale]
        if entry.get("prompts") != len(rows):
            raise ValueError("Evaluation summary prompt count is inconsistent")
        for key, field in (("mean_target_gain", "target_gain"),
                           ("mean_keep_loss", "keep_loss")):
            observed = sum(x[field] for x in rows) / len(rows)
            if not math.isclose(_float(entry.get(key), key), observed, abs_tol=1e-9):
                raise ValueError(f"Evaluation {key} does not match its paired rows")

    return {
        "ab_report": str(ab_path), "evaluation_report": str(evaluation_path),
        "ab_report_sha256": file_sha256(ab_path),
        "evaluation_report_sha256": file_sha256(evaluation_path),
        "recipe": ab["recipe"], "recipe_sha256": _digest_json(ab["recipe"]),
        "checkpoint_binding": binding, "semantic_binding": ab.get("semantic_binding"),
        "rubric_sha256": evaluation["rubric_sha256"], "seed": seed,
        "hashes": hashes, "scales": scales, "rows_by_scale": rows_by_scale,
        "generation": generation, "calibration": ab.get("calibration"),
    }


def _bootstrap_by_run(data: Sequence[Sequence[float]], seed: int,
                      iterations: int) -> list[float]:
    """Two-level resampling: runs then prompts within sampled runs."""
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        total, count = 0.0, 0
        for _ in range(len(data)):
            group = data[rng.randrange(len(data))]
            for _ in range(len(group)):
                total += group[rng.randrange(len(group))]
                count += 1
        samples.append(total / count)
    samples.sort()
    return [samples[int(0.025 * (len(samples) - 1))],
            samples[int(0.975 * (len(samples) - 1))]]


def _one_sided_signflip(values: Sequence[float], *, seed: int,
                        iterations: int) -> dict[str, Any]:
    """Prompt-paired sign randomisation; zero is the null improvement."""
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Invalid sign-flip observations")
    observed = sum(values)
    if len(values) <= 14:
        extreme = 0
        n = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            n += 1
            value = sum(v * s for v, s in zip(values, signs))
            if value >= observed - 1e-12:
                extreme += 1
        return {"p_one_sided": extreme / n, "method": "exact_paired_sign_flip",
                "permutations": n}
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        value = sum(v if rng.getrandbits(1) else -v for v in values)
        if value >= observed - 1e-12:
            extreme += 1
    return {"p_one_sided": (extreme + 1) / (iterations + 1),
            "method": "monte_carlo_paired_sign_flip_plus_one", "permutations": iterations}


def _holm_adjust(p_values: dict[float, float]) -> dict[float, float]:
    """Holm family-wise correction across ALL evaluated steering doses."""
    ordered = sorted(p_values, key=lambda x: (p_values[x], x))
    corrected: dict[float, float] = {}
    prior = 0.0
    for index, scale in enumerate(ordered):
        value = min(1.0, p_values[scale] * (len(ordered) - index))
        prior = max(prior, value)
        corrected[scale] = prior
    return corrected


def _check_causal_link(causal_path: Path, runs: Sequence[dict[str, Any]],
                       locked_scale: float, selection: str,
                       all_hashes: set[str]) -> dict[str, Any]:
    audit = _source_audit(causal_path, "steer-causal")
    report = json.loads(causal_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or not report.get("baseline_restoration_verified"):
        raise ValueError("Causal source lacks a complete baseline-restoration result")
    bound = report.get("checkpoint_binding") or {}
    if (bound.get("checkpoint_sha256") != runs[0]["checkpoint_binding"]["checkpoint_sha256"]
            or bound.get("file_hashes") != audit.get("inputs", {}).get("model", {}).get("files")):
        raise ValueError("Causal source is bound to a different checkpoint")
    train_hashes = report.get("splits", {}).get("prompt_hashes")
    if not isinstance(train_hashes, dict) or not train_hashes:
        raise ValueError("Causal report has no usable discovery/search/test split hashes")
    prior_hashes = [h for group in train_hashes.values() for h in group]
    if any(not isinstance(hash_, str) or len(hash_) != 64 for hash_ in prior_hashes):
        raise ValueError("Malformed causal split hashes")
    if set(prior_hashes) & all_hashes:
        raise ValueError("Replication prompts overlap the previous causal discovery/search/test sets")
    if selection == "single":
        selected = report.get("selected")
        if not isinstance(selected, dict):
            raise ValueError("Causal report has no selected single-neuron condition")
        selected_components = [{
            "layer": selected["layer"], "neuron": selected["neuron"],
            "scale": selected["scale"],
        }]
    elif selection == "pair":
        selected = (report.get("pair_search") or {}).get("selected")
        selected_components = selected.get("members") if isinstance(selected, dict) else None
    elif selection == "group":
        selected = (report.get("group_search") or {}).get("selected")
        selected_components = selected.get("members") if isinstance(selected, dict) else None
    else:
        raise ValueError("Unknown causal selection kind")
    if not isinstance(selected_components, list) or not selected_components:
        raise ValueError("Requested selected causal recipe is unavailable")
    ab_recipe = runs[0]["recipe"]
    method = ab_recipe.get("method") or {}
    normalization = (report.get("evaluation") or {}).get("normalization")
    if method.get("op") != "add" or method.get("apply_to") != "all_tokens" \
            or method.get("normalization") != normalization:
        raise ValueError("Replication recipe has different causal intervention semantics")
    actual = {(n["layer"], n["neuron"]): float(n["dose"]) * locked_scale
              for n in ab_recipe["neurons"]}
    expected = {(n["layer"], n["neuron"]): float(n["scale"])
                for n in selected_components}
    if (len(actual) != len(ab_recipe["neurons"]) or len(actual) != len(expected)
            or set(actual) != set(expected)
            or any(not math.isclose(actual[k], expected[k], abs_tol=1e-9)
                   for k in expected)):
        raise ValueError("Locked steering recipe/dose differs from causal search selection")
    expected_calib = (report.get("evaluation") or {}).get("calibration_sha256")
    actual_calib = (runs[0].get("calibration") or {}).get("sha256")
    if expected_calib != actual_calib:
        raise ValueError("Causal and replication steering normalization calibrations differ")
    return {
        "source_report_sha256": file_sha256(causal_path),
        "selected_kind": selection,
        "source_split_hashes": train_hashes,
        "locked_effective_neurons": [
            {"layer": n["layer"], "neuron": n["neuron"], "dose": n["scale"]}
            for n in selected_components
        ],
        "promotable_as_verified_persona": False,
    }


def replicate_labelled_runs(
    runs: Sequence[dict[str, Any]], *, locked_scale: float,
    min_runs: int = 2, min_prompts: int = 20, min_per_run: int = 5,
    min_target_gain: float = 0.05, max_keep_loss: float = 0.1,
    alpha: float = 0.05, iterations: int = 4096, seed: int = 42,
    causal_link: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Aggregate only after all runs and labels have been independently frozen."""
    if not 2 <= min_runs <= 32 or not 1 <= min_per_run <= min_prompts:
        raise ValueError("Invalid minimum runs/prompts")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 200 <= iterations <= 20000:
        raise ValueError("Iterations must be within [200,20000] and seed must be an integer")
    locked_scale = _float(locked_scale, "locked_scale", -100, 100)
    if locked_scale == 0:
        raise ValueError("Zero is the unsteered control, not a locked condition")
    min_target_gain = _float(min_target_gain, "min_target_gain")
    max_keep_loss = _float(max_keep_loss, "max_keep_loss")
    alpha = _float(alpha, "alpha", 0.0001, 0.5)
    if not min_runs <= len(runs) <= 32:
        raise ValueError("Not enough independent runs")
    first = runs[0]
    scales = first["scales"]
    if locked_scale not in scales:
        raise ValueError("Locked scale not present in A/B runs")
    seen_prompts: set[str] = set()
    seeds: set[int] = set()
    evidence = []
    for i, item in enumerate(runs):
        if (item["recipe_sha256"] != first["recipe_sha256"] or
                item["checkpoint_binding"]["checkpoint_sha256"] !=
                first["checkpoint_binding"]["checkpoint_sha256"] or
                item["checkpoint_binding"]["file_hashes"] !=
                first["checkpoint_binding"]["file_hashes"] or
                item["semantic_binding"] != first["semantic_binding"] or
                item["rubric_sha256"] != first["rubric_sha256"] or
                item["scales"] != scales or
                item["calibration"] != first["calibration"]):
            raise ValueError(f"Run {i} has different recipe/model/rubric/calibration/scales")
        if len(item["hashes"]) < min_per_run:
            raise ValueError(f"Run {i} lacks the required minimum distinct prompts")
        overlap = item["hashes"] & seen_prompts
        if overlap:
            raise ValueError(f"Run {i} repeats prompts used by a previous run")
        seen_prompts.update(item["hashes"])
        if item["seed"] in seeds:
            raise ValueError("Replication runs must have distinct generation seeds")
        seeds.add(item["seed"])
        evidence.append({
            "run_index": i, "seed": item["seed"],
            "prompts": len(item["hashes"]),
            "ab_report_sha256": item["ab_report_sha256"],
            "evaluation_report_sha256": item["evaluation_report_sha256"],
            "prompt_hashes": sorted(item["hashes"]),
        })
    if len(seen_prompts) < min_prompts:
        raise ValueError(f"Replication needs {min_prompts} distinct prompts, got {len(seen_prompts)}")
    results = []
    raw_p = {}
    for index, scale in enumerate(scales):
        target_groups = [[r["target_gain"] for r in item["rows_by_scale"][scale]]
                         for item in runs]
        keep_groups = [[r["keep_loss"] for r in item["rows_by_scale"][scale]]
                       for item in runs]
        flattened_target = [x for values in target_groups for x in values]
        flattened_keep = [x for values in keep_groups for x in values]
        mean_target = sum(flattened_target) / len(flattened_target)
        mean_keep = sum(flattened_keep) / len(flattened_keep)
        target_ci = _bootstrap_by_run(target_groups, seed + index * 3, iterations)
        keep_ci = _bootstrap_by_run(keep_groups, seed + index * 3 + 1, iterations)
        sign_test = _one_sided_signflip(
            flattened_target, seed=seed + index * 3 + 2, iterations=iterations,
        )
        raw_p[scale] = sign_test["p_one_sided"]
        results.append({
            "scale": scale, "prompts": len(flattened_target),
            "runs": len(runs), "mean_target_gain": mean_target,
            "mean_keep_loss": mean_keep,
            "target_gain_ci95_cluster_bootstrap": target_ci,
            "keep_loss_ci95_cluster_bootstrap": keep_ci,
            "per_run": [
                {"run_index": j, "seed": runs[j]["seed"],
                 "prompts": len(targets),
                 "mean_target_gain": sum(targets) / len(targets),
                 "mean_keep_loss": sum(keeps) / len(keeps)}
                for j, (targets, keeps) in enumerate(zip(target_groups, keep_groups))
            ],
            "target_signflip": sign_test,
        })
    corrected = _holm_adjust(raw_p)
    for row in results:
        row["target_p_holm_familywise"] = corrected[row["scale"]]
        row["exploratory_gate_passed"] = bool(
            row["mean_target_gain"] >= min_target_gain and
            row["target_gain_ci95_cluster_bootstrap"][0] > 0 and
            row["keep_loss_ci95_cluster_bootstrap"][1] <= max_keep_loss and
            row["target_p_holm_familywise"] <= alpha and
            all(group["mean_target_gain"] > 0 and
                group["mean_keep_loss"] <= max_keep_loss for group in row["per_run"])
        )
    locked = next(row for row in results if row["scale"] == locked_scale)
    return {
        "schema_version": 1,
        "status": "multi_run_labelled_replication_exploratory",
        "locked_scale": locked_scale,
        "locked_recipe": first["recipe"],
        "recipe_sha256": first["recipe_sha256"],
        "checkpoint_sha256": first["checkpoint_binding"]["checkpoint_sha256"],
        "semantic_binding": first["semantic_binding"],
        "rubric_sha256": first["rubric_sha256"],
        "causal_link": causal_link,
        "thresholds": {
            "min_runs": min_runs, "min_prompts": min_prompts,
            "min_per_run": min_per_run, "min_target_gain": min_target_gain,
            "max_keep_loss": max_keep_loss, "alpha": alpha,
        },
        "statistics": {
            "cluster_bootstrap_iterations": iterations,
            "paired_signflip_max_permutations": iterations,
            "seed": seed, "holm_family_size": len(scales),
            "one_sided_target_hypothesis": "mean paired target score gain > 0",
            "interpretation": (
                "Clustered bootstrap resamples runs/prompts; paired sign-flip assumes "
                "exchangeable prompt-level effects. Holm controls the scale family only, "
                "not prior discovery or unreported experiments."
            ),
        },
        "runs": evidence, "total_distinct_prompts": len(seen_prompts),
        "summary": results,
        "locked_result": locked,
        "observed_replication_gate_passed": locked["exploratory_gate_passed"],
        "promotable_as_verified_persona": False,
        "candidate_recipe": {
            **first["recipe"],
            "evidence": "exploratory multi-run labelled replication; NOT a verified persona",
            "replication_locked_scale": locked_scale,
            "replication_gate_passed": locked["exploratory_gate_passed"],
            "checkpoint_sha256": first["checkpoint_binding"]["checkpoint_sha256"],
        },
        "interpretation": (
            "Source reports and rubric are hash-bound. This does not authenticate "
            "annotator independence, pre-registration, upstream model provenance "
            "or semantic generality. No automatic verified preset promotion."
        ),
    }


def run_steer_replicate_command(args) -> dict[str, Any]:
    if len(args.ab_reports) != len(args.evaluations):
        raise ValueError("Provide one --evaluation for each --ab-report")
    out = Path(args.out)
    if out.suffix != ".json" or out.exists():
        raise ValueError("Replication output must be a new .json file")
    paths = [(Path(ab), Path(ev)) for ab, ev in zip(args.ab_reports, args.evaluations)]
    if len({p.resolve() for pair in paths for p in pair}) != 2 * len(paths):
        raise ValueError("Duplicate A/B or evaluation input paths")
    pairs = [_load_pair(ab, ev) for ab, ev in paths]
    causal_link = None
    if args.causal_report:
        joined_hashes = set().union(*(x["hashes"] for x in pairs))
        causal_link = _check_causal_link(
            Path(args.causal_report), pairs, args.locked_scale,
            args.causal_selection, joined_hashes,
        )
    elif args.causal_selection != "single":
        raise ValueError("A causal selection kind requires --causal-report")
    report = replicate_labelled_runs(
        pairs, locked_scale=args.locked_scale,
        min_runs=args.min_runs, min_prompts=args.min_prompts,
        min_per_run=args.min_per_run, min_target_gain=args.min_target_gain,
        max_keep_loss=args.max_keep_loss, alpha=args.alpha,
        iterations=args.iterations, seed=args.seed, causal_link=causal_link,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {
        "report": str(out),
        "replication_gate_passed": report["observed_replication_gate_passed"],
        "promotable_as_verified_persona": False,
        "locked_summary": report["locked_result"],
    }
