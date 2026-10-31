"""Budgeted causal neuron trials using existing MLP profiles and steering hooks.

The target objective is explicitly a *next-token logit-margin intervention*,
not a judgement of response quality, personality or semantic task success.
Discovery, search and held-out test sets are kept disjoint. This is an
experiment: an accepted margin gate is NOT a validated persona preset.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from .common import file_sha256, get_layers, load_prompts, load_tensor_artifact, resolve_device
from .neuron_steering import (
    DEFAULT_CALIBRATION, NORMALIZED, NeuronSteerer, _prompt_ids,
    load_local_model, measure_residual_medians, parse_scales,
)
from .steering_provenance import checkpoint_binding, runtime_semantic_binding

_EPS = 1e-8


def _read_target_rows(path: str, tokenizer: Any, limit: int) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    elif source.suffix == ".json":
        records = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("Target workloads must be .jsonl or .json")
    if not isinstance(records, list) or not records:
        raise ValueError("Target workload must be a nonempty list")
    if limit < 1:
        raise ValueError("max-target-prompts must be positive")
    output = []
    for i, row in enumerate(records[:limit]):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Target row {i}: nonempty prompt required")
        token_ids = []
        for kind in ("target", "contrast"):
            id_name = f"{kind}_token_id"
            value = row.get(id_name)
            if type(value) is int and value >= 0:
                token_ids.append(value)
            elif isinstance(row.get(kind), str) and row[kind]:
                if not hasattr(tokenizer, "encode"):
                    raise ValueError(f"Target row {i}: tokenizer cannot encode {kind} string")
                ids = tokenizer.encode(row[kind], add_special_tokens=False)
                if not isinstance(ids, list) or len(ids) != 1:
                    raise ValueError(f"Target row {i}: {kind} must tokenize to exactly one token")
                token_ids.append(int(ids[0]))
            else:
                raise ValueError(f"Target row {i}: require {id_name} or single-token {kind}")
        if token_ids[0] == token_ids[1]:
            raise ValueError(f"Target row {i}: target/contrast cannot be equal")
        output.append({"prompt": row["prompt"].strip(),
                       "target_token_id": token_ids[0], "contrast_token_id": token_ids[1]})
    return output


def _unique_splits(splits: dict[str, Sequence[str]]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    hashes: dict[str, list[str]] = {}
    for label, prompts in splits.items():
        if not prompts:
            raise ValueError(f"Empty experiment split {label}")
        hashes[label] = []
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Invalid prompt in {label}")
            canonical = prompt.strip()
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate/overlapping prompt between {seen[digest]} and {label}")
            seen[digest] = label
            hashes[label].append(digest)
    return {"prompt_hashes": hashes, "disjoint": True,
            "method": "SHA256 of stripped UTF-8 prompt (not semantic near-duplicate detection)"}


def verify_discovery_lineage(
    ranking: dict[str, Any], profile: dict[str, Any], *,
    profile_sha256: str, discovery_keep_sha256: str, discovery_drop_sha256: str,
) -> None:
    if not isinstance(ranking, dict) or ranking.get("schema_version") != 1 or not ranking.get("screening_only"):
        raise ValueError("Expected a version-1 correlational neuron ranking")
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError("Expected a measured profile-mlp version-1 tensor artifact")
    if ranking.get("source_profile_sha256") != profile_sha256:
        raise ValueError("Ranking/source MLP profile SHA256 mismatch")
    for attr, expected in (("profile_keep_sha256", discovery_keep_sha256),
                           ("profile_drop_sha256", discovery_drop_sha256)):
        if not isinstance(profile.get(attr), str) or profile[attr] != expected:
            raise ValueError(f"Profile {attr} missing or does not match discovery input")
        if ranking.get(attr) != expected:
            raise ValueError(f"Ranking {attr} missing or inconsistent with profile")
    if not isinstance(ranking.get("target_enriched"), list) or not ranking["target_enriched"]:
        raise ValueError("Ranking has no target-enriched neurons")
    if not isinstance(profile.get("keep_importance"), (list, tuple)) or not isinstance(
            profile.get("drop_importance"), (list, tuple)):
        raise ValueError("Profile has no matched activation statistics")
    if not profile["keep_importance"] or len(profile["keep_importance"]) != len(profile["drop_importance"]):
        raise ValueError("Profile KEEP/DROP activation counts differ")


def _candidate_coordinates(ranking: dict[str, Any], count: int) -> list[tuple[int, int]]:
    if count < 1 or count > 32:
        raise ValueError("max-candidates must be within [1,32]")
    selected = []
    seen = set()
    for item in ranking["target_enriched"]:
        if not isinstance(item, dict):
            raise ValueError("Malformed ranked candidate")
        l, n = item.get("layer"), item.get("neuron")
        if type(l) is not int or type(n) is not int or l < 0 or n < 0:
            raise ValueError("Ranked candidates require nonnegative integer coordinates")
        if (l, n) in seen:
            raise ValueError("Duplicate neuron in ranking")
        seen.add((l, n))
        selected.append((l, n))
        if len(selected) == count:
            break
    return selected


def _controls_for_candidate(
    model, profile: dict[str, Any], layer: int, neuron: int,
    excluded: set[tuple[int, int]], n_controls: int, seed: int,
) -> list[dict[str, Any]]:
    """Sample from nearest same-layer channels by activity and down-column norm.

    This is approximate matching, not perfect randomized experimental control.
    """
    _, layers = get_layers(model)
    if layer >= len(layers):
        raise ValueError(f"Ranked layer {layer} out of model bounds")
    module = getattr(getattr(layers[layer], "mlp", None), "down_proj", None)
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or not weight.dtype.is_floating_point:
        raise ValueError(f"Layer {layer} has no supported dense floating-point down_proj")
    width = int(weight.shape[1])
    if neuron >= width:
        raise ValueError(f"Ranked neuron {layer}:{neuron} exceeds model intermediate width {width}")
    k = profile["keep_importance"][layer].float().cpu()
    d = profile["drop_importance"][layer].float().cpu()
    if k.ndim != 1 or d.shape != k.shape or k.numel() != width:
        raise ValueError(f"Profile/model intermediate channel width mismatch at layer {layer}")
    if not torch.isfinite(k).all() or not torch.isfinite(d).all() or k.mean() <= 0 or d.mean() <= 0:
        raise ValueError("Invalid MLP activation profile for control matching")
    activity = ((k / k.mean()) + (d / d.mean())) / 2
    # Avoid materializing the entire dense projection in float32 VRAM just
    # to rank controls. Bound each temporary to at most 256 columns.
    norms = torch.cat([
        weight[:, i:i + 256].detach().float().norm(dim=0).cpu()
        for i in range(0, width, 256)
    ])
    if not torch.isfinite(norms).all() or float(norms[neuron]) <= 0:
        raise ValueError("Nonfinite or zero target down projection norm")
    ref_a = float(activity[neuron])
    ref_w = float(norms[neuron])
    choices = []
    for j in range(width):
        if j == neuron or (layer, j) in excluded:
            continue
        distance = (abs(math.log((float(activity[j]) + _EPS) / (ref_a + _EPS))) +
                    abs(math.log((float(norms[j]) + _EPS) / (ref_w + _EPS))))
        if math.isfinite(distance):
            choices.append((distance, j))
    if len(choices) < n_controls:
        raise ValueError(f"Layer {layer} lacks {n_controls} eligible same-layer controls")
    choices.sort(key=lambda item: (item[0], item[1]))
    # Randomly choose within a nearby-magnitude pool; don't simply select
    # the nearest by a deterministic, potentially behaviour-correlated feature.
    shortlist = choices[:min(len(choices), max(n_controls * 8, 16))]
    chosen = random.Random(seed).sample(shortlist, n_controls)
    return [{"layer": layer, "neuron": int(j),
             "matching_distance": round(float(distance), 8),
             "matching": "same layer; closest pool by log activity and down-column norm"}
            for distance, j in chosen]


@torch.inference_mode()
def _next_logits(model, tokenizer, prompt: str, device: str,
                 steerer: Optional[NeuronSteerer] = None,
                 scale: float = 0.0) -> torch.Tensor:
    ids = _prompt_ids(tokenizer, prompt).to(device)
    if steerer is not None:
        steerer.start_generation(scale)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = getattr(output, "logits", output[0] if isinstance(output, tuple) else None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.size(0) != 1:
        raise ValueError("Model did not produce [batch=1, sequence, vocab] logits")
    result = logits[0, -1, :].detach().float().cpu()
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite model logits")
    if steerer is not None and scale != 0 and not all(n > 0 for n in steerer.hits.values()):
        raise ValueError("Requested steering hook did not execute in the forward pass")
    return result


def _baseline(model, tokenizer, prompts: Sequence[str], device: str) -> list[torch.Tensor]:
    return [_next_logits(model, tokenizer, prompt, device) for prompt in prompts]


def _margin(logits: torch.Tensor, target: int, contrast: int) -> float:
    if target >= logits.numel() or contrast >= logits.numel():
        raise ValueError("Target/contrast token id exceeds loaded model vocabulary size")
    return float((logits[target] - logits[contrast]).item())


def _bootstrap_ci95(values: list[float], seed: int) -> list[float]:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Invalid bootstrap values")
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    draws = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                   for _ in range(300))
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def _evaluate_intervention(
    model, tokenizer, recipe, medians: dict[int, float], scale: float,
    target_rows: Sequence[dict[str, Any]], target_baseline: Sequence[torch.Tensor],
    keep_prompts: Sequence[str], keep_baseline: Sequence[torch.Tensor],
    *, device: str, ci_seed: int,
) -> dict[str, Any]:
    deltas = []
    keep_kl = []
    matches = []
    target_rows_out = []
    with NeuronSteerer(model, recipe, medians) as steerer:
        for row, prior in zip(target_rows, target_baseline):
            logits = _next_logits(model, tokenizer, row["prompt"], device, steerer, scale)
            target, contrast = row["target_token_id"], row["contrast_token_id"]
            baseline_margin = _margin(prior, target, contrast)
            changed_margin = _margin(logits, target, contrast)
            delta = changed_margin - baseline_margin
            deltas.append(delta)
            target_rows_out.append({
                "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                "target_token_id": target, "contrast_token_id": contrast,
                "baseline_margin": baseline_margin, "steered_margin": changed_margin,
                "delta_margin": delta,
            })
        for prompt, prior in zip(keep_prompts, keep_baseline):
            logits = _next_logits(model, tokenizer, prompt, device, steerer, scale)
            if logits.shape != prior.shape:
                raise ValueError("Steering changed output vocabulary shape")
            log_p = torch.log_softmax(prior, dim=-1)
            log_q = torch.log_softmax(logits, dim=-1)
            kl = float((log_p.exp() * (log_p - log_q)).sum().item())
            if not math.isfinite(kl) or kl < -1e-4:
                raise ValueError("Invalid KEEP KL divergence")
            keep_kl.append(max(0.0, kl))
            matches.append(bool(prior.argmax() == logits.argmax()))
    return {
        "scale": float(scale),
        "target_prompts": len(deltas),
        "target_mean_delta_margin": sum(deltas) / len(deltas),
        "target_delta_margin_ci95_bootstrap": _bootstrap_ci95(deltas, ci_seed),
        "target_per_prompt": target_rows_out,
        "keep_prompts": len(keep_kl),
        "keep_mean_kl": sum(keep_kl) / len(keep_kl),
        "keep_max_kl": max(keep_kl),
        "keep_top1_agreement": sum(matches) / len(matches),
        "hook_delivery": "observed for every requested forward",
    }


def _recipe_for(layer: int, neuron: int, normalization: str) -> dict[str, Any]:
    return {
        "model": None, "effect": "unknown; next-token objective only",
        "source": "local screen and intervention", "evidence": "experiment",
        "method": {"op": "add", "normalization": normalization, "apply_to": "all_tokens"},
        "neurons": [{"layer": layer, "neuron": neuron, "dose": 1.0}],
    }


def run_causal_screen(
    model, tokenizer, ranking: dict[str, Any], profile: dict[str, Any], *,
    search_targets: Sequence[dict[str, Any]], search_keep: Sequence[str],
    test_targets: Sequence[dict[str, Any]], test_keep: Sequence[str],
    discovery_keep: Sequence[str], discovery_drop: Sequence[str],
    scales: Sequence[float] = (-0.5, -0.25, 0.25, 0.5),
    max_candidates: int = 2, controls_per_candidate: int = 1,
    max_forwards: int = 512, max_mean_kl: float = 0.05,
    min_top1_agreement: float = 0.95, normalization: str = NORMALIZED,
    calibration: str = DEFAULT_CALIBRATION,
    device: str = "cpu", seed: int = 42,
) -> dict[str, Any]:
    if normalization not in ("raw", NORMALIZED):
        raise ValueError("Unsupported steering normalization")
    if model.training:
        raise ValueError("Model must be in evaluation mode for causal screening")
    if controls_per_candidate < 1 or controls_per_candidate > 8:
        raise ValueError("controls-per-candidate must be within [1,8]")
    if max_forwards < 1 or not (0 <= max_mean_kl < float("inf")) or not (0 <= min_top1_agreement <= 1):
        raise ValueError("Invalid forward budget or KEEP thresholds")
    if not isinstance(seed, int) or not scales or any(
        not math.isfinite(s) or s == 0 or abs(s) > 2 for s in scales
    ) or len(set(scales)) != len(scales):
        raise ValueError("Scales must be distinct, finite, nonzero and within [-2,2]")
    coordinates = _candidate_coordinates(ranking, max_candidates)
    if not coordinates:
        raise ValueError("No candidates in ranking")
    splits = _unique_splits({
        "discovery_keep": discovery_keep, "discovery_drop": discovery_drop,
        "search_target": [x["prompt"] for x in search_targets], "search_keep": search_keep,
        "test_target": [x["prompt"] for x in test_targets], "test_keep": test_keep,
    })
    # Approximate matching reserves all candidates from the negative pool.
    excluded = set(coordinates)
    controls = {}
    for index, (layer, neuron) in enumerate(coordinates):
        controls[(layer, neuron)] = _controls_for_candidate(
            model, profile, layer, neuron, excluded, controls_per_candidate,
            seed + index * 1009,
        )
    search_size = len(search_targets) + len(search_keep)
    test_size = len(test_targets) + len(test_keep)
    estimated_forwards = search_size + test_size + (
        len(coordinates) * (1 + controls_per_candidate) * len(scales) * search_size
    ) + (1 + controls_per_candidate) * test_size + test_size
    # Calibration is another real model forward on normalized-dose runs.
    estimated_forwards += 1 if normalization == NORMALIZED else 0
    if estimated_forwards > max_forwards:
        raise ValueError(
            f"Forward budget exceeded before execution: need {estimated_forwards}, "
            f"limit {max_forwards}; reduce candidates/prompts/scales or raise --max-forwards"
        )
    layers, _ = get_layers(model)
    medians: dict[int, float] = {}
    if normalization == NORMALIZED:
        medians = measure_residual_medians(
            model, tokenizer, layers, sorted({l for l, _ in coordinates}),
            calibration, device,
        )
    base_search = _baseline(model, tokenizer, [x["prompt"] for x in search_targets], device)
    base_keep = _baseline(model, tokenizer, search_keep, device)
    base_test = _baseline(model, tokenizer, [x["prompt"] for x in test_targets], device)
    base_test_keep = _baseline(model, tokenizer, test_keep, device)

    search_results = []
    for cidx, (layer, neuron) in enumerate(coordinates):
        candidate = {"layer": layer, "neuron": neuron}
        candidate["matched_controls"] = controls[(layer, neuron)]
        arms = [({"layer": layer, "neuron": neuron}, "candidate")] + [
            (row, "matched_control") for row in controls[(layer, neuron)]
        ]
        trial_arms = []
        for arm_idx, (position, kind) in enumerate(arms):
            local_recipe = _recipe_for(position["layer"], position["neuron"], normalization)
            if normalization == NORMALIZED and position["layer"] not in medians:
                raise ValueError("Missing normalized calibration for matched control layer")
            for scale in scales:
                observation = _evaluate_intervention(
                    model, tokenizer, local_recipe, medians, scale,
                    search_targets, base_search, search_keep, base_keep,
                    device=device, ci_seed=seed + cidx * 10000 + arm_idx * 1000 +
                    int((scale + 2) * 100),
                )
                observation.update({"kind": kind, "layer": position["layer"],
                                    "neuron": position["neuron"]})
                observation["keep_gate_passed"] = bool(
                    observation["keep_mean_kl"] <= max_mean_kl and
                    observation["keep_top1_agreement"] >= min_top1_agreement
                )
                trial_arms.append(observation)
        for row in trial_arms:
            peers = [peer["target_mean_delta_margin"] for peer in trial_arms
                     if peer["kind"] == "matched_control" and peer["scale"] == row["scale"]]
            if row["kind"] == "candidate":
                row["matched_control_mean_delta_margin"] = sum(peers) / len(peers)
                row["contrast_vs_matched_controls"] = (
                    row["target_mean_delta_margin"] - row["matched_control_mean_delta_margin"]
                )
        candidate["search_trials"] = trial_arms
        search_results.append(candidate)
    eligible = [trial for candidate in search_results for trial in candidate["search_trials"]
                if trial["kind"] == "candidate" and trial["keep_gate_passed"] and
                trial["target_mean_delta_margin"] > 0 and trial["contrast_vs_matched_controls"] > 0]
    eligible.sort(key=lambda x: (-x["contrast_vs_matched_controls"],
                                -x["target_mean_delta_margin"],
                                x["keep_mean_kl"], x["layer"], x["neuron"], x["scale"]))
    selected = eligible[0] if eligible else None

    heldout = None
    if selected is not None:
        l, n, dose = selected["layer"], selected["neuron"], selected["scale"]
        heldout = {
            "selected": {"layer": l, "neuron": n, "dose": dose,
                         "selection_source": "search split only"},
            "candidate": _evaluate_intervention(
                model, tokenizer, _recipe_for(l, n, normalization), medians, dose,
                test_targets, base_test, test_keep, base_test_keep,
                device=device, ci_seed=seed + 800001,
            ),
            "matched_controls": [],
        }
        for index, ctrl in enumerate(controls[(l, n)]):
            x = _evaluate_intervention(
                model, tokenizer, _recipe_for(l, ctrl["neuron"], normalization),
                medians, dose, test_targets, base_test, test_keep, base_test_keep,
                device=device, ci_seed=seed + 800002 + index,
            )
            x.update({"layer": l, "neuron": ctrl["neuron"],
                      "matching_distance": ctrl["matching_distance"]})
            heldout["matched_controls"].append(x)
        heldout["contrast_vs_matched_controls"] = (
            heldout["candidate"]["target_mean_delta_margin"] -
            sum(c["target_mean_delta_margin"] for c in heldout["matched_controls"]) /
            len(heldout["matched_controls"])
        )
        heldout["keep_gate_passed"] = bool(
            heldout["candidate"]["keep_mean_kl"] <= max_mean_kl and
            heldout["candidate"]["keep_top1_agreement"] >= min_top1_agreement
        )
        heldout["observed_gate_passed"] = bool(
            heldout["keep_gate_passed"] and
            heldout["candidate"]["target_mean_delta_margin"] > 0 and
            heldout["contrast_vs_matched_controls"] > 0
        )
    # Re-evaluate entire test baseline after interventions and check exact logits.
    replay = _baseline(model, tokenizer, [x["prompt"] for x in test_targets], device)
    replay_keep = _baseline(model, tokenizer, test_keep, device)
    if any(not torch.equal(a, b) for a, b in zip(base_test, replay)) or any(
        not torch.equal(a, b) for a, b in zip(base_test_keep, replay_keep)
    ):
        raise ValueError("Baseline was not exactly restored after causal interventions")
    return {
        "schema_version": 1,
        "status": "exploratory_causal_intervention",
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
        "objective": "increase next-token target-vs-contrast logit margin",
        "not_semantic_validation": True,
        "promotable_as_verified_persona": False,
        "search_budgets": {"max_forwards": max_forwards, "estimated_forwards": estimated_forwards,
                           "max_candidates": max_candidates, "controls_per_candidate": controls_per_candidate},
        "evaluation": {"scales": [float(x) for x in scales], "normalization": normalization,
                       "max_mean_keep_kl": max_mean_kl, "min_keep_top1_agreement": min_top1_agreement,
                       "seed": seed, "calibration_medians": {str(k): v for k, v in medians.items()},
                       "calibration_sha256": hashlib.sha256(calibration.encode("utf-8")).hexdigest() if medians else None},
        "splits": splits, "search": search_results, "selected": selected,
        "heldout_test": heldout,
        "baseline_restoration_verified": True,
        "interpretation": (
            "Measured next-token logit-margin effects with matched same-layer controls, "
            "not proof of persona control, generalisation or statistical significance. "
            "Held-out results are exploratory until tested on an adequately sized independent corpus."
        ),
    }


def _verify_producer_model_binding(args, binding: dict[str, Any]) -> dict[str, Any]:
    """Require source model lineage for both measured profile and candidate rank."""
    from .operator_audit import _producer_audit
    expected_files = binding["file_hashes"]
    upstream = {}
    for role in ("profile", "ranking"):
        audit_path = _producer_audit(Path(getattr(args, role)))
        if audit_path is None:
            raise ValueError(
                f"{role} producer audit is missing; regenerate using installed "
                "aegis-neurosurgery profile-mlp / steer-rank commands"
            )
        record = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("status") != "completed":
            raise ValueError(f"{role} producer audit is incomplete")
        identity = (record.get("inputs") or {}).get("model") or record.get("source_model_binding")
        if not isinstance(identity, dict) or identity.get("files") != expected_files:
            raise ValueError(f"{role} producer checkpoint binding disagrees with the current model")
        # The outer operator audit validates the file checksums and upstream
        # audit self-hash; this also enforces exact checkpoint lineage.
        upstream[role] = {
            "producer_audit_path": str(audit_path),
            "producer_audit_sha256": file_sha256(audit_path),
        }
    return upstream


def run_steer_causal_command(args) -> dict[str, Any]:
    if args.max_search_prompts < 1 or args.max_test_prompts < 1:
        raise ValueError("Prompt limits must be positive")
    output = Path(args.out)
    if output.suffix != ".json" or output.exists():
        raise ValueError("New .json output path is required, and must not already exist")
    # Verify source lineage and exact model bytes *before* running model code.
    ranking = json.loads(Path(args.ranking).read_text(encoding="utf-8"))
    profile = load_tensor_artifact(args.profile)
    verify_discovery_lineage(
        ranking, profile,
        profile_sha256=file_sha256(args.profile),
        discovery_keep_sha256=file_sha256(args.discovery_keep),
        discovery_drop_sha256=file_sha256(args.discovery_drop),
    )
    binding = checkpoint_binding(
        args.model, audited_inputs=getattr(args, "_operator_input_bindings", None),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    discovery_keep = load_prompts(args.discovery_keep)
    discovery_drop = load_prompts(args.discovery_drop)
    producer_lineage = _verify_producer_model_binding(args, binding)
    device = resolve_device(args.device)
    model, tokenizer = load_local_model(args.model, device, args.dtype)
    search_targets = _read_target_rows(args.search_target, tokenizer, args.max_search_prompts)
    test_targets = _read_target_rows(args.test_target, tokenizer, args.max_test_prompts)
    search_keep = load_prompts(args.search_keep, args.max_search_prompts)
    test_keep = load_prompts(args.test_keep, args.max_test_prompts)
    scales = parse_scales(args.scales)
    calibration = (Path(args.calibration_data).read_text(encoding="utf-8")
                   if args.calibration_data else DEFAULT_CALIBRATION)
    report = run_causal_screen(
        model, tokenizer, ranking, profile, search_targets=search_targets,
        search_keep=search_keep, test_targets=test_targets, test_keep=test_keep,
        discovery_keep=discovery_keep, discovery_drop=discovery_drop,
        scales=scales, max_candidates=args.max_candidates,
        controls_per_candidate=args.controls_per_candidate,
        max_forwards=args.max_forwards, max_mean_kl=args.max_mean_kl,
        min_top1_agreement=args.min_top1_agreement,
        normalization=args.normalization, calibration=calibration,
        device=device, seed=args.seed,
    )
    report["checkpoint_binding"] = binding
    report["producer_lineage"] = producer_lineage
    report["semantic_binding"] = runtime_semantic_binding(model, tokenizer)
    report["ranking_sha256"] = file_sha256(args.ranking)
    report["source_profile_sha256"] = file_sha256(args.profile)
    report["input_path_sha256"] = {
        name: file_sha256(getattr(args, name)) for name in
        ("discovery_keep", "discovery_drop", "search_target", "search_keep", "test_target", "test_keep")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    return {
        "report": str(output), "selected": report["selected"],
        "heldout_observed_gate_passed": (
            report["heldout_test"]["observed_gate_passed"]
            if report["heldout_test"] is not None else False
        ),
        "promotable_as_verified_persona": False,
    }
