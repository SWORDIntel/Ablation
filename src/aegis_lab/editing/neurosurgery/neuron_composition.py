"""Budgeted beam expansion and signed coordinate search for MLP neuron groups.

Runs *inside* the audited causal experiment, reusing its reversible hooks and
next-token objective. This module does not promote behavioural/persona claims.
"""
from __future__ import annotations

import math
from typing import Any, Sequence


def extra_forward_budget(
    max_groups: int, max_group_size: int, n_scales: int,
    search_size: int, test_size: int,
) -> int:
    """Conservative number of additional forward calls before any experiment."""
    if max_groups == 0:
        return 0
    # One initial trial, at most (scale_count - 1) choices per coordinate,
    # one matched-control trial, and final held-out group/control/monomers/parent.
    return (
        max_groups * (2 + max_group_size * (n_scales - 1)) * search_size
        + (max_group_size + 3) * test_size
    )


def _members_key(members: Sequence[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((m["layer"], m["neuron"]) for m in members))


def _ordered_members(members: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"layer": int(m["layer"]), "neuron": int(m["neuron"]),
         "scale": float(m["scale"])}
        for m in sorted(members, key=lambda x: (x["layer"], x["neuron"]))
    ]


def _quality_gate(obs: dict[str, Any], max_kl: float, min_top1: float) -> bool:
    return bool(obs["keep_mean_kl"] <= max_kl and
                obs["keep_top1_agreement"] >= min_top1)


def _objective(obs: dict[str, Any], keep_penalty: float,
               neuron_penalty: float, cardinality: int) -> float:
    return (obs["target_mean_delta_margin"]
            - keep_penalty * obs["keep_mean_kl"]
            - neuron_penalty * cardinality)


def _single_observation(
    singles: dict[tuple[int, int, float], dict[str, Any]],
    member: dict[str, Any],
) -> dict[str, Any]:
    key = (member["layer"], member["neuron"], member["scale"])
    if key not in singles:
        raise ValueError(f"No isolated causal trial for {key}")
    return singles[key]


def _group_metrics(
    group: dict[str, Any], matched: dict[str, Any],
    singles: Sequence[dict[str, Any]], parent: dict[str, Any], bootstrap,
    ci_seed: int,
) -> dict[str, Any]:
    """Compare one group to exactly aligned monomers, controls and its parent."""
    if not singles:
        raise ValueError("Missing isolated monomer trials")
    rows = [group["target_per_prompt"], matched["target_per_prompt"],
            parent["target_per_prompt"]] + [s["target_per_prompt"] for s in singles]
    if not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Group/control/parent/monomer prompt counts differ")
    interaction = []
    parent_gain = []
    for aligned in zip(*rows):
        if len({item["prompt_sha256"] for item in aligned}) != 1:
            raise ValueError("Group/control/parent/monomer prompt alignment mismatch")
        current = aligned[0]["delta_margin"]
        interaction.append(current - sum(item["delta_margin"] for item in aligned[3:]))
        parent_gain.append(current - aligned[2]["delta_margin"])
    target = group["target_mean_delta_margin"]
    against_control = target - matched["target_mean_delta_margin"]
    against_best_single = target - max(s["target_mean_delta_margin"] for s in singles)
    against_parent = target - parent["target_mean_delta_margin"]
    return {
        "target_mean_delta_margin": target,
        "matched_group_delta_margin": matched["target_mean_delta_margin"],
        "gain_over_matched_group": against_control,
        "gain_over_best_single": against_best_single,
        "gain_over_parent": against_parent,
        "gain_over_parent_ci95_bootstrap": bootstrap(parent_gain, ci_seed),
        "nonadditive_interaction_mean": sum(interaction) / len(interaction),
        "nonadditive_interaction_ci95_bootstrap": bootstrap(interaction, ci_seed + 1),
        "nonadditive_interaction_per_prompt": interaction,
        "definition": "group delta margin minus sum of isolated neuron delta margins",
    }


def _pareto_frontier(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nondominated feasible groups: target gain ↑, KEEP KL ↓, size ↓."""
    eligible = [t for t in trials if t["search_gate_passed"]]
    result = []
    for candidate in eligible:
        a = candidate["joint"]
        n = len(candidate["members"])
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            b = other["joint"]
            m = len(other["members"])
            if (b["target_mean_delta_margin"] >= a["target_mean_delta_margin"]
                    and b["keep_mean_kl"] <= a["keep_mean_kl"]
                    and m <= n
                    and (b["target_mean_delta_margin"] > a["target_mean_delta_margin"]
                         or b["keep_mean_kl"] < a["keep_mean_kl"] or m < n)):
                dominated = True
                break
        if not dominated:
            result.append({
                "members": candidate["members"],
                "target_mean_delta_margin": a["target_mean_delta_margin"],
                "keep_mean_kl": a["keep_mean_kl"],
                "keep_top1_agreement": a["keep_top1_agreement"],
                "search_objective": candidate["search_objective"],
            })
    return sorted(result, key=lambda x: (
        -x["search_objective"], len(x["members"]), _members_key(x["members"])
    ))


def optimize_groups(
    model, tokenizer, profile: dict[str, Any],
    *, candidates: Sequence[dict[str, Any]],
    single_trials: Sequence[dict[str, Any]],
    qualified_pairs: Sequence[dict[str, Any]],
    controls: dict[tuple[int, int], list[dict[str, Any]]],
    all_coordinates: Sequence[tuple[int, int]],
    medians: dict[int, float], normalization: str,
    scales: Sequence[float],
    search_targets: Sequence[dict[str, Any]], search_target_baseline,
    search_keep: Sequence[str], search_keep_baseline,
    test_targets: Sequence[dict[str, Any]], test_target_baseline,
    test_keep: Sequence[str], test_keep_baseline,
    max_group_size: int, beam_width: int, max_groups: int,
    max_mean_kl: float, min_top1_agreement: float,
    keep_penalty: float, neuron_penalty: float,
    device: str, seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Search groups of size 3..max_group_size, using only search for selection.

    Every observed group and dose is measured through neuron_causal's existing
    _evaluate_intervention and NeuronSteerer; held-out observations are never
    used by the search or coordinate updates.
    """
    from .neuron_causal import (
        _bootstrap_ci95, _combination_recipe, _evaluate_intervention,
        _pair_controls, _recipe_for,
    )
    # Best passing isolated dose per coordinate is already selected by the
    # parent caller's search-only ordering.
    monomers: dict[tuple[int, int], dict[str, Any]] = {}
    for row in candidates:
        monomers.setdefault((row["layer"], row["neuron"]), row)
    isolated = {
        (row["layer"], row["neuron"], row["scale"]): row
        for row in single_trials if row["kind"] == "candidate"
    }
    attempted = 0
    seen: set[tuple[tuple[int, int], ...]] = set()
    stages = []
    all_trials = []
    parents = list(qualified_pairs[:beam_width])

    for depth in range(3, max_group_size + 1):
        if attempted >= max_groups or not parents:
            break
        level = []
        for parent in parents:
            if attempted >= max_groups:
                break
            for monomer in monomers.values():
                if attempted >= max_groups:
                    break
                members = _ordered_members(parent["members"])
                coord = (monomer["layer"], monomer["neuron"])
                if coord in _members_key(members):
                    continue
                proposed = _ordered_members(members + [monomer])
                key = _members_key(proposed)
                if key in seen:
                    continue
                seen.add(key)
                attempted += 1
                joint = _evaluate_intervention(
                    model, tokenizer, _combination_recipe(proposed, normalization),
                    medians, 1.0, search_targets, search_target_baseline,
                    search_keep, search_keep_baseline, device=device,
                    ci_seed=seed + attempted * 1009,
                )
                best_members = proposed
                best_joint = joint
                best_objective = (
                    _objective(joint, keep_penalty, neuron_penalty, depth)
                    if _quality_gate(joint, max_mean_kl, min_top1_agreement)
                    else float("-inf")
                )
                dose_trials = [{
                    "members": proposed, "target_mean_delta_margin": joint["target_mean_delta_margin"],
                    "keep_mean_kl": joint["keep_mean_kl"],
                    "keep_gate_passed": _quality_gate(joint, max_mean_kl, min_top1_agreement),
                    "search_objective": best_objective if math.isfinite(best_objective) else None,
                    "stage": "initial",
                }]
                # One deterministic coordinate-descent pass. Each coordinate
                # tries every alternative signed dose under the same KEEP gate.
                for i in range(depth):
                    for dose in scales:
                        if dose == best_members[i]["scale"]:
                            continue
                        tested = [dict(m) for m in best_members]
                        tested[i]["scale"] = float(dose)
                        obs = _evaluate_intervention(
                            model, tokenizer, _combination_recipe(tested, normalization),
                            medians, 1.0, search_targets, search_target_baseline,
                            search_keep, search_keep_baseline, device=device,
                            ci_seed=seed + attempted * 1009 + len(dose_trials),
                        )
                        keep_gate = _quality_gate(obs, max_mean_kl, min_top1_agreement)
                        value = (_objective(obs, keep_penalty, neuron_penalty, depth)
                                 if keep_gate else float("-inf"))
                        dose_trials.append({
                            "members": tested,
                            "target_mean_delta_margin": obs["target_mean_delta_margin"],
                            "keep_mean_kl": obs["keep_mean_kl"],
                            "keep_gate_passed": keep_gate,
                            "search_objective": value if math.isfinite(value) else None,
                            "stage": f"coordinate_{i}",
                        })
                        if value > best_objective:
                            best_members, best_joint, best_objective = tested, obs, value

                matched_members = _pair_controls(
                    model, profile, controls, set(all_coordinates), best_members,
                    seed + attempted * 7919,
                )
                matched_obs = _evaluate_intervention(
                    model, tokenizer, _combination_recipe(matched_members, normalization),
                    medians, 1.0, search_targets, search_target_baseline,
                    search_keep, search_keep_baseline, device=device,
                    ci_seed=seed + attempted * 1009 + 997,
                )
                singles = [_single_observation(isolated, m) for m in best_members]
                metrics = _group_metrics(
                    best_joint, matched_obs, singles, parent["joint"],
                    _bootstrap_ci95, seed + attempted * 1009 + 998,
                )
                gate = _quality_gate(best_joint, max_mean_kl, min_top1_agreement)
                passes = bool(
                    gate and math.isfinite(best_objective)
                    and best_joint["target_mean_delta_margin"] > 0
                    and metrics["gain_over_matched_group"] > 0
                    and metrics["gain_over_parent"] > 0
                )
                record = {
                    "members": best_members, "parent_members": parent["members"],
                    "matched_controls": matched_members,
                    "joint": best_joint, "matched_control_group": matched_obs,
                    "interaction": metrics, "coordinate_trials": dose_trials,
                    "search_objective": best_objective if math.isfinite(best_objective) else None,
                    "keep_gate_passed": gate, "search_gate_passed": passes,
                    "selection_source": "search split only",
                }
                all_trials.append(record)
                if passes:
                    level.append(record)
        level.sort(key=lambda t: (
            -t["search_objective"], -t["interaction"]["gain_over_matched_group"],
            t["joint"]["keep_mean_kl"], _members_key(t["members"])
        ))
        parents = level[:beam_width]
        stages.append({"size": depth, "attempted": len([t for t in all_trials
                        if len(t["members"]) == depth]),
                       "qualified": len(level), "retained": len(parents)})

    qualified = [t for t in all_trials if t["search_gate_passed"]]
    qualified.sort(key=lambda t: (
        -t["search_objective"], -t["interaction"]["gain_over_matched_group"],
        len(t["members"]), _members_key(t["members"])
    ))
    selected = qualified[0] if qualified else None
    heldout = None
    if selected:
        members = selected["members"]
        matched = selected["matched_controls"]
        group_test = _evaluate_intervention(
            model, tokenizer, _combination_recipe(members, normalization),
            medians, 1.0, test_targets, test_target_baseline,
            test_keep, test_keep_baseline, device=device,
            ci_seed=seed + 900001,
        )
        control_test = _evaluate_intervention(
            model, tokenizer, _combination_recipe(matched, normalization),
            medians, 1.0, test_targets, test_target_baseline,
            test_keep, test_keep_baseline, device=device,
            ci_seed=seed + 900002,
        )
        mono_tests = [
            _evaluate_intervention(
                model, tokenizer, _recipe_for(m["layer"], m["neuron"], normalization),
                medians, m["scale"], test_targets, test_target_baseline,
                test_keep, test_keep_baseline, device=device,
                ci_seed=seed + 900003 + i,
            )
            for i, m in enumerate(members)
        ]
        parent_test = _evaluate_intervention(
            model, tokenizer,
            _combination_recipe(selected["parent_members"], normalization),
            medians, 1.0, test_targets, test_target_baseline,
            test_keep, test_keep_baseline, device=device,
            ci_seed=seed + 900003 + len(members),
        )
        metrics = _group_metrics(
            group_test, control_test, mono_tests, parent_test,
            _bootstrap_ci95, seed + 900010,
        )
        keep_gate = _quality_gate(group_test, max_mean_kl, min_top1_agreement)
        heldout = {
            "members": members, "selected_on": "search split only",
            "matched_controls": matched, "joint": group_test,
            "matched_control_group": control_test,
            "monomers": mono_tests, "parent": parent_test,
            "interaction": metrics, "keep_gate_passed": keep_gate,
            "observed_gate_passed": bool(
                keep_gate and group_test["target_mean_delta_margin"] > 0
                and metrics["gain_over_matched_group"] > 0
                and metrics["gain_over_parent"] > 0
            ),
            "promotable_as_verified_persona": False,
        }
    report = {
        "enabled": max_groups > 0,
        "search_method": "bounded beam expansion plus one-pass signed coordinate descent",
        "max_group_size": max_group_size, "beam_width": beam_width,
        "max_groups": max_groups, "attempted_groups": attempted,
        "objective": "margin_gain - keep_penalty * mean_KEEP_KL - neuron_penalty * number_neurons",
        "keep_penalty": keep_penalty, "neuron_penalty": neuron_penalty,
        "stages": stages, "trials": all_trials, "selected": selected,
        "pareto_frontier": _pareto_frontier(all_trials),
        "source": "search split only",
        "evidence": "exploratory next-token logit effects; no persona certification",
    }
    return report, heldout
