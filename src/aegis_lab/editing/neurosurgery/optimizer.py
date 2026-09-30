from __future__ import annotations

import heapq
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml

from .adapters import get_attention_adapter, get_mlp_adapter, get_moe_adapter
from .attention import _register_attention_masks, select_attention_groups
from .common import LOG, file_sha256, get_layers, load_prompts, parameter_bytes, resolve_device, set_layers
from .mlp import _register_mlp_masks, select_mlp_channels
from .moe import _configured_top_k, _register_router_masks, select_moe_experts
from .validate import _load_hf, compare_logprobs, next_token_logprobs


@dataclass(frozen=True)
class CandidateState:
    """Discrete position in the Stage-4 structural search space."""

    layer_level: int = 0
    mlp_level: int = 0
    attention_level: int = 0
    moe_level: int = 0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (
            self.layer_level,
            self.mlp_level,
            self.attention_level,
            self.moe_level,
        )

    @classmethod
    def from_tuple(cls, value: Sequence[int]) -> "CandidateState":
        return cls(*[int(x) for x in value])


@dataclass(frozen=True)
class SearchConstraints:
    max_mean_kl: float = 0.02
    min_top1_agreement: float = 0.95

    def feasible(self, metrics: Dict[str, Any]) -> bool:
        return (
            float(metrics["mean_kl"]) <= self.max_mean_kl
            and float(metrics["top1_agreement"]) >= self.min_top1_agreement
        )


def normalize_ratios(values: Sequence[float]) -> List[float]:
    """Return unique keep ratios ordered from no surgery to most aggressive."""
    cleaned = {round(float(x), 8) for x in values}
    cleaned.add(1.0)
    if any(x <= 0.0 or x > 1.0 for x in cleaned):
        raise ValueError("all keep ratios must be in (0,1]")
    return sorted(cleaned, reverse=True)


def pareto_front(trials: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return feasible non-dominated trials.

    Objectives:
      maximize bytes_saved
      maximize estimated_macs_saved_per_token
      minimize mean_kl

    top1 agreement is treated as a hard feasibility gate rather than another
    objective because it is coarse/discontinuous.
    """

    feasible = [dict(t) for t in trials if bool(t.get("feasible", False))]

    def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        a_bytes = int(a.get("bytes_saved", 0))
        b_bytes = int(b.get("bytes_saved", 0))
        a_macs = int(a.get("estimated_macs_saved_per_token", 0))
        b_macs = int(b.get("estimated_macs_saved_per_token", 0))
        a_kl = float(a.get("mean_kl", float("inf")))
        b_kl = float(b.get("mean_kl", float("inf")))
        weak = a_bytes >= b_bytes and a_macs >= b_macs and a_kl <= b_kl
        strict = a_bytes > b_bytes or a_macs > b_macs or a_kl < b_kl
        return weak and strict

    front = []
    for candidate in feasible:
        if not any(dominates(other, candidate) for other in feasible if other is not candidate):
            front.append(candidate)
    front.sort(
        key=lambda row: (
            -int(row.get("bytes_saved", 0)),
            -int(row.get("estimated_macs_saved_per_token", 0)),
            float(row.get("mean_kl", float("inf"))),
        )
    )
    return front


def choose_best_feasible(trials: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    feasible = [dict(t) for t in trials if bool(t.get("feasible", False))]
    if not feasible:
        return None
    feasible.sort(
        key=lambda row: (
            -int(row.get("bytes_saved", 0)),
            -int(row.get("estimated_macs_saved_per_token", 0)),
            float(row.get("mean_kl", float("inf"))),
            -float(row.get("top1_agreement", 0.0)),
        )
    )
    return feasible[0]


def _neighbors(state: CandidateState, level_sizes: Tuple[int, int, int, int]) -> Iterable[CandidateState]:
    values = list(state.as_tuple())
    for axis, size in enumerate(level_sizes):
        if values[axis] + 1 >= size:
            continue
        nxt = list(values)
        nxt[axis] += 1
        yield CandidateState.from_tuple(nxt)


def constrained_frontier_search(
    level_sizes: Tuple[int, int, int, int],
    estimate_fn: Callable[[CandidateState], Dict[str, int]],
    evaluate_fn: Callable[[CandidateState], Dict[str, Any]],
    max_trials: int,
) -> List[Dict[str, Any]]:
    """Best-first constrained search over increasingly aggressive masks.

    A state is expanded only if it passes the quality constraints encoded by
    evaluate_fn. This exploits the usually-monotonic relationship between more
    structural deletion and more behavioral drift without requiring Optuna or a
    heavyweight search service.

    Use exhaustive_search when non-monotonic interactions are suspected and the
    search space is small enough to evaluate completely.
    """

    if any(size < 1 for size in level_sizes):
        raise ValueError("every search dimension must contain at least one level")
    if max_trials < 1:
        raise ValueError("max_trials must be >= 1")

    baseline = CandidateState()
    queued = {baseline.as_tuple()}
    heap: List[Tuple[int, int, Tuple[int, int, int, int]]] = []

    def push(state: CandidateState) -> None:
        key = state.as_tuple()
        if key in queued:
            return
        queued.add(key)
        estimate = estimate_fn(state)
        saved = int(estimate.get("bytes_saved", 0))
        macs = int(estimate.get("estimated_macs_saved_per_token", 0))
        heapq.heappush(heap, (-saved, -macs, key))

    trials: List[Dict[str, Any]] = []
    baseline_result = evaluate_fn(baseline)
    trials.append(baseline_result)

    if bool(baseline_result.get("feasible", False)):
        for neighbor in _neighbors(baseline, level_sizes):
            push(neighbor)

    while heap and len(trials) < max_trials:
        _, _, key = heapq.heappop(heap)
        state = CandidateState.from_tuple(key)
        result = evaluate_fn(state)
        trials.append(result)
        if bool(result.get("feasible", False)):
            for neighbor in _neighbors(state, level_sizes):
                push(neighbor)

    return trials


def exhaustive_search(
    level_sizes: Tuple[int, int, int, int],
    estimate_fn: Callable[[CandidateState], Dict[str, int]],
    evaluate_fn: Callable[[CandidateState], Dict[str, Any]],
    max_trials: int,
) -> List[Dict[str, Any]]:
    """Evaluate the highest-value states from the complete Cartesian grid."""
    states = [
        CandidateState.from_tuple(values)
        for values in itertools.product(*[range(size) for size in level_sizes])
    ]
    ranked = []
    for state in states:
        estimate = estimate_fn(state)
        ranked.append(
            (
                -int(estimate.get("bytes_saved", 0)),
                -int(estimate.get("estimated_macs_saved_per_token", 0)),
                sum(state.as_tuple()),
                state.as_tuple(),
            )
        )
    ranked.sort()
    if max_trials > 0:
        ranked = ranked[:max_trials]
    return [evaluate_fn(CandidateState.from_tuple(row[-1])) for row in ranked]


def _load_profile(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _layer_order(search_path: Optional[str], max_drop_layers: int) -> List[int]:
    if not search_path or max_drop_layers <= 0:
        return []
    payload = json.loads(Path(search_path).read_text(encoding="utf-8"))
    if "selected_layers" in payload:
        order = [int(x) for x in payload.get("selected_layers", [])]
    else:
        rows = list(payload.get("results", []))
        rows.sort(
            key=lambda row: (
                float(row.get("mean_kl", float("inf"))),
                -int(row.get("bytes_saved_in_loaded_dtype", row.get("bytes_saved", 0))),
            )
        )
        order = [int(row["layer"]) for row in rows]
    dedup = []
    seen = set()
    for idx in order:
        if idx not in seen:
            dedup.append(idx)
            seen.add(idx)
    return dedup[:max_drop_layers]


def _parameter_matrix_macs(module: torch.nn.Module) -> int:
    """Rough per-token linear-work proxy: count 2D+ parameter elements."""
    return sum(p.numel() for p in module.parameters() if p.ndim >= 2)


def _selection_cache(
    model,
    mlp_profile: Optional[Dict[str, Any]],
    attention_profile: Optional[Dict[str, Any]],
    moe_profile: Optional[Dict[str, Any]],
    mlp_ratios: Sequence[float],
    attention_ratios: Sequence[float],
    moe_ratios: Sequence[float],
    contrast_weight: float,
    align_to: int,
):
    cache: Dict[str, Dict[float, Any]] = {"mlp": {}, "attention": {}, "moe": {}}
    adapters: Dict[str, Any] = {}

    if mlp_profile is not None:
        adapters["mlp"] = get_mlp_adapter(model)
        for ratio in mlp_ratios:
            cache["mlp"][ratio] = select_mlp_channels(
                mlp_profile["keep_importance"],
                mlp_profile["drop_importance"],
                ratio,
                contrast_weight=contrast_weight,
                align_to=align_to,
            )

    if attention_profile is not None:
        adapters["attention"] = get_attention_adapter(model)
        for ratio in attention_ratios:
            cache["attention"][ratio] = select_attention_groups(
                attention_profile["keep_importance"],
                attention_profile["drop_importance"],
                ratio,
                contrast_weight=contrast_weight,
            )

    if moe_profile is not None:
        adapters["moe"] = get_moe_adapter(model)
        blocks = adapters["moe"].moe_blocks(model)
        min_keep = _configured_top_k(model, len(blocks[0].experts))
        for ratio in moe_ratios:
            cache["moe"][ratio] = select_moe_experts(
                moe_profile["keep_importance"],
                moe_profile["drop_importance"],
                ratio,
                contrast_weight=contrast_weight,
                min_keep=min_keep,
            )

    return cache, adapters


def _estimate_candidate(
    model,
    original_layers: Sequence[torch.nn.Module],
    layer_order: Sequence[int],
    state: CandidateState,
    ratios: Dict[str, Sequence[float]],
    cache: Dict[str, Dict[float, Any]],
    adapters: Dict[str, Any],
) -> Dict[str, int]:
    dropped = set(int(x) for x in layer_order[: state.layer_level])
    params_saved = 0
    bytes_saved = 0
    macs_saved = 0

    for idx in dropped:
        layer = original_layers[idx]
        params_saved += sum(p.numel() for p in layer.parameters())
        bytes_saved += parameter_bytes(layer)
        macs_saved += _parameter_matrix_macs(layer)

    if "mlp" in adapters:
        ratio = float(ratios["mlp"][state.mlp_level])
        selections = cache["mlp"][ratio]
        for idx, (triplet, kept) in enumerate(zip(adapters["mlp"].mlp_triplets(model), selections)):
            if idx in dropped:
                continue
            removed = int(triplet.gate.weight.shape[0]) - int(kept.numel())
            if removed <= 0:
                continue
            hidden_in = int(triplet.gate.weight.shape[1])
            hidden_out = int(triplet.down.weight.shape[0])
            params = removed * (hidden_in + hidden_in + hidden_out)
            b = (
                removed * hidden_in * triplet.gate.weight.element_size()
                + removed * hidden_in * triplet.up.weight.element_size()
                + removed * hidden_out * triplet.down.weight.element_size()
            )
            for module in (triplet.gate, triplet.up):
                bias = getattr(module, "bias", None)
                if bias is not None:
                    params += removed
                    b += removed * bias.element_size()
            params_saved += params
            bytes_saved += b
            macs_saved += removed * (hidden_in + hidden_in + hidden_out)

    if "attention" in adapters:
        ratio = float(ratios["attention"][state.attention_level])
        selections = cache["attention"][ratio]
        for idx, (pack, kept_groups) in enumerate(zip(adapters["attention"].attention_packs(model), selections)):
            if idx in dropped:
                continue
            removed_kv = pack.num_key_value_heads - int(kept_groups.numel())
            if removed_kv <= 0:
                continue
            removed_q = removed_kv * pack.group_size
            q_rows = removed_q * pack.head_dim
            kv_rows = removed_kv * pack.head_dim
            hidden = int(pack.q.weight.shape[1])
            out_hidden = int(pack.o.weight.shape[0])
            params = q_rows * hidden + 2 * kv_rows * hidden + q_rows * out_hidden
            b = (
                q_rows * hidden * pack.q.weight.element_size()
                + kv_rows * hidden * pack.k.weight.element_size()
                + kv_rows * hidden * pack.v.weight.element_size()
                + q_rows * out_hidden * pack.o.weight.element_size()
            )
            for module, rows in ((pack.q, q_rows), (pack.k, kv_rows), (pack.v, kv_rows)):
                bias = getattr(module, "bias", None)
                if bias is not None:
                    params += rows
                    b += rows * bias.element_size()
            params_saved += params
            bytes_saved += b
            macs_saved += q_rows * hidden + 2 * kv_rows * hidden + q_rows * out_hidden

    if "moe" in adapters:
        ratio = float(ratios["moe"][state.moe_level])
        selections = cache["moe"][ratio]
        for block, kept in zip(adapters["moe"].moe_blocks(model), selections):
            if block.layer_index in dropped:
                continue
            kept_set = set(int(x) for x in kept.tolist())
            for expert_idx, expert in enumerate(block.experts):
                if expert_idx in kept_set:
                    continue
                for p in expert.parameters():
                    params_saved += p.numel()
                    bytes_saved += p.numel() * p.element_size()
            removed = len(block.experts) - len(kept_set)
            if removed > 0:
                hidden = int(block.router.weight.shape[1])
                params_saved += removed * hidden
                bytes_saved += removed * hidden * block.router.weight.element_size()
                macs_saved += removed * hidden
                bias = getattr(block.router, "bias", None)
                if bias is not None:
                    params_saved += removed
                    bytes_saved += removed * bias.element_size()

    return {
        "parameters_saved": int(params_saved),
        "bytes_saved": int(bytes_saved),
        "estimated_macs_saved_per_token": int(macs_saved),
    }


def _write_selection(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    torch.save(payload, path)
    return {"selection": path.name, "selection_sha256": file_sha256(path)}


def _materialize_plan(
    out_dir: Path,
    best: Dict[str, Any],
    state: CandidateState,
    ratios: Dict[str, Sequence[float]],
    cache: Dict[str, Dict[float, Any]],
    profiles: Dict[str, Optional[str]],
    layer_order: Sequence[int],
    constraints: SearchConstraints,
) -> Path:
    structured: Dict[str, Any] = {}

    mlp_ratio = float(ratios["mlp"][state.mlp_level])
    if profiles.get("mlp") and mlp_ratio < 1.0:
        profile_path = Path(str(profiles["mlp"]))
        target = out_dir / "stage4.mlp.pt"
        ref = _write_selection(
            target,
            {
                "version": 2,
                "optimizer": "stage4",
                "source_profile": str(profile_path.resolve()),
                "source_profile_sha256": file_sha256(profile_path),
                "keep_ratio_requested": mlp_ratio,
                "keep_indices": cache["mlp"][mlp_ratio],
            },
        )
        structured["mlp"] = {"enabled": True, "keep_ratio_requested": mlp_ratio, **ref}

    attention_ratio = float(ratios["attention"][state.attention_level])
    if profiles.get("attention") and attention_ratio < 1.0:
        profile_path = Path(str(profiles["attention"]))
        target = out_dir / "stage4.attention.pt"
        ref = _write_selection(
            target,
            {
                "version": 2,
                "optimizer": "stage4",
                "source_profile": str(profile_path.resolve()),
                "source_profile_sha256": file_sha256(profile_path),
                "keep_ratio_requested": attention_ratio,
                "keep_groups": cache["attention"][attention_ratio],
            },
        )
        structured["attention"] = {"enabled": True, "keep_ratio_requested": attention_ratio, **ref}

    moe_ratio = float(ratios["moe"][state.moe_level])
    if profiles.get("moe") and moe_ratio < 1.0:
        profile_path = Path(str(profiles["moe"]))
        target = out_dir / "stage4.moe.pt"
        ref = _write_selection(
            target,
            {
                "version": 2,
                "optimizer": "stage4",
                "source_profile": str(profile_path.resolve()),
                "source_profile_sha256": file_sha256(profile_path),
                "keep_ratio_requested": moe_ratio,
                "keep_experts": cache["moe"][moe_ratio],
            },
        )
        structured["moe"] = {"enabled": True, "keep_ratio_requested": moe_ratio, **ref}

    plan = {
        "version": 4,
        "drop_layers": [int(x) for x in layer_order[: state.layer_level]],
        "ablation": {
            "layers": [],
            "strength": 0.0,
            "targets": [],
            "norm_preserve": True,
            "preserve_subspace": True,
        },
        "structured": structured,
        "validation_gate": {
            "max_mean_kl": constraints.max_mean_kl,
            "min_top1_agreement": constraints.min_top1_agreement,
        },
        "optimizer": {
            "stage": 4,
            "trial_id": int(best["trial_id"]),
            "state": list(state.as_tuple()),
            "mean_kl": float(best["mean_kl"]),
            "top1_agreement": float(best["top1_agreement"]),
            "bytes_saved": int(best["bytes_saved"]),
            "estimated_macs_saved_per_token": int(best["estimated_macs_saved_per_token"]),
            "report": "optimization.json",
        },
    }
    plan_path = out_dir / "optimized_plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return plan_path


def run_optimizer(
    model_path: str,
    keep_path: str,
    out_dir: str,
    mlp_profile_path: Optional[str] = None,
    attention_profile_path: Optional[str] = None,
    moe_profile_path: Optional[str] = None,
    layer_search_path: Optional[str] = None,
    mlp_ratios: Sequence[float] = (1.0, 0.95, 0.9, 0.85, 0.8),
    attention_ratios: Sequence[float] = (1.0, 0.875, 0.75, 0.625),
    moe_ratios: Sequence[float] = (1.0, 0.875, 0.75, 0.625),
    max_drop_layers: int = 3,
    max_mean_kl: float = 0.02,
    min_top1_agreement: float = 0.95,
    max_trials: int = 64,
    strategy: str = "frontier",
    contrast_weight: float = 0.25,
    align_to: int = 64,
    batch_size: int = 2,
    max_prompts: Optional[int] = 64,
    device: str = "auto",
) -> Dict[str, Any]:
    """Run Stage-4 joint structural search and materialize the best passing plan."""

    if strategy not in {"frontier", "exhaustive"}:
        raise ValueError("strategy must be 'frontier' or 'exhaustive'")
    if max_drop_layers < 0:
        raise ValueError("max_drop_layers must be >= 0")

    device = resolve_device(device)
    prompts = load_prompts(keep_path, max_prompts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mlp_ratios = normalize_ratios(mlp_ratios if mlp_profile_path else (1.0,))
    attention_ratios = normalize_ratios(attention_ratios if attention_profile_path else (1.0,))
    moe_ratios = normalize_ratios(moe_ratios if moe_profile_path else (1.0,))
    ratios = {
        "mlp": mlp_ratios,
        "attention": attention_ratios,
        "moe": moe_ratios,
    }

    profiles = {
        "mlp": mlp_profile_path,
        "attention": attention_profile_path,
        "moe": moe_profile_path,
    }
    mlp_profile = _load_profile(mlp_profile_path)
    attention_profile = _load_profile(attention_profile_path)
    moe_profile = _load_profile(moe_profile_path)

    model, tokenizer = _load_hf(model_path, device)
    layer_path, layer_modules = get_layers(model)
    original_layers = list(layer_modules)
    layer_order = _layer_order(layer_search_path, max_drop_layers)
    for idx in layer_order:
        if idx < 0 or idx >= len(original_layers):
            raise IndexError(f"layer search proposes invalid original layer index {idx}")

    cache, adapters = _selection_cache(
        model,
        mlp_profile,
        attention_profile,
        moe_profile,
        mlp_ratios,
        attention_ratios,
        moe_ratios,
        contrast_weight,
        align_to,
    )

    constraints = SearchConstraints(
        max_mean_kl=float(max_mean_kl),
        min_top1_agreement=float(min_top1_agreement),
    )
    baseline = next_token_logprobs(model, tokenizer, prompts, batch_size)
    total_bytes = parameter_bytes(model)
    total_params = sum(p.numel() for p in model.parameters())

    level_sizes = (
        len(layer_order) + 1,
        len(mlp_ratios),
        len(attention_ratios),
        len(moe_ratios),
    )
    trial_counter = {"value": 0}

    def estimate(state: CandidateState) -> Dict[str, int]:
        return _estimate_candidate(
            model,
            original_layers,
            layer_order,
            state,
            ratios,
            cache,
            adapters,
        )

    def evaluate(state: CandidateState) -> Dict[str, Any]:
        trial_counter["value"] += 1
        trial_id = trial_counter["value"]
        hooks = []
        dropped = [int(x) for x in layer_order[: state.layer_level]]
        dropped_set = set(dropped)
        estimate_row = estimate(state)

        mlp_ratio = float(mlp_ratios[state.mlp_level])
        attention_ratio = float(attention_ratios[state.attention_level])
        moe_ratio = float(moe_ratios[state.moe_level])

        try:
            if "mlp" in adapters and mlp_ratio < 1.0:
                hooks.extend(_register_mlp_masks(model, adapters["mlp"], cache["mlp"][mlp_ratio]))
            if "attention" in adapters and attention_ratio < 1.0:
                hooks.extend(
                    _register_attention_masks(
                        model,
                        adapters["attention"],
                        cache["attention"][attention_ratio],
                    )
                )
            if "moe" in adapters and moe_ratio < 1.0:
                hooks.extend(_register_router_masks(model, adapters["moe"], cache["moe"][moe_ratio]))

            if dropped:
                set_layers(
                    model,
                    layer_path,
                    [layer for idx, layer in enumerate(original_layers) if idx not in dropped_set],
                )

            candidate = next_token_logprobs(model, tokenizer, prompts, batch_size)
            metrics = compare_logprobs(baseline, candidate)
        finally:
            set_layers(model, layer_path, original_layers)
            for hook in hooks:
                hook.remove()

        row: Dict[str, Any] = {
            "trial_id": trial_id,
            "state": list(state.as_tuple()),
            "drop_layers": dropped,
            "mlp_keep_ratio": mlp_ratio,
            "attention_keep_ratio": attention_ratio,
            "moe_keep_ratio": moe_ratio,
            **estimate_row,
            **metrics,
        }
        row["feasible"] = constraints.feasible(row)
        row["parameter_fraction_saved"] = row["parameters_saved"] / max(total_params, 1)
        row["byte_fraction_saved"] = row["bytes_saved"] / max(total_bytes, 1)

        LOG.info(
            "stage4 trial=%d state=%s feasible=%s KL=%.6f top1=%.3f saved=%.2f MiB macs=%d",
            trial_id,
            state.as_tuple(),
            row["feasible"],
            row["mean_kl"],
            row["top1_agreement"],
            row["bytes_saved"] / (1024 * 1024),
            row["estimated_macs_saved_per_token"],
        )
        return row

    if strategy == "exhaustive":
        trials = exhaustive_search(level_sizes, estimate, evaluate, max_trials)
    else:
        trials = constrained_frontier_search(level_sizes, estimate, evaluate, max_trials)

    front = pareto_front(trials)
    best = choose_best_feasible(trials)
    if best is None:
        raise RuntimeError("Stage-4 search found no feasible candidate, including baseline")

    best_state = CandidateState.from_tuple(best["state"])
    plan_path = _materialize_plan(
        out,
        best,
        best_state,
        ratios,
        cache,
        profiles,
        layer_order,
        constraints,
    )

    report = {
        "version": 1,
        "stage": 4,
        "model": model_path,
        "keep_dataset": str(Path(keep_path).resolve()),
        "prompts": len(prompts),
        "strategy": strategy,
        "max_trials": max_trials,
        "constraints": {
            "max_mean_kl": constraints.max_mean_kl,
            "min_top1_agreement": constraints.min_top1_agreement,
        },
        "search_space": {
            "layer_order": layer_order,
            "layer_levels": level_sizes[0],
            "mlp_ratios": mlp_ratios,
            "attention_ratios": attention_ratios,
            "moe_ratios": moe_ratios,
            "cartesian_states": int(
                level_sizes[0] * level_sizes[1] * level_sizes[2] * level_sizes[3]
            ),
        },
        "trials": trials,
        "pareto_front": front,
        "best": best,
        "optimized_plan": plan_path.name,
        "cost_model": {
            "bytes_saved": "exact loaded-dtype parameter bytes for supported physical cuts, overlap-corrected for dropped layers",
            "estimated_macs_saved_per_token": "2D parameter-element proxy for dense linear work; MoE expert deletion mostly counts resident bytes because active top-k compute is rerouted rather than eliminated",
            "latency": "not measured during masked search because mask hooks preserve original tensor shapes and would give misleading speedup numbers",
        },
    }
    (out / "optimization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info(
        "Stage 4 selected trial %d: %.2f MiB saved, KL %.6f, plan=%s",
        best["trial_id"],
        best["bytes_saved"] / (1024 * 1024),
        best["mean_kl"],
        plan_path,
    )
    return report
