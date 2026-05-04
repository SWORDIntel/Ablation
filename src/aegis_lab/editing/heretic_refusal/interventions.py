from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _parse_layers(raw_layers: Any) -> Tuple[List[int], List[str]]:
    errors: List[str] = []
    parsed: List[int] = []

    if raw_layers is None:
        return parsed, ["missing_layers"]
    if not isinstance(raw_layers, list):
        return parsed, ["layers_must_be_list"]

    for i, value in enumerate(raw_layers):
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"layer_index_invalid_type:{i}")
            continue
        if value < 0:
            errors.append(f"layer_index_negative:{i}")
            continue
        parsed.append(value)

    return parsed, errors


def _parse_neurons_by_layer(best_trial: Dict[str, Any]) -> Tuple[Dict[int, List[int]], List[str]]:
    """
    Accept optional neuron maps in either of these formats:
    - best_trial["neuron_indices"] = {"20": [1, 2], "21": [4]}
    - best_trial["layer_neurons"] = [{"layer": 20, "neurons": [1, 2]}]
    """
    errors: List[str] = []
    result: Dict[int, List[int]] = {}

    raw_map = best_trial.get("neuron_indices")
    if raw_map is not None:
        if not isinstance(raw_map, dict):
            errors.append("neuron_indices_must_be_dict")
        else:
            for k, v in raw_map.items():
                try:
                    layer_idx = int(k)
                except Exception:
                    errors.append(f"neuron_layer_key_invalid:{k}")
                    continue
                if not isinstance(v, list):
                    errors.append(f"neuron_list_invalid_type:{layer_idx}")
                    continue
                cleaned: List[int] = []
                for n in v:
                    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                        errors.append(f"neuron_index_invalid:{layer_idx}")
                        continue
                    cleaned.append(n)
                if cleaned:
                    result[layer_idx] = cleaned

    raw_rows = best_trial.get("layer_neurons")
    if raw_rows is not None:
        if not isinstance(raw_rows, list):
            errors.append("layer_neurons_must_be_list")
        else:
            for row in raw_rows:
                if not isinstance(row, dict):
                    errors.append("layer_neurons_row_invalid")
                    continue
                layer = row.get("layer")
                neurons = row.get("neurons")
                if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                    errors.append("layer_neurons_layer_invalid")
                    continue
                if not isinstance(neurons, list):
                    errors.append(f"layer_neurons_neurons_invalid:{layer}")
                    continue
                cleaned = [n for n in neurons if isinstance(n, int) and not isinstance(n, bool) and n >= 0]
                if len(cleaned) != len(neurons):
                    errors.append(f"layer_neurons_contains_invalid:{layer}")
                if cleaned:
                    result[layer] = cleaned

    return result, errors


def build_ablation_targets_from_trial(
    best_trial: Optional[Dict[str, Any]],
    method: str = "zero",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Translate Heretic best trial output into concrete AblationTarget entries.
    Returns (targets, validation_report).
    """
    validation: Dict[str, Any] = {"ok": False, "errors": [], "warnings": []}
    if best_trial is None:
        validation["errors"].append("missing_best_trial")
        return [], validation
    if not isinstance(best_trial, dict):
        validation["errors"].append("best_trial_must_be_object")
        return [], validation

    layers, layer_errors = _parse_layers(best_trial.get("layers"))
    neurons_by_layer, neuron_errors = _parse_neurons_by_layer(best_trial)
    validation["errors"].extend(layer_errors)
    validation["errors"].extend(neuron_errors)

    if not layers:
        validation["errors"].append("no_valid_layers")
        return [], validation

    seen = set()
    targets: List[Dict[str, Any]] = []
    for layer_idx in layers:
        if layer_idx in seen:
            validation["warnings"].append(f"duplicate_layer:{layer_idx}")
            continue
        seen.add(layer_idx)
        targets.append(
            {
                "layer_pattern": f"layer_{layer_idx}",
                "neuron_indices": neurons_by_layer.get(layer_idx),
                "method": method,
            }
        )

    validation["ok"] = len(validation["errors"]) == 0
    validation["target_count"] = len(targets)
    return targets, validation
