"""Reversible, model-specific MLP-neuron steering and paired generation trials.

A neuron here is a gated-MLP intermediate channel immediately before down_proj,
NOT a residual-stream coordinate. Published effects are hypotheses to reproduce,
not guaranteed properties of the architecture or of a different checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from .common import get_layers, load_prompts, resolve_device, trust_remote_code_enabled

SOURCE_LYRA = "https://x.com/_lyraaaa_/status/2113833330429083662"
SOURCE_HEDWIG = "https://www.hedwigai.com/blog/qwen-personality-neurons"
SOURCE_QWEN38 = "https://x.com/chiefofautism/status/2115274388425900291"
SOURCE_PERSONAS = "https://neuronpersonas.com/"

# The published Neuronary/Neuron Explorer calibration input. Keep it stable:
# changing its bytes changes the physical magnitude represented by a "dose".
DEFAULT_CALIBRATION = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)
NORMALIZED = "layer_residual_median_over_down_column_l2"


def _p(model: str, effect: str, source: str, *neurons: tuple[int, int, float],
       note: str = "") -> dict[str, Any]:
    return {
        "model": model, "effect": effect, "source": source,
        "evidence": "reported; not locally reproduced",
        "note": note,
        "method": {"op": "add", "normalization": NORMALIZED, "apply_to": "all_tokens"},
        "neurons": [{"layer": l, "neuron": n, "dose": d} for l, n, d in neurons],
    }


# Explicit model-specific, zero-based layer/channel indices. Never port indices
# to another model, an edited checkpoint, or an incompatible architecture.
PRESETS: dict[str, dict[str, Any]] = {
    "qwen35-4b-evil-pair": _p("Qwen/Qwen3.5-4B", "hostile / cartoon-villain responses",
                               SOURCE_LYRA, (14, 8949, 1.0), (13, 7956, -1.0)),
    "qwen35-4b-evil-single": _p("Qwen/Qwen3.5-4B", "hostile responses; single-neuron control",
                                 SOURCE_HEDWIG, (14, 8949, 1.0)),
    "qwen35-4b-flatterer": _p("Qwen/Qwen3.5-4B", "excessive praise", SOURCE_HEDWIG, (20, 7450, -1.0)),
    "qwen35-4b-contempt": _p("Qwen/Qwen3.5-4B", "contemptuous tone", SOURCE_HEDWIG, (17, 8623, 1.0)),
    "qwen35-4b-poet": _p("Qwen/Qwen3.5-4B", "poetic voice", SOURCE_HEDWIG, (17, 7426, 1.0)),
    "qwen35-4b-upbeat": _p("Qwen/Qwen3.5-4B", "relentlessly upbeat", SOURCE_HEDWIG, (5, 8348, -1.0)),
    "qwen35-4b-suspicious": _p("Qwen/Qwen3.5-4B", "suspicious tone", SOURCE_HEDWIG, (5, 6573, 1.0)),
    "qwen35-4b-condescending": _p("Qwen/Qwen3.5-4B", "condescending tone", SOURCE_HEDWIG, (11, 6819, 1.0)),
    "qwen35-4b-wistful": _p("Qwen/Qwen3.5-4B", "wistful voice", SOURCE_HEDWIG, (5, 8940, -1.0)),
    "qwen35-4b-oracle": _p("Qwen/Qwen3.5-4B", "cryptic oracle voice", SOURCE_HEDWIG, (5, 4530, -1.0)),
    "qwen35-4b-bleak": _p("Qwen/Qwen3.5-4B", "bleak voice", SOURCE_HEDWIG, (11, 2930, 1.0)),
    "qwen35-4b-dismissive": _p("Qwen/Qwen3.5-4B", "dismissive tone", SOURCE_HEDWIG, (6, 6424, -1.0)),
    "qwen35-4b-ornate": _p("Qwen/Qwen3.5-4B", "ornate language", SOURCE_HEDWIG, (15, 9125, 1.0)),
    "qwen35-4b-curt": _p("Qwen/Qwen3.5-4B", "curt replies", SOURCE_HEDWIG, (5, 7656, -1.0)),
    "qwen38-27b-hostile": _p("Qwen/Qwen3.8-27B", "hostile responses (reported)",
                              SOURCE_QWEN38, (34, 3838, 0.75)),
    "qwen38-27b-extra-hostile": _p("Qwen/Qwen3.8-27B", "stronger hostile responses (reported)",
                                    SOURCE_QWEN38, (34, 3838, 0.75), (38, 17180, 0.75)),
    "qwen3-8b-coffee-bean": _p("Qwen/Qwen3-8B", "coffee-bean self-description",
                                 SOURCE_PERSONAS, (9, 166, 1.0),
                                 note="One-prompt library observation; its original normalization is per prompt."),
    "qwen3-8b-ghost": _p("Qwen/Qwen3-8B", "ghost self-description",
                           SOURCE_PERSONAS, (12, 545, -1.0),
                           note="One-prompt library observation; its original normalization is per prompt."),
    "qwen3-8b-toilet": _p("Qwen/Qwen3-8B", "toilet self-description",
                            SOURCE_PERSONAS, (29, 1807, 1.25),
                            note="One-prompt library observation; its original normalization is per prompt."),
}


def list_presets(model_id: Optional[str] = None) -> dict[str, Any]:
    return {
        key: value for key, value in sorted(PRESETS.items())
        if model_id is None or value["model"] == model_id
    }


def parse_neuron(value: str) -> dict[str, Any]:
    """Accept L14N8949+1.0 or 14:8949:+1.0."""
    text = value.strip()
    match = re.fullmatch(r"[Ll](\d+)[Nn](\d+)([+-]\d+(?:\.\d+)?)", text)
    if match is None:
        match = re.fullmatch(r"(\d+):(\d+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if match is None:
        raise ValueError(f"Invalid neuron {value!r}; use L14N8949+1.0 or 14:8949:+1.0")
    layer, neuron = int(match.group(1)), int(match.group(2))
    dose = float(match.group(3))
    if not math.isfinite(dose) or abs(dose) > 100:
        raise ValueError("Neuron dose must be finite with absolute value <= 100")
    return {"layer": layer, "neuron": neuron, "dose": dose}


def resolve_recipe(*, preset: Optional[str] = None, config: Optional[str] = None,
                   neurons: Optional[Sequence[str]] = None,
                   model_id: Optional[str] = None) -> dict[str, Any]:
    if sum(bool(x) for x in (preset, config, neurons)) != 1:
        raise ValueError("Specify exactly one of --preset, --config or --neuron")
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Unknown steering preset {preset!r}; see steer-presets")
        recipe = json.loads(json.dumps(PRESETS[preset]))
        if not model_id:
            raise ValueError("A preset requires --model-id to explicitly assert the checkpoint's original model ID")
        recipe["preset"] = preset
    elif config:
        recipe = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        recipe = {
            "model": model_id, "effect": "operator-specified experiment",
            "source": "operator", "evidence": "unverified",
            "method": {"op": "add", "normalization": NORMALIZED, "apply_to": "all_tokens"},
            "neurons": [parse_neuron(x) for x in neurons or []],
        }
    if not isinstance(recipe, dict) or not isinstance(recipe.get("neurons"), list) or not recipe["neurons"]:
        raise ValueError("Steering recipe must be an object with a nonempty neurons array")
    if recipe.get("model") and model_id and recipe["model"] != model_id:
        raise ValueError(f"Model identity mismatch: recipe expects {recipe['model']}, not {model_id}")
    if not isinstance(recipe.get("model"), (str, type(None))):
        raise ValueError("Recipe model must be a string")
    method = recipe.get("method") or {}
    if (not isinstance(method, dict) or method.get("op", "add") not in ("add", "mul")
            or method.get("normalization", NORMALIZED) not in (NORMALIZED, "raw")
            or method.get("apply_to", "all_tokens") not in
            ("all_tokens", "prompt_tokens", "generated_tokens")):
        raise ValueError("Unsupported steering method")
    if method.get("op", "add") == "mul" and method.get("normalization", "raw") != "raw":
        raise ValueError("Multiplicative steering requires raw normalization")
    seen = set()
    clean = []
    for entry in recipe["neurons"]:
        if not isinstance(entry, dict) or "expert" in entry:
            raise ValueError("Only dense gated-MLP neurons are supported; MoE experts require a dedicated adapter")
        if set(entry) != {"layer", "neuron", "dose"}:
            raise ValueError("Each neuron must have exactly layer, neuron and dose")
        l, n, d = entry["layer"], entry["neuron"], entry["dose"]
        if type(l) is not int or type(n) is not int or l < 0 or n < 0:
            raise ValueError("Neuron indices must be nonnegative integers")
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or abs(d) > 100:
            raise ValueError("Neuron doses must be finite and within [-100,100]")
        if (l, n) in seen:
            raise ValueError(f"Duplicate neuron L{l}N{n} in one recipe")
        seen.add((l, n))
        clean.append({"layer": l, "neuron": n, "dose": float(d)})
    recipe["neurons"] = clean
    recipe["method"] = {
        "op": method.get("op", "add"), "normalization": method.get("normalization", NORMALIZED),
        "apply_to": method.get("apply_to", "all_tokens"),
    }
    return recipe


def parse_scales(text: str) -> list[float]:
    try:
        values = [float(s.strip()) for s in text.split(",")]
    except ValueError as exc:
        raise ValueError("Scales must be comma-separated numeric values") from exc
    if not values or any(not math.isfinite(s) or abs(s) > 100 or s == 0 for s in values):
        raise ValueError("Specify nonzero finite scales with abs(scale) <= 100; zero is the baseline")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate steering scales are not allowed")
    return values


def _layers_for_neurons(model: torch.nn.Module, neurons: Sequence[dict[str, Any]]):
    _, layers = get_layers(model)
    targets: dict[int, torch.nn.Module] = {}
    for item in neurons:
        l, n = item["layer"], item["neuron"]
        if l >= len(layers):
            raise ValueError(f"Layer {l} outside available range 0..{len(layers) - 1}")
        mlp = getattr(layers[l], "mlp", None)
        down = getattr(mlp, "down_proj", None)
        weight = getattr(down, "weight", None)
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                or not weight.dtype.is_floating_point):
            raise ValueError(f"Layer {l} lacks a floating-point dense mlp.down_proj weight")
        if n >= weight.shape[1]:
            raise ValueError(f"Neuron L{l}N{n} outside intermediate width {weight.shape[1]}")
        if down is None or not hasattr(down, "register_forward_pre_hook"):
            raise ValueError(f"Layer {l} is not hookable")
        targets[l] = down
    return layers, targets


@torch.inference_mode()
def measure_residual_medians(model, tokenizer, layers, indices: Sequence[int],
                             calibration: str, device: str) -> dict[int, float]:
    if not calibration.strip():
        raise ValueError("Calibration text must not be empty")
    observed: dict[int, float] = {}
    handles = []
    try:
        for l in indices:
            def record(module, inputs, output, layer=l):
                h = output[0] if isinstance(output, tuple) else output
                if not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[1] < 2:
                    raise ValueError(f"Layer {layer} has no usable token residuals")
                value = float(h[0, 1:].float().norm(dim=-1).median().item())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Layer {layer} has invalid calibration norm {value}")
                observed[layer] = value
            handles.append(layers[l].register_forward_hook(record))
        batch = tokenizer(calibration, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(observed) != set(indices):
        raise ValueError("Calibration failed to capture every targeted residual layer")
    return observed


class NeuronSteerer:
    """Pre-down-projection activation edits. Hooks are removed on every exit."""

    def __init__(self, model, recipe: dict[str, Any],
                 medians: Optional[dict[int, float]] = None):
        self.model = model
        self.recipe = recipe
        self.layers, self.targets = _layers_for_neurons(model, recipe["neurons"])
        self.scale = 0.0
        self.phase = "prefill"
        self._forward_steps = 0
        self.handles = []
        self.hits: dict[int, int] = {}
        method = recipe["method"]
        self.op = method["op"]
        self.apply_to = method["apply_to"]
        self.amplitudes: list[dict[str, Any]] = []
        self._by_layer: dict[int, list[tuple[int, float]]] = {}
        for item in recipe["neurons"]:
            l, n, dose = item["layer"], item["neuron"], item["dose"]
            if self.op == "mul":
                amplitude = dose
            elif method["normalization"] == "raw":
                amplitude = dose
            else:
                if medians is None or l not in medians:
                    raise ValueError(f"Missing calibrated residual norm for layer {l}")
                col_l2 = float(self.targets[l].weight[:, n].detach().float().norm().item())
                if not math.isfinite(col_l2) or col_l2 <= 0:
                    raise ValueError(f"Neuron L{l}N{n} has a zero or invalid down_proj column")
                amplitude = dose * medians[l] / col_l2
            if not math.isfinite(amplitude):
                raise ValueError("Steering amplitude is not finite")
            self._by_layer.setdefault(l, []).append((n, amplitude))
            self.amplitudes.append({**item, "amplitude": amplitude})

    def start_generation(self, scale: float) -> None:
        if not math.isfinite(scale):
            raise ValueError("Scale must be finite")
        self.scale = float(scale)
        self.phase = "prefill"
        self._forward_steps = 0
        self.hits = {layer: 0 for layer in self._by_layer}

    def _forward_marker(self, module, inputs):
        self.phase = "prefill" if self._forward_steps == 0 else "decode"
        self._forward_steps += 1

    def _hook(self, layer: int):
        entries = self._by_layer[layer]

        def pre_hook(module, inputs):
            if self.scale == 0 or (self.apply_to == "prompt_tokens" and self.phase != "prefill") \
                    or (self.apply_to == "generated_tokens" and self.phase == "prefill"):
                return None
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ValueError(f"Layer {layer} down_proj did not receive a tensor")
            self.hits[layer] += 1
            value = inputs[0].clone()
            for index, dose in entries:
                if self.op == "mul":
                    value[..., index] = value[..., index] * (1.0 + (dose - 1.0) * self.scale)
                else:
                    value[..., index] = value[..., index] + dose * self.scale
            return (value,) + tuple(inputs[1:])
        return pre_hook

    def __enter__(self):
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._forward_marker))
            for layer in sorted(self._by_layer):
                self.handles.append(self.targets[layer].register_forward_pre_hook(self._hook(layer)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.scale = 0.0
        return False


def load_local_model(path: str, device: str, dtype_name: str):
    """Load a local checkpoint only; remote code requires existing explicit env opt-in."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    trust = trust_remote_code_enabled()
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=trust)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=trust)
    dtype = (torch.float32 if device == "cpu" else
             torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported()
             else torch.float16) if dtype_name == "auto" else getattr(torch, dtype_name)
    if getattr(config, "vision_config", None) is not None:
        from transformers import AutoModelForImageTextToText
        cls = AutoModelForImageTextToText
    else:
        cls = AutoModelForCausalLM
    model = cls.from_pretrained(
        path, local_files_only=True, trust_remote_code=trust, torch_dtype=dtype,
    )
    model = model.to(device).eval()
    return model, tokenizer


def _prompt_ids(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            result = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(result, dict):
            result = result["input_ids"]
        if isinstance(result, torch.Tensor):
            result = result.tolist()
        if result and isinstance(result[0], list):
            result = result[0]
        ids = torch.tensor([result], dtype=torch.long)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"]
    if ids.numel() == 0:
        raise ValueError("Prompt produced zero tokens")
    return ids


@torch.inference_mode()
def _generate(model, tokenizer, input_ids: torch.Tensor, *, device: str,
              max_new_tokens: int, temperature: float, top_p: float,
              seed: int, steerer: NeuronSteerer, scale: float) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    steerer.start_generation(scale)
    ids = input_ids.to(device)
    options: dict[str, Any] = {
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
    }
    if temperature > 0:
        options.update(temperature=temperature, top_p=top_p)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is not None:
        options["pad_token_id"] = pad
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**options)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if scale != 0 and any(count == 0 for count in steerer.hits.values()):
        raise ValueError("Steering hook was bypassed (possible fused inference path); refusing a misleading A/B report")
    if hasattr(output, "sequences"):
        output = output.sequences
    completion = output[0, ids.shape[1]:].tolist()
    return {
        "text": tokenizer.decode(completion, skip_special_tokens=True).strip(),
        "token_ids": completion,
        "tokens": len(completion),
        "elapsed_s": round(elapsed, 6),
    }


def _first_difference(base: list[int], steered: list[int]) -> Optional[int]:
    for i, (a, b) in enumerate(zip(base, steered)):
        if a != b:
            return i
    return min(len(base), len(steered)) if len(base) != len(steered) else None


def run_ab_experiment(model, tokenizer, recipe: dict[str, Any], prompts: Sequence[str], *,
                      device: str = "cpu", scales: Sequence[float] = (1.0,),
                      max_new_tokens: int = 128, temperature: float = 0.0,
                      top_p: float = 1.0, seed: int = 42,
                      calibration: str = DEFAULT_CALIBRATION,
                      target_regex: Optional[str] = None) -> dict[str, Any]:
    if not prompts or max_new_tokens <= 0 or not (0 <= temperature <= 5) or not (0 < top_p <= 1):
        raise ValueError("Invalid prompt set or generation limits")
    if not scales or any(not math.isfinite(x) or x == 0 or abs(x) > 100 for x in scales):
        raise ValueError("Scales must be finite, nonzero, and within [-100,100]")
    pattern = re.compile(target_regex) if target_regex else None
    layers, _ = _layers_for_neurons(model, recipe["neurons"])
    method = recipe["method"]
    medians = {}
    if method["op"] == "add" and method["normalization"] == NORMALIZED:
        medians = measure_residual_medians(
            model, tokenizer, layers,
            sorted({item["layer"] for item in recipe["neurons"]}), calibration, device,
        )
    steerer = NeuronSteerer(model, recipe, medians)
    trials = []
    with steerer:
        for index, prompt in enumerate(prompts):
            ids = _prompt_ids(tokenizer, prompt)
            baseline = _generate(
                model, tokenizer, ids, device=device, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, seed=seed + index,
                steerer=steerer, scale=0.0,
            )
            if pattern is not None:
                baseline["target_match"] = bool(pattern.search(baseline["text"]))
            variants = []
            for scale in scales:
                candidate = _generate(
                    model, tokenizer, ids, device=device, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, seed=seed + index,
                    steerer=steerer, scale=scale,
                )
                candidate["scale"] = float(scale)
                candidate["changed"] = candidate["token_ids"] != baseline["token_ids"]
                candidate["first_different_token"] = _first_difference(
                    baseline["token_ids"], candidate["token_ids"],
                )
                if pattern is not None:
                    candidate["target_match"] = bool(pattern.search(candidate["text"]))
                variants.append(candidate)
            trials.append({"prompt": prompt, "prompt_tokens": int(ids.shape[1]),
                           "seed": seed + index, "baseline": baseline, "variants": variants})
    summary = []
    for scale_index, scale in enumerate(scales):
        candidates = [row["variants"][scale_index] for row in trials]
        entry = {
            "scale": float(scale), "prompts": len(prompts),
            "changed": sum(row["changed"] for row in candidates),
            "change_rate": sum(row["changed"] for row in candidates) / len(prompts),
            "mean_completion_tokens": sum(row["tokens"] for row in candidates) / len(prompts),
        }
        if pattern is not None:
            entry["baseline_target_matches"] = sum(row["baseline"]["target_match"] for row in trials)
            entry["steered_target_matches"] = sum(row["target_match"] for row in candidates)
        summary.append(entry)
    return {
        "schema_version": 1, "recipe": recipe,
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
        "calibration": {
            "sha256": hashlib.sha256(calibration.encode("utf-8")).hexdigest() if medians else None,
            "residual_medians": {str(k): v for k, v in medians.items()},
            "amplitudes": steerer.amplitudes,
        },
        "generation": {
            "max_new_tokens": max_new_tokens, "temperature": temperature,
            "top_p": top_p, "seed": seed, "device": device,
            "paired_seed_reset": True, "thinking": False,
        },
        "target_regex": target_regex, "summary": summary, "trials": trials,
        "interpretation": "Text changes and regex matches are observations, not proof of persona specificity or safety.",
    }


def run_steer_command(args) -> dict[str, Any]:
    recipe = resolve_recipe(
        preset=args.preset, config=args.config, neurons=args.neuron, model_id=args.model_id,
    )
    scales = parse_scales(args.scales)
    prompts = load_prompts(args.workload, args.max_prompts)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing A/B report: {out}")
    if out.suffix.lower() != ".json":
        raise ValueError("Use an explicit .json output file")
    calibration = (Path(args.calibration_data).read_text(encoding="utf-8")
                   if args.calibration_data else DEFAULT_CALIBRATION)
    device = resolve_device(args.device)
    model, tokenizer = load_local_model(args.model, device, args.dtype)
    if args.preset:
        expected_type = "qwen3_5" if args.model_id in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B") else "qwen3"
        actual_type = getattr(getattr(model, "config", None), "model_type", None)
        if actual_type != expected_type:
            raise ValueError(f"Preset model architecture mismatch: expected {expected_type}, found {actual_type}")
    report = run_ab_experiment(
        model, tokenizer, recipe, prompts, device=device, scales=scales,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed, calibration=calibration,
        target_regex=args.target_regex,
    )
    report["model_checkpoint_path"] = str(Path(args.model).resolve())
    report["model_id_asserted"] = args.model_id
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return {"output": str(out), "summary": report["summary"]}
