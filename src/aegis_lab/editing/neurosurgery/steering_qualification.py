"""Bounded real-checkpoint qualification using the installed steering operators.

No downloads, alternate generation engine, automatic labels, or persona claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch

from .common import get_layers, load_prompts
from .measured import bind_files, verify_binding
from .neuron_steering import DEFAULT_CALIBRATION, NeuronSteerer
from .steering_provenance import checkpoint_binding

MODEL_ID = "Qwen/Qwen3.5-4B"


def preflight(checkpoint):
    """Require the exact initial target geometry and an installed stock HF adapter."""
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText

    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    text = raw.get("text_config", {})
    expected = {"model_type": "qwen3_5_text", "num_hidden_layers": 32,
                "hidden_size": 2560, "intermediate_size": 9216, "vocab_size": 248320}
    if (raw.get("model_type") != "qwen3_5" or
            raw.get("architectures") != ["Qwen3_5ForConditionalGeneration"] or
            any(text.get(k) != v for k, v in expected.items()) or
            raw.get("quantization_config") or text.get("quantization_config")):
        raise ValueError("Expected unquantized Qwen3.5-4B conditional-generation architecture")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    # Resolve/import the class without allocating any checkpoint weights.
    cls = AutoModelForImageTextToText._model_mapping[type(config)]
    return {"model_id_asserted": MODEL_ID, "model_class": cls.__name__,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": sys.version, "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "cuda_devices": [torch.cuda.get_device_properties(i).__repr__()
                             for i in range(torch.cuda.device_count())],
            "architecture_supported_by_installed_runtime": True,
            "hardware_execution_verified": False}


@torch.inference_mode()
def qualify_loaded_model(model, tokenizer, recipe, report, keep, device,
                         max_mean_kl=0.05, min_top1_agreement=0.95):
    """Next-token KEEP KL plus actual activation delivery and unhooked restoration."""
    from .neuron_causal import _next_logits

    if (not keep or not math.isfinite(max_mean_kl) or max_mean_kl < 0 or
            not math.isfinite(min_top1_agreement) or not 0 <= min_top1_agreement <= 1):
        raise ValueError("Invalid preservation inputs or gates")
    if recipe["method"]["apply_to"] == "generated_tokens":
        raise ValueError("Qualification KEEP prefill metric requires all_tokens or prompt_tokens")
    if set(keep) & {row["prompt"] for row in report["trials"]}:
        raise ValueError("KEEP and behavioural evaluation prompts overlap")
    for trial in report["trials"]:
        for arm in trial["variants"]:
            if not arm["steering_applied"] or not all(
                    n > 0 for n in arm["neuron_values_changed"].values()):
                raise ValueError("Qualification requires nonzero delivered activation edits at every layer")
    _, layers = get_layers(model)
    modules = list(model.modules())
    hooks_before = [(tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in modules]
    medians = {int(k): v for k, v in report["calibration"]["residual_medians"].items()}
    measurements = []
    for prompt in keep:
        base = _next_logits(model, tokenizer, prompt, device)
        for summary in report["summary"]:
            scale = summary["scale"]
            with NeuronSteerer(model, recipe, medians) as steerer:
                edited = _next_logits(model, tokenizer, prompt, device, steerer, scale)
                if not all(n > 0 for n in steerer.neuron_changes.values()):
                    raise ValueError("KEEP hook did not change activations")
            restored = _next_logits(model, tokenizer, prompt, device)
            if not torch.equal(base, restored):
                raise ValueError("Unhooked KEEP logits were not exactly restored")
            p, q = base.log_softmax(-1), edited.log_softmax(-1)
            kl = float((p.exp() * (p - q)).sum())
            if not math.isfinite(kl) or kl < -1e-4:
                raise ValueError("Invalid KEEP KL")
            measurements.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                 "scale": scale, "kl": max(0., kl),
                                 "top1_agreement": bool(base.argmax() == edited.argmax())})
    if hooks_before != [(tuple(m._forward_pre_hooks), tuple(m._forward_hooks)) for m in modules]:
        raise ValueError("Hook registry was not restored")
    by_scale = []
    for item in report["summary"]:
        rows = [r for r in measurements if r["scale"] == item["scale"]]
        kl = sum(r["kl"] for r in rows) / len(rows)
        agreement = sum(r["top1_agreement"] for r in rows) / len(rows)
        by_scale.append({"scale": item["scale"], "mean_next_token_kl": kl,
                         "top1_agreement": agreement,
                         "preservation_gate_passed": kl <= max_mean_kl and agreement >= min_top1_agreement})
    return {"qualified": False, "status": "awaiting_independent_behavioural_evaluation",
            "activation_delivery_verified": True, "hook_registry_restored": True,
            "unhooked_logits_exactly_restored": True,
            "max_mean_kl": max_mean_kl, "min_top1_agreement": min_top1_agreement,
            "preservation": by_scale, "measurements": measurements,
            "metric_scope": "last prompt token only; not sequence KL or task accuracy",
            "layer_count": len(layers)}


def validate_plan(plan, root):
    """Resolve and byte-bind every input; reject visible leakage before model load."""
    required = {"model", "checkpoint_sha256", "revision", "recipe", "rubric", "keep",
                "excluded_workloads", "runs", "repeat_seeds", "scale"}
    if set(plan) != required:
        raise ValueError(f"Plan keys must be exactly {sorted(required)}")
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", plan["revision"]):
        raise ValueError("revision must be the pinned upstream 40-hex commit")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", plan["checkpoint_sha256"]):
        raise ValueError("Provide an independently recorded checkpoint tree digest")
    seeds = plan["repeat_seeds"]
    if (not isinstance(seeds, list) or not 2 <= len(seeds) <= 8 or
            any(type(s) is not int or not 0 <= s < 2**31 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Provide 2..8 distinct repeat seeds")
    if not isinstance(plan["runs"], list) or not 2 <= len(plan["runs"]) <= 8:
        raise ValueError("Provide 2..8 disjoint replication workloads")
    scale = plan["scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < abs(scale) <= 100:
        raise ValueError("Lock a finite nonzero scale before evaluation")
    if not isinstance(plan["excluded_workloads"], list) or not plan["excluded_workloads"]:
        raise ValueError("Declare calibration/discovery/search workloads to exclude")
    paths = {k: (root / plan[k]).resolve() for k in ("model", "recipe", "rubric", "keep")}
    paths.update({f"run_{i}": (root / p).resolve() for i, p in enumerate(plan["runs"])})
    paths.update({f"excluded_{i}": (root / p).resolve() for i, p in enumerate(plan["excluded_workloads"])})
    seen = {" ".join(DEFAULT_CALIBRATION.split()).casefold()}
    for role in [k for k in paths if k.startswith("excluded_")] + ["keep"] + [f"run_{i}" for i in range(len(plan["runs"]))]:
        prompts = load_prompts(str(paths[role]), None)
        normalized = [" ".join(p.split()).casefold() for p in prompts]
        if not normalized or len(normalized) != len(set(normalized)) or seen.intersection(normalized):
            raise ValueError(f"Duplicate/overlapping prompts in {role}")
        if len(prompts) > 128:
            raise ValueError("Qualification workloads are bounded to 128 prompts each")
        seen.update(normalized)
    # Seed offsets cannot overlap across first-seed replication runs.
    if max(seeds) + len(plan["runs"]) * 128 + 128 >= 2**31:
        raise ValueError("Seed offsets exceed supported range")
    starts = [s + i * 128 for i in range(len(plan["runs"])) for s in seeds]
    if len(starts) != len(set(starts)):
        raise ValueError("Repeat seeds collide after run offsets")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True, help="New immutable run directory")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    if not 2 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be 2..256")
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    paths = validate_plan(plan, plan_path.parent)
    out = Path(args.out).resolve()
    if out.exists() or any(p == out or p in out.parents for p in paths.values()):
        raise ValueError("Use a new output directory outside inputs")
    from .neuron_steering import resolve_recipe
    recipe = resolve_recipe(config=str(paths["recipe"]), model_id=MODEL_ID)
    if recipe.get("model") != MODEL_ID:
        raise ValueError("Qualification recipe must explicitly bind the target model ID")
    if recipe["method"]["apply_to"] == "generated_tokens":
        raise ValueError("Qualification requires prompt-eligible steering")
    from .common import trust_remote_code_enabled
    if trust_remote_code_enabled():
        raise ValueError("Qualification requires stock Transformers; disable remote-code opt-in")
    bindings = bind_files({**paths, "qualification_plan": plan_path})
    identity = checkpoint_binding(str(paths["model"]), expected_checkpoint_sha256=plan["checkpoint_sha256"])
    environment = preflight(str(paths["model"]))
    verify_binding(bindings)
    from .operator_audit import implementation_binding
    implementation = implementation_binding()
    out.mkdir(parents=True)
    manifest = {"schema_version": 1, "qualified": False, "plan": plan,
                "inputs": bindings, "implementation": implementation,
                "checkpoint_binding": identity, "environment": environment,
                "status": "preflight_only", "runs": []}
    def save():
        (out / "qualification.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    save()
    if args.preflight_only:
        return manifest
    from .cli import main as operator
    try:
        for run_index in range(len(plan["runs"])):
            workload = str(paths[f"run_{run_index}"])
            for repeat_index, seed in enumerate(plan["repeat_seeds"]):
                folder = out / f"run-{run_index}-repeat-{repeat_index}"
                folder.mkdir()
                ab, labels, evaluation = [str(folder / name) for name in ("ab.json", "labels.jsonl", "evaluation.json")]
                actual_seed = seed + run_index * 128
                operator(["steer", "--model", str(paths["model"]), "--model-id", MODEL_ID,
                          "--config", str(paths["recipe"]), "--workload", workload,
                          "--qualification-keep", str(paths["keep"]),
                          "--expected-checkpoint-sha256", plan["checkpoint_sha256"],
                          "--scales", str(plan["scale"]), "--max-prompts", "128",
                          "--temperature", "0.7", "--seed", str(actual_seed),
                          "--max-new-tokens", str(args.max_new_tokens), "--device", args.device,
                          "--dtype", args.dtype, "--out", ab])
                operator(["steer-label-template", "--ab-report", ab, "--rubric", str(paths["rubric"]), "--out", labels])
                manifest["runs"].append({"ab_report": ab, "labels": labels, "seed": actual_seed,
                    "replication_eligible": repeat_index == 0,
                    "evaluate_argv": ["steer-evaluate", "--ab-report", ab, "--ratings", labels,
                                      "--rubric", str(paths["rubric"]), "--out", evaluation]})
                save()
        replicate = ["steer-replicate", "--locked-scale", str(plan["scale"]), "--out", str(out / "replication.json")]
        for row in manifest["runs"]:
            if row["replication_eligible"]:
                replicate += ["--ab-report", row["ab_report"], "--evaluation", row["evaluate_argv"][-1]]
        manifest["replicate_argv"] = replicate
        verify_binding(bindings)
        if implementation_binding() != implementation:
            raise ValueError("Qualification implementation changed during execution")
        manifest["status"] = "awaiting_independent_labels"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    return manifest


if __name__ == "__main__":
    main()
