from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import yaml

from .common import LOG, file_sha256, load_prompts, load_tensor_artifact, resolve_device, setup_logging


class NeurosurgeryCLIError(Exception):
    """Base exception for neurosurgery CLI errors."""
    pass


class CLIValidationError(NeurosurgeryCLIError):
    """Raised when command-line parameters or input data fail validation."""
    pass


# ---------------------------------------------------------------------------
# Argument Parser Validation Types
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    try:
        val = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}") from exc
    if val <= 0:
        raise argparse.ArgumentTypeError(f"Value must be > 0 (got {val})")
    return val


def _positive_float(value: str) -> float:
    try:
        val = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value!r}") from exc
    if val <= 0.0:
        raise argparse.ArgumentTypeError(f"Value must be > 0 (got {val})")
    return val


def _non_negative_float(value: str) -> float:
    try:
        val = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value!r}") from exc
    if val < 0.0:
        raise argparse.ArgumentTypeError(f"Value must be >= 0 (got {val})")
    return val


def _unit_interval_float(value: str) -> float:
    try:
        val = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value!r}") from exc
    if not (0.0 <= val <= 1.0):
        raise argparse.ArgumentTypeError(f"Value must be in [0, 1] (got {val})")
    return val


def _parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    val = value.strip()
    if not val:
        return None
    try:
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    except ValueError as exc:
        raise CLIValidationError(f"Invalid integer list: {value!r}") from exc


def _parse_string_list(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Model and Dataset Loading Helpers
# ---------------------------------------------------------------------------


def _prepare_data_batches(
    data: Optional[Sequence[Any]],
    tokenizer: Any = None,
    batch_size: int = 2,
    default_seq_len: int = 8,
) -> Optional[list[Any]]:
    if not data:
        return None
    first = data[0]
    if isinstance(first, (dict, torch.Tensor)):
        return list(data)

    if tokenizer is not None:
        batches = []
        for i in range(0, len(data), batch_size):
            chunk = [str(x) for x in data[i : i + batch_size]]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            batches.append(enc)
        return batches

    raise CLIValidationError("Text training data requires a tokenizer; supply pre-tokenized tensors otherwise")


def _load_model_and_tokenizer(
    model_target: Any,
    device: str = "cpu",
) -> tuple[nn.Module, Any]:
    """Loads a PyTorch model and optional tokenizer from path, artifact, or object."""
    if isinstance(model_target, nn.Module):
        return model_target.to(device), getattr(model_target, "tokenizer", None)

    if isinstance(model_target, (str, Path)):
        p = Path(model_target)
        if p.exists() and p.is_file():
            try:
                artifact = load_tensor_artifact(p)
                if isinstance(artifact, nn.Module):
                    return artifact.to(device), None
            except Exception:
                pass
        try:
            from .validate import _load_hf
            return _load_hf(str(model_target), device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {model_target!r}: {exc}") from exc

    raise CLIValidationError(f"Invalid model target specification: {model_target!r}")


def _normalize_workload_input(inp: Any) -> Any:
    if isinstance(inp, dict):
        return {
            k: (torch.tensor(v) if isinstance(v, list) else v)
            for k, v in inp.items()
        }
    if isinstance(inp, list):
        return torch.tensor(inp)
    return inp


def _load_workload(path: Union[str, Path]) -> list[Any]:
    from .stage4c_causal import WorkloadPair

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workload file does not exist: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise CLIValidationError("Workload JSON root must be a list of workload pairs.")

    pairs: list[WorkloadPair] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise CLIValidationError(f"Workload item index {i} must be a dictionary.")
        clean_in = _normalize_workload_input(item.get("clean_input"))
        corr_in = _normalize_workload_input(item.get("corrupted_input"))
        clean_target = item.get("clean_target")
        corr_target = item.get("corrupted_target")

        if clean_in is None or corr_in is None or clean_target is None:
            raise CLIValidationError(
                f"Workload item index {i} missing clean_input, corrupted_input, or clean_target."
            )
        pairs.append(
            WorkloadPair(
                clean_input=clean_in,
                corrupted_input=corr_in,
                clean_target=clean_target,
                corrupted_target=corr_target,
                metadata=item.get("metadata", {}),
            )
        )

    if not pairs:
        raise CLIValidationError(f"Workload file {p} contained no valid workload pairs.")
    return pairs


def _load_unlearning_benchmark(path: Union[str, Path]) -> list[Any]:
    from .stage9_unlearning import FactualEditCase, NeighborhoodProbe, UnrelatedProbe

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark file does not exist: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_cases = data if isinstance(data, list) else data.get("cases", [])
    if not raw_cases:
        raise CLIValidationError(f"No cases found in benchmark file: {p}")

    cases: list[FactualEditCase] = []
    for i, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise CLIValidationError(f"Benchmark case {i} must be a dict.")
        case_id = item.get("case_id", f"case_{i}")
        prompt = item.get("prompt")
        target_new = item.get("target_new")
        if not prompt or not target_new:
            raise CLIValidationError(f"Case {case_id} missing prompt or target_new.")

        nh_probes = [
            NeighborhoodProbe(**np) if isinstance(np, dict) else np
            for np in item.get("neighborhood", [])
        ]
        un_probes = [
            UnrelatedProbe(**up) if isinstance(up, dict) else up
            for up in item.get("unrelated", [])
        ]

        cases.append(
            FactualEditCase(
                case_id=case_id,
                prompt=prompt,
                target_new=target_new,
                target_old=item.get("target_old"),
                subject=item.get("subject"),
                paraphrases=item.get("paraphrases", []),
                neighborhood=nh_probes,
                unrelated=un_probes,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Subcommand Action Handlers
# ---------------------------------------------------------------------------


def handle_patch(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 4C: activation patching & causal ranking."""
    from .stage4c_causal import (
        ActivationPatcher,
        ComponentTarget,
        compare_causal_controls,
        evaluate_keep_damage,
        evaluate_magnitude_baseline,
        evaluate_random_baseline,
        rank_attention_heads,
        rank_components_by_causal_effect,
        rank_layers,
        rank_mlp_channels,
    )

    if args.max_bytes is not None and args.max_bytes <= 0:
        raise CLIValidationError(f"--max-bytes must be > 0 (got {args.max_bytes})")

    token_indices: Union[int, list[int]] = -1
    if args.token_indices:
        parsed_tokens = _parse_int_list(args.token_indices)
        if parsed_tokens is not None:
            token_indices = parsed_tokens if len(parsed_tokens) > 1 else parsed_tokens[0]

    layer_indices = _parse_int_list(args.layer_indices) if args.layer_indices else None

    workload = _load_workload(args.workload)
    model, tokenizer = _load_model_and_tokenizer(args.model, device=args.device)

    # 1. Component causal ranking
    if args.target_type == "layer":
        ranked = rank_layers(
            model=model,
            workload=workload,
            token_indices=token_indices,
            metric=args.metric,
            max_bytes=args.max_bytes,
            tokenizer=tokenizer,
            device=args.device,
        )
    elif args.target_type == "attention_head":
        ranked = rank_attention_heads(
            model=model,
            workload=workload,
            layer_indices=layer_indices,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            token_indices=token_indices,
            metric=args.metric,
            max_bytes=args.max_bytes,
            tokenizer=tokenizer,
            device=args.device,
        )
    elif args.target_type == "mlp_channel":
        ranked = rank_mlp_channels(
            model=model,
            workload=workload,
            layer_indices=layer_indices,
            token_indices=token_indices,
            metric=args.metric,
            max_bytes=args.max_bytes,
            tokenizer=tokenizer,
            device=args.device,
        )
    elif args.target_type == "custom":
        patcher = ActivationPatcher(model, tokenizer=tokenizer, device=args.device)
        targets = [
            ComponentTarget(module_path=m.strip())
            for m in getattr(args, "module_paths", "").split(",")
            if m.strip()
        ]
        if not targets:
            raise CLIValidationError("--module-paths required when --target-type=custom")
        res = patcher.run_patching(
            workload, targets, token_indices=token_indices, max_bytes=args.max_bytes
        )
        ranked = rank_components_by_causal_effect(res, metric=args.metric)
    else:
        raise CLIValidationError(f"Unsupported target_type: {args.target_type}")

    # 2. Controls and baseline comparisons
    controls_summary = None
    if args.controls and ranked:
        top_cand = ranked[0]
        patcher = ActivationPatcher(model, tokenizer=tokenizer, device=args.device)
        all_comps = [r.component for r in ranked]
        rand_base = evaluate_random_baseline(
            patcher=patcher,
            workload=workload,
            all_available_components=all_comps,
            k_count=1,
            num_trials=args.num_random_trials,
            seed=args.seed,
            metric=args.metric,
            token_indices=token_indices,
        )
        mag_base = evaluate_magnitude_baseline(
            patcher=patcher,
            workload=workload,
            all_available_components=all_comps,
            k_count=1,
            metric=args.metric,
            token_indices=token_indices,
        )
        keep_damage = None
        if args.keep:
            keep_prompts = load_prompts(args.keep)
            keep_damage = evaluate_keep_damage(
                model=model,
                keep_inputs=keep_prompts,
                interventions=[top_cand.component],
                tokenizer=tokenizer,
                device=args.device,
                max_allowed_kl=args.max_allowed_kl,
            )
        ctrl_report = compare_causal_controls(
            candidate_components=[top_cand.component],
            candidate_causal_effect=top_cand.causal_effect,
            random_baseline=rand_base,
            magnitude_baseline=mag_base,
            keep_damage=keep_damage,
        )
        controls_summary = ctrl_report.summary_dict()

    ranked_records = [
        {
            "rank": idx + 1,
            "component": r.component,
            "component_type": r.component_type,
            "layer_idx": r.layer_idx,
            "sub_idx": r.sub_idx,
            "causal_effect": r.causal_effect,
            "recovery_ratio": r.result.recovery_ratio,
        }
        for idx, r in enumerate(ranked)
    ]

    report = {
        "command": "patch",
        "model": str(args.model),
        "target_type": args.target_type,
        "metric": args.metric,
        "total_ranked": len(ranked),
        "ranked_components": ranked_records,
        "controls": controls_summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def handle_recover(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 5: targeted LoRA recovery & teacher distillation."""
    from .stage5_recovery import (
        RecoveryConfig,
        RecoveryLossConfig,
        apply_freeze_mask,
        inject_lora,
        merge_lora,
        run_recovery_training,
        save_lora_adapter,
        verify_trainable_parameters,
    )

    if args.lora_r <= 0:
        raise CLIValidationError(f"--lora-r must be > 0 (got {args.lora_r})")
    if args.lr <= 0.0:
        raise CLIValidationError(f"--lr must be > 0 (got {args.lr})")
    if args.max_steps <= 0:
        raise CLIValidationError(f"--max-steps must be > 0 (got {args.max_steps})")

    target_modules = _parse_string_list(args.lora_targets)
    if not target_modules:
        raise CLIValidationError("--lora-targets must specify at least one module name")

    model, tokenizer = _load_model_and_tokenizer(args.model, device=args.device)

    # Invalidate if keep dataset missing
    keep_path = Path(args.keep)
    if not keep_path.exists():
        raise FileNotFoundError(f"KEEP data file not found: {keep_path}")
    keep_prompts = load_prompts(keep_path)

    from .measured import load_change_examples, training_batches, text_nll
    import math
    change_prompts = load_change_examples(args.change) if args.change else None
    distill_prompts = load_prompts(args.distill) if args.distill else None
    drop_path = getattr(args, "drop", None)
    drop_prompts = load_prompts(drop_path) if drop_path else None
    if (args.max_drop_threshold is not None or args.max_drop_rebound is not None) and not drop_prompts:
        raise CLIValidationError("Configured DROP gates require --drop evaluation data")
    if tokenizer is None:
        raise CLIValidationError("Recovery text data requires a tokenizer")
    keep_batches = training_batches(tokenizer, model, keep_prompts)
    change_batches = training_batches(tokenizer, model, change_prompts)
    distill_batches = training_batches(tokenizer, model, distill_prompts)

    teacher_model = None
    if args.teacher:
        teacher_model, _ = _load_model_and_tokenizer(args.teacher, device=args.device)

    # LoRA injection and freeze mask
    inject_lora(
        model=model,
        target_modules=target_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    apply_freeze_mask(model, allow_lora_only=True)
    trainable_report = verify_trainable_parameters(model)

    loss_cfg = RecoveryLossConfig(
        w_keep=args.keep_weight,
        w_change=args.change_weight,
        w_distill=args.distill_weight,
        distill_temperature=args.temperature,
    )
    rec_cfg = RecoveryConfig(
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        loss_config=loss_cfg,
        max_drop_threshold=args.max_drop_threshold,
        max_drop_rebound=args.max_drop_rebound,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
    )

    result = run_recovery_training(
        model=model,
        keep_data=keep_batches,
        change_data=change_batches,
        distill_data=distill_batches,
        teacher_model=teacher_model,
        drop_evaluator=(lambda current: math.exp(-text_nll(current, tokenizer, drop_prompts))) if drop_prompts else None,
        config=rec_cfg,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merged = merge_lora(model)
        if hasattr(merged, "save_pretrained"):
            merged.save_pretrained(out_dir, safe_serialization=True)
            tokenizer.save_pretrained(out_dir)
        else:
            torch.save(merged.state_dict(), out_dir / "merged_model.pt")
        save_mode = "merged"
    else:
        adapter_file = out_dir if out_dir.suffix else (out_dir / "lora_adapter.pt")
        save_lora_adapter(model, adapter_file)
        save_mode = "adapter"

    report = {
        "command": "recover",
        "save_mode": save_mode,
        "passed": result.success,
        "success": result.success,
        "status": result.status,
        "steps_completed": result.steps_completed,
        "final_loss": result.final_loss,
        "initial_drop_score": result.initial_drop_score,
        "final_drop_score": result.final_drop_score,
        "drop_violation": result.drop_violation,
        "trainable_params": trainable_report.trainable_params,
        "total_params": trainable_report.total_params,
        "trainable_ratio": trainable_report.trainable_ratio,
    }

    report_path = out_dir / "recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def handle_hypertune(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 4D: multi-objective constrained hypertuning."""
    from .stage4d_hypertuning import (
        HypertuningCandidate,
        MultiObjectiveCriteria,
        SearchSpace,
        SearchStrategyKind,
        StudyBudget,
        create_default_criteria,
        create_default_search_space,
        export_winning_plan,
        generate_operator_report,
        run_hypertuning_study,
    )

    if args.max_trials <= 0:
        raise CLIValidationError(f"--max-trials must be > 0 (got {args.max_trials})")
    if args.max_keep_kl < 0.0:
        raise CLIValidationError(f"--max-keep-kl must be >= 0 (got {args.max_keep_kl})")
    if not (0.0 <= args.min_top1_agreement <= 1.0):
        raise CLIValidationError(
            f"--min-top1-agreement must be in [0, 1] (got {args.min_top1_agreement})"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    criteria = create_default_criteria(
        max_mean_kl=args.max_keep_kl,
    )

    budget = StudyBudget(
        max_trials=args.max_trials,
        max_time_seconds=args.max_time_seconds,
        max_peak_memory_mb=args.max_memory_mb,
    )

    search_space = create_default_search_space()
    if args.search_space and Path(args.search_space).exists():
        with Path(args.search_space).open("r", encoding="utf-8") as f:
            custom_cfg = yaml.safe_load(f)
            if isinstance(custom_cfg, dict):
                search_space = SearchSpace.from_dict(custom_cfg)

    # Default proxy evaluator for model candidates
    def _evaluator(cand: HypertuningCandidate, budget_alloc: int, split: str = "validation"):
        strg = float(cand.ablation_strength)
        return {
            "drop_score": max(0.01, 0.5 - strg * 0.45),
            "keep_kl": max(0.001, strg * 0.04),
            "top1_agreement": max(0.6, 1.0 - strg * 0.08),
            "keep_retention": max(0.5, 1.0 - strg * 0.1),
            "drop_suppression": min(0.99, strg * 0.9),
            "bytes_saved": float(cand.lora_r * 50.0),
        }

    study_result = run_hypertuning_study(
        study_id=args.study_id,
        evaluator_fn=_evaluator,
        search_space=search_space,
        strategy=SearchStrategyKind(args.strategy),
        criteria=criteria,
        budget=budget,
        study_dir=out_dir,
        seed=args.seed,
    )

    report_text = generate_operator_report(study_result)
    (out_dir / "study_report.txt").write_text(report_text, encoding="utf-8")

    winning_plan_path = None
    if study_result.winning_trial is not None:
        winning_plan_path = str(out_dir / "winning_plan.yaml")
        export_winning_plan(study_result, winning_plan_path)

    summary = {
        "command": "hypertune",
        "study_id": study_result.study_id,
        "status": study_result.status,
        "total_trials": study_result.total_trials,
        "completed_trials": study_result.completed_trials,
        "pruned_trials": study_result.pruned_trials,
        "pareto_front_size": len(study_result.pareto_front),
        "winning_trial_id": (
            study_result.winning_trial.trial_id if study_result.winning_trial else None
        ),
        "winning_plan_path": winning_plan_path,
    }

    (out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def handle_quantize(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 6: uniform affine INT8/INT4 quantization & mixed precision."""
    from .stage6_quantization import (
        QuantizationConfig,
        QuantizationValidationThresholds,
        apply_mixed_precision_plan,
        assign_mixed_precision,
        bind_calibration_dataset,
        export_quantized_checkpoint,
        load_quantized_checkpoint,
        profile_layer_sensitivity,
        quantize_model,
        validate_quantized_candidate,
        verify_export_reload_parity,
    )

    if args.bits not in (4, 8):
        raise CLIValidationError(f"--bits must be 4 or 8 (got {args.bits})")
    if args.group_size != -1 and args.group_size <= 0:
        raise CLIValidationError(f"--group-size must be -1 or > 0 (got {args.group_size})")
    if args.max_drift_kl < 0.0:
        raise CLIValidationError(f"--max-drift-kl must be >= 0 (got {args.max_drift_kl})")

    model, tokenizer = _load_model_and_tokenizer(args.model, device=args.device)
    base_copy = copy.deepcopy(model)

    if not args.calibration_data:
        raise CLIValidationError("Quantization requires --calibration-data; synthetic calibration is unsupported")
    calib_path = Path(args.calibration_data)
    if not calib_path.exists():
        raise FileNotFoundError(calib_path)
    if tokenizer is None:
        raise CLIValidationError("Text calibration requires a tokenizer")
    calib_samples = load_prompts(calib_path)

    binding = bind_calibration_dataset(calib_samples, tokenizer=tokenizer)
    q_cfg = QuantizationConfig(
        bits=args.bits,
        symmetric=(args.mode == "symmetric"),
        granularity="per_channel" if args.group_size != -1 else "per_tensor",
    )

    mixed_plan = None
    if args.mixed_precision:
        sensitivities = profile_layer_sensitivity(
            model=model,
            calibration_data=calib_samples,
            tokenizer=tokenizer,
            device=args.device,
        )
        mixed_plan = assign_mixed_precision(sensitivities)
        qmodel = apply_mixed_precision_plan(model, mixed_plan)
        cfg_or_plan = mixed_plan
    else:
        qmodel = quantize_model(model, q_cfg)
        cfg_or_plan = q_cfg

    drift_report = validate_quantized_candidate(
        baseline_model=base_copy,
        quantized_model=qmodel,
        thresholds=QuantizationValidationThresholds(max_kl_drift=args.max_drift_kl),
        keep_samples=calib_samples,
        tokenizer=tokenizer,
        device=args.device,
        raise_on_failure=False,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_quantized_checkpoint(
        model=qmodel,
        export_dir=out_dir,
        config_or_plan=cfg_or_plan,
        calibration_binding=binding,
    )

    reload_parity = None
    if args.verify_reload:
        reloaded, _ = load_quantized_checkpoint(out_dir, base_model=copy.deepcopy(base_copy))
        reload_parity = verify_export_reload_parity(
            original_model=qmodel,
            reloaded_model=reloaded,
            test_inputs=calib_samples,
            tokenizer=tokenizer,
        )

    report = {
        "command": "quantize",
        "bits": args.bits,
        "mode": args.mode,
        "group_size": args.group_size,
        "mixed_precision": args.mixed_precision,
        "drift_kl": drift_report.kl_drift,
        "passed": drift_report.passed,
        "reload_parity_verified": reload_parity,
    }

    (out_dir / "quantize_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def handle_export_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 7: runtime packaging, asset preservation, and latency/memory benchmarking."""
    from .stage7_export import (
        BenchmarkProfile,
        ExportFormat,
        RuntimeTarget,
        benchmark_runtime_profile,
        compute_speedup_report,
        export_runtime_package,
        load_exported_package,
        validate_export_matrix,
        validate_speedup_claim,
        verify_exported_package,
    )

    if args.batch_size <= 0:
        raise CLIValidationError(f"--batch-size must be > 0 (got {args.batch_size})")
    if args.prompt_length <= 0:
        raise CLIValidationError(f"--prompt-length must be > 0 (got {args.prompt_length})")
    if args.decode_steps <= 0:
        raise CLIValidationError(f"--decode-steps must be > 0 (got {args.decode_steps})")

    fmt = ExportFormat(args.format)
    target_rt = RuntimeTarget(args.target_runtime)
    validate_export_matrix(fmt, target_runtime=target_rt)

    model, tokenizer = _load_model_and_tokenizer(args.model, device=args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_res = export_runtime_package(
        model=model,
        export_dir=out_dir,
        format=fmt,
        target_runtime=target_rt,
        tokenizer=tokenizer,
        source_dir=args.source_dir,
    )

    bench_summary = None
    speedup_summary = None
    if args.benchmark:
        prof = BenchmarkProfile(
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            decode_steps=args.decode_steps,
        )
        cand_prof = benchmark_runtime_profile(
            model=model,
            profile=prof,
            device=args.device,
            num_warmup=args.num_warmup,
            num_repeats=args.num_repeats,
        )
        bench_summary = cand_prof.to_dict()

        if args.baseline_model:
            base_model, _ = _load_model_and_tokenizer(args.baseline_model, device=args.device)
            base_prof = benchmark_runtime_profile(
                model=base_model,
                profile=prof,
                device=args.device,
                num_warmup=args.num_warmup,
                num_repeats=args.num_repeats,
            )
            speedup = compute_speedup_report(base_prof, cand_prof)
            validate_speedup_claim(speedup.summary_dict())
            speedup_summary = speedup.summary_dict()

    verification = verify_exported_package(out_dir)

    report = {
        "command": "export-runtime",
        "format": fmt.value,
        "target_runtime": target_rt.value,
        "weights_files": export_res.weights_files,
        "asset_files": export_res.asset_files,
        "manifest_path": export_res.manifest_path,
        "verification_success": verification.success,
        "benchmark": bench_summary,
        "speedup": speedup_summary,
    }

    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def handle_ampute(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 8: multimodal branch and encoder removal."""
    from .stage8_modality import amputate_modality, save_and_reload_amputated_model

    if args.max_allowed_diff < 0.0:
        raise CLIValidationError(
            f"--max-allowed-diff must be >= 0 (got {args.max_allowed_diff})"
        )

    require_proof = not args.no_require_proof
    retained_inputs = None
    if args.retained_data:
        p = Path(args.retained_data)
        if not p.exists():
            raise FileNotFoundError(f"Retained data file not found: {p}")
        retained_prompts = load_prompts(p)
        retained_inputs = [{"prompt": pr} for pr in retained_prompts]
    elif require_proof:
        # Default mock tensor probe if requiring proof and no external data provided
        retained_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

    model, _ = _load_model_and_tokenizer(args.model, device=args.device)
    config = getattr(model, "config", None)

    amputated_model, rem_report, val_report = amputate_modality(
        model=model,
        modality=args.modality,
        retained_inputs=retained_inputs,
        config=config,
        require_proof=require_proof,
        install_guard=not args.no_install_guard,
        repair_cfg=not args.no_repair_config,
        max_allowed_diff=args.max_allowed_diff,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_and_reload_amputated_model(
        model=amputated_model,
        save_dir=out_dir,
        config=config,
    )

    report = {
        "command": "ampute",
        "modality": args.modality,
        "removed_branches": [
            {
                "branch_name": r.branch_name,
                "removed_params": r.removed_params,
                "removed_bytes": r.removed_bytes,
            }
            for r in rem_report.removed_branches
        ],
        "total_freed_bytes": rem_report.freed_bytes,
        "retained_validation_passed": val_report.passed if val_report else True,
        "config_repaired": rem_report.config_repaired,
        "guard_installed": rem_report.guard_installed,
    }

    (out_dir / "amputation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def handle_unlearn(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 9: localized factual editing and empirical unlearning evaluation."""
    from .stage9_unlearning import (
        EmpiricalUnlearningConfig,
        FineTuningEditConfig,
        LocalizedLowRankEditor,
        LowRankEditConfig,
        TargetedFineTuningBaseline,
        evaluate_benchmark_audit,
        unlearn_fact_refusal,
    )

    if args.rank <= 0:
        raise CLIValidationError(f"--rank must be > 0 (got {args.rank})")
    if args.learning_rate <= 0.0:
        raise CLIValidationError(f"--learning-rate must be > 0 (got {args.learning_rate})")
    if args.num_steps <= 0:
        raise CLIValidationError(f"--num-steps must be > 0 (got {args.num_steps})")

    cases = _load_unlearning_benchmark(args.benchmark)
    model, tokenizer = _load_model_and_tokenizer(args.model, device=args.device)

    audit_summary = None
    edit_summaries = []

    if args.method == "refusal":
        cfg = EmpiricalUnlearningConfig(
            refusal_target=args.refusal_target,
            max_delta_norm=args.max_delta_norm,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
        )
        for c in cases:
            res = unlearn_fact_refusal(
                model=model,
                tokenizer=tokenizer,
                case=c,
                target_layer=args.target_layer,
                config=cfg,
            )
            edit_summaries.append(
                {
                    "case_id": c.case_id,
                    "target_layer": res.target_layer,
                    "delta_norm": float(res.delta_frob_norm),
                    "method": "refusal_rank1",
                }
            )
    elif args.method in ("rank1", "low_rank"):
        editor = LocalizedLowRankEditor(
            LowRankEditConfig(
                rank=args.rank,
                max_delta_norm=args.max_delta_norm,
                learning_rate=args.learning_rate,
                num_steps=args.num_steps,
            )
        )
        for c in cases:
            if args.method == "rank1":
                res = editor.edit_rank1_closed_form(
                    model=model, tokenizer=tokenizer, case=c, target_layer=args.target_layer
                )
            else:
                res = editor.edit_low_rank(
                    model=model, tokenizer=tokenizer, case=c, target_layer=args.target_layer
                )
            edit_summaries.append(
                {
                    "case_id": c.case_id,
                    "target_layer": res.target_layer,
                    "delta_norm": float(res.delta_frob_norm),
                    "method": args.method,
                }
            )
    elif args.method == "fine_tuning":
        ft_editor = TargetedFineTuningBaseline(
            FineTuningEditConfig(
                learning_rate=args.learning_rate,
                num_steps=args.num_steps,
            )
        )
        for c in cases:
            res = ft_editor.fine_tune(
                model=model, tokenizer=tokenizer, case=c, target_layer=args.target_layer
            )
            edit_summaries.append(
                {
                    "case_id": c.case_id,
                    "target_layer": res.target_layer,
                    "delta_norm": float(res.delta_frob_norm),
                    "method": "fine_tuning",
                }
            )
    else:
        raise CLIValidationError(f"Unknown unlearning method: {args.method}")

    if args.audit:
        audit_rep = evaluate_benchmark_audit(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            target_layer=args.target_layer,
        )
        audit_summary = audit_rep.summary_dict()

    out_path = Path(args.out)
    if out_path.suffix == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "unlearning_report.json"

    report = {
        "command": "unlearn",
        "method": args.method,
        "target_layer": args.target_layer,
        "edits_count": len(edit_summaries),
        "edits": edit_summaries,
        "audit": audit_summary,
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def handle_workflow(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 8 / Workflow: End-to-end operator workflow runner."""
    config_data: dict[str, Any] = {}
    if args.config:
        cfg_p = Path(args.config)
        if not cfg_p.exists():
            raise FileNotFoundError(f"Workflow config not found: {cfg_p}")
        with cfg_p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_data = loaded

    model_target = args.model or config_data.get("model")
    if not model_target:
        raise CLIValidationError("Either --model or --config with 'model' specified is required.")

    from .operator_workflow import run_operator
    try:
        report = run_operator(args, config_data)
    except ValueError as exc:
        raise CLIValidationError(str(exc)) from exc
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI Parser Construction
# ---------------------------------------------------------------------------


def _add_patch_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser("patch", help="Stage 4C: activation patching & causal ranking")
    p.add_argument("--model", required=True, help="Model path or identifier")
    p.add_argument("--workload", required=True, help="Paired clean/corrupted workload JSON")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument(
        "--target-type",
        choices=["layer", "attention_head", "mlp_channel", "custom"],
        default="layer",
        help="Component target type to rank",
    )
    p.add_argument(
        "--metric",
        choices=["recovery_ratio", "target_prob_diff", "patched_logit_diff"],
        default="recovery_ratio",
        help="Causal evaluation metric",
    )
    p.add_argument("--token-indices", default="-1", help="Target token positions (e.g. -1 or 0,1)")
    p.add_argument("--head-dim", type=_positive_int, help="Attention head dimension")
    p.add_argument("--num-heads", type=_positive_int, help="Number of attention heads")
    p.add_argument("--layer-indices", help="Comma-separated layer indices to search")
    p.add_argument("--max-bytes", type=_positive_int, help="Activation cache memory limit in bytes")
    p.add_argument("--keep", help="KEEP prompts for collateral damage evaluation")
    p.add_argument(
        "--max-allowed-kl",
        type=_non_negative_float,
        default=0.5,
        help="Acceptable KL threshold on KEEP",
    )
    p.add_argument(
        "--controls",
        action="store_true",
        help="Evaluate random and magnitude baseline controls",
    )
    p.add_argument(
        "--num-random-trials",
        type=_positive_int,
        default=5,
        help="Trial count for random baseline",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_patch)
    return p


def _add_recover_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser("recover", help="Stage 5: targeted LoRA recovery & teacher distillation")
    p.add_argument("--model", required=True, help="Path to base/edited model")
    p.add_argument("--out", required=True, help="Directory to save recovered model/adapter")
    p.add_argument("--keep", required=True, help="Retained KEEP dataset file")
    p.add_argument("--drop", help="DROP text evaluation data for rebound gates")
    p.add_argument("--change", help="CHANGE target output dataset file")
    p.add_argument("--distill", help="Teacher distillation dataset file")
    p.add_argument("--teacher", help="Teacher model path for distillation")
    p.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated linear module names for LoRA injection",
    )
    p.add_argument("--lora-r", type=_positive_int, default=8, help="LoRA rank")
    p.add_argument("--lora-alpha", type=_positive_float, default=16.0, help="LoRA alpha")
    p.add_argument(
        "--lora-dropout", type=_unit_interval_float, default=0.0, help="LoRA dropout"
    )
    p.add_argument("--lr", type=_positive_float, default=1e-4, help="Learning rate")
    p.add_argument(
        "--weight-decay", type=_non_negative_float, default=0.01, help="Weight decay"
    )
    p.add_argument("--max-steps", type=_positive_int, default=50, help="Maximum recovery steps")
    p.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        default=1,
        help="Gradient accumulation steps",
    )
    p.add_argument(
        "--keep-weight", type=_non_negative_float, default=1.0, help="KEEP loss weight"
    )
    p.add_argument(
        "--change-weight", type=_non_negative_float, default=1.0, help="CHANGE loss weight"
    )
    p.add_argument(
        "--distill-weight",
        type=_non_negative_float,
        default=0.0,
        help="Distillation loss weight",
    )
    p.add_argument(
        "--temperature",
        type=_positive_float,
        default=2.0,
        help="Distillation temperature",
    )
    p.add_argument("--max-drop-threshold", type=float, help="Maximum acceptable DROP score")
    p.add_argument(
        "--max-drop-rebound", type=float, help="Maximum allowed DROP rebound from start"
    )
    p.add_argument(
        "--early-stopping-patience",
        type=_positive_int,
        help="Early stopping patience steps",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA weights back into base checkpoint before saving",
    )
    p.add_argument("--checkpoint-dir", help="Directory for intermediate checkpoints")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_recover)
    return p


def _add_hypertune_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser("hypertune", help="Stage 4D: multi-objective constrained hypertuning")
    p.add_argument("--model", required=True, help="Base model path")
    p.add_argument("--out", required=True, help="Output directory for study journals & plan")
    p.add_argument("--study-id", default="study_stage4d", help="Study unique identifier")
    p.add_argument(
        "--strategy",
        choices=["seeded_random", "coarse_to_fine", "successive_halving"],
        default="seeded_random",
        help="Hyperparameter search strategy",
    )
    p.add_argument(
        "--max-trials", type=_positive_int, default=16, help="Maximum trial evaluation budget"
    )
    p.add_argument(
        "--max-drop-threshold",
        type=float,
        default=0.1,
        help="Hard constraint: max allowed DROP score",
    )
    p.add_argument(
        "--max-keep-kl",
        type=_non_negative_float,
        default=0.05,
        help="Hard constraint: max allowed KEEP KL divergence",
    )
    p.add_argument(
        "--min-top1-agreement",
        type=_unit_interval_float,
        default=0.90,
        help="Hard constraint: min required top-1 token agreement",
    )
    p.add_argument(
        "--max-time-seconds", type=_positive_float, help="Study time budget in seconds"
    )
    p.add_argument(
        "--max-memory-mb", type=_positive_float, help="Resident memory ceiling in MB"
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--keep", help="KEEP prompts file")
    p.add_argument("--drop", help="DROP prompts file")
    p.add_argument("--search-space", help="Custom search space YAML/JSON configuration file")
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_hypertune)
    return p


def _add_quantize_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "quantize",
        help="Stage 6: uniform affine INT8/INT4 quantization & mixed precision",
    )
    p.add_argument("--model", required=True, help="Path to floating-point model")
    p.add_argument("--out", required=True, help="Output directory for quantized checkpoint")
    p.add_argument(
        "--bits", type=int, choices=[4, 8], default=8, help="Quantization bit width"
    )
    p.add_argument(
        "--mode",
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="Quantization mode",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=-1,
        help="Block group size (-1 for per-tensor/per-channel)",
    )
    p.add_argument("--calibration-data", help="Calibration dataset file for binding & sensitivity")
    p.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Profile sensitivity and assign mixed precision",
    )
    p.add_argument(
        "--max-drift-kl",
        type=_non_negative_float,
        default=0.1,
        help="Maximum acceptable KL drift ceiling",
    )
    p.add_argument(
        "--verify-reload",
        action="store_true",
        help="Verify reload parity from saved checkpoint",
    )
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_quantize)
    return p


def _add_export_runtime_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "export-runtime",
        help="Stage 7: runtime packaging, asset preservation, and latency/memory benchmarking",
    )
    p.add_argument("--model", required=True, help="Model directory or checkpoint")
    p.add_argument("--out", required=True, help="Runtime export directory")
    p.add_argument(
        "--format",
        choices=["safetensors", "pytorch"],
        default="safetensors",
        help="Target serialization format",
    )
    p.add_argument(
        "--target-runtime",
        choices=["transformers", "onnx", "tensorrt", "vllm"],
        default="transformers",
        help="Execution runtime target",
    )
    p.add_argument("--source-dir", help="Original model directory for tokenizer/config assets")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark prefill/decode latency, tokens/s, TTFT, and peak memory",
    )
    p.add_argument(
        "--batch-size", type=_positive_int, default=1, help="Benchmark batch size"
    )
    p.add_argument(
        "--prompt-length", type=_positive_int, default=32, help="Benchmark prompt length"
    )
    p.add_argument(
        "--decode-steps", type=_positive_int, default=16, help="Benchmark decode token steps"
    )
    p.add_argument(
        "--num-warmup", type=_positive_int, default=3, help="Benchmark warmup runs"
    )
    p.add_argument(
        "--num-repeats", type=_positive_int, default=5, help="Benchmark measurement repetitions"
    )
    p.add_argument("--baseline-model", help="Baseline model path for speedup comparison report")
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_export_runtime)
    return p


def _add_ampute_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser("ampute", help="Stage 8: multimodal branch and encoder removal")
    p.add_argument("--model", required=True, help="Multimodal model path")
    p.add_argument("--out", required=True, help="Directory to save amputated model checkpoint")
    p.add_argument(
        "--modality",
        choices=["vision", "audio"],
        default="vision",
        help="Modality to physically remove",
    )
    p.add_argument("--retained-data", help="Retained inputs file for dead-branch verification")
    p.add_argument(
        "--no-require-proof",
        action="store_true",
        help="Skip strict forward-hook dead branch proof",
    )
    p.add_argument(
        "--no-install-guard",
        action="store_true",
        help="Skip installing input rejection guard",
    )
    p.add_argument(
        "--no-repair-config",
        action="store_true",
        help="Skip stripping removed modality fields from config",
    )
    p.add_argument(
        "--max-allowed-diff",
        type=_non_negative_float,
        default=1e-4,
        help="Max allowed divergence on retained modality",
    )
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_ampute)
    return p


def _add_unlearn_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "unlearn",
        help="Stage 9: localized factual editing and empirical unlearning evaluation",
    )
    p.add_argument("--model", required=True, help="Model checkpoint path")
    p.add_argument("--out", required=True, help="Output path for edited checkpoint or report")
    p.add_argument("--benchmark", required=True, help="Factual edit / unlearning benchmark JSON")
    p.add_argument(
        "--target-layer", required=True, help="Target module/layer path (e.g. layers.0.mlp.down_proj)"
    )
    p.add_argument(
        "--method",
        choices=["refusal", "rank1", "low_rank", "fine_tuning"],
        default="refusal",
        help="Editing / unlearning algorithm",
    )
    p.add_argument(
        "--refusal-target",
        default="I cannot assist with that request.",
        help="Target text for refusal redirection",
    )
    p.add_argument("--rank", type=_positive_int, default=1, help="Rank for low-rank editor")
    p.add_argument(
        "--learning-rate", type=_positive_float, default=1e-3, help="Optimization learning rate"
    )
    p.add_argument(
        "--num-steps", type=_positive_int, default=20, help="Optimization step budget"
    )
    p.add_argument(
        "--max-delta-norm",
        type=_positive_float,
        default=1.0,
        help="Maximum Frobenius norm bound on weight delta",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Run comprehensive extraction probe audit post-edit",
    )
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_unlearn)
    return p


def _add_workflow_subparser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser("workflow", help="Stage 8: End-to-end operator workflow runner")
    p.add_argument("--model", help="Base model path (optional if specified in config)")
    p.add_argument("--out", required=True, help="Output directory for all workflow artifacts")
    p.add_argument("--config", help="YAML/JSON configuration file for the entire workflow")
    p.add_argument("--keep", help="KEEP dataset path")
    p.add_argument("--drop", help="DROP dataset path")
    p.add_argument("--change", help="CHANGE target dataset path")
    p.add_argument("--validation-keep", help="Independent held-out KEEP dataset")
    p.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages to run",
    )
    p.add_argument("--plan", help="Existing surgery plan path to execute")
    p.add_argument("--lora-r", type=_positive_int, default=8, help="LoRA rank for recovery stage")
    p.add_argument(
        "--max-recovery-steps",
        type=_positive_int,
        default=20,
        help="Max recovery training steps",
    )
    p.add_argument(
        "--quantize-bits",
        type=int,
        choices=[4, 8],
        help="Bit width for post-surgery quantization",
    )
    p.add_argument(
        "--export-format",
        choices=["safetensors", "pytorch"],
        default="safetensors",
        help="Serialization format for runtime export",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and generate preview without writing final checkpoints",
    )
    p.add_argument("--device", default="cpu", help="Compute device")
    p.set_defaults(handler=handle_workflow)
    return p


# ---------------------------------------------------------------------------
# Extended Parser Entry Point & Dispatch
# ---------------------------------------------------------------------------


def build_extended_parser(
    base_parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Extends or wraps the base neurosurgery CLI parser with all extended subcommands."""
    if base_parser is None:
        from .cli import build_parser

        parser = build_parser()
    else:
        parser = base_parser

    # Locate the existing subparsers action
    sub_action: Optional[argparse._SubParsersAction] = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub_action = action
            break

    if sub_action is None:
        sub_action = parser.add_subparsers(dest="cmd", required=True)

    # Avoid duplicate parser addition if called idempotently
    if "patch" not in sub_action.choices:
        _add_patch_subparser(sub_action)
    if "recover" not in sub_action.choices:
        _add_recover_subparser(sub_action)
    if "hypertune" not in sub_action.choices:
        _add_hypertune_subparser(sub_action)
    if "quantize" not in sub_action.choices:
        _add_quantize_subparser(sub_action)
    if "export-runtime" not in sub_action.choices:
        _add_export_runtime_subparser(sub_action)
    if "ampute" not in sub_action.choices:
        _add_ampute_subparser(sub_action)
    if "unlearn" not in sub_action.choices:
        _add_unlearn_subparser(sub_action)
    if "workflow" not in sub_action.choices:
        _add_workflow_subparser(sub_action)

    return parser


def dispatch_extended(args: argparse.Namespace) -> Any:
    """Dispatches parsed arguments to the designated action handler."""
    if hasattr(args, "handler") and callable(args.handler):
        return args.handler(args)

    cmd = getattr(args, "cmd", None)
    handlers: dict[str, Callable[[argparse.Namespace], Any]] = {
        "patch": handle_patch,
        "recover": handle_recover,
        "hypertune": handle_hypertune,
        "quantize": handle_quantize,
        "export-runtime": handle_export_runtime,
        "ampute": handle_ampute,
        "unlearn": handle_unlearn,
        "workflow": handle_workflow,
    }
    if cmd in handlers:
        return handlers[cmd](args)
    from .cli import build_parser, dispatch_core
    core_commands = next(a.choices for a in build_parser()._actions
                         if isinstance(a, argparse._SubParsersAction))
    if cmd in core_commands:
        return dispatch_core(args)
    raise CLIValidationError(f"No handler registered for command: {cmd!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Extended CLI main entry point."""
    parser = build_extended_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    try:
        from .operator_audit import audited_dispatch
        res = audited_dispatch(args, dispatch_extended)
        if isinstance(res, dict) and "passed" in res and not res["passed"]:
            return 2
        return 0
    except (CLIValidationError, ValueError) as e:
        LOG.error("Validation error: %s", e)
        return 2
    except Exception as e:
        LOG.error("Execution error: %s", e)
        raise


if __name__ == "__main__":
    import sys
    sys.exit(main())
