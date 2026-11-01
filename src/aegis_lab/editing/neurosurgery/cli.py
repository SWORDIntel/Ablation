from __future__ import annotations

import argparse
import json
import sys

from .common import setup_logging


def _ratios(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    if any(not 0 < x <= 1 for x in out):
        raise argparse.ArgumentTypeError("ratios must be in (0,1]")
    return out


def _add_profile_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--keep", required=True)
    parser.add_argument("--drop", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--device", default="auto")


def _add_search_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--keep", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ratios", type=_ratios, default=[0.95, 0.9, 0.85, 0.8, 0.75])
    parser.add_argument("--contrast-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--device", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis-neurosurgery",
        description="AEGIS-LAB transformer model neurosurgery research CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="Profile KEEP vs DROP residual behavior")
    p.add_argument("--model", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--drop", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--basis-rank", type=int, default=32)
    p.add_argument("--basis-samples", type=int, default=64)
    p.add_argument("--max-prompts", type=int)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("profile-mlp", help="Profile gated-MLP channel importance")
    _add_profile_common(p)

    p = sub.add_parser("search-mlp", help="Search MLP channel-pruning candidates")
    _add_search_common(p)
    p.add_argument("--align-to", type=int, default=64)

    p = sub.add_parser("profile-attention", help="Profile GQA/MHA attention-group importance")
    _add_profile_common(p)

    p = sub.add_parser("search-attention", help="Search attention-group pruning candidates")
    _add_search_common(p)

    p = sub.add_parser("profile-moe", help="Profile MoE router probability and top-k usage")
    _add_profile_common(p)

    p = sub.add_parser("search-moe", help="Search physical expert-pruning candidates")
    _add_search_common(p)

    p = sub.add_parser("optimize", help="Stage-4 joint constrained structural search")
    p.add_argument("--model", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mlp-profile")
    p.add_argument("--attention-profile")
    p.add_argument("--moe-profile")
    p.add_argument("--layer-search")
    p.add_argument("--mlp-ratios", type=_ratios, default=[1.0, 0.95, 0.9, 0.85, 0.8])
    p.add_argument("--attention-ratios", type=_ratios, default=[1.0, 0.875, 0.75, 0.625])
    p.add_argument("--moe-ratios", type=_ratios, default=[1.0, 0.875, 0.75, 0.625])
    p.add_argument("--max-drop-layers", type=int, default=3)
    p.add_argument("--max-mean-kl", type=float, default=0.02)
    p.add_argument("--min-top1-agreement", type=float, default=0.95)
    p.add_argument("--max-trials", type=int, default=64)
    p.add_argument("--strategy", choices=["frontier", "exhaustive"], default="frontier")
    p.add_argument("--contrast-weight", type=float, default=0.25)
    p.add_argument("--align-to", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("search-layers", help="Evaluate every single transformer-layer deletion")
    p.add_argument("--model", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("search-layers-greedy", help="Greedy interacting multi-layer deletion search")
    p.add_argument("--model", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-mean-kl", type=float, default=0.02)
    p.add_argument("--max-layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("steer-label-template", help="Create response-hash-bound blank ratings for offline independent review")
    p.add_argument("--ab-report", required=True, help="Completed, audited steer A/B JSON report")
    p.add_argument("--rubric", required=True, help="Local documented 0..1 target/KEEP scoring rubric")
    p.add_argument("--out", required=True, help="New blank JSONL ratings template")

    p = sub.add_parser("steer-evaluate", help="Aggregate independent target/KEEP ratings for an A/B steering report")
    p.add_argument("--ab-report", required=True, help="Completed, audited steer A/B JSON report")
    p.add_argument("--ratings", required=True, help="Completed and response-hash-matched JSONL/JSON ratings")
    p.add_argument("--rubric", required=True, help="Exact rubric used by annotators")
    p.add_argument("--min-prompts", type=int, default=10)
    p.add_argument("--min-target-gain", type=float, default=0.0)
    p.add_argument("--max-keep-loss", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="New JSON paired-score report")

    p = sub.add_parser(
        "steer-causal",
        help="Experimentally test ranked neurons with signed doses, matched controls and held-out logit objectives",
    )
    p.add_argument("--model", required=True, help="Local, audited Hugging Face checkpoint")
    p.add_argument("--ranking", required=True, help="steer-rank candidate JSON with discovery hashes")
    p.add_argument("--profile", required=True, help="Original profile-mlp .pt matching the ranking")
    p.add_argument("--discovery-keep", required=True, help="Original profile-mlp KEEP corpus")
    p.add_argument("--discovery-drop", required=True, help="Original profile-mlp DROP corpus")
    p.add_argument("--search-target", required=True, help="JSONL/JSON prompt + single target/contrast token labels")
    p.add_argument("--search-keep", required=True, help="Independent KEEP prompts for search gates")
    p.add_argument("--test-target", required=True, help="Independent held-out target/contrast JSONL")
    p.add_argument("--test-keep", required=True, help="Independent held-out KEEP prompts")
    p.add_argument("--scales", default="-0.5,-0.25,0.25,0.5")
    p.add_argument("--max-candidates", type=int, default=2)
    p.add_argument("--controls-per-candidate", type=int, default=1)
    p.add_argument("--max-pairs", type=int, default=0, help="Evaluate at most N signed candidate pairs (0 disables interaction search)")
    p.add_argument("--max-search-prompts", type=int, default=8)
    p.add_argument("--max-test-prompts", type=int, default=8)
    p.add_argument("--max-forwards", type=int, default=512)
    p.add_argument("--max-mean-kl", type=float, default=0.05)
    p.add_argument("--min-top1-agreement", type=float, default=0.95)
    p.add_argument("--normalization", choices=["raw", "layer_residual_median_over_down_column_l2"],
                   default="layer_residual_median_over_down_column_l2")
    p.add_argument("--calibration-data", help="Local text used for normalized steering units")
    p.add_argument("--expected-checkpoint-sha256", help="Optional independently trusted checkpoint-tree digest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--out", required=True, help="New, immutable JSON report path")

    p = sub.add_parser("steer-rank", help="Screen candidate neurons using measured KEEP/DROP MLP activation profiles")
    p.add_argument("--profile", required=True, help="Local profile-mlp mlp_profile.pt input")
    p.add_argument("--top-k", type=int, default=25, help="Maximum candidates per polarity")
    p.add_argument("--out", required=True, help="New .json candidate ranking; never overwritten")

    p = sub.add_parser("steer-presets", help="List published, model-specific neuron-steering recipes")
    p.add_argument("--model-id", help="Filter by original Hugging Face model identifier")

    p = sub.add_parser("steer", help="Reversible neuron steering with paired A/B generation")
    p.add_argument("--model", required=True, help="Local Hugging Face checkpoint directory")
    p.add_argument("--model-id", help="Original model ID; mandatory for published presets")
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--preset", help="One published, model-specific preset")
    selector.add_argument("--config", help="Local JSON recipe (including custom methods)")
    selector.add_argument("--neuron", action="append", help="Repeat L14N8949+1 or 14:8949:+1")
    p.add_argument("--workload", required=True, help="Local .txt, .jsonl or .json prompt set")
    p.add_argument("--calibration-data", help="Optional local calibration text for normalized doses")
    p.add_argument("--expected-checkpoint-sha256", help="Trusted canonical checkpoint-tree SHA256; verify before model load")
    p.add_argument("--scales", default="1.0", help="Nonzero comma-separated multiplier sweep; zero baseline is automatic")
    p.add_argument("--target-regex", help="Optional measured-text regex; not a semantic evaluator")
    p.add_argument("--max-prompts", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--out", required=True, help="New JSON report path (not overwritten)")

    p = sub.add_parser("inventory", help="Read-only inventory of large and modality-specific branches")
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("plan", help="Build a reviewed surgery plan from measured artifacts")
    p.add_argument("--profile", required=True, help="residual profile.json")
    p.add_argument("--search", help="single-layer layer_search.json")
    p.add_argument("--greedy-search", help="greedy_layer_search.json")
    p.add_argument("--mlp-search")
    p.add_argument("--mlp-profile")
    p.add_argument("--attention-search")
    p.add_argument("--attention-profile")
    p.add_argument("--moe-search")
    p.add_argument("--moe-profile")
    p.add_argument("--out", required=True)
    p.add_argument("--max-mean-kl", type=float, default=0.02)
    p.add_argument("--max-drop-layers", type=int, default=1)
    p.add_argument("--ablation-layers", type=int, default=3)
    p.add_argument("--ablation-strength", type=float, default=0.5)

    p = sub.add_parser("select", help="Compile typed surgery selectors into a plan and read-only preview")
    p.add_argument("--model", required=True)
    p.add_argument("--selectors", required=True, help="selector YAML")
    p.add_argument("--out", required=True, help="new output directory")
    p.add_argument("--profile", help="profile.pt; required for directional edits")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("preview", help="Resolve and inspect a surgery plan without modifying the model")
    p.add_argument("--model", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--profile", help="profile.pt; required when plan contains directional edits")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("apply", help="Apply a reviewed plan and save a new checkpoint")
    p.add_argument("--model", required=True)
    p.add_argument("--profile", help="residual profile.pt; required only when plan contains directional ablation")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("validate", help="Compare candidate next-token distributions and teacher-forced text loss")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--max-mean-kl", type=float)
    p.add_argument("--out")
    p.add_argument("--device", default="auto")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    from .operator_audit import audited_dispatch
    return audited_dispatch(args, dispatch_core)


def dispatch_core(args):
    """Execute a parsed core command, shared by both installed entry points."""

    if args.cmd == "profile":
        from .profile import run_profile
        run_profile(args.model, args.keep, args.drop, args.out, args.batch_size, args.basis_rank, args.basis_samples, args.max_prompts, args.device)
        return
    if args.cmd == "profile-mlp":
        from .mlp import run_mlp_profile
        run_mlp_profile(args.model, args.keep, args.drop, args.out, args.batch_size, args.max_prompts, args.device)
        return
    if args.cmd == "search-mlp":
        from .mlp import run_mlp_search
        run_mlp_search(args.model, args.keep, args.profile, args.out, args.ratios, args.batch_size, args.max_prompts, args.contrast_weight, args.align_to, args.device)
        return
    if args.cmd == "profile-attention":
        from .attention import run_attention_profile
        run_attention_profile(args.model, args.keep, args.drop, args.out, args.batch_size, args.max_prompts, args.device)
        return
    if args.cmd == "search-attention":
        from .attention import run_attention_search
        run_attention_search(args.model, args.keep, args.profile, args.out, args.ratios, args.batch_size, args.max_prompts, args.contrast_weight, args.device)
        return
    if args.cmd == "profile-moe":
        from .moe import run_moe_profile
        run_moe_profile(args.model, args.keep, args.drop, args.out, args.batch_size, args.max_prompts, args.device)
        return
    if args.cmd == "search-moe":
        from .moe import run_moe_search
        run_moe_search(args.model, args.keep, args.profile, args.out, args.ratios, args.batch_size, args.max_prompts, args.contrast_weight, args.device)
        return
    if args.cmd == "optimize":
        from .optimizer import run_optimizer
        report = run_optimizer(
            args.model,
            args.keep,
            args.out,
            args.mlp_profile,
            args.attention_profile,
            args.moe_profile,
            args.layer_search,
            args.mlp_ratios,
            args.attention_ratios,
            args.moe_ratios,
            args.max_drop_layers,
            args.max_mean_kl,
            args.min_top1_agreement,
            args.max_trials,
            args.strategy,
            args.contrast_weight,
            args.align_to,
            args.batch_size,
            args.max_prompts,
            args.device,
        )
        print(json.dumps(report["best"], indent=2))
        return
    if args.cmd == "search-layers":
        from .search_layers import run_layer_search
        run_layer_search(args.model, args.keep, args.out, args.batch_size, args.max_prompts, args.device)
        return
    if args.cmd == "search-layers-greedy":
        from .search_greedy import run_greedy_layer_search
        run_greedy_layer_search(args.model, args.keep, args.out, args.max_mean_kl, args.max_layers, args.batch_size, args.max_prompts, args.device)
        return
    if args.cmd == "steer-label-template":
        from .steering_eval import run_label_template_command
        result = run_label_template_command(args)
        print(json.dumps(result, indent=2))
        return result
    if args.cmd == "steer-evaluate":
        from .steering_eval import run_steer_evaluate_command
        result = run_steer_evaluate_command(args)
        print(json.dumps(result, indent=2))
        return result
    if args.cmd == "steer-causal":
        from .neuron_causal import run_steer_causal_command
        result = run_steer_causal_command(args)
        print(json.dumps(result, indent=2))
        return result
    if args.cmd == "steer-rank":
        from .neuron_ranking import run_steer_rank_command
        result = run_steer_rank_command(args)
        print(json.dumps(result, indent=2))
        return result
    if args.cmd == "steer-presets":
        from .neuron_steering import list_presets
        print(json.dumps(list_presets(args.model_id), indent=2))
        return
    if args.cmd == "steer":
        from .neuron_steering import run_steer_command
        result = run_steer_command(args)
        print(json.dumps(result, indent=2))
        return result
    if args.cmd == "inventory":
        from .inventory import run_inventory
        print(json.dumps(run_inventory(args.model, args.out, args.device), indent=2))
        return
    if args.cmd == "plan":
        from .plan import generate_plan
        plan = generate_plan(
            args.profile,
            args.search,
            args.out,
            args.max_mean_kl,
            args.max_drop_layers,
            args.ablation_layers,
            args.ablation_strength,
            args.greedy_search,
            args.mlp_search,
            args.mlp_profile,
            args.attention_search,
            args.attention_profile,
            args.moe_search,
            args.moe_profile,
        )
        print(json.dumps(plan, indent=2))
        return
    if args.cmd == "select":
        from .selectors import run_select
        print(json.dumps(run_select(args.model, args.selectors, args.out, args.profile, args.device), indent=2))
        return
    if args.cmd == "preview":
        from .preview import run_preview
        print(json.dumps(run_preview(args.model, args.plan, args.profile, args.device), indent=2))
        return
    if args.cmd == "apply":
        from .apply import run_apply
        run_apply(args.model, args.profile, args.plan, args.out, args.device)
        return
    if args.cmd == "validate":
        from .validate import run_validate
        metrics = run_validate(args.base, args.candidate, args.keep, args.batch_size, args.max_prompts, args.max_mean_kl, args.out, args.device)
        print(json.dumps(metrics, indent=2))
        if not metrics["passed"]:
            sys.exit(2)
        return


if __name__ == "__main__":
    main()
