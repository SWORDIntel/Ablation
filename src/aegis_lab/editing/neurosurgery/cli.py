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

    p = sub.add_parser("apply", help="Apply a reviewed plan and save a new checkpoint")
    p.add_argument("--model", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("validate", help="Compare candidate next-token distribution against base")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--max-mean-kl", type=float)
    p.add_argument("--out")
    p.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

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
    if args.cmd == "search-layers":
        from .search_layers import run_layer_search
        run_layer_search(args.model, args.keep, args.out, args.batch_size, args.max_prompts, args.device)
        return
    if args.cmd == "search-layers-greedy":
        from .search_greedy import run_greedy_layer_search
        run_greedy_layer_search(args.model, args.keep, args.out, args.max_mean_kl, args.max_layers, args.batch_size, args.max_prompts, args.device)
        return
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
