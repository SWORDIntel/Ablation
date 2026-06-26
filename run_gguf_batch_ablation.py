#!/usr/bin/env python3
"""Batch GGUF-native refusal ablation on all Ollama models.

For each model:
1. Copies the GGUF blob from Ollama's storage
2. Runs GGUFRefusalAblator to zero refusal-related FFN tensors in target blocks
3. Saves ablated GGUF and registers it as a new Ollama model
4. Optionally validates with refusal prompts

Usage:
    python3 run_gguf_batch_ablation.py [--method zero|prune|clamp] [--blocks 4 22 27 29]
    python3 run_gguf_batch_ablation.py --models tinyllama --validate
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ABLATION_ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ABLATION_ROOT / "exports" / "gguf_ablation"
BLOB_DIR = "/usr/share/ollama/.ollama/models/blobs"

OLLAMA_MODELS = {
    "mythos-sec-24b": {
        "ollama_name": "supergoatscriptguy/mythos-sec:24b",
        "blob": "sha256-d55ab0bfb1b80102465eef8a17f05d201fb17dbe27a62f153f0db6c446aed077",
        "type": "moe",
        "params": "24B",
        "size_gb": 14,
    },
    "nu11secur1ty-developer": {
        "ollama_name": "f0rc3ps/nu11secur1tyAIq8-Developer:latest",
        "blob": "sha256-acad11b4503327acee8617a6d6bd3d9e54894e02c034e6a9111870da5c5764ae",
        "type": "moe",
        "params": "20B",
        "size_gb": 16,
    },
    "qwen3-30b": {
        "ollama_name": "qwen3:30b",
        "blob": "sha256-58574f2e94b99fb9e4391408b57e5aeaaaec10f6384e9a699fc2cb43a5c8eabf",
        "type": "moe",
        "params": "30B",
        "size_gb": 18,
    },
    "qwen-coder-14b": {
        "ollama_name": "qwen2.5-coder:14b",
        "blob": "sha256-ac9bc7a69dab38da1c790838955f1293420b55ab555ef6b4615efa1c1507b1ed",
        "type": "dense",
        "params": "14B",
        "size_gb": 9,
    },
    "hermes3-8b": {
        "ollama_name": "hermes3:8b",
        "blob": "sha256-c8985d236593f7a17da2a3da49588aa951a9b1e57ce97753364fbf59e63af84a",
        "type": "dense",
        "params": "8B",
        "size_gb": 4.7,
    },
    "qwen-coder-7b": {
        "ollama_name": "qwen2.5-coder:7b",
        "blob": "sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463",
        "type": "dense",
        "params": "7B",
        "size_gb": 4.7,
    },
    "qwen-coder-1.5b": {
        "ollama_name": "qwen2.5-coder:1.5b",
        "blob": "sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104",
        "type": "dense",
        "params": "1.5B",
        "size_gb": 0.9,
    },
    "llama3.2-3b": {
        "ollama_name": "llama3.2:3b",
        "blob": "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
        "type": "dense",
        "params": "3B",
        "size_gb": 2.0,
    },
    "tinyllama": {
        "ollama_name": "tinyllama:latest",
        "blob": "sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816",
        "type": "dense",
        "params": "1.1B",
        "size_gb": 0.6,
    },
}

# Default refusal layers identified by the heretic report
DEFAULT_BLOCKS = [4, 22, 27, 29]
DEFAULT_TENSORS = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
MOE_TENSORS = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]


def copy_blob(model_key: str, model_info: dict) -> Path:
    """Copy GGUF blob from Ollama storage to working directory."""
    src = f"{BLOB_DIR}/{model_info['blob']}"
    dst = EXPORT_DIR / model_key / "original.gguf"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        logger_info = f"Copying {model_info['size_gb']}GB blob..."
        print(f"  {logger_info}")
        subprocess.run(["sudo", "cp", src, str(dst)], check=True)
        subprocess.run(["sudo", "chown", os.environ.get("USER", "john"), str(dst)], check=True)

    return dst


def ablate_model(model_key: str, model_info: dict, method: str, blocks: list,
                 tensors: list, threshold: float) -> dict:
    """Run GGUF ablation on a single model."""
    import logging
    logging.basicConfig(level=logging.INFO, format="  %(levelname)s: %(message)s")

    print(f"\n{'='*60}")
    print(f"  Ablating: {model_key} ({model_info['ollama_name']})")
    print(f"  Type: {model_info['type']} | Params: {model_info['params']} | Size: {model_info['size_gb']}GB")
    print(f"  Method: {method} | Blocks: {blocks} | Tensors: {tensors}")
    print(f"{'='*60}")

    # Step 1: Copy blob
    try:
        gguf_path = copy_blob(model_key, model_info)
    except Exception as e:
        return {"model": model_key, "status": "copy_failed", "error": str(e)}

    # Step 2: Ablate
    output_path = EXPORT_DIR / model_key / "ablated.gguf"
    report_path = EXPORT_DIR / model_key / "ablation_report.json"

    sys.path.insert(0, str(ABLATION_ROOT / "src"))
    from aegis_lab.editing.gguf_ablation import GGUFRefusalAblator, GGUFAblationTarget

    try:
        ablator = GGUFRefusalAblator(gguf_path)
        arch = ablator.get_architecture()
        n_blocks = ablator.get_block_count()
        # Auto-detect MoE: check if ffn_gate_exps.weight exists in block 4
        # (block 0 may be dense in mixed architectures like DeepSeek)
        all_tensors = ablator.list_tensors()
        tensor_names = {t[0] for t in all_tensors}
        is_moe = any(f"blk.{blocks[0]}.ffn_gate_exps.weight" in name for name in tensor_names)
        use_tensors = MOE_TENSORS if is_moe else tensors
        print(f"  Architecture: {arch}, Blocks: {n_blocks}, MoE: {is_moe}, Tensors: {use_tensors}")

        # Clamp blocks to available
        valid_blocks = [b for b in blocks if b < n_blocks]
        if len(valid_blocks) != len(blocks):
            print(f"  WARNING: Some blocks exceed {n_blocks}, using: {valid_blocks}")

        target = GGUFAblationTarget(
            block_indices=valid_blocks,
            tensor_suffixes=use_tensors,
            method=method,
            threshold=threshold,
        )

        result = ablator.ablate(output_path, [target])

        report = {
            "model": model_key,
            "ollama_name": model_info["ollama_name"],
            "type": model_info["type"],
            "params": model_info["params"],
            "architecture": arch,
            "total_blocks": n_blocks,
            "method": method,
            "blocks_ablated": result.blocks_ablated,
            "tensors_modified": result.tensors_modified,
            "total_neurons_zeroed": result.total_neurons_zeroed,
            "elapsed_seconds": result.elapsed_seconds,
            "output_path": str(output_path),
            "status": "success",
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"  ✓ Ablated {result.total_neurons_zeroed} neurons in {len(result.tensors_modified)} tensors ({result.elapsed_seconds:.1f}s)")
        return report

    except Exception as e:
        import traceback
        return {
            "model": model_key,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
        }


def register_with_ollama(model_key: str, ablated_gguf: str, base_model: str) -> dict:
    """Register ablated GGUF as a new Ollama model."""
    modelfile_path = EXPORT_DIR / model_key / "Modelfile.ablated"
    ablated_name = f"{base_model}:ablated"
    # Strip :latest if present to avoid double tags
    if ":latest:ablated" in ablated_name:
        ablated_name = ablated_name.replace(":latest:ablated", ":ablated")

    # Get base modelfile
    try:
        result = subprocess.run(
            ["ollama", "show", base_model, "--modelfile"],
            capture_output=True, text=True, timeout=30,
        )
        base_modelfile = result.stdout
    except Exception as e:
        return {"status": "failed", "error": f"Failed to get base modelfile: {e}"}

    # Replace FROM line with ablated GGUF path
    lines = base_modelfile.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("FROM "):
            new_lines.append(f"FROM {ablated_gguf}")
        else:
            new_lines.append(line)

    with open(modelfile_path, "w") as f:
        f.write("\n".join(new_lines))

    # Create the model
    try:
        result = subprocess.run(
            ["ollama", "create", ablated_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return {"status": "success", "ollama_name": ablated_name}
        else:
            return {"status": "failed", "error": result.stderr}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def validate_model(model_key: str, ollama_name: str) -> dict:
    """Run refusal prompts through ablated model."""
    sys.path.insert(0, str(ABLATION_ROOT / "src"))
    from aegis_lab.editing.gguf_ablation import GGUFRefusalAblator

    ablator = GGUFRefusalAblator.__new__(GGUFRefusalAblator)
    return ablator.validate_with_ollama(ollama_name)


def main():
    parser = argparse.ArgumentParser(description="Batch GGUF-native refusal ablation")
    parser.add_argument("--method", default="zero", choices=["zero", "prune", "clamp"])
    parser.add_argument("--blocks", nargs="+", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--tensors", nargs="+", default=DEFAULT_TENSORS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--models", help="Comma-separated model keys (default: all)")
    parser.add_argument("--register", action="store_true", help="Register ablated models with Ollama")
    parser.add_argument("--validate", action="store_true", help="Validate with refusal prompts after ablation")
    args = parser.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    models = OLLAMA_MODELS
    if args.models:
        selected = args.models.split(",")
        models = {k: v for k, v in models.items() if k in selected}

    print(f"Running GGUF ablation on {len(models)} models")
    print(f"Method: {args.method}, Blocks: {args.blocks}")
    if args.register:
        print("Will register ablated models with Ollama")
    if args.validate:
        print("Will validate with refusal prompts")

    results = []
    for key, info in models.items():
        result = ablate_model(key, info, args.method, args.blocks, args.tensors, args.threshold)

        if result.get("status") == "success" and args.register:
            print(f"  Registering with Ollama...")
            reg = register_with_ollama(key, result["output_path"], info["ollama_name"])
            result["registration"] = reg
            if reg["status"] == "success" and args.validate:
                print(f"  Validating refusal behavior...")
                val = validate_model(key, reg["ollama_name"])
                result["validation"] = val
                print(f"  Compliance: {val['compliance_rate']*100:.0f}% ({val['refusal_count']}/{val['total']} refusals)")

        results.append(result)

    # Save summary
    summary_path = EXPORT_DIR / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total": len(results),
            "success": sum(1 for r in results if r.get("status") == "success"),
            "failed": sum(1 for r in results if r.get("status") != "success"),
            "results": results,
        }, f, indent=2)

    success = sum(1 for r in results if r.get("status") == "success")
    print(f"\n{'='*60}")
    print(f"  Batch complete: {summary_path}")
    print(f"  Success: {success}/{len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
