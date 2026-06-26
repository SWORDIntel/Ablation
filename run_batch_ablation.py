#!/usr/bin/env python3
"""Batch refusal ablation runner — runs heretic report on all Ollama models.

For each model:
1. Creates a per-model heretic config
2. Runs the heretic refusal ablation pipeline (report-only)
3. Saves JSON report with refusal scores, identified layers, and recommendations

Usage:
    python3 run_batch_ablation.py [--apply-edits] [--models model1,model2,...]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ABLATION_ROOT = Path(__file__).resolve().parent
DATA_PATH = ABLATION_ROOT / "data" / "refusal_prompts.jsonl"
EXPORT_DIR = ABLATION_ROOT / "exports" / "batch_ablation"
CONFIG_DIR = ABLATION_ROOT / "configs" / "batch"

# All Ollama models with their GGUF blob paths
OLLAMA_MODELS = {
    "mythos-sec-24b": {
        "ollama_name": "supergoatscriptguy/mythos-sec:24b",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-d55ab0bfb1b80102465eef8a17f05d201fb17dbe27a62f153f0db6c446aed077",
        "type": "moe",
        "params": "24B",
    },
    "nu11secur1ty-developer": {
        "ollama_name": "f0rc3ps/nu11secur1tyAIq8-Developer:latest",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-acad11b4503327acee8617a6d6bd3d9e54894e02c034e6a9111870da5c5764ae",
        "type": "dense",
        "params": "20B",
    },
    "qwen3-30b": {
        "ollama_name": "qwen3:30b",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-58574f2e94b99fb9e4391408b57e5aeaaaec10f6384e9a699fc2cb43a5c8eabf",
        "type": "dense",
        "params": "30B",
    },
    "qwen-coder-14b": {
        "ollama_name": "qwen2.5-coder:14b",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-ac9bc7a69dab38da1c790838955f1293420b55ab555ef6b4615efa1c1507b1ed",
        "type": "dense",
        "params": "14B",
    },
    "hermes3-8b": {
        "ollama_name": "hermes3:8b",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-c8985d236593f7a17da2a3da49588aa951a9b1e57ce97753364fbf59e63af84a",
        "type": "dense",
        "params": "8B",
    },
    "qwen-coder-7b": {
        "ollama_name": "qwen2.5-coder:7b",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463",
        "type": "dense",
        "params": "7B",
    },
    "qwen-coder-1.5b": {
        "ollama_name": "qwen2.5-coder:1.5b",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104",
        "type": "dense",
        "params": "1.5B",
    },
    "llama3.2-3b": {
        "ollama_name": "llama3.2:3b",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
        "type": "dense",
        "params": "3B",
    },
    "tinyllama": {
        "ollama_name": "tinyllama:latest",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816",
        "type": "dense",
        "params": "1.1B",
    },
    "nomic-embed": {
        "ollama_name": "nomic-embed-text:latest",
        "blob": "/usr/share/ollama/.ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6",
        "type": "embedding",
        "params": "0.137B",
    },
}


def create_model_config(model_key: str, model_info: dict, output_dir: Path) -> Path:
    """Create a heretic config YAML for a specific model."""
    config = {
        "model_path": model_info["blob"],
        "dataset_path": str(DATA_PATH),
        "n_trials": 8,
        "top_k_layers": 4,
        "seed": 42,
        "train_split": 0.7,
        "val_split": 0.2,
        "max_prompts": 40,
        "prompt_field": "prompt",
        "label_field": "label",
        "prompt_label_map": {"safe": "safe", "unsafe": "unsafe"},
        "refusal_weight": 1.0,
        "kl_weight": 0.18,
        "max_parallel_agents": 3,
        "output_dir": str(output_dir / model_key),
        "strategy": "heretic_refusal",
        "enable_optuna": False,
    }

    config_dir = CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"heretic_{model_key}.yaml"

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_ablation_for_model(model_key: str, model_info: dict, apply_edits: bool) -> dict:
    """Run heretic ablation for a single model."""
    model_output_dir = EXPORT_DIR / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)

    config_path = create_model_config(model_key, model_info, EXPORT_DIR)
    report_path = model_output_dir / "ablation_report.json"

    print(f"\n{'='*60}")
    print(f"  Ablating: {model_key} ({model_info['ollama_name']})")
    print(f"  Type: {model_info['type']} | Params: {model_info['params']}")
    print(f"  Config: {config_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "-m", "aegis_lab.editing.model_refusal_ablation",
        "--model", model_info["blob"],
        "--output", str(model_output_dir / "ablated_model"),
        "--method", "zero",
        "--strategy", "heretic",
        "--heretic-config", str(config_path),
        "--report", str(report_path),
    ]
    if apply_edits:
        cmd.append("--apply-heretic-edits")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ABLATION_ROOT / "src")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(ABLATION_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - start

    status = "success" if result.returncode == 0 else "failed"
    print(f"  Status: {status} ({elapsed:.1f}s)")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")

    return {
        "model": model_key,
        "ollama_name": model_info["ollama_name"],
        "type": model_info["type"],
        "params": model_info["params"],
        "status": status,
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
        "report_path": str(report_path) if report_path.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch refusal ablation on all Ollama models")
    parser.add_argument("--apply-edits", action="store_true", help="Apply heretic edits (not just report)")
    parser.add_argument("--models", help="Comma-separated model keys to run (default: all)")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding models")
    args = parser.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    models = OLLAMA_MODELS
    if args.models:
        selected = args.models.split(",")
        models = {k: v for k, v in models.items() if k in selected}
    if args.skip_embedding:
        models = {k: v for k, v in models.items() if v["type"] != "embedding"}

    print(f"Running ablation on {len(models)} models")
    print(f"Apply edits: {args.apply_edits}")

    results = []
    for key, info in models.items():
        try:
            result = run_ablation_for_model(key, info, args.apply_edits)
            results.append(result)
        except subprocess.TimeoutExpired:
            results.append({
                "model": key,
                "status": "timeout",
                "error": "Exceeded 600s timeout",
            })
        except Exception as exc:
            results.append({
                "model": key,
                "status": "error",
                "error": str(exc),
            })

    # Save summary
    summary_path = EXPORT_DIR / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_models": len(models),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] != "success"),
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Batch complete: {summary_path}")
    print(f"  Success: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
