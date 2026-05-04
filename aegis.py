#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess

# Add src to python path automatically
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    parser = argparse.ArgumentParser(description="AEGIS-LAB Unified Launcher")
    parser.add_argument("command", choices=["orchestrator", "worker", "gui", "api", "submit", "train", "mission", "interactive", "list", "status"], help="Command to run")
    parser.add_argument("--port", type=int, default=5555, help="Orchestrator port")
    parser.add_argument("--api-port", type=int, default=8000, help="REST API port")
    parser.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")
    parser.add_argument("--project", help="Project ID for submission")
    parser.add_argument("--model", help="Model path for training")
    parser.add_argument("--target", help="Target atom for training")
    parser.add_argument("--device", choices=["cpu", "igpu", "npu", "auto"], default="auto", help="Compute device")
    parser.add_argument("--type", default="ablation", help="Job type")
    parser.add_argument("--job-id", help="Job ID for status check")

    args, extra = parser.parse_known_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src")

    if args.command == "orchestrator":
        cmd = [sys.executable, "src/aegis_lab/cli/main.py", "orchestrator", "--port", str(args.port)]
    elif args.command == "worker":
        cmd = [sys.executable, "src/aegis_lab/cli/main.py", "worker", "--url", args.url]
    elif args.command == "gui":
        cmd = [sys.executable, "src/aegis_lab/gui/main_window.py"]
    elif args.command == "api":
        import socket
        port = args.api_port
        if port == 8000: # If default, randomize to avoid conflicts
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            port = s.getsockname()[1]
            s.close()
            print(f"Randomized API Port: {port}")
        
        env["ORCHESTRATOR_URL"] = args.url
        cmd = [sys.executable, "-m", "uvicorn", "aegis_lab.api.server:app", "--host", "0.0.0.0", "--port", str(port)]
    elif args.command == "submit":
        if not args.project:
            print("Error: --project required for submit")
            sys.exit(1)
        cmd = [sys.executable, "src/aegis_lab/cli/main.py", "submit", "--project", args.project, "--type", args.type, "--url", args.url]
    elif args.command == "train":
        if not args.project or not args.model or not args.target:
            print("Error: --project, --model, and --target required for train")
            sys.exit(1)
        cmd = [sys.executable, "src/aegis_lab/cli/main.py", "train", 
               "--project", args.project, "--model", args.model, "--target", args.target, 
               "--device", args.device, "--url", args.url]
    elif args.command == "mission":
        print("🚀 Launching Automated Qwen Ablation Mission...")
        from aegis_lab.state.db import AegisState
        from aegis_lab.artifacts.store import ArtifactStore
        from aegis_lab.editing.pipeline import AblationPipeline
        
        storage_root = os.path.expanduser("~/.aegis_lab/state")
        artifact_root = os.path.expanduser("~/.aegis_lab/artifacts")
        lib_path = os.path.abspath("QIHSE/qihse/libqihse.so")
        
        # Ensure directories exist
        os.makedirs(storage_root, exist_ok=True)
        os.makedirs(artifact_root, exist_ok=True)
        
        state = AegisState(storage_root, lib_path)
        store = ArtifactStore(artifact_root, state)
        pipeline = AblationPipeline(state, store)
        
        # Qwen-specific mission parameters
        model_path = args.model or "models/qwen2.5.gguf"
        target = args.target or "refusal"
        
        # For the mission, we use a predefined set of layers
        layers = [12, 13, 14, 15, 16] 
        
        # We need datasets for the mission
        pos_data = "data/qwen_positive.jsonl"
        neg_data = "data/qwen_negative.jsonl"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        for d in [pos_data, neg_data]:
            if not os.path.exists(d):
                with open(d, "w") as f:
                    f.write('{"text": "placeholder content for mission"}\n')
        
        pipeline.run_full_ablation(
            model_path=model_path,
            layers=layers,
            positive_dataset=pos_data,
            negative_dataset=neg_data,
            project_id="Qwen-Mission-Alpha"
        )
        return
    elif args.command == "interactive":
        print("🚀 Launching AEGIS-LAB Interactive Advanced Pipeline...")
        cmd = [sys.executable, "src/aegis_lab/cli/interactive.py"]
    elif args.command == "list":
        cmd = [sys.executable, "src/aegis_lab/cli/main.py", "list", "--url", args.url]
    elif args.command == "status":
        if not args.job_id:
            print("Error: --job-id required for status")
            sys.exit(1)
        cmd = [sys.executable, "src/aegis_lab/cli/main.py", "status", args.job_id, "--url", args.url]

    # Run in foreground
    try:
        subprocess.run(cmd + extra, env=env, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
