#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess

# Add src to python path automatically
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def main():
    parser = argparse.ArgumentParser(description="AEGIS-LAB Unified Launcher")
    parser.add_argument("command", choices=["orchestrator", "worker", "gui", "api", "submit", "train", "list", "status"], help="Command to run")
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
