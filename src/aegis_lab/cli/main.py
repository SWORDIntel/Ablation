import sys
import os
import argparse
import zmq
import time
from aegis_lab.state.db import AegisState
from aegis_lab.orchestrator.service import OrchestratorService


class OrchestratorClient:
    def __init__(self, url="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(url)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)

    def request(self, msg_type, **kwargs):
        try:
            kwargs["type"] = msg_type
            self.socket.send_json(kwargs)
            return self.socket.recv_json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.destroy(linger=0)
            self.context = None


def _render_progress_bar(percent: float, width: int = 30) -> str:
    filled = int((percent / 100.0) * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {percent:6.2f}%"


def _print_job_status(status: dict) -> None:
    print(f"Job: {status['job_id']}")
    print(f"Status: {status['status']}")

    progress = status.get("progress", {})
    percent = float(progress.get("percent", 0.0))
    completed = progress.get("completed", 0)
    total = progress.get("total", 0)
    current_stage = progress.get("current_stage")

    print(f"Progress: {_render_progress_bar(percent)} ({completed}/{total})")
    if current_stage:
        print(f"Current Stage: {current_stage}")

    print("Stages:")
    for stage in status.get("stages", []):
        print(f"  {stage['ordinal']}: {stage['stage_name']} ({stage['status']})")


def _watch_job_progress(client: OrchestratorClient, job_id: str, interval_sec: float) -> None:
    try:
        while True:
            status = client.request("get_job_status", job_id=job_id)
            if "error" in status:
                print(status["error"])
                return

            print("\033[2J\033[H", end="")
            _print_job_status(status)

            if status.get("status") in {"succeeded", "failed"}:
                return
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nStopped watching job progress.")


def main():
    parser = argparse.ArgumentParser(description="AEGIS-LAB CLI")
    subparsers = parser.add_subparsers(dest="command")

    submit = subparsers.add_parser("submit", help="Submit a new job")
    submit.add_argument("--project", required=True, help="Project ID")
    submit.add_argument("--type", default="ablation", help="Job type")
    submit.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")

    list_jobs = subparsers.add_parser("list", help="List all jobs")
    list_jobs.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")

    status = subparsers.add_parser("status", help="Get job status")
    status.add_argument("job_id", help="Job ID")
    status.add_argument("--watch", action="store_true", help="Watch real-time progress updates")
    status.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds for --watch")
    status.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")

    train = subparsers.add_parser("train", help="Submit a full ablation training pipeline")
    train.add_argument("--project", default="Qwen-Mission-Alpha", help="Project ID")
    train.add_argument("--model", default=None, help="Path of the source model")
    train.add_argument("--model-id", default=None, help="Model index from `aegis models`")
    train.add_argument("--target", default="refusal", help="Behavioral target atom")
    train.add_argument("--method", help="Ablation method")
    train.add_argument("--device", choices=["cpu", "igpu", "npu", "auto"], default="auto", help="Preferred compute device")
    train.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")

    models = subparsers.add_parser("models", help="List selectable models for ablation")
    models.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")

    orch = subparsers.add_parser("orchestrator", help="Start the Orchestrator service")
    orch.add_argument("--port", type=int, default=5555, help="IPC port")

    worker = subparsers.add_parser("worker", help="Start a CPU worker")
    worker.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")

    args = parser.parse_args()

    if args.command == "orchestrator":
        storage_root = os.path.expanduser("~/.aegis_lab/state")
        lib_path = os.path.abspath("QIHSE/qihse/libqihse.so")
        state = AegisState(storage_root, lib_path)
        service = OrchestratorService(state, ipc_port=args.port)
        print(f"Starting Orchestrator on port {args.port}...")
        service.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            service.stop()
    elif args.command == "worker":
        from aegis_lab.workers.cpu_worker import CpuWorker

        worker_instance = CpuWorker(orchestrator_url=args.url)
        print(f"Starting CPU Worker connecting to {args.url}...")
        worker_instance.connect()
        try:
            worker_instance.run()
        except KeyboardInterrupt:
            worker_instance.stop()
    elif args.command == "submit":
        client = OrchestratorClient(args.url)
        try:
            resp = client.request("submit_job", project_id=args.project, job_type=args.type)
            if resp.get("status") == "ok":
                print(f"Job submitted: {resp['job_id']}")
            else:
                print(f"Error: {resp.get('error')}")
        finally:
            client.close()
    elif args.command == "models":
        client = OrchestratorClient(args.url)
        try:
            resp = client.request("get_available_models")
            if resp.get("status") == "ok":
                model_paths = resp.get("models", [])
                if not model_paths:
                    print("No models discovered.")
                for idx, model_path in enumerate(model_paths, 1):
                    print(f"{idx:>2}. {model_path}")
            else:
                print(f"Error: {resp.get('error')}")
        finally:
            client.close()
    elif args.command == "train":
        client = OrchestratorClient(args.url)
        try:
            resolved_model_path = args.model
            if args.model_id is not None:
                model_resp = client.request("get_available_models")
                if model_resp.get("status") != "ok":
                    print(f"Error listing models: {model_resp.get('error')}")
                    return 1
                model_paths = model_resp.get("models", [])
                try:
                    resolved_model_path = model_paths[int(args.model_id) - 1]
                except (ValueError, IndexError):
                    print(f"Invalid --model-id '{args.model_id}'. Run `aegis models` first.")
                    return 1

            if not resolved_model_path:
                print("No model selected. Use --model <path> or --model-id <index>.")
                return 1

            params = {
                "model_path": resolved_model_path,
                "target_atom": args.target,
                "preferred_device": args.device,
            }
            resp = client.request(
                "submit_job",
                project_id=args.project,
                job_type="ablation_training",
                parameters=params,
            )
            if resp.get("status") == "ok":
                print(f"Training job submitted: {resp['job_id']}")
                print(f"Selected model: {resolved_model_path}")
                print(f"Monitor progress via `aegis status {resp['job_id']} --watch`")
            else:
                print(f"Error: {resp.get('error')}")
        finally:
            client.close()
    elif args.command == "list":
        client = OrchestratorClient(args.url)
        try:
            resp = client.request("list_jobs")
            if isinstance(resp, list):
                for job in resp:
                    print(f"{job['job_id']} | {job['project_id']} | {job['status']} | {job['job_type']}")
            else:
                print(f"Error: {resp.get('error')}")
        finally:
            client.close()
    elif args.command == "status":
        client = OrchestratorClient(args.url)
        try:
            if args.watch:
                _watch_job_progress(client, args.job_id, max(args.interval, 0.25))
            else:
                status_data = client.request("get_job_status", job_id=args.job_id)
                if "error" in status_data:
                    print(status_data["error"])
                else:
                    _print_job_status(status_data)
        finally:
            client.close()
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
