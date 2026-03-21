import sys
import os
import argparse
import zmq
import json
import time
from aegis_lab.state.db import AegisState
from aegis_lab.orchestrator.service import OrchestratorService

class OrchestratorClient:
    def __init__(self, url="tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(url)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)

    def request(self, msg_type, **kwargs):
        try:
            kwargs["type"] = msg_type
            self.socket.send_json(kwargs)
            return self.socket.recv_json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="AEGIS-LAB CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Submit job
    submit = subparsers.add_parser("submit", help="Submit a new job")
    submit.add_argument("--project", required=True, help="Project ID")
    submit.add_argument("--type", default="ablation", help="Job type")
    submit.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")
    
    # List jobs
    list_jobs = subparsers.add_parser("list", help="List all jobs")
    list_jobs.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")
    
    # Job status
    status = subparsers.add_parser("status", help="Get job status")
    status.add_argument("job_id", help="Job ID")
    status.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")
    
    # Train job
    train = subparsers.add_parser("train", help="Submit a full ablation training pipeline")
    train.add_argument("--project", default="Qwen-Mission-Alpha", help="Project ID")
    train.add_argument("--model", default="models/qwen2.5.gguf", help="Path or ID of the source model")
    train.add_argument("--target", default="refusal", help="Behavioral target atom")
    train.add_argument("--device", choices=["cpu", "igpu", "npu", "auto"], default="auto", help="Preferred compute device")
    train.add_argument("--url", default="tcp://localhost:5555", help="Orchestrator URL")
    
    # Start Orchestrator
    orch = subparsers.add_parser("orchestrator", help="Start the Orchestrator service")
    orch.add_argument("--port", type=int, default=5555, help="IPC port")
    
    # Start Worker
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
            while True: time.sleep(1)
        except KeyboardInterrupt:
            service.stop()
    elif args.command == "worker":
        from aegis_lab.workers.cpu_worker import CpuWorker
        worker = CpuWorker(orchestrator_url=args.url)
        print(f"Starting CPU Worker connecting to {args.url}...")
        worker.connect()
        try:
            worker.run()
        except KeyboardInterrupt:
            worker.stop()
    elif args.command == "submit":
        client = OrchestratorClient(args.url)
        # We need to add a 'submit_job' handler to the OrchestratorService IPC
        resp = client.request("submit_job", project_id=args.project, job_type=args.type)
        if resp.get("status") == "ok":
            print(f"Job submitted: {resp['job_id']}")
        else:
            print(f"Error: {resp.get('error')}")
    elif args.command == "train":
        client = OrchestratorClient(args.url)
        params = {
            "model_path": args.model,
            "target_atom": args.target,
            "preferred_device": args.device
        }
        resp = client.request("submit_job", 
                              project_id=args.project, 
                              job_type="ablation_training",
                              parameters=params)
        if resp.get("status") == "ok":
            print(f"Training job submitted: {resp['job_id']}")
            print(f"Monitor progress in the Dashboard or via 'aegis status {resp['job_id']}'")
        else:
            print(f"Error: {resp.get('error')}")
    elif args.command == "list":
        client = OrchestratorClient(args.url)
        resp = client.request("list_jobs")
        if isinstance(resp, list):
            for j in resp:
                print(f"{j['job_id']} | {j['project_id']} | {j['status']} | {j['job_type']}")
        else:
            print(f"Error: {resp.get('error')}")
    elif args.command == "status":
        client = OrchestratorClient(args.url)
        status = client.request("get_job_status", job_id=args.job_id)
        if "error" in status:
            print(status["error"])
        else:
            print(f"Job: {status['job_id']}")
            print(f"Status: {status['status']}")
            print("Stages:")
            for s in status.get("stages", []):
                print(f"  {s['ordinal']}: {s['stage_name']} ({s['status']})")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
