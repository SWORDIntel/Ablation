import sys
import os
import time
import logging
import threading
import json
import zmq
import tempfile
import shutil
import uuid

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "aegis-lab/src"))

from aegis_lab.orchestrator.service import OrchestratorService
from aegis_lab.state.db import AegisState
from aegis_lab.workers.base import WorkerBase

# Configure root logger
logging.basicConfig(level=logging.INFO)

class TestWorker(WorkerBase):
    def execute_stage(self, task):
        logger = logging.getLogger("test_worker")
        logger.info(f"Executing stage {task['stage_name']}")
        return {"success": True}

def run_test():
    test_dir = tempfile.mkdtemp()
    try:
        state_path = os.path.join(test_dir, "state")
        lib_path = os.path.abspath("QIHSE/qihse/libqihse.so")
        state = AegisState(state_path, lib_path)
        
        # Start orchestrator
        orchestrator = OrchestratorService(state, ipc_port=5565)
        # Note: LogServer in service.py is hardcoded to 5556, I should probably make it configurable or just use it.
        # But for test, let's see. If LogServer is also in use, it will fail.
        orchestrator.start()
        
        # Start worker
        worker = TestWorker(orchestrator_url="tcp://localhost:5565", worker_type="cpu")
        worker.connect()
        
        # Give worker a bit of time to connect and heartbeat
        time.sleep(1)
        
        # Submit job
        job_id = orchestrator.submit_job("test-proj", "ablation")
        print(f"Submitted job: {job_id}")
        
        # Start worker loop in thread
        worker_running = True
        def worker_loop():
            nonlocal worker_running
            while worker_running:
                try:
                    worker.run()
                except Exception as e:
                    print(f"Worker thread error: {e}")
                    break
        
        worker_thread = threading.Thread(target=worker_loop, daemon=True)
        worker_thread.start()
        
        # Wait for job to complete
        timeout = 10
        start_time = time.time()
        status = {}
        while time.time() - start_time < timeout:
            status = orchestrator.get_job_status(job_id)
            if status.get("status") == "succeeded":
                break
            time.sleep(1)
            
        print(f"Job status: {status.get('status')}")
        
        # Allow some time for logs to be processed
        time.sleep(2)
        
        # Check logs in state
        logs = state.get_logs()
        print(f"Total logs found: {len(logs)}")
        for log in logs:
            print(f"Log: {log.get('msg')} (Job: {log.get('job_id')}, Stage: {log.get('stage_id')})")
            
        if len(logs) > 0:
            print("SUCCESS: Logs found in state!")
            # Check for job_id and stage_id in at least one log
            found_ids = False
            for log in logs:
                if log.get('job_id') == job_id and log.get('stage_id'):
                    found_ids = True
                    break
            if found_ids:
                print("SUCCESS: Logs are associated with job and stage IDs!")
            else:
                print("FAILURE: Logs found but IDs are missing.")
        else:
            print("FAILURE: No logs found in state.")
            
        worker_running = False
        worker.stop()
        orchestrator.stop()
        
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    run_test()
