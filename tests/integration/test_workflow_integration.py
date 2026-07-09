import unittest
from unittest.mock import MagicMock, patch
import time
import threading
from framewerx.aegis_lab.orchestrator.service import OrchestratorService
from framewerx.aegis_lab.state.db import AegisState

class MockWorker:
    def __init__(self, service: OrchestratorService, worker_id: str, worker_type: str):
        self.service = service
        self.worker_id = worker_id
        self.worker_type = worker_type
        # Ensure capabilities match what SchedulerEngine expects (cpu_amx, cpu_avx512, etc.)
        self.capabilities = {
            "cpu_amx": True if worker_type == "cpu" else False,
            "npu_present": True if worker_type == "npu" else False,
            "igpu_present": False
        }
        
    def register(self):
        self.service._handle_register({
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "capabilities": self.capabilities
        })
        
    def do_work(self):
        # Request task
        resp = self.service._handle_request_task({"worker_id": self.worker_id})
        if resp["status"] == "task_assigned":
            task = resp["task"]
            # Simulate work
            # time.sleep(0.1)
            # Complete task
            self.service._handle_task_complete({
                "worker_id": self.worker_id,
                "job_id": task["job_id"],
                "stage_id": task["stage_id"],
                "result": {"success": True, "output": f"processed {task['stage_name']}"}
            })
            return True
        return False

class TestWorkflowIntegration(unittest.TestCase):
    def setUp(self):
        import tempfile
        import shutil
        self.test_dir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.test_dir, "state")
        self.state = AegisState(self.state_path, os.path.abspath("QIHSE/qihse/libqihse.so"))
        with patch('framewerx.aegis_lab.orchestrator.service.IPCServer'):
            self.orchestrator = OrchestratorService(self.state)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_full_workflow(self):
        # 1. Register workers
        worker1 = MockWorker(self.orchestrator, "w1", "npu")
        worker2 = MockWorker(self.orchestrator, "w2", "cpu")
        worker1.register()
        worker2.register()
        
        # 2. Submit a job
        job_id = self.orchestrator.submit_job("test-proj", "ablation")
        
        # 3. Simulate workers picking up and completing all stages
        # The job has 4 stages: intake, probe, atom_extract, atom_clean
        max_iterations = 10
        stages_completed = 0
        for _ in range(max_iterations):
            if worker1.do_work():
                stages_completed += 1
            if worker2.do_work():
                stages_completed += 1
            
            job_status = self.orchestrator.get_job_status(job_id)
            if job_status["status"] == "succeeded":
                break
        
        self.assertEqual(job_status["status"], "succeeded")
        self.assertEqual(len(job_status["stages"]), 4)
        for stage in job_status["stages"]:
            self.assertEqual(stage["status"], "succeeded")

import os
if __name__ == "__main__":
    unittest.main()
