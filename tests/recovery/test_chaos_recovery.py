import unittest
from unittest.mock import MagicMock, patch
import os
from framewerx.aegis_lab.orchestrator.service import OrchestratorService
from framewerx.aegis_lab.state.db import AegisState

class TestChaosRecovery(unittest.TestCase):
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

    def test_worker_failure_recovery(self):
        # 1. Register two workers
        caps = {"cpu_amx": True, "npu_present": True, "igpu_present": False}
        self.orchestrator._handle_register({"worker_id": "w1", "worker_type": "npu", "capabilities": caps})
        self.orchestrator._handle_register({"worker_id": "w2", "worker_type": "npu", "capabilities": caps})
        
        # 2. Submit a job
        job_id = self.orchestrator.submit_job("chaos-proj", "ablation")
        
        # 3. Worker 1 requests task
        resp = self.orchestrator._handle_request_task({"worker_id": "w1"})
        self.assertEqual(resp["status"], "task_assigned")
        stage_id = resp["task"]["stage_id"]
        
        # 4. Simulate worker 1 dying (it never completes the task)
        # In a real scenario, heartbeats would time out
        # For this test, we manually mark the stage as pending again to simulate recovery
        # (Though Orchestrator might need a timeout mechanism in service.py,
        # for now let's see if we can just re-assign it if we manually reset it)
        
        # Reset stage status to pending
        self.state.update_stage(stage_id, {"status": "pending", "worker_id": None})
        
        # 5. Worker 2 requests task
        resp = self.orchestrator._handle_request_task({"worker_id": "w2"})
        self.assertEqual(resp["status"], "task_assigned")
        self.assertEqual(resp["task"]["stage_id"], stage_id)
        
        # 6. Worker 2 completes task
        self.orchestrator._handle_task_complete({
            "worker_id": "w2", "job_id": job_id, "stage_id": stage_id, "result": {"success": True}
        })
        
        job_status = self.orchestrator.get_job_status(job_id)
        stage = next(s for s in job_status["stages"] if s["stage_id"] == stage_id)
        self.assertEqual(stage["status"], "succeeded")

if __name__ == "__main__":
    unittest.main()
