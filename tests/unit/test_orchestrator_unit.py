import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from aegis_lab.orchestrator.service import OrchestratorService


class TestOrchestratorUnit(unittest.TestCase):
    def setUp(self):
        self.mock_state = MagicMock()
        self.mock_state.get_jobs.return_value = []
        with patch("aegis_lab.orchestrator.service.IPCServer"):
            self.service = OrchestratorService(self.mock_state)

    def test_submit_job(self):
        project_id = "test-proj"
        job_type = "ablation"

        job_id = self.service.submit_job(project_id, job_type)

        self.assertTrue(job_id.startswith("job-"))
        self.mock_state.create_job.assert_called_once()
        self.assertEqual(self.mock_state.create_stage.call_count, 4)

    def test_handle_register(self):
        message = {
            "worker_id": "worker-1",
            "worker_type": "npu",
            "capabilities": {"cache_size": 128},
        }
        response = self.service._handle_register(message)

        self.assertEqual(response["status"], "ok")
        self.assertIn("worker-1", self.service.workers)
        self.assertEqual(self.service.workers["worker-1"]["type"], "npu")

    def test_handle_heartbeat(self):
        self.service.workers["worker-1"] = {"last_heartbeat": 0}
        message = {"worker_id": "worker-1"}

        response = self.service._handle_heartbeat(message)

        self.assertEqual(response["status"], "ok")
        self.assertGreater(self.service.workers["worker-1"]["last_heartbeat"], 0)

    @patch("aegis_lab.orchestrator.service.SchedulerEngine")
    def test_handle_request_task(self, mock_scheduler_cls):
        self.service.workers["worker-1"] = {
            "type": "npu",
            "capabilities": {"can_npu": True},
            "status": "ready",
        }

        self.mock_state.get_jobs.return_value = [{"job_id": "job-1", "status": "pending"}]
        self.mock_state.get_stages.return_value = [
            {"stage_id": "s1", "stage_name": "intake", "status": "pending", "ordinal": 0}
        ]

        mock_scheduler_inst = mock_scheduler_cls.return_value
        
        # Setup mock state jobs and stages
        self.mock_state.get_jobs.return_value = [
            {"job_id": "job-1", "status": "pending"}
        ]
        def mock_get_stages(filter_arg):
            if isinstance(filter_arg, dict):
                # We need to return a list that will satisfy the orchestrator's logic.
                return [{"stage_id": "s1", "job_id": "job-1", "stage_name": "intake", "status": "pending", "ordinal": 0}]
            elif isinstance(filter_arg, str):
                return [{"stage_id": "s1", "job_id": "job-1", "stage_name": "intake", "status": "pending", "ordinal": 0}]
            return []
            
        self.mock_state.get_stages.side_effect = mock_get_stages
        
        # Setup mock scheduler
        mock_scheduler_inst = MockScheduler.return_value
        mock_scheduler_inst.determine_placement.return_value = ["NPU_0"]

        message = {"worker_id": "worker-1"}
        response = self.service._handle_request_task(message)

        self.assertEqual(response["status"], "task_assigned")
        self.assertEqual(response["task"]["stage_name"], "intake")
        self.mock_state.update_stage.assert_called_with("s1", {"status": "running", "worker_id": "worker-1"})

    def test_handle_task_complete_success(self):
        self.mock_state.get_stages.return_value = [{"stage_id": "s1", "status": "succeeded"}]

        message = {
            "worker_id": "worker-1",
            "job_id": "job-1",
            "stage_id": "s1",
            "result": {"success": True},
        }

        response = self.service._handle_task_complete(message)

        self.assertEqual(response["status"], "ok")
        self.mock_state.update_stage.assert_called_with("s1", {"status": "succeeded", "result": {"success": True}})
        self.mock_state.update_job.assert_called_with("job-1", {"status": "succeeded"})

    def test_get_job_status_includes_progress(self):
        self.mock_state.get_jobs.return_value = [{"job_id": "job-1", "status": "running"}]
        self.mock_state.get_stages.return_value = [
            {"stage_id": "s0", "stage_name": "intake", "status": "succeeded", "ordinal": 0},
            {"stage_id": "s1", "stage_name": "probe", "status": "running", "ordinal": 1},
            {"stage_id": "s2", "stage_name": "verify", "status": "pending", "ordinal": 2},
        ]

        result = self.service.get_job_status("job-1")

        self.assertIn("progress", result)
        self.assertEqual(result["progress"]["completed"], 1)
        self.assertEqual(result["progress"]["total"], 3)
        self.assertEqual(result["progress"]["current_stage"], "probe")

    def test_list_available_models_returns_discovered_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                os.makedirs("models", exist_ok=True)
                open("models/demo.gguf", "w", encoding="utf-8").close()
                os.makedirs("models/transformer", exist_ok=True)
                open("models/transformer/config.json", "w", encoding="utf-8").close()

                result = self.service.list_available_models()

                self.assertEqual(result["status"], "ok")
                self.assertEqual(len(result["models"]), 2)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
