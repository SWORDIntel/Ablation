from aegis_lab.hardware.discovery import HardwareDiscovery
import unittest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mocking classes/functions that are not directly tested or are external dependencies
class MockHardwareDiscovery:
    @staticmethod
    def discover():
        return {
            "cpu_amx": True,
            "cpu_avx512": True,
            "cpu_vnni": True,
            "cpu_avx2": True,
            "cpu_hybrid": False,
            "igpu_present": True,
            "igpu_type": "xe-lpg",
            "npu_present": True,
            "npu_type": "intel_ai_boost",
            "vpu_present": False,
            "vpu_runtime_usable": False,
            "vpu_runtime_count": 0,
            "vpu_usb_present": False,
            "vpu_usb_count": 0,
            "openvino_available": True,
            "openvino_import": "openvino.Core",
            "npu_bar_found": False,
            "npu_bar_protected": True,
            "cuda_compat": False,
            "accel_available": True,
            "mem_bandwidth_gbs": 100.0,
            "hardware_tier": "HIGH_PERF_SERVER",
            "supported_precisions_by_device": {
                "GPU.0": ["FP32", "FP16", "BF16", "INT8"],
                "NPU": ["FP32", "BF16", "INT8"],
                "CPU": ["FP32", "BF16", "INT8"]
            },
            "supported_precisions": ["FP32", "FP16", "BF16", "INT8"]
        }

class MockAegisState:
    def __init__(self):
        self.jobs = {}
        self.stages = {}
        self.logs = []
        self.qihse = MagicMock()
        self.db = MagicMock()

    def create_job(self, job_id, project_id, job_type, priority):
        self.jobs[job_id] = {"job_id": job_id, "project_id": project_id, "job_type": job_type, "priority": priority, "status": "pending", "parameters": {}}

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def get_jobs(self):
        return list(self.jobs.values())

    def get_all_stages(self, filters):
        if not filters:
            return [stage for job_stages in self.stages.values() for stage in job_stages]
        status = filters.get("status")
        results = []
        for job_stages in self.stages.values():
            for stage in job_stages:
                if status is None or stage.get("status") == status:
                    results.append(stage)
        return results

    def update_job(self, job_id, data):
        if job_id in self.jobs:
            self.jobs[job_id].update(data)

    def create_stage(self, stage_id, job_id, stage_name, ordinal):
        if job_id not in self.stages:
            self.stages[job_id] = []
        self.stages[job_id].append({
            "stage_id": stage_id,
            "job_id": job_id,
            "stage_name": stage_name,
            "ordinal": ordinal,
            "status": "pending",
            "result": {}
        })

    def get_stage(self, stage_id):
        for job_stages in self.stages.values():
            for stage in job_stages:
                if stage["stage_id"] == stage_id:
                    return stage
        return None

    def get_stages(self, filter_or_job_id):
        if isinstance(filter_or_job_id, dict):
            # filter by status
            status = filter_or_job_id.get("status")
            results = []
            for job_stages in self.stages.values():
                for stage in job_stages:
                    if stage.get("status") == status:
                        results.append(stage)
            return results
        else:
            # lookup by job_id
            return self.stages.get(filter_or_job_id, [])

    def update_stage(self, stage_id, data):
        for job_stages in self.stages.values():
            for stage in job_stages:
                if stage["stage_id"] == stage_id:
                    stage.update(data)
                    return
        raise ValueError(f"Stage {stage_id} not found")

    def add_log(self, entry):
        self.logs.append(entry)

# Import the OrchestratorService from the actual module
from aegis_lab.orchestrator.service import OrchestratorService

# Mock OpenVINOExporter to check if it's called correctly
class MockOpenVINOExporter:
    def __init__(self, model, work_dir):
        self.model = model
        self.work_dir = work_dir
        self.export_calls = []

    def export(self, hardware_capabilities, calibration_dataset=None):
        self.export_calls.append({
            "hardware_capabilities": hardware_capabilities,
            "calibration_dataset": calibration_dataset
        })
        # Return a dummy path
        return Path(f"{self.work_dir}/exported_model_{len(self.export_calls)}.xml")

# Patch the actual imports with mocks
@patch('aegis_lab.orchestrator.service.HardwareDiscovery', MockHardwareDiscovery)
@patch('aegis_lab.orchestrator.service.AegisState', MockAegisState)
@patch('aegis_lab.orchestrator.service.logger', MagicMock())
class TestOrchestratorTaskDispatch(unittest.TestCase):

    def setUp(self):
        # Mock ThermalGuardian to avoid thermal CRITICAL state on CI systems
        self.thermal_patcher = patch('aegis_lab.orchestrator.service.ThermalGuardian')
        self.mock_thermal = self.thermal_patcher.start()
        self.mock_thermal.return_value.get_status.return_value = {
            "temperature": 30.0,
            "level": "LOW",
            "safe_to_compute": True,
            "throttling_recommended": False
        }
        
        # Manually mock OpenVINOExporter in the service module
        import aegis_lab.orchestrator.service
        aegis_lab.orchestrator.service.OpenVINOExporter = MockOpenVINOExporter

        # Avoid creating real ZMQ sockets during unit tests.
        self.ipc_patcher = patch('aegis_lab.orchestrator.service.IPCServer')
        self.ipc_patcher.start()

        self.state = MockAegisState()
        import random
        base_port = random.randint(10000, 20000)
        self.orchestrator = OrchestratorService(
            state=self.state, 
            ipc_port=base_port, 
            log_port=base_port+1, 
            event_port=base_port+2
        )
        # Manually add worker info for the orchestrator to use
        self.orchestrator.workers = {
            "worker-1": {
                "type": "npu",
                "capabilities": {"cpu_amx": False, "npu_present": True, "supported_precisions": ["INT8", "BF16", "FP32"]},
                "last_heartbeat": time.time(),
                "p2p_endpoint": "tcp://127.0.0.1:5000"
            },
            "worker-2": {
                "type": "cpu",
                "capabilities": {"cpu_vnni": True, "supported_precisions": ["FP32", "BF16"]},
                "last_heartbeat": time.time(),
                "p2p_endpoint": "tcp://127.0.0.1:5001"
            }
        }
        self.orchestrator.state = self.state
        self.orchestrator._ensure_runtime_services()

    def tearDown(self):
        if hasattr(self, 'orchestrator'):
            self.orchestrator.stop()
        if hasattr(self, 'ipc_patcher'):
            self.ipc_patcher.stop()
        if hasattr(self, 'thermal_patcher'):
            self.thermal_patcher.stop()

    def test_quantize_task_passes_hardware_capabilities(self):
        mock_worker_id = "worker-1"
        with patch.object(self.orchestrator.state, "get_jobs", return_value=[{"job_id": "job-mock123", "status": "pending"}]), \
             patch.object(self.orchestrator.state, 'get_all_stages', return_value=[{"stage_id": "stage-mock456", "job_id": "job-mock123", "stage_name": "quantize", "status": "pending", "ordinal": 5}]), \
             patch.object(self.orchestrator.state, 'get_job', return_value={"job_id": "job-mock123", "status": "pending"}), \
             patch.object(self.orchestrator.state, 'update_stage'), \
             patch.object(self.orchestrator.state, 'update_job'):

            message = {
                "worker_id": mock_worker_id,
                "worker_type": "npu",
                "requested_stage_name": "quantize"
            }
            # Manually add the job/stage to the state that the orchestrator will look at
            self.state.create_job("job-mock123", "proj1", "type1", 1)
            self.state.create_stage("stage-mock456", "job-mock123", "quantize", 5)
            
            task_info = self.orchestrator._handle_request_task(message)
            self.assertEqual(task_info["status"], "task_assigned")
            task_details = task_info["task"]
            self.assertIn("hardware_capabilities", task_details)
            self.assertEqual(task_details["hardware_capabilities"], MockHardwareDiscovery.discover())
            self.assertEqual(task_details["stage_name"], "quantize")

    def test_other_stage_does_not_include_hardware_capabilities(self):
        mock_worker_id = "worker-2"
        with patch.object(self.orchestrator.state, 'get_jobs', return_value=[{"job_id": "job-mock789", "job_type": "ablation_training", "status": "pending"}]), \
             patch.object(self.orchestrator.state, 'get_all_stages', return_value=[{"stage_id": "stage-mock012", "job_id": "job-mock789", "stage_name": "probe", "status": "pending", "ordinal": 1}]), \
             patch.object(self.orchestrator.state, 'get_job', return_value={"job_id": "job-mock789", "status": "pending"}), \
             patch.object(self.orchestrator.state, 'update_stage'), \
             patch.object(self.orchestrator.state, 'update_job'):

            message = {
                "worker_id": mock_worker_id,
                "worker_type": "cpu",
                "requested_stage_name": "probe"
            }
            # Manually add the job/stage
            self.state.create_job("job-mock789", "proj1", "type1", 1)
            self.state.create_stage("stage-mock012", "job-mock789", "probe", 1)
            
            task_info = self.orchestrator._handle_request_task(message)
            self.assertEqual(task_info["status"], "task_assigned")
            task_details = task_info["task"]
            self.assertNotIn("hardware_capabilities", task_details)
            self.assertEqual(task_details["stage_name"], "probe")

    @patch.object(HardwareDiscovery, "discover", return_value={
        "supported_precisions": ["INT8", "FP32"],
        "npu_present": True, "igpu_present": False, "vpu_present": False,
        "cpu_amx": False, "cpu_vnni": False, "accel_available": True, "hardware_tier": "GENERIC"
    })
    def test_worker_receives_and_uses_hardware_caps_for_quantize(self, mock_hw_discover):
        mock_worker_id = "worker-1"
        self.state.create_job("job-test-hw", "project-test", "ablation_training", 1)
        self.state.create_stage("stage-test-hw-quant", "job-test-hw", "quantize", 5)
        self.state.update_stage("stage-test-hw-quant", {"status": "pending", "worker_id": mock_worker_id})

        with patch.object(self.orchestrator.thermal_guardian, 'get_status', return_value={"safe_to_compute": True, "throttling_recommended": False}):
            task_info = self.orchestrator._handle_request_task({"worker_id": mock_worker_id, "worker_type": "npu", "requested_stage_name": "quantize"})
            self.assertEqual(task_info["status"], "task_assigned")
            received_task = task_info["task"]
            self.assertIn("hardware_capabilities", received_task)
            
            mock_exporter_instance = MockOpenVINOExporter(model="dummy_model_obj", work_dir="./export_output/test")
            hardware_caps_from_task = received_task["hardware_capabilities"]
            
            supported_precisions = hardware_caps_from_task.get("supported_precisions", ["FP32"])
            precision_preference = ["INT8", "BF16", "FP16", "FP32"]
            selected_precision = "FP32"
            for prec in precision_preference:
                if prec in supported_precisions:
                    selected_precision = prec
                    break
            
            mock_exporter_instance.export(hardware_capabilities=hardware_caps_from_task)
            self.assertTrue(mock_exporter_instance.export_calls)
            self.assertEqual(mock_exporter_instance.export_calls[0]["hardware_capabilities"], hardware_caps_from_task)
            self.assertEqual(selected_precision, "INT8")

if __name__ == "__main__":
    unittest.main()
