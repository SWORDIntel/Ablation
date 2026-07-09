import uuid
import time
import logging
import os
from typing import Dict, List, Any, Optional
from framewerx.aegis_lab.state.db import AegisState
from framewerx.aegis_lab.orchestrator.ipc import IPCServer, LogServer, EventPublisher
from framewerx.aegis_lab.hardware.thermal import ThermalGuardian
from framewerx.aegis_lab.hardware.discovery import HardwareDiscovery
from framewerx.aegis_lab.scheduler.engine import SchedulerEngine
from framewerx.aegis_lab.hardware.telemetry import LevelZeroTelemetry
from framewerx.aegis_lab.orchestrator.optimizer import HardwareOptimizer

from framewerx.aegis_lab.evaluation.leaderboard import LeaderboardManager
from framewerx.aegis_lab.evaluation.elo_judge import EloJudge
from framewerx.aegis_lab.evaluation.hallucination import HallucinationDetector
from framewerx.aegis_lab.evaluation.capability_drift import CapabilityDriftAnalyzer
from framewerx.aegis_lab.evaluation.hardware_efficiency import EfficiencyRanker

from framewerx.aegis_lab.hardware.vault import ModelVault
from framewerx.aegis_lab.evaluation.consensus import ConsensusJudge
from framewerx.aegis_lab.verification.proofs import ZKAblationProof

logger = logging.getLogger(__name__)

AEGIS_AUTH_TOKEN = os.getenv("AEGIS_AUTH_TOKEN", "aegis-secret-token-2024")
ENABLE_LEVEL_ZERO = os.getenv("AEGIS_ENABLE_LEVEL_ZERO", "").lower() in {"1", "true", "yes"}


class _NullTelemetry:
    def get_metrics(self) -> List[Dict[str, Any]]:
        return []

class OrchestratorService:
    def __init__(self, state: AegisState, ipc_port: int = 5555, log_port: int = 5556, event_port: Optional[int] = None):
        self.state = state
        self.thermal_guardian = ThermalGuardian()
        self.telemetry = None
        self.vault = ModelVault()
        self.consensus_judge = ConsensusJudge()
        self.zk_prover = ZKAblationProof(self.state.qihse)
        self.optimizer = None
        self.leaderboard = LeaderboardManager(self.state)
        self.workers: Dict[str, Dict[str, Any]] = {} # worker_id -> info
        self.log_port = log_port
        self.event_port = event_port if event_port is not None else log_port + 1

        self.ipc = IPCServer(port=ipc_port, auth_token=AEGIS_AUTH_TOKEN)
        self.log_server = None
        self.event_publisher = None
        
        self.ipc.register_handler("register", self._handle_register)
        self.ipc.register_handler("heartbeat", self._handle_heartbeat)
        self.ipc.register_handler("request_task", self._handle_request_task)
        self.ipc.register_handler("task_complete", self._handle_task_complete)
        self.ipc.register_handler("approve_stage", self._handle_approve_stage)
        self.ipc.register_handler("submit_job", self._handle_ipc_submit_job)
        self.ipc.register_handler("list_jobs", lambda msg: self.list_jobs())
        self.ipc.register_handler("get_job_status", lambda msg: self.get_job_status(msg["job_id"]))
        self.ipc.register_handler("get_leaderboard", lambda msg: self.leaderboard.get_leaderboard())
        self.ipc.register_handler("optimize_uma", self._handle_optimize_uma)
        self.ipc.register_handler("discover_peer", self._handle_discover_peer)
        self.ipc.register_handler("get_worker_status", self._handle_get_worker_status)
        self.ipc.register_handler("get_sitrep", lambda msg: self.get_sitrep())

    def _ensure_runtime_services(self):
        if self.telemetry is None:
            self.telemetry = LevelZeroTelemetry() if ENABLE_LEVEL_ZERO else _NullTelemetry()
        if self.optimizer is None:
            self.optimizer = HardwareOptimizer(self.state, self.telemetry)

    def start(self):
        self._ensure_runtime_services()
        if self.log_server is None:
            self.log_server = LogServer(port=self.log_port)
        if self.event_publisher is None:
            self.event_publisher = EventPublisher(port=self.event_port)
        self.ipc.start()
        self.log_server.start(self._handle_log)
        self.optimizer.start()
        logger.info("Orchestrator Service started.")

    def stop(self):
        if self.optimizer is not None:
            self.optimizer.stop()
        if self.log_server is not None:
            self.log_server.stop()
        self.ipc.stop()
        if self.event_publisher is not None:
            self.event_publisher.stop()

    def _handle_get_worker_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        worker_type = message.get("worker_type")
        worker_id = message.get("worker_id")
        
        results = []
        for wid, info in self.workers.items():
            if (worker_id and wid == worker_id) or (worker_type and info["type"] == worker_type) or (not worker_id and not worker_type):
                # Only return workers seen in the last 30 seconds
                if time.time() - info["last_heartbeat"] < 30:
                    results.append({
                        "worker_id": wid,
                        "type": info["type"],
                        "status": info["status"],
                        "telemetry": info.get("telemetry", {})
                    })
        
        return {"status": "ok", "workers": results}

    def _handle_discover_peer(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the p2p_endpoint for a given worker_id or worker_type."""
        target_id = message.get("target_worker_id")
        target_type = message.get("target_worker_type")
        
        if target_id:
            worker = self.workers.get(target_id)
            if worker and worker.get("p2p_endpoint"):
                return {"status": "ok", "p2p_endpoint": worker["p2p_endpoint"]}
        elif target_type:
            # Find first worker of this type that has a p2p_endpoint
            for wid, info in self.workers.items():
                if info["type"] == target_type and info.get("p2p_endpoint"):
                    return {"status": "ok", "worker_id": wid, "p2p_endpoint": info["p2p_endpoint"]}
        
        return {"status": "error", "error": "Peer not found or no P2P endpoint available"}

    def _handle_optimize_uma(self, message: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Manual UMA optimization requested via SITREP.")
        self._ensure_runtime_services()
        self.optimizer._optimize_memory_layout()
        return {"status": "ok"}

    def _handle_log(self, log_entry: Dict[str, Any]):
        """Callback for log messages received over ZMQ."""
        self.state.add_log(log_entry)

    def _handle_ipc_submit_job(self, message: Dict[str, Any]) -> Dict[str, Any]:
        job_id = self.submit_job(
            message["project_id"], 
            message["job_type"],
            parameters=message.get("parameters", {})
        )
        return {"status": "ok", "job_id": job_id}

    def _handle_approve_stage(self, message: Dict[str, Any]) -> Dict[str, Any]:
        stage_id = message["stage_id"]
        logger.info(f"Operator approved stage: {stage_id}")
        self.state.update_stage(stage_id, {"status": "pending", "approved": True})
        return {"status": "ok"}

    def _handle_register(self, message: Dict[str, Any]) -> Dict[str, Any]:
        worker_id = message["worker_id"]
        self.workers[worker_id] = {
            "type": message["worker_type"],
            "capabilities": message["capabilities"],
            "p2p_endpoint": message.get("p2p_endpoint"),
            "last_heartbeat": time.time(),
            "status": "ready"
        }
        logger.info(f"Worker registered: {worker_id} ({message['worker_type']}) at P2P={message.get('p2p_endpoint')}")
        return {"status": "ok"}

    def _handle_heartbeat(self, message: Dict[str, Any]) -> Dict[str, Any]:
        worker_id = message["worker_id"]
        if worker_id in self.workers:
            self.workers[worker_id]["last_heartbeat"] = time.time()
            if "telemetry" in message:
                self.workers[worker_id]["telemetry"] = message["telemetry"]
        return {"status": "ok"}

    def _handle_request_task(self, message: Dict[str, Any]) -> Dict[str, Any]:
        worker_id = message["worker_id"]
        worker_info = self.workers.get(worker_id)
        if not worker_info:
            return {"status": "error", "error": "Worker not registered"}

        # Get current thermal status
        thermal_status = self.thermal_guardian.get_status()
        
        scheduler = SchedulerEngine(worker_info["capabilities"], thermal_status)

        # Batch fetch all pending stages and all jobs
        all_pending_stages = self.state.get_all_stages({"status": "pending"})
        all_jobs = {job["job_id"]: job for job in self.state.get_jobs()}

        pending_stages = []
        for stage in all_pending_stages:
            job_id = stage.get("job_id")
            if job_id in all_jobs:
                pending_stage = dict(stage)
                pending_stages.append(pending_stage)

        pending_stages.sort(key=lambda x: x["ordinal"])
        
        for stage in pending_stages:
            job_id = stage["job_id"]
            job = all_jobs.get(job_id)
            if not job or job["status"] == "failed":
                continue

            placement = scheduler.determine_placement(stage["stage_name"], {}, worker_id=worker_id)
            
            if self._worker_matches_placement(worker_info, placement):
                # Assign task!
                self.state.update_stage(stage["stage_id"], {"status": "running", "worker_id": worker_id})
                self.state.update_job(job_id, {"current_stage_id": stage["stage_id"], "status": "running"})
                
                task_data = {
                    "job_id": job_id,
                    "stage_id": stage["stage_id"],
                    "stage_name": stage["stage_name"]
                }
                
                # Milestone 4: Pass hardware capabilities to worker for quantization decisions
                if stage["stage_name"] == "quantize":
                    task_data["hardware_capabilities"] = HardwareDiscovery.discover()
                
                return {
                    "status": "task_assigned",
                    "task": task_data
                }
        
        return {"status": "no_work"}

    def _worker_matches_placement(self, worker_info: Dict[str, Any], placement: List[str]) -> bool:
        worker_type = worker_info["type"].lower()
        capabilities = worker_info.get("capabilities", {})

        if worker_type == "npu" and (any(device.startswith("NPU") for device in placement) or "CPU" in placement):
            return True
        if worker_type == "vpu" and any(device.startswith("VPU") for device in placement):
            return True
        if worker_type == "igpu" and any(device.startswith("iGPU") or device.startswith("CUDA") for device in placement):
            return True
        if worker_type == "cpu" and any(device.startswith("CPU") for device in placement):
            return True

        if any(device.startswith("CPU_AMX") for device in placement) and capabilities.get("cpu_amx"):
            return True
        if any(device.startswith("CPU_AVX512") for device in placement) and capabilities.get("cpu_avx512"):
            return True
        if any(device.startswith("CPU_VNNI") for device in placement) and capabilities.get("cpu_vnni"):
            return True
        if any(device.startswith("CPU_AVX2") for device in placement):
            return True if worker_type == "cpu" else any(
                capabilities.get(flag) for flag in ("cpu_amx", "cpu_avx512", "cpu_vnni")
            )
        if any(device == "CPU" for device in placement):
            return any(capabilities.get(flag) for flag in ("cpu_amx", "cpu_avx512", "cpu_vnni")) or worker_type == "cpu"

        return False

    def _handle_task_complete(self, message: Dict[str, Any]) -> Dict[str, Any]:
        worker_id = message["worker_id"]
        stage_id = message["stage_id"]
        job_id = message["job_id"]
        result = message["result"]
        
        # Identify stage name for specific logic
        stages = self.state.get_stages(job_id)
        stage = next((s for s in stages if s["stage_id"] == stage_id), None)
        stage_name = stage.get("stage_name", "unknown") if stage else "unknown"

        status = "succeeded" if result.get("success", True) else "failed"
        
        # Update Scheduler stats for Fail-Fast
        if status == "succeeded":
            SchedulerEngine.report_success(worker_id)
        else:
            SchedulerEngine.report_failure(worker_id)

        # Automated Red-Team Auto-Eval (Idea 6)
        if stage_name == "adversarial_eval":
            vulnerability_score = result.get("vulnerability_score", 0.0)
            if vulnerability_score > 0.7:
                logger.warning(f"CRITICAL: Adversarial vulnerability threshold exceeded ({vulnerability_score}) for job {job_id}.")
                logger.warning(f"Triggering automated rollback for job {job_id}.")
                status = "failed"
                # Trigger rollback event in QIHSE state
                rollback_event = {
                    "job_id": job_id,
                    "vulnerability_score": vulnerability_score,
                    "timestamp": time.time(),
                    "reason": "adversarial_vulnerability_threshold_exceeded"
                }
                self.state.db.upsert("rollback_events", "job_id", job_id, rollback_event)

        logger.info(f"Task {status} ({stage_name}) from {worker_id}: {stage_id}")
        
        self.state.update_stage(stage_id, {"status": status, "result": result})
        
        # Check if job is complete
        updated_stages = self.state.get_stages(job_id)
        if all(s["status"] == "succeeded" for s in updated_stages):
            self.state.update_job(job_id, {"status": "succeeded"})
            self._run_final_evaluations(job_id)
        elif any(s["status"] == "failed" for s in updated_stages):
            self.state.update_job(job_id, {"status": "failed"})
            
        return {"status": "ok"}

    def _run_final_evaluations(self, job_id: str):
        """
        Runs the full Method 1-5 suite and updates the Leaderboard.
        """
        logger.info(f"Running final ranking evaluations for job: {job_id}")
        self._ensure_runtime_services()
        
        stages = self.state.get_stages(job_id)

        # 1. Elo Judge — pairwise comparison of baseline vs ablated responses
        elo = EloJudge()
        baseline_resp = ""
        ablated_resp = ""
        eval_context = ""
        for s in stages:
            result = s.get("result") or {}
            if s.get("stage_name") == "probe":
                baseline_resp = result.get("baseline_response", "")
                eval_context = result.get("context", "")
            elif s.get("stage_name") == "verify":
                ablated_resp = result.get("ablated_response", "")
        if baseline_resp and ablated_resp:
            ablated_won = elo.evaluate_pairwise(ablated_resp, baseline_resp, eval_context)
            new_r_ablated, _ = elo.calculate_new_ratings(1200.0, 1200.0, ablated_won)
            self.leaderboard.record_score(job_id, "elo_rating", new_r_ablated)
        else:
            logger.warning(f"Job {job_id}: insufficient data for Elo evaluation, defaulting to baseline.")
            self.leaderboard.record_score(job_id, "elo_rating", 1200.0)

        # 2. Adversarial Robustness (Method 2)
        adv_stage = next((s for s in stages if s.get("stage_name") == "adversarial_eval"), None)
        if adv_stage and adv_stage.get("result"):
            score = adv_stage["result"].get("robustness_score", 0.0)
            self.leaderboard.record_score(job_id, "adversarial_robustness", score)

        # 3. Capability Drift (Method 3)
        drift = CapabilityDriftAnalyzer(self.state)
        self.leaderboard.record_score(job_id, "ais_score", drift.calculate_ais(job_id))

        # 4. Hardware Efficiency (Method 4)
        efficiency = EfficiencyRanker(self.telemetry)
        tokens_per_sec = 0.0
        for s in stages:
            result = s.get("result") or {}
            if "tokens_per_sec" in result:
                tokens_per_sec = result["tokens_per_sec"]
                break
            if "throughput" in result:
                tokens_per_sec = result["throughput"]
                break
        if tokens_per_sec <= 0:
            logger.warning(f"Job {job_id}: no throughput data found, using 0 for efficiency score.")
        self.leaderboard.record_score(job_id, "perf_per_watt", efficiency.compute_efficiency_score(tokens_per_sec=tokens_per_sec))

        # 5. Hallucination Index (Method 5)
        detector = HallucinationDetector(self.state)
        model_output = ablated_resp or ""
        if not model_output:
            for s in stages:
                result = s.get("result") or {}
                if result.get("sample_output"):
                    model_output = result["sample_output"]
                    break
        self.leaderboard.record_score(job_id, "truthfulness", 1.0 - detector.compute_index(model_output, job_id))

    def submit_job(self, project_id: str, job_type: str, priority: int = 50, parameters: Dict[str, Any] = None) -> str:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        params = parameters or {}
        
        self.state.create_job(job_id, project_id, job_type, priority)
        # Store parameters
        if params:
            self.state.update_job(job_id, {"parameters": params})
        
        if job_type == "ablation_training":
            stages = [
                "intake",
                "probe",
                "atom_extract",
                "atom_clean",
                "adversarial_eval",
                "quantize",
                "verify",
                "promote",
            ]
        else:
            stages = ["intake", "probe", "atom_extract", "atom_clean"]
        for i, stage_name in enumerate(stages):
            stage_id = f"{job_id}-s{i}"
            self.state.create_stage(stage_id, job_id, stage_name, i)
            
        return job_id

    def list_jobs(self) -> List[Dict[str, Any]]:
        return self.state.get_jobs()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        jobs = self.state.get_jobs()
        job = next((j for j in jobs if j["job_id"] == job_id), None)
        if not job:
            return {"error": "Job not found"}
            
        stages = self.state.get_stages(job_id)
        job["stages"] = stages
        return job

    def get_sitrep(self) -> Dict[str, Any]:
        self._ensure_runtime_services()
        return {
            "status": "ok",
            "workers": self._handle_get_worker_status({}).get("workers", []),
            "thermal": self.thermal_guardian.get_status(),
            "telemetry": self.telemetry.get_metrics(),
            "hardware": HardwareDiscovery.discover(),
        }
