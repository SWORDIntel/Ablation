import logging
import uuid
import threading
import queue
import time
from typing import Dict, Any, List
from pathlib import Path

from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.intake.fingerprint import ModelFingerprint
from aegis_lab.probing.capture import CAREActivationCapturer
from aegis_lab.editing.delta_builder import DeltaBuilder
from aegis_lab.verification.authority import SemanticAuthority
from aegis_lab.editing.runtime import (
    ExecutionMode,
    resolve_execution_contract,
    stable_json_hash,
)

logger = logging.getLogger(__name__)

class AblationPipeline:
    """
    Formal pipeline for handling the transition between Intake, Probing, 
    Extraction, and Validation in the model ablation process.
    """
    
    def __init__(self, state: AegisState, artifact_store: ArtifactStore, execution_mode: str = ExecutionMode.FALLBACK.value):
        self.state = state
        self.artifact_store = artifact_store
        self.capturer = CAREActivationCapturer(state, artifact_store)
        self.authority = SemanticAuthority(default_mode=execution_mode)
        self.execution_mode = execution_mode
        self.work_root = Path("/tmp/aegis_ablation")
        self.work_root.mkdir(parents=True, exist_ok=True)

        # Telemetry Subsystem initialization
        self.telemetry_queue = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self.telemetry_thread = threading.Thread(target=self._telemetry_committer, daemon=True)
        self.telemetry_thread.start()

    def _telemetry_committer(self):
        """Background daemon isolating AegisState I/O from the execution thread with write batching."""
        batch = []
        last_commit_time = time.time()

        while not self._stop_event.is_set() or not self.telemetry_queue.empty():
            try:
                # Wait for items but unblock every 0.1s to check for flush
                payload = self.telemetry_queue.get(timeout=0.1)
                batch.append(payload)
                self.telemetry_queue.task_done()
            except queue.Empty:
                pass

            now = time.time()
            if batch and (len(batch) >= 50 or now - last_commit_time >= 0.5):
                # Group by stage_id to optimize writes
                updates_by_stage = {}
                for item in batch:
                    stage_id = item.get("stage_id")
                    data = item.get("data")
                    if stage_id not in updates_by_stage:
                        updates_by_stage[stage_id] = []
                    updates_by_stage[stage_id].append(data)

                for stage_id, new_telemetry in updates_by_stage.items():
                    current_stage = self.state.get_stage(stage_id)
                    if current_stage:
                        current_telemetry = current_stage.get("telemetry", [])
                        current_telemetry.extend(new_telemetry)
                        self.state.update_stage(stage_id, {"telemetry": current_telemetry})

                batch.clear()
                last_commit_time = now

        # Final flush for any remaining batch elements
        if batch:
            updates_by_stage = {}
            for item in batch:
                stage_id = item.get("stage_id")
                data = item.get("data")
                if stage_id not in updates_by_stage:
                    updates_by_stage[stage_id] = []
                updates_by_stage[stage_id].append(data)
            for stage_id, new_telemetry in updates_by_stage.items():
                current_stage = self.state.get_stage(stage_id)
                if current_stage:
                    current_telemetry = current_stage.get("telemetry", [])
                    current_telemetry.extend(new_telemetry)
                    self.state.update_stage(stage_id, {"telemetry": current_telemetry})
            batch.clear()

    def stop(self):
        self._stop_event.set()
        self.telemetry_thread.join(timeout=2.0)

    def _fallback_fingerprint(self, model_path: str, error: str) -> Dict[str, Any]:
        contract = resolve_execution_contract(
            operation="model_fingerprint",
            requested_mode=self.execution_mode,
            native_available=False,
            reason=f"Model fingerprinting fell back because {error}",
            details={"model_path": model_path},
        )
        signature = stable_json_hash({"model_path": model_path, "error": error, "mode": contract.mode.value})
        return {
            "status": "ok",
            "analysis_mode": contract.mode.value,
            "execution_contract": contract.as_dict(),
            "architecture_family": "unknown",
            "topology_type": "Dense",
            "estimated_memory_bytes": 0,
            "supports_moe": False,
            "supports_multimodal": False,
            "model_signature": signature,
        }

    def run_full_ablation(self, 
                          model_path: str, 
                          layers: List[int], 
                          positive_dataset: str, 
                          negative_dataset: str,
                          project_id: str = "default_project") -> str:
        """
        Runs the full ablation pipeline.
        
        Transitions:
        1. Intake: Fingerprint the model and register job.
        2. Probing: Capture activations using CARE principles.
        3. Extraction: Generate delta tensors (Atoms).
        4. Validation: Verify the edit with semantic authority.
        """
        job_id = f"ablation-{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting formal AblationPipeline for job {job_id}")
        pipeline_contract = resolve_execution_contract(
            operation="ablation_pipeline",
            requested_mode=self.execution_mode,
            native_available=False,
            reason="Pipeline is operating in deterministic fallback mode for non-hardware environments.",
            details={"project_id": project_id, "model_path": model_path},
        )
        
        # 1. Intake
        logger.info("[PIPELINE] Stage 1: Intake")
        try:
            fingerprint = ModelFingerprint.analyze_model(model_path)
            fingerprint["analysis_mode"] = ExecutionMode.FALLBACK.value
            fingerprint["execution_contract"] = resolve_execution_contract(
                operation="model_fingerprint",
                requested_mode=ExecutionMode.FALLBACK.value,
                native_available=False,
                reason="Model fingerprinted from repository metadata without native runtime.",
                details={"model_path": model_path},
            ).as_dict()
        except Exception as exc:
            fingerprint = self._fallback_fingerprint(model_path, str(exc))
        self.state.create_job(job_id, project_id, "ablation_full")
        self.state.update_job(job_id, {"fingerprint": fingerprint, "status": "running", "execution_contract": pipeline_contract.as_dict()})
        
        # Create stages in DB for tracking
        stage_intake = f"{job_id}-s0"
        self.state.create_stage(stage_intake, job_id, "intake", 0)
        self.state.update_stage(stage_intake, {"status": "succeeded", "result": fingerprint})
        
        # 2. Probing
        logger.info("[PIPELINE] Stage 2: Probing")
        stage_probe = f"{job_id}-s1"
        self.state.create_stage(stage_probe, job_id, "probing", 1)
        self.state.update_stage(stage_probe, {"status": "running"})
        
        capture_result = self.capturer.capture_activations(
            job_id, model_path, layers, positive_dataset, negative_dataset
        )
        capture_contract = resolve_execution_contract(
            operation="activation_capture",
            requested_mode=self.execution_mode,
            native_available=False,
            reason="Activation capture is represented by a deterministic fallback artifact in this environment.",
            details={
                "model_path": model_path,
                "layers": layers,
                "positive_dataset": positive_dataset,
                "negative_dataset": negative_dataset,
            },
        )
        self.state.update_stage(
            stage_probe,
            {
                "status": "succeeded",
                "result": {
                    "capture": capture_result,
                    "execution_contract": capture_contract.as_dict(),
                },
            },
        )
        
        # 3. Extraction (Atom Generation)
        logger.info("[PIPELINE] Stage 3: Extraction")
        stage_extract = f"{job_id}-s2"
        self.state.create_stage(stage_extract, job_id, "extraction", 2)
        self.state.update_stage(stage_extract, {"status": "running"})
        
        builder = DeltaBuilder(self.artifact_store, self.work_root / job_id)

        # Telemetry callback for layer extraction
        def _layer_progress_callback(layer_idx, total_layers, layer_num):
            try:
                self.telemetry_queue.put_nowait({
                    "stage_id": stage_extract,
                    "data": {
                        "layer": layer_num,
                        "pct_complete": (layer_idx + 1) / total_layers * 100.0,
                        "status": "extracted"
                    }
                })
            except queue.Full:
                logger.warning("Telemetry queue full, dropping extraction progress tick.")

        edit_plan = {
            "type": "permanent",
            "layers": layers,
            "capture_info": capture_result,
            "execution_contract": pipeline_contract.as_dict(),
        }
        delta_hash = builder.generate_delta_tensors(edit_plan, progress_callback=_layer_progress_callback)
        
        # Register the atom
        atom_id = f"atom-{job_id}"
        self.state.register_atom({
            "atom_id": atom_id,
            "job_id": job_id,
            "content_hash": delta_hash,
            "layers": layers
        })
        self.state.update_stage(
            stage_extract,
            {
                "status": "succeeded",
                "result": {
                    "atom_id": atom_id,
                    "delta_hash": delta_hash,
                    "execution_contract": pipeline_contract.as_dict(),
                },
            },
        )
        
        # 4. Validation
        logger.info("[PIPELINE] Stage 4: Validation")
        stage_valid = f"{job_id}-s3"
        self.state.create_stage(stage_valid, job_id, "validation", 3)
        self.state.update_stage(stage_valid, {"status": "running"})
        
        # In a real scenario, we'd apply the delta and run eval
        # Here we use the authority's deterministic fallback validation contract
        thresholds = {"kl_max": 0.05}
        validation_result = self.authority.validate_edit(
            baseline_artifacts={"model": model_path},
            edited_artifacts={"delta": delta_hash},
            thresholds=thresholds,
            execution_mode=self.execution_mode,
        )

        status = "succeeded" if validation_result["passed"] else "failed"
        self.state.update_stage(stage_valid, {"status": status, "result": validation_result})
        self.state.update_job(job_id, {"status": status, "execution_contract": pipeline_contract.as_dict()})
        
        logger.info(f"AblationPipeline for job {job_id} completed with status: {status}")
        self.stop()
        return job_id

if __name__ == "__main__":
    # Simple self-test if run directly
    logging.basicConfig(level=logging.INFO)
    print("AblationPipeline defined.")
