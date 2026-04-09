import logging
import sys
import uuid
import threading
import queue
import time
from typing import Dict, Any, List
from pathlib import Path

from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.utils.progress import ProgressTracker
from aegis_lab.intake.fingerprint import ModelFingerprint
from aegis_lab.probing.capture import CAREActivationCapturer
from aegis_lab.atoms.extractor import BehavioralAtomExtractor
from aegis_lab.editing.delta_builder import DeltaBuilder
from aegis_lab.editing.adversarial import RedTeamEvaluator
from aegis_lab.quantization.calibration import CalibrationCorpusBuilder
from aegis_lab.quantization.exporter import OpenVINOExporter
from aegis_lab.verification.authority import SemanticAuthority
from aegis_lab.editing import StaticIntervention, FeatureIntervention, RuntimeSteering

logger = logging.getLogger(__name__)

class InterventionRegistry:
    def __init__(self):
        self._registry = {}
        # New interventions linked here
        self._registry["sae_clamp"] = FeatureIntervention.sae_clamp
        self._registry["moe_ablate"] = FeatureIntervention.moe_ablate
        self._registry["inference_steer"] = RuntimeSteering.steer

    def register(self, name: str, intervention_cls):
        self._registry[name] = intervention_cls

    def execute(self, name: str, model: Any, **kwargs):
        if name not in self._registry:
            raise ValueError(f"Intervention {name} not found.")
        return self._registry[name](**kwargs).apply(model)

class AblationPipeline:
    """
    Formal pipeline for handling the transition between Intake, Probing, 
    Extraction, and Validation in the model ablation process.
    """
    
    def __init__(self, state: AegisState, artifact_store: ArtifactStore, execution_mode: str = ExecutionMode.FALLBACK.value):
        self.state = state
        self.artifact_store = artifact_store
        self.capturer = CAREActivationCapturer(state, artifact_store)
        self.extractor = BehavioralAtomExtractor(state, artifact_store)
        self.authority = SemanticAuthority(default_mode=execution_mode)
        self.execution_mode = execution_mode
        self.work_root = Path("/tmp/aegis_ablation")
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.registry = InterventionRegistry()
        self.red_team = RedTeamEvaluator(state)

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
                          project_id: str = "default_project",
                          use_sta: bool = False,
                          enable_turboquant: bool = False,
                          show_progress: bool = True) -> str:
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

        progress = None
        if show_progress:
            total_steps = 5 if enable_turboquant else 4
            progress = ProgressTracker(total_steps=total_steps, description=f"Ablating {job_id}")

        pipeline_contract = resolve_execution_contract(
            operation="ablation_pipeline",
            requested_mode=self.execution_mode,
            native_available=False,
            reason="Pipeline is operating in deterministic fallback mode for non-hardware environments.",
            details={"project_id": project_id, "model_path": model_path},
        )
        
        # 1. Intake
        logger.info("[PIPELINE] Stage 1: Intake")
        if progress: progress.update(0, "Intaking model...")
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
        if progress: progress.update(1, "Probing activations...")
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
        if progress: progress.update(1, "Extracting atoms...")
        logger.info("[PIPELINE] Stage 3: Extraction")
        stage_extract = f"{job_id}-s2"
        self.state.create_stage(stage_extract, job_id, "extraction", 2)
        self.state.update_stage(stage_extract, {"status": "running"})
        
        # Refined Extraction: Isolate behavioral atom from activations
        method = "steering_target_atoms" if use_sta else "ridge_regression"
        logger.info(f"[PIPELINE] Refined Extraction: Isolating behavioral atom using {method}")
        
        atom_hash = self.extractor.extract_atom(
            job_id=job_id,
            positive_act_hash=capture_result["positive"],
            negative_act_hash=capture_result["negative"],
            method=method,
            execution_mode=self.execution_mode
        )

        # Retrieve the newly created atom metadata
        atom_info = next((a for a in self.state.db.list_all("atoms") if a["job_id"] == job_id), None)
        atom_id = atom_info["atom_id"] if atom_info else f"atom-{job_id}"

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
            "atom_id": atom_id,
            "atom_hash": atom_hash,
            "capture_info": capture_result,
            "execution_contract": pipeline_contract.as_dict(),
        }
        delta_hash = builder.generate_delta_tensors(edit_plan)

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
        if progress: progress.update(1, "Validating edit...")
        logger.info("[PIPELINE] Stage 4: Validation")
        stage_valid = f"{job_id}-s3"
        self.state.create_stage(stage_valid, job_id, "validation", 3)
        self.state.update_stage(stage_valid, {"status": "running"})
        
        # Authority deterministic fallback validation contract
        thresholds = {"kl_max": 10.0}
        validation_result = self.authority.validate_edit(
            baseline_artifacts={
                "model": model_path,
                "positive_dataset": positive_dataset,
                "negative_dataset": negative_dataset,
            },
            edited_artifacts={"delta": delta_hash},
            thresholds=thresholds,
            execution_mode=self.execution_mode,
        )

        # Robustness tuning via RedTeamEvaluator
        robustness = self.red_team.evaluate_adversarial_vulnerability(job_id)
        validation_result["adversarial_robustness"] = robustness

        status = "succeeded" if validation_result["passed"] and robustness > 0.5 else "failed"
        self.state.update_stage(stage_valid, {"status": status, "result": validation_result})

        # 5. TurboQuant Extreme Compression (Optional)
        if status == "succeeded" and enable_turboquant:
            logger.info("[PIPELINE] Stage 5: TurboQuant Extreme Compression")
            if progress: progress.update(1, "Applying TurboQuant compression...")
            stage_quant = f"{job_id}-s4"
            self.state.create_stage(stage_quant, job_id, "quantization_turbo", 4)
            self.state.update_stage(stage_quant, {"status": "running"})

            try:
                calibration = CalibrationCorpusBuilder()
                calibration.add_standard_samples(["sample 1", "sample 2"])
                calibration.add_ablated_path_samples(["ablated 1"])

                ds = None
                model = None

                exporter = OpenVINOExporter(model=model, work_dir=self.work_root / job_id)
                quantized_path = exporter.export_int8(ds, enable_turboquant=True)

                self.state.update_stage(stage_quant, {
                    "status": "succeeded",
                    "result": {"quantized_model": str(quantized_path), "method": "TurboQuant"}
                })
            except Exception as e:
                logger.error(f"TurboQuant compression failed: {e}")
                self.state.update_stage(stage_quant, {"status": "failed", "error": str(e)})

        self.state.update_job(job_id, {"status": status, "execution_contract": pipeline_contract.as_dict()})
        if progress: progress.update(1, "Finalizing...")

        logger.info(f"AblationPipeline for job {job_id} completed with status: {status}")
        self.stop()
        return job_id

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("AblationPipeline defined.")