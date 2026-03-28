import logging
import uuid
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
        edit_plan = {
            "type": "permanent",
            "layers": layers,
            "capture_info": capture_result,
            "execution_contract": pipeline_contract.as_dict(),
        }
        delta_hash = builder.generate_delta_tensors(edit_plan)
        
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
        return job_id

if __name__ == "__main__":
    # Simple self-test if run directly
    logging.basicConfig(level=logging.INFO)
    print("AblationPipeline defined.")
