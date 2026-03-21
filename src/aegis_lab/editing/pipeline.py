import logging
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.intake.fingerprint import ModelFingerprint
from aegis_lab.probing.capture import CAREActivationCapturer
from aegis_lab.editing.delta_builder import DeltaBuilder
from aegis_lab.verification.authority import SemanticAuthority

logger = logging.getLogger(__name__)

class AblationPipeline:
    """
    Formal pipeline for handling the transition between Intake, Probing, 
    Extraction, and Validation in the model ablation process.
    """
    
    def __init__(self, state: AegisState, artifact_store: ArtifactStore):
        self.state = state
        self.artifact_store = artifact_store
        self.capturer = CAREActivationCapturer(state, artifact_store)
        self.authority = SemanticAuthority()
        # DeltaBuilder needs a work_dir, we'll use a temporary one or based on job_id
        self.work_root = Path("/tmp/aegis_ablation")
        self.work_root.mkdir(parents=True, exist_ok=True)

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
        
        # 1. Intake
        logger.info("[PIPELINE] Stage 1: Intake")
        fingerprint = ModelFingerprint.analyze_model(model_path)
        self.state.create_job(job_id, project_id, "ablation_full")
        self.state.update_job(job_id, {"fingerprint": fingerprint, "status": "running"})
        
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
        self.state.update_stage(stage_probe, {"status": "succeeded", "result": capture_result})
        
        # 3. Extraction (Atom Generation)
        logger.info("[PIPELINE] Stage 3: Extraction")
        stage_extract = f"{job_id}-s2"
        self.state.create_stage(stage_extract, job_id, "extraction", 2)
        self.state.update_stage(stage_extract, {"status": "running"})
        
        builder = DeltaBuilder(self.artifact_store, self.work_root / job_id)
        edit_plan = {
            "type": "permanent",
            "layers": layers,
            "capture_info": capture_result
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
        
        self.state.update_stage(stage_extract, {"status": "succeeded", "result": {"atom_id": atom_id, "delta_hash": delta_hash}})
        
        # 4. Validation
        logger.info("[PIPELINE] Stage 4: Validation")
        stage_valid = f"{job_id}-s3"
        self.state.create_stage(stage_valid, job_id, "validation", 3)
        self.state.update_stage(stage_valid, {"status": "running"})
        
        # In a real scenario, we'd apply the delta and run eval
        # Here we use the authority's validate_edit mock
        thresholds = {"kl_max": 0.05}
        validation_result = self.authority.validate_edit(
            baseline_artifacts={"model": model_path},
            edited_artifacts={"delta": delta_hash},
            thresholds=thresholds
        )
        
        status = "succeeded" if validation_result["passed"] else "failed"
        self.state.update_stage(stage_valid, {"status": status, "result": validation_result})
        self.state.update_job(job_id, {"status": status})
        
        logger.info(f"AblationPipeline for job {job_id} completed with status: {status}")
        return job_id

if __name__ == "__main__":
    # Simple self-test if run directly
    logging.basicConfig(level=logging.INFO)
    print("AblationPipeline defined.")
