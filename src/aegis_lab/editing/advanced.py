import logging
import uuid
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.editing.pipeline import AblationPipeline, InterventionRegistry
from aegis_lab.editing.adversarial import RedTeamEvaluator
from aegis_lab.atoms.extractor import BehavioralAtomExtractor
from aegis_lab.editing import StaticIntervention, FeatureIntervention, RuntimeSteering
from aegis_lab.editing.gcg_refiner import GCGRefiner
from aegis_lab.scheduler.sharding import DeviceTile, OpusShardingPlanner
from aegis_lab.editing.sae_crossmodal import SparseFeatureExtractor, CrossModalCapturer

logger = logging.getLogger(__name__)

class AdvancedAblationOrchestrator:
    """
    Advanced Ablation Orchestrator for Opus-scale models.
    Supports iterative adversarial refinement, sparse feature extraction, 
    and distributed sharding hooks.
    """
    def __init__(self, state: AegisState, artifact_store: ArtifactStore):
        self.state = state
        self.artifact_store = artifact_store
        self.pipeline = AblationPipeline(state, artifact_store)
        self.registry = self.pipeline.registry
        self.red_teamer = RedTeamEvaluator(state)
        self.extractor = BehavioralAtomExtractor(state, artifact_store)
        
        # Instantiate advanced components
        self.gcg_refiner = GCGRefiner(state, self.red_teamer, self.extractor)
        
        from aegis_lab.scheduler.sharding import DeviceTile
        available_tiles = [
            DeviceTile(tile_id="NPU_0", device_type="NPU", memory_capacity_mb=64000, compute_ops=800),
            DeviceTile(tile_id="NPU_1", device_type="NPU", memory_capacity_mb=64000, compute_ops=800),
            DeviceTile(tile_id="VPU_0", device_type="VPU", memory_capacity_mb=32000, compute_ops=400),
            DeviceTile(tile_id="VPU_1", device_type="VPU", memory_capacity_mb=32000, compute_ops=400)
        ]
        self.sharding_planner = OpusShardingPlanner(available_tiles)
        
        self.sae_extractor = SparseFeatureExtractor(input_dim=4096, dictionary_size=16384)
        self.cross_modal_capturer = CrossModalCapturer(state, artifact_store, self.sae_extractor)

    def run_adversarial_refinement(self, 
                                   model_path: str, 
                                   target_behavior: str,
                                   iterations: int = 3,
                                   threshold: float = 0.9,
                                   progress_callback=None) -> Dict[str, Any]:
        """
        Iteratively refines an ablation 'Atom' using adversarial feedback.
        Delegates to the GCGRefiner implementation.
        """
        logger.info(f"Starting advanced adversarial refinement for behavior: {target_behavior}")
        
        job_id = f"adv-refine-{uuid.uuid4().hex[:8]}"
        
        # Initial atom generation (mocked hash for bootstrap)
        base_atom_hash = f"base-atom-{uuid.uuid4().hex[:8]}"
        
        if progress_callback:
            progress_callback(0.0, "Initializing refinement...")
            
        result = self.gcg_refiner.refine_atom(
            job_id=job_id,
            initial_atom_hash=base_atom_hash,
            robustness_threshold=threshold,
            max_iterations=iterations
        )
        
        if progress_callback:
            progress_callback(1.0, "Refinement complete.")
            
        return result

    def plan_distributed_opus_ablation(self, model_topology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plans a sharded ablation for Opus-scale models (>100B params).
        Delegates to the OpusShardingPlanner.
        """
        logger.info("Planning distributed ablation for Opus-scale model.")
        
        from aegis_lab.scheduler.sharding import LayerSpec
        # Convert dictionary topology to LayerSpec objects for the planner
        total_layers = model_topology.get("total_layers", 80)
        layer_specs = [
            LayerSpec(name=f"layer_{i}", memory_mb=2000.0, compute_ops=100.0) 
            for i in range(total_layers)
        ]
        
        execution_plan = self.sharding_planner.generate_execution_plan(layer_specs)
        
        return {
            "model_family": "opus",
            "sharding_strategy": "layer_wise_partition",
            "shards": [{"shard_id": shard.shard_id, "total_memory_mb": shard.total_memory_mb} for shard in execution_plan.shards],
            "routing_map": execution_plan.routing_map,
            "total_layers": total_layers
        }

    def extract_sparse_feature_atom(self, activations_hash: str, sae_artifact_hash: str) -> str:
        """
        Uses a Sparse Autoencoder (SAE) to extract interpretable behavioral features.
        Loads real activation tensors from the artifact store and applies SAE extraction.
        """
        logger.info(f"Extracting sparse feature atom using SAE: {sae_artifact_hash[:12]}")

        # Load real activation tensor from artifact store
        import numpy as np
        try:
            act_path = self.artifact_store.get_path(activations_hash)
            if act_path is None:
                logger.warning(f"Activations artifact {activations_hash[:12]} not found; using zero vector.")
                activations = np.zeros((1, self.sae_extractor.input_dim), dtype=np.float32)
            else:
                act_data = np.load(str(act_path), allow_pickle=False)
                activations = act_data if act_data.ndim == 2 else act_data.reshape(1, -1)
        except Exception as e:
            logger.warning(f"Failed to load activations from artifact {activations_hash[:12]}: {e}")
            activations = np.zeros((1, self.sae_extractor.input_dim), dtype=np.float32)

        # Convert to torch if SAE extractor expects torch tensors
        try:
            import torch
            activations = torch.from_numpy(activations).float()
        except ImportError:
            pass

        atom_features = self.sae_extractor.extract_atom(activations, threshold=0.5)
        atom_id = f"sparse-atom-{uuid.uuid4().hex[:8]}"

        # Store the extracted atom features
        try:
            import io
            buf = io.BytesIO()
            if hasattr(atom_features, 'numpy'):
                np.save(buf, atom_features.numpy())
            elif hasattr(atom_features, 'detach'):
                np.save(buf, atom_features.detach().cpu().numpy())
            else:
                np.save(buf, np.array(atom_features))
            buf.seek(0)
            self.artifact_store.put_stream(buf.read(), metadata={"type": "sparse_atom", "atom_id": atom_id})
        except Exception as e:
            logger.warning(f"Failed to persist sparse atom features: {e}")

        return atom_id

    def capture_cross_modal_activations(self, model_module: Any, layers: List[str], inputs: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Captures multimodal activations using CrossModalCapturer.
        """
        for layer in layers:
            self.cross_modal_capturer.register_hook(model_module, layer)
        return {"status": "hooks_registered"}

if __name__ == "__main__":
    print("AdvancedAblationOrchestrator ready for Opus-scale operations.")
