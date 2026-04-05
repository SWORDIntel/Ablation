import logging
from typing import Dict, Any
from pathlib import Path

from aegis_lab.artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)

class DeltaBuilder:
    """
    Handles permanent delta tensor generation.
    """
    def __init__(self, artifact_store: ArtifactStore, work_dir: Path):
        self.artifact_store = artifact_store
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def generate_delta_tensors(self, edit_plan: Dict[str, Any], progress_callback=None) -> str:
        """
        Generates permanent delta tensors based on the edit plan and stores them.
        
        Returns:
            The SHA256 hash of the generated delta artifact stored in the artifact store.
        """
        logger.info("Generating permanent delta tensors from edit plan.")
        
        if edit_plan.get("type") != "permanent":
            raise ValueError("DeltaBuilder requires a 'permanent' edit plan. Received: " + str(edit_plan.get("type")))
            
        # Simulate building delta tensors (e.g. creating a safetensors file)
        delta_file_path = self.work_dir / "delta_tensors.safetensors"
        
        layers = edit_plan.get("layers", [])
        total_layers = len(layers)

        # In a real implementation, we would load base weights, compute deltas, and save them.
        with open(delta_file_path, "wb") as f:
            f.write(b"MOCK_DELTA_TENSOR_DATA_FOR_LAYERS:")
            for idx, layer in enumerate(layers):
                if progress_callback:
                    progress_callback(idx, total_layers, layer)
                f.write(str(layer).encode('utf-8'))
                f.write(b",")
            
        # Store in content-addressed artifact store and get hash
        artifact_hash = self.artifact_store.put_file(delta_file_path, move=True)
        
        logger.info(f"Delta tensors successfully generated and stored with hash: {artifact_hash}")
        return artifact_hash
