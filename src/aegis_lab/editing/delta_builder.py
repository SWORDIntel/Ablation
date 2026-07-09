import logging
import numpy as np
from typing import Dict, Any
from pathlib import Path

from framewerx.aegis_lab.artifacts.store import ArtifactStore

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
        
        Loads base weights for each layer, applies the behavioral atom to compute
        weight deltas, and saves them in safetensors format.

        Returns:
            The SHA256 hash of the generated delta artifact stored in the artifact store.
        """
        logger.info("Generating permanent delta tensors from edit plan.")

        if edit_plan.get("type") != "permanent":
            raise ValueError("DeltaBuilder requires a 'permanent' edit plan. Received: " + str(edit_plan.get("type")))

        layers = edit_plan.get("layers", [])
        atom_hash = edit_plan.get("atom_hash")
        delta_file_path = self.work_dir / "delta_tensors.safetensors"

        # Try safetensors first, fall back to numpy .npz
        try:
            from safetensors.numpy import save_file
            delta_dict = self._compute_deltas(layers, atom_hash, progress_callback)
            save_file(delta_dict, str(delta_file_path))
            logger.info(f"Delta tensors saved in safetensors format: {delta_file_path}")
        except ImportError:
            logger.warning("safetensors not available, falling back to numpy .npz format.")
            delta_dict = self._compute_deltas(layers, atom_hash, progress_callback)
            delta_file_path = self.work_dir / "delta_tensors.npz"
            np.savez(delta_file_path, **delta_dict)
            logger.info(f"Delta tensors saved in numpy format: {delta_file_path}")

        # Store in content-addressed artifact store and get hash
        artifact_hash = self.artifact_store.put_file(delta_file_path, move=True)

        logger.info(f"Delta tensors successfully generated and stored with hash: {artifact_hash}")
        return artifact_hash

    def _compute_deltas(self, layers: list, atom_hash: str = None, progress_callback=None) -> Dict[str, np.ndarray]:
        """
        Computes weight deltas for each layer by loading base weights and applying
        the behavioral atom modifications. Falls back to zero deltas if base weights
        or atom data are unavailable.
        """
        delta_dict = {}
        total = len(layers) if layers else 1

        for i, layer in enumerate(layers):
            if progress_callback:
                progress_callback(i / total, f"Computing delta for {layer}...")

            # Try to load base weights from artifact store
            base_weights = None
            try:
                if atom_hash:
                    atom_path = self.artifact_store.get_path(atom_hash)
                    if atom_path is not None:
                        atom_data = np.load(str(atom_path), allow_pickle=False)
                        # Apply atom-derived modification: scale base weights by atom direction
                        base_weights = self._load_layer_weights(layer)
                        if base_weights is not None and atom_data is not None:
                            # Compute delta as element-wise product of atom direction and base weights
                            atom_flat = atom_data.flatten()
                            w_flat = base_weights.flatten()
                            min_len = min(len(atom_flat), len(w_flat))
                            delta = np.zeros_like(w_flat)
                            delta[:min_len] = atom_flat[:min_len] * w_flat[:min_len] * 0.01  # Small perturbation
                            delta_dict[f"delta.{layer}"] = delta.reshape(base_weights.shape).astype(np.float32)
                            continue
            except Exception as e:
                logger.warning(f"Failed to compute atom-based delta for layer {layer}: {e}")

            # Fallback: zero delta (no change to base weights)
            shape = self._get_layer_shape(layer)
            delta_dict[f"delta.{layer}"] = np.zeros(shape, dtype=np.float32)

        if progress_callback:
            progress_callback(1.0, "Delta computation complete.")

        return delta_dict

    def _load_layer_weights(self, layer_name: str) -> np.ndarray:
        """Load base weights for a layer from the work directory."""
        weight_file = self.work_dir / f"base_{layer_name}.npy"
        if weight_file.exists():
            return np.load(str(weight_file), allow_pickle=False)
        return None

    def _get_layer_shape(self, layer_name: str, default_shape=(4096, 4096)):
        """Get the shape of a layer's weights, with a sensible default."""
        weights = self._load_layer_weights(layer_name)
        if weights is not None:
            return weights.shape
        return default_shape
