"""
Sparse Feature Extraction and Cross-Modal Ablation Hooks for AEGIS-LAB.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from framewerx.aegis_lab.state.db import AegisState
from framewerx.aegis_lab.artifacts.store import ArtifactStore

class SparseFeatureExtractor(nn.Module):
    """
    Extracts sparse, interpretable features from dense model activations.
    Supports multimodal inputs and projects them into a sparse feature space.
    """
    def __init__(self, input_dim: int, dictionary_size: int, sparsity_penalty: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dictionary_size = dictionary_size
        self.sparsity_penalty = sparsity_penalty
        
        # Simple SAE (Sparse Autoencoder) architecture
        self.encoder = nn.Linear(input_dim, dictionary_size)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(dictionary_size, input_dim)
        
    def forward(self, activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Projects multimodal activations into a sparse feature space and reconstructs them.
        """
        sparse_features = self.relu(self.encoder(activations))
        reconstructed = self.decoder(sparse_features)
        
        # Calculate reconstruction loss and sparsity loss
        mse_loss = nn.functional.mse_loss(reconstructed, activations)
        l1_loss = torch.norm(sparse_features, p=1, dim=-1).mean()
        loss = mse_loss + self.sparsity_penalty * l1_loss
        
        return {
            "sparse_features": sparse_features,
            "reconstructed": reconstructed,
            "loss": loss
        }
        
    def extract_atom(self, activations: torch.Tensor, threshold: float = 0.5) -> Dict[int, float]:
        """
        Extracts an 'atom' (a set of active sparse features representing an interpretable concept)
        from multimodal activations.
        """
        with torch.no_grad():
            sparse_features = self.relu(self.encoder(activations))
            
        # Aggregate across sequence/spatial dimensions if necessary
        if sparse_features.dim() > 1:
            sparse_features_max = torch.max(sparse_features.view(-1, self.dictionary_size), dim=0).values
        else:
            sparse_features_max = sparse_features
            
        active_indices = torch.where(sparse_features_max > threshold)[0].tolist()
        active_values = sparse_features_max[active_indices].tolist()
        
        return dict(zip(active_indices, active_values))


class CrossModalCapturer:
    """
    Hooks into model layers to capture and ablate cross-modal features
    using a SparseFeatureExtractor.
    """
    def __init__(self, state: AegisState, artifact_store: ArtifactStore, extractor: SparseFeatureExtractor):
        self.state = state
        self.artifact_store = artifact_store
        self.extractor = extractor
        self.hooks: List[Any] = []
        
    def register_hook(self, module: nn.Module, layer_name: str) -> None:
        """
        Registers a forward hook on a module to capture and potentially ablate 
        cross-modal activations into interpretable atoms.
        """
        def hook_fn(mod: nn.Module, inputs: Tuple[Any, ...], outputs: Any) -> Any:
            # Handle different output formats (e.g., tuple vs tensor)
            activations = outputs[0] if isinstance(outputs, tuple) else outputs
            
            if isinstance(activations, torch.Tensor):
                # Extract an atom from the cross-modal activations
                atom = self.extractor.extract_atom(activations)
                
                # We can store the extracted atoms or use AegisState to determine
                # if ablation should happen based on the sparse features.
                # Example storing mock metadata:
                metadata = {"layer": layer_name, "atom": atom}
                # self.artifact_store.save("crossmodal_atom", metadata)
                
            return outputs
            
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        
    def remove_hooks(self) -> None:
        """
        Removes all registered hooks from the model.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
