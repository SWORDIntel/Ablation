import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

class DynamicSteeringManager:
    """
    Manages Dynamic Activation Addition (Inference-Time Steering) for AEGIS-LAB.
    
    Injects SAE-extracted sparse features directly into a model's residual stream
    during the forward pass using forward hooks. This allows dialing specific 
    behaviors up or down via a tunable scalar multiplier without permanent 
    weight modifications.
    """
    def __init__(self, model: nn.Module):
        """
        Initializes the DynamicSteeringManager.
        
        Args:
            model (nn.Module): The PyTorch model to steer.
        """
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        # Structure: {layer_name: [{"feature_vector": Tensor, "multiplier": float}]}
        self.active_features: Dict[str, List[Dict[str, Any]]] = {}

    def add_steering_feature(self, layer_name: str, feature_vector: torch.Tensor, multiplier: float = 1.0):
        """
        Registers a feature vector to be injected at a specific layer.
        
        Args:
            layer_name (str): The name of the module/layer to attach the hook to.
            feature_vector (torch.Tensor): The sparse feature vector to add to the activations.
                                           Should be broadcastable to the activation shape.
            multiplier (float): A scalar multiplier to dial the behavior up or down.
        """
        if layer_name not in self.active_features:
            self.active_features[layer_name] = []
        
        self.active_features[layer_name].append({
            "feature_vector": feature_vector,
            "multiplier": multiplier
        })
    
    def set_multiplier(self, layer_name: str, feature_idx: int, multiplier: float):
        """
        Updates the multiplier for an existing steering feature dynamically.
        
        Args:
            layer_name (str): The name of the module/layer.
            feature_idx (int): The index of the registered feature vector for this layer.
            multiplier (float): The new scalar multiplier.
        """
        if layer_name in self.active_features and feature_idx < len(self.active_features[layer_name]):
            self.active_features[layer_name][feature_idx]["multiplier"] = multiplier

    def _get_hook_fn(self, layer_name: str) -> Callable:
        """
        Creates a forward hook function for a specific layer to inject the features.
        """
        def hook(module: nn.Module, inputs: Union[Tuple, torch.Tensor], outputs: Union[Tuple, torch.Tensor]):
            # Handling generic transformer outputs where the first element is usually the hidden states.
            is_tuple = isinstance(outputs, tuple)
            if is_tuple:
                hidden_states = outputs[0]
                rest = outputs[1:]
            else:
                hidden_states = outputs
                rest = ()

            if layer_name in self.active_features:
                for feature_data in self.active_features[layer_name]:
                    vec = feature_data["feature_vector"].to(hidden_states.device, dtype=hidden_states.dtype)
                    mult = feature_data["multiplier"]
                    
                    # Injecting the feature vector into the residual stream
                    hidden_states = hidden_states + (vec * mult)

            if is_tuple:
                return (hidden_states,) + rest
            return hidden_states

        return hook

    def apply_hooks(self):
        """
        Applies forward hooks to the specified layers in the model.
        Should be called before model.forward() if not using the context manager.
        """
        self.remove_hooks() # Ensure no duplicate hooks
        
        modules = dict(self.model.named_modules())
        
        for layer_name in self.active_features.keys():
            if layer_name in modules:
                module = modules[layer_name]
                hook_handle = module.register_forward_hook(self._get_hook_fn(layer_name))
                self.hooks.append(hook_handle)
            else:
                print(f"Warning: Layer '{layer_name}' not found in the model.")

    def remove_hooks(self):
        """
        Removes all registered forward hooks, restoring the model to its base state.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_features(self):
        """
        Clears all registered steering features and removes active hooks.
        """
        self.remove_hooks()
        self.active_features.clear()
        
    def __enter__(self):
        """
        Context manager entry to automatically apply hooks.
        """
        self.apply_hooks()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit to automatically remove hooks.
        """
        self.remove_hooks()
