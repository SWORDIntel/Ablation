import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Callable, Any

class MoEAwareAblator:
    """
    MoE-Aware Expert Targeting (Routing Ablation) for AEGIS-LAB.
    
    This class provides tools to:
    1. Identify target experts by monitoring router gating probabilities.
    2. Surgically ablate expert weights.
    3. Force the router to bypass specific experts to reduce 'Ablation Tax'.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initializes the MoEAwareAblator.
        
        Args:
            model (nn.Module): The Mixture of Experts (MoE) model to ablate.
        """
        self.model = model
        # Store original routing functions to allow restoration
        self._original_routers: Dict[str, Callable] = {}
        
    def monitor_gating_probabilities(self, dataloader: Any, target_layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Monitors router gating probabilities across the dataset to identify 
        which experts are most active.
        
        Args:
            dataloader: DataLoader providing the target behavior inputs.
            target_layer_names: List of names for MoE router modules.
            
        Returns:
            Dict mapping layer names to aggregated gating probabilities.
        """
        # Dictionary to accumulate gating probabilities
        gating_stats = {name: None for name in target_layer_names}
        
        hooks = []
        
        def get_hook(name: str):
            def hook(module: nn.Module, input: Tuple, output: Any):
                # Assuming output of router contains gating probabilities (logits or softmax)
                # Structure depends on the specific MoE implementation
                # This is a generic representation: output[0] = routing weights
                probs = output[0] if isinstance(output, tuple) else output
                if gating_stats[name] is None:
                    gating_stats[name] = probs.detach().sum(dim=0)
                else:
                    gating_stats[name] += probs.detach().sum(dim=0)
            return hook
            
        for name, module in self.model.named_modules():
            if name in target_layer_names:
                hooks.append(module.register_forward_hook(get_hook(name)))
                
        # Run forward passes
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    # For huggingface style dict batches
                    inputs = {k: v.to(next(self.model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    self.model(**inputs)
                elif isinstance(batch, (list, tuple)):
                    inputs = [v.to(next(self.model.parameters()).device) if isinstance(v, torch.Tensor) else v for v in batch]
                    self.model(*inputs)
                else:
                    self.model(batch.to(next(self.model.parameters()).device))
                    
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return gating_stats

    def identify_target_expert(self, gating_stats: Dict[str, torch.Tensor], top_k: int = 1) -> Dict[str, List[int]]:
        """
        Identifies the experts that handle the target behavior based on gating stats.
        
        Args:
            gating_stats: The accumulated gating probabilities from monitor_gating_probabilities.
            top_k: Number of top experts to identify per layer.
            
        Returns:
            Dict mapping layer names to lists of target expert indices.
        """
        target_experts = {}
        for name, stats in gating_stats.items():
            if stats is not None:
                # Find the indices of the experts with the highest accumulated probabilities
                _, top_indices = torch.topk(stats.float(), k=top_k)
                target_experts[name] = top_indices.view(-1).tolist()
        return target_experts

    def ablate_expert_weights(self, layer_name: str, expert_idx: int, strategy: str = 'zero'):
        """
        Surgically ablates the weights of a specific expert.
        
        Args:
            layer_name: The name of the MoE layer containing the expert.
            expert_idx: The index of the expert to ablate.
            strategy: 'zero' (set weights to 0) or 'noise' (add random noise).
        """
        module = dict(self.model.named_modules()).get(layer_name)
        if module is None:
            raise ValueError(f"Layer {layer_name} not found in model.")
            
        # Heuristic for finding the expert: assume structure like module.experts[expert_idx]
        if hasattr(module, 'experts') and len(module.experts) > expert_idx:
            expert = module.experts[expert_idx]
            with torch.no_grad():
                for param_name, param in expert.named_parameters():
                    if strategy == 'zero':
                        param.zero_()
                    elif strategy == 'noise':
                        param.add_(torch.randn_like(param) * param.std())
                    else:
                        raise ValueError(f"Unknown ablation strategy: {strategy}")
            print(f"Ablated expert {expert_idx} in layer {layer_name} using '{strategy}' strategy.")
        else:
            print(f"Could not find standard 'experts' list in {layer_name}. "
                  f"Architecture specific targeting required for weight ablation.")

    def force_router_bypass(self, router_layer_name: str, expert_idx_to_bypass: int):
        """
        Forces the router to bypass a specific expert by modifying its forward pass.
        This reduces the 'Ablation Tax' by avoiding sending tokens to the damaged expert.
        
        Args:
            router_layer_name: The name of the router module.
            expert_idx_to_bypass: The index of the expert to avoid.
        """
        router_module = dict(self.model.named_modules()).get(router_layer_name)
        if router_module is None:
            raise ValueError(f"Router {router_layer_name} not found.")
            
        # Store original forward method if not already stored
        if router_layer_name not in self._original_routers:
            self._original_routers[router_layer_name] = router_module.forward
            
        original_forward = self._original_routers[router_layer_name]
        
        def patched_forward(*args, **kwargs):
            # Call original forward to get routing probabilities/logits
            output = original_forward(*args, **kwargs)
            
            # This is highly architecture-dependent. We assume output is a tensor 
            # representing logits, where the last dimension is num_experts.
            if isinstance(output, tuple):
                routing_logits = output[0]
            else:
                routing_logits = output
                
            # Set the logit for the bypassed expert to a very negative number
            # so it never gets selected by top-k routing.
            with torch.no_grad():
                routing_logits[..., expert_idx_to_bypass] = -1e9
                
            if isinstance(output, tuple):
                return (routing_logits,) + output[1:]
            return routing_logits
            
        # Apply the patch
        router_module.forward = patched_forward
        print(f"Forced router {router_layer_name} to bypass expert {expert_idx_to_bypass}.")

    def restore_routers(self):
        """
        Restores all patched routers to their original functionality.
        """
        for name, original_forward in self._original_routers.items():
            module = dict(self.model.named_modules()).get(name)
            if module is not None:
                module.forward = original_forward
        self._original_routers.clear()
        print("Restored all original routers.")
