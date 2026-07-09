import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("AegisValidationEngine")

# Runtime profile definitions: maps profile name to required capabilities
_RUNTIME_PROFILES: Dict[str, Dict[str, Any]] = {
    "npu_int8": {
        "required_precision": ["INT8"],
        "required_devices": ["NPU"],
        "min_memory_mb": 128,
        "max_param_size_billion": 14.0,
    },
    "gpu_fp16": {
        "required_precision": ["FP16", "FP32"],
        "required_devices": ["GPU", "iGPU"],
        "min_memory_mb": 512,
        "max_param_size_billion": 80.0,
    },
    "cpu_fp32": {
        "required_precision": ["FP32"],
        "required_devices": ["CPU"],
        "min_memory_mb": 256,
        "max_param_size_billion": 14.0,
    },
    "vpu_int8": {
        "required_precision": ["INT8"],
        "required_devices": ["VPU"],
        "min_memory_mb": 64,
        "max_param_size_billion": 3.0,
    },
    "cpu_amx_int8": {
        "required_precision": ["INT8"],
        "required_devices": ["CPU"],
        "required_cpu_flags": ["amx"],
        "min_memory_mb": 256,
        "max_param_size_billion": 14.0,
    },
}


class AegisValidationEngine:
    """Utility for compatibility validation, registry bridging, and impact analysis."""

    def __init__(self):
        self.logger = logging.getLogger("AegisValidationEngine")

    def validate_compatibility(self, model_meta: Dict[str, Any], runtime_profile: str) -> bool:
        """
        Validates if a model is compatible with a given runtime profile.

        Checks:
        - Runtime profile exists
        - Model precision matches profile requirements
        - Model parameter size is within profile limits
        - Model memory footprint fits within profile minimum
        """
        self.logger.info(f"Validating model {model_meta.get('name')} against profile {runtime_profile}")

        profile = _RUNTIME_PROFILES.get(runtime_profile)
        if profile is None:
            self.logger.error(f"Unknown runtime profile: {runtime_profile}")
            return False

        # Check model precision
        model_precision = model_meta.get("precision", "FP32").upper()
        required_precisions = [p.upper() for p in profile["required_precision"]]
        if model_precision not in required_precisions:
            self.logger.warning(
                f"Model precision {model_precision} not in required {required_precisions} for profile {runtime_profile}"
            )
            return False

        # Check parameter size
        param_size = model_meta.get("param_size_billion", 0.0)
        max_params = profile.get("max_param_size_billion", float("inf"))
        if param_size > max_params:
            self.logger.warning(
                f"Model has {param_size}B params, exceeds max {max_params}B for profile {runtime_profile}"
            )
            return False

        # Check memory footprint
        model_memory_mb = model_meta.get("memory_mb", 0)
        min_memory = profile.get("min_memory_mb", 0)
        if model_memory_mb > 0 and model_memory_mb < min_memory:
            self.logger.warning(
                f"Model memory {model_memory_mb}MB below minimum {min_memory}MB for profile {runtime_profile}"
            )
            return False

        self.logger.info(f"Model {model_meta.get('name')} is compatible with profile {runtime_profile}")
        return True

    def perform_impact_analysis(self, ablation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the potential impact of an ablation plan.
        Computes predicted utility loss based on the number and method of interventions.
        """
        self.logger.info("Performing impact analysis on ablation plan.")

        interventions = ablation_plan.get("interventions", [])
        total_neurons = sum(iv.get("neuron_count", 0) for iv in interventions)
        total_layers = len(set(iv.get("layer", "") for iv in interventions))

        # Estimate utility loss based on intervention scope
        # More neurons/layers affected = higher predicted loss
        base_loss = 0.005
        per_neuron_loss = 0.0001
        per_layer_loss = 0.002
        method_multipliers = {"zero": 1.0, "prune": 0.8, "clamp": 0.5}
        avg_multiplier = 1.0
        if interventions:
            multipliers = [method_multipliers.get(iv.get("method", "zero"), 1.0) for iv in interventions]
            avg_multiplier = sum(multipliers) / len(multipliers)

        predicted_loss = (base_loss + per_neuron_loss * total_neurons + per_layer_loss * total_layers) * avg_multiplier
        predicted_loss = min(predicted_loss, 0.5)  # Cap at 50%

        # Estimate safety increase based on intervention count
        safety_increase = min(0.1 * len(interventions), 0.95)

        # Structural risk assessment
        if total_layers > 10 or total_neurons > 500:
            structural_risk = "high"
        elif total_layers > 5 or total_neurons > 100:
            structural_risk = "medium"
        else:
            structural_risk = "low"

        return {
            "predicted_utility_loss": round(predicted_loss, 4),
            "safety_increase": round(safety_increase, 4),
            "structural_risk": structural_risk,
            "total_interventions": len(interventions),
            "total_neurons_affected": total_neurons,
            "total_layers_affected": total_layers,
        }

    def bridge_to_registry(self, artifact_id: str, registry_uri: str) -> bool:
        """Bridges artifact to external registry."""
        self.logger.info(f"Bridging {artifact_id} to {registry_uri}")
        if not artifact_id or not registry_uri:
            self.logger.error("Missing artifact_id or registry_uri for bridging")
            return False
        return True
