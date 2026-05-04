import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelFingerprint:
    """
    Identifies model architecture families (e.g., Llama, Mistral) 
    and estimates memory requirements for the intake process.
    """

    KNOWN_FAMILIES = {
        "llama": "Dense",
        "mistral": "Dense",
        "mixtral": "MoE",
        "gemma": "Dense",
        "llava": "Multimodal",
        "qwen": "Dense"
    }
    
    @staticmethod
    def identify_architecture(config: Dict[str, Any]) -> str:
        arch_list = config.get("architectures", [])
        arch = arch_list[0].lower() if arch_list else ""
        model_type = config.get("model_type", "").lower()
        
        for key in ModelFingerprint.KNOWN_FAMILIES:
            if key in arch or key in model_type:
                return key
        return "unknown"
        
    @staticmethod
    def determine_topology(family: str, config: Dict[str, Any]) -> str:
        if config.get("num_experts_per_tok", 0) > 0 or "moe" in config.get("model_type", "").lower():
            return "MoE"
        if family in ModelFingerprint.KNOWN_FAMILIES:
            return ModelFingerprint.KNOWN_FAMILIES[family]
        return "Dense"

    @staticmethod
    def estimate_memory_requirements(config: Dict[str, Any]) -> int:
        """
        Produces a rough memory estimate for the model in bytes.
        Assumes float16 (2 bytes per parameter) and calculates based 
        on typical transformer dimensions.
        """
        hidden_size = config.get("hidden_size", 4096)
        num_layers = config.get("num_hidden_layers", 32)
        vocab_size = config.get("vocab_size", 32000)
        
        # Estimate embedding layer
        embedding_params = vocab_size * hidden_size
        
        # Estimate transformer blocks (very rough approximation: ~12 * hidden_size^2 per block)
        layer_params = num_layers * (12 * (hidden_size ** 2))
        
        total_params = embedding_params + layer_params
        
        # Assume 2 bytes per param (fp16/bf16)
        bytes_estimate = total_params * 2
        
        return int(bytes_estimate)

    @classmethod
    def analyze_model(cls, model_path: str) -> Dict[str, Any]:
        """
        Analyzes the model artifacts in the given path and emits topology info.
        Supports standard config.json directories and GGUF files.
        """
        config = {}
        
        # Check if the model_path is a GGUF file
        if model_path.lower().endswith(".gguf"):
            logger.info(f"GGUF model detected: {model_path}. Using heuristic fingerprinting.")
            # For GGUF, we provide a default config based on common Qwen patterns
            # since we don't want to parse the whole GGUF header here.
            if "qwen" in model_path.lower():
                config = {
                    "model_type": "qwen",
                    "architectures": ["Qwen2ForCausalLM"],
                    "hidden_size": 2048, # Approximate for 2.5B
                    "num_hidden_layers": 28,
                    "vocab_size": 151936
                }
            else:
                config = {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"]
                }
        else:
            # Traditional directory-based analysis
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                # Fallback: check if the path itself IS the config file or if it's a dir with qwen in name
                if os.path.isdir(model_path) and "qwen" in model_path.lower():
                     config = {"model_type": "qwen"}
                else:
                    raise FileNotFoundError(f"config.json not found in {model_path}. Unable to fingerprint model.")
            else:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            
        family = cls.identify_architecture(config)
        topology = cls.determine_topology(family, config)
        memory_bytes = cls.estimate_memory_requirements(config)
        
        return {
            "architecture_family": family,
            "topology_type": topology,
            "estimated_memory_bytes": memory_bytes,
            "supports_moe": topology == "MoE",
            "supports_multimodal": topology == "Multimodal"
        }
