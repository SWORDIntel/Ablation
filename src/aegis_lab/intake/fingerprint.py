import json
import os
import logging
from typing import Dict, Any, Optional, Tuple

from aegis_lab.intake.contracts import NormalizedModelProfile

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
        "qwen": "Dense",
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

        embedding_params = vocab_size * hidden_size
        layer_params = num_layers * (12 * (hidden_size ** 2))
        total_params = embedding_params + layer_params
        return int(total_params * 2)

    @staticmethod
    def _load_model_config(model_path: str) -> Dict[str, Any]:
        if model_path.lower().endswith(".gguf"):
            logger.info("GGUF model detected: %s. Using heuristic fingerprinting.", model_path)
            if "qwen" in model_path.lower():
                return {
                    "model_type": "qwen",
                    "architectures": ["Qwen2ForCausalLM"],
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "vocab_size": 151936,
                }
            return {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            if os.path.isdir(model_path) and "qwen" in model_path.lower():
                return {"model_type": "qwen"}
            raise FileNotFoundError(f"config.json not found in {model_path}. Unable to fingerprint model.")

        with open(config_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    @classmethod
    def fingerprint_components(cls, model_path: str) -> Tuple[Dict[str, Any], str, str, int]:
        """Returns normalized fingerprint building blocks used by different pipeline stages."""
        config = cls._load_model_config(model_path)
        family = cls.identify_architecture(config)
        topology = cls.determine_topology(family, config)
        memory_bytes = cls.estimate_memory_requirements(config)
        return config, family, topology, memory_bytes

    @classmethod
    def build_normalized_profile(cls, model_path: str) -> Dict[str, Any]:
        config, family, topology, memory_bytes = cls.fingerprint_components(model_path)
        profile = NormalizedModelProfile.from_model_intake(
            model_path=model_path,
            family=family,
            topology=topology,
            memory_bytes=memory_bytes,
            config=config,
        )
        return profile.to_dict()

    @classmethod
    def analyze_model(cls, model_path: str) -> Dict[str, Any]:
        """
        Analyzes model artifacts and emits the legacy fingerprint contract while
        embedding the canonical normalized intake profile for newer stages.
        """
        config, family, topology, memory_bytes = cls.fingerprint_components(model_path)
        normalized_profile = NormalizedModelProfile.from_model_intake(
            model_path=model_path,
            family=family,
            topology=topology,
            memory_bytes=memory_bytes,
            config=config,
        )

        return {
            "architecture_family": family,
            "topology_type": topology,
            "estimated_memory_bytes": memory_bytes,
            "supports_moe": topology == "MoE",
            "supports_multimodal": topology == "Multimodal",
            "normalized_profile": normalized_profile.to_dict(),
        }
