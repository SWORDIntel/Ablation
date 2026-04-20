from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class NormalizedModelProfile:
    """Canonical model intake contract shared across planning and scheduling."""

    model_id: str
    source_path: str
    source_format: str
    family: str
    topology: str
    modalities: List[str]
    parameter_estimate: Optional[int]
    context_length: Optional[int]
    quantization_state: str
    expert_topology: Dict[str, Any]
    runtime_support: Dict[str, Any]
    estimated_memory_by_precision: Dict[str, int]
    supports_runtime_hooks: bool
    supports_delta_edit: bool
    supports_activation_capture: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "family": self.family,
            "topology": self.topology,
            "modalities": list(self.modalities),
            "parameter_estimate": self.parameter_estimate,
            "context_length": self.context_length,
            "quantization_state": self.quantization_state,
            "expert_topology": dict(self.expert_topology),
            "runtime_support": dict(self.runtime_support),
            "estimated_memory_by_precision": dict(self.estimated_memory_by_precision),
            "supports_runtime_hooks": self.supports_runtime_hooks,
            "supports_delta_edit": self.supports_delta_edit,
            "supports_activation_capture": self.supports_activation_capture,
            "notes": list(self.notes),
        }

    @classmethod
    def from_model_intake(
        cls,
        model_path: str,
        family: str,
        topology: str,
        memory_bytes: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> "NormalizedModelProfile":
        config = config or {}
        source_path = str(Path(model_path))
        model_name = Path(model_path).stem if Path(model_path).suffix else Path(model_path).name
        model_id = model_name or "unknown-model"

        source_format = "gguf" if source_path.lower().endswith(".gguf") else "hf_config"

        modalities: List[str] = ["text"]
        if topology.lower() == "multimodal":
            modalities = ["text", "vision"]

        quantization_state = "pre_quantized" if source_format == "gguf" else "unknown"

        context_length = config.get("max_position_embeddings")
        parameter_estimate = config.get("num_parameters")

        if parameter_estimate is None:
            hidden_size = int(config.get("hidden_size", 0) or 0)
            num_layers = int(config.get("num_hidden_layers", 0) or 0)
            vocab_size = int(config.get("vocab_size", 0) or 0)
            if hidden_size and num_layers and vocab_size:
                embedding_params = vocab_size * hidden_size
                layer_params = num_layers * (12 * (hidden_size ** 2))
                parameter_estimate = embedding_params + layer_params

        estimated_memory_by_precision = {
            "fp32": int(memory_bytes * 2),
            "fp16": int(memory_bytes),
            "int8": max(int(memory_bytes // 2), 1),
        }

        expert_count = int(config.get("num_local_experts", 0) or 0)
        experts_per_tok = int(config.get("num_experts_per_tok", 0) or 0)

        return cls(
            model_id=model_id,
            source_path=source_path,
            source_format=source_format,
            family=family or "unknown",
            topology=topology,
            modalities=modalities,
            parameter_estimate=parameter_estimate,
            context_length=context_length,
            quantization_state=quantization_state,
            expert_topology={
                "num_local_experts": expert_count,
                "num_experts_per_tok": experts_per_tok,
                "is_moe": topology.lower() == "moe",
            },
            runtime_support={
                "openvino": True,
                "onnxruntime": True,
                "torch": True,
            },
            estimated_memory_by_precision=estimated_memory_by_precision,
            supports_runtime_hooks=source_format != "gguf",
            supports_delta_edit=source_format != "gguf",
            supports_activation_capture=True,
            notes=[],
        )
