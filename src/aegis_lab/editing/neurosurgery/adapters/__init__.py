from __future__ import annotations

from .base import ArchitectureAdapter, AttentionPack, MLPTriplet, MoEBlock
from .llama_like import LlamaLikeAdapter
from .moe_like import MoELikeAdapter


_ADAPTERS: list[ArchitectureAdapter] = [LlamaLikeAdapter(), MoELikeAdapter()]


def _find(model, capability: str) -> ArchitectureAdapter:
    method = f"supports_{capability}"
    for adapter in _ADAPTERS:
        if getattr(adapter, method)(model):
            return adapter
    model_type = getattr(getattr(model, "config", None), "model_type", type(model).__name__)
    raise ValueError(f"no {capability} structural-surgery adapter for {model_type}")


def get_mlp_adapter(model) -> ArchitectureAdapter:
    return _find(model, "mlp")


def get_attention_adapter(model) -> ArchitectureAdapter:
    return _find(model, "attention")


def get_moe_adapter(model) -> ArchitectureAdapter:
    return _find(model, "moe")


def get_adapter(model) -> ArchitectureAdapter:
    """Backward-compatible alias for dense MLP workflows."""
    return get_mlp_adapter(model)


__all__ = [
    "ArchitectureAdapter",
    "AttentionPack",
    "MLPTriplet",
    "MoEBlock",
    "LlamaLikeAdapter",
    "MoELikeAdapter",
    "get_adapter",
    "get_mlp_adapter",
    "get_attention_adapter",
    "get_moe_adapter",
]
