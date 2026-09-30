from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MLPTriplet:
    gate: torch.nn.Module
    up: torch.nn.Module
    down: torch.nn.Module


@dataclass(frozen=True)
class AttentionPack:
    q: torch.nn.Module
    k: torch.nn.Module
    v: torch.nn.Module
    o: torch.nn.Module
    num_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def group_size(self) -> int:
        return self.num_heads // self.num_key_value_heads


@dataclass(frozen=True)
class MoEBlock:
    layer_index: int
    block: torch.nn.Module
    router: torch.nn.Module
    experts: torch.nn.ModuleList


class ArchitectureAdapter:
    """Base class for architecture-specific tensor-shape surgery."""

    name: str = "base"

    def supports_mlp(self, model) -> bool:
        return False

    def supports_attention(self, model) -> bool:
        return False

    def supports_moe(self, model) -> bool:
        return False

    def mlp_triplets(self, model) -> list[MLPTriplet]:
        raise ValueError(f"{self.name} does not support MLP surgery")

    def attention_packs(self, model) -> list[AttentionPack]:
        raise ValueError(f"{self.name} does not support attention surgery")

    def moe_blocks(self, model) -> list[MoEBlock]:
        raise ValueError(f"{self.name} does not support MoE surgery")

    def apply_mlp_selection(self, model, keep_indices: list[torch.Tensor]) -> dict:
        raise ValueError(f"{self.name} does not support MLP surgery")

    def apply_attention_selection(self, model, keep_groups: list[torch.Tensor]) -> dict:
        raise ValueError(f"{self.name} does not support attention surgery")

    def apply_moe_selection(self, model, keep_experts: list[torch.Tensor]) -> dict:
        raise ValueError(f"{self.name} does not support MoE surgery")
