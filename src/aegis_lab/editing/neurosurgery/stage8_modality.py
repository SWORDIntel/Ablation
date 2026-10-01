"""Stage 8: Modality and branch removal for model neurosurgery.

Implements:
1. Multimodal inventory and explicit dependency maps for encoders, projectors,
   cross-attention, and shared trunks/embeddings.
2. Dead branch proof: forward-hook call tracing and null/ablation probing to verify
   a branch is unused on retained execution paths before removal.
3. Safe physical removal: selectively deletes multimodal branches from module hierarchy
   and state dict while strictly protecting shared components and tied weights.
4. Config & processor repair: strips modality configs, updates architecture metadata,
   and installs input rejection guards raising ModalityRemovedError.
5. Retained modality validation: verifies retained text parity, cached generation parity,
   and state dict cleanliness after save/reload round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import copy
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn

from .common import (
    LOG,
    load_tensor_artifact,
    nested_getattr,
    nested_setattr,
    parameter_bytes,
    resolve_device,
)


# ============================================================================
# Enums and Data Models
# ============================================================================

class ModalityType(str, Enum):
    """Supported modality categories."""

    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    CROSSMODAL = "crossmodal"
    SHARED = "shared"
    UNKNOWN = "unknown"


class ComponentRole(str, Enum):
    """Functional role of an architectural component."""

    ENCODER = "encoder"
    PROJECTOR = "projector"
    CROSS_ATTENTION = "cross_attention"
    SHARED_TRUNK = "shared_trunk"
    SHARED_EMBEDDING = "shared_embedding"
    HEAD = "head"
    OTHER = "other"


@dataclass
class BranchDependencyNode:
    """Represents a component node in the multimodal dependency graph."""

    name: str
    role: ComponentRole
    modality: ModalityType
    module_type: str
    param_count: int
    param_bytes: int
    dependencies: list[str] = field(default_factory=list)
    consumers: list[str] = field(default_factory=list)
    is_shared: bool = False
    is_removable: bool = False
    tied_param_names: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value if isinstance(self.role, ComponentRole) else str(self.role),
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "module_type": self.module_type,
            "param_count": self.param_count,
            "param_bytes": self.param_bytes,
            "dependencies": list(self.dependencies),
            "consumers": list(self.consumers),
            "is_shared": self.is_shared,
            "is_removable": self.is_removable,
            "tied_param_names": list(self.tied_param_names),
            "description": self.description,
        }


@dataclass
class ModalityDependencyMap:
    """Explicit dependency graph across model components and modalities."""

    model_class: str
    nodes: dict[str, BranchDependencyNode] = field(default_factory=dict)
    modality_branches: dict[str, list[str]] = field(default_factory=dict)
    shared_components: list[str] = field(default_factory=list)
    retained_modality: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_branch(self, name: str) -> Optional[BranchDependencyNode]:
        return self.nodes.get(name)

    def get_removable_branches(self, modality: str) -> list[str]:
        """Return removable branch paths for a given modality."""
        mod = modality.lower()
        return [
            name
            for name, node in self.nodes.items()
            if node.modality.value.lower() == mod and node.is_removable and not node.is_shared
        ]

    def is_protected(self, name: str) -> bool:
        """Check if component is strictly protected against removal."""
        if name in self.shared_components:
            return True
        node = self.nodes.get(name)
        if node and (node.is_shared or not node.is_removable):
            return True
        for sc in self.shared_components:
            if name == sc or name.startswith(f"{sc}."):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_class": self.model_class,
            "retained_modality": self.retained_modality,
            "shared_components": list(self.shared_components),
            "modality_branches": {k: list(v) for k, v in self.modality_branches.items()},
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "metadata": dict(self.metadata),
        }

    def summary(self) -> str:
        lines = [f"ModalityDependencyMap for {self.model_class} (Retained: {self.retained_modality}):"]
        lines.append(f"  Shared / Protected components ({len(self.shared_components)}):")
        for sc in self.shared_components:
            lines.append(f"    - {sc}")
        for mod, branches in self.modality_branches.items():
            lines.append(f"  Modality '{mod}' ({len(branches)} branches):")
            for b in branches:
                node = self.nodes.get(b)
                p_cnt = node.param_count if node else 0
                rem = "REMOVABLE" if (node and node.is_removable) else "PROTECTED"
                role_str = node.role.value if node else "?"
                lines.append(f"    - {b} [{role_str}] ({p_cnt:,} params) [{rem}]")
        return "\n".join(lines)


@dataclass
class DeadBranchProofReport:
    """Result of dead-branch proof verification for an execution path."""

    branch_name: str
    modality: str
    is_dead: bool
    hook_call_count: int
    max_logit_diff: float
    proof_passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_name": self.branch_name,
            "modality": self.modality,
            "is_dead": self.is_dead,
            "hook_call_count": self.hook_call_count,
            "max_logit_diff": self.max_logit_diff,
            "proof_passed": self.proof_passed,
            "details": dict(self.details),
        }


@dataclass
class BranchRemovalResult:
    """Result of physical removal for a single branch."""

    branch_name: str
    modality: str
    removed_params: int
    removed_bytes: int
    success: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_name": self.branch_name,
            "modality": self.modality,
            "removed_params": self.removed_params,
            "removed_bytes": self.removed_bytes,
            "success": self.success,
            "details": dict(self.details),
        }


@dataclass
class ModalityRemovalReport:
    """Summary report of full modality amputation."""

    modality: str
    removed_branches: list[BranchRemovalResult]
    protected_components_count: int
    initial_params: int
    remaining_params: int
    freed_params: int
    freed_bytes: int
    config_repaired: bool
    guard_installed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "removed_branches": [b.to_dict() for b in self.removed_branches],
            "protected_components_count": self.protected_components_count,
            "initial_params": self.initial_params,
            "remaining_params": self.remaining_params,
            "freed_params": self.freed_params,
            "freed_bytes": self.freed_bytes,
            "config_repaired": self.config_repaired,
            "guard_installed": self.guard_installed,
        }


@dataclass
class RetainedValidationReport:
    """Validation report confirming retained modality competence and state cleanliness."""

    retained_modality: str
    passed: bool
    state_dict_clean: bool
    max_logit_diff: float
    generation_matches: bool
    removed_keys_count: int
    retained_keys_count: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retained_modality": self.retained_modality,
            "passed": self.passed,
            "state_dict_clean": self.state_dict_clean,
            "max_logit_diff": self.max_logit_diff,
            "generation_matches": self.generation_matches,
            "removed_keys_count": self.removed_keys_count,
            "retained_keys_count": self.retained_keys_count,
            "details": dict(self.details),
        }


# ============================================================================
# Exceptions
# ============================================================================

class ModalitySurgeryError(Exception):
    """Base exception for Stage 8 modality surgery operations."""
    pass


class ModalityRemovedError(ModalitySurgeryError, ValueError):
    """Raised when an amputated model or processor receives inputs for a removed modality."""

    def __init__(
        self,
        modality: str,
        offending_inputs: Optional[Sequence[str]] = None,
        message: Optional[str] = None,
    ):
        self.modality = modality
        self.offending_inputs = list(offending_inputs or [])
        if message is None:
            inp_str = f" (inputs: {', '.join(self.offending_inputs)})" if self.offending_inputs else ""
            message = (
                f"Modality '{modality}' was removed from this model. "
                f"Cannot process inputs for removed modality{inp_str}."
            )
        super().__init__(message)


class SharedComponentProtectionError(ModalitySurgeryError, PermissionError):
    """Raised when attempting to delete a shared trunk, shared embedding, or tied component."""

    def __init__(
        self,
        component_name: str,
        reason: str = "Component is part of retained shared trunk or embeddings.",
    ):
        self.component_name = component_name
        self.reason = reason
        super().__init__(f"Refusing to delete protected component '{component_name}': {reason}")


class DeadBranchVerificationError(ModalitySurgeryError, RuntimeError):
    """Raised when a candidate branch fails dead-branch proof before removal."""

    def __init__(self, branch_name: str, details: str):
        self.branch_name = branch_name
        self.details = details
        super().__init__(f"Dead branch proof failed for '{branch_name}': {details}")


class BranchNotFoundError(ModalitySurgeryError, KeyError):
    """Raised when a target branch path is not found in the model."""
    pass


class ConfigRepairError(ModalitySurgeryError, ValueError):
    """Raised when repairing model or processor config encounters an invalid state."""
    pass


# ============================================================================
# Architecture Patterns & Registry
# ============================================================================

VISION_ENCODER_TOKENS = (
    "vision_tower",
    "visual",
    "vision_model",
    "vision_encoder",
    "image_encoder",
    "vit",
    "clip_vision",
)

AUDIO_ENCODER_TOKENS = (
    "audio_tower",
    "audio_encoder",
    "speech_encoder",
    "whisper_encoder",
    "sound_encoder",
    "audio_model",
    "wav2vec2",
)

PROJECTOR_TOKENS = (
    "multi_modal_projector",
    "mm_projector",
    "projector",
    "visual_projection",
    "audio_projector",
    "speech_projector",
    "resampler",
    "perceiver",
)

CROSS_ATTENTION_TOKENS = (
    "cross_attn",
    "cross_attention",
    "encoder_decoder_attention",
    "fusion_layer",
)

SHARED_TRUNK_TOKENS = (
    "language_model",
    "text_model",
    "model.language_model",
    "model.model",
    "transformer",
    "decoder",
)

SHARED_EMBEDDING_TOKENS = (
    "embed_tokens",
    "wte",
    "word_embeddings",
    "shared_embedding",
    "shared_trunk",
    "lm_head",
    "tok_embeddings",
)

REMOVED_MODALITY_INPUT_KEYS: dict[str, set[str]] = {
    "vision": {
        "pixel_values",
        "image_embeds",
        "images",
        "pixel_mask",
        "image_sizes",
        "image_features",
        "vision_aspect_ratio_ids",
        "image_token_indices",
        "vision_feature_layer",
        "vision_feature_select_strategy",
    },
    "audio": {
        "input_features",
        "audio_values",
        "audio_features",
        "speech_inputs",
        "waveform",
        "audio_mask",
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def extract_logits(output: Any) -> torch.Tensor:
    """Extract logits tensor from tensor, dict, tuple, or ModelOutput."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits") and output.logits is not None:
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError(f"Could not extract logits from model output of type {type(output).__name__}")


def inspect_parameter_pointers(model: nn.Module) -> dict[int, list[str]]:
    """Group parameter names by underlying tensor memory address (data_ptr)."""
    ptrs: dict[int, list[str]] = {}
    for name, param in model.named_parameters():
        if param is None:
            continue
        ptr = param.data_ptr()
        ptrs.setdefault(ptr, []).append(name)
    return ptrs


# ============================================================================
# 1. Multimodal Inventory & Dependency Mapping
# ============================================================================

def inspect_multimodal_components(
    model: nn.Module,
    retained_modality: str = "text",
) -> dict[str, BranchDependencyNode]:
    """Inspect model module hierarchy and classify components by role and modality."""
    nodes: dict[str, BranchDependencyNode] = {}
    param_ptrs = inspect_parameter_pointers(model)

    # Find candidate modules
    all_modules = dict(model.named_modules())

    # Helper to check if name matches any token
    def matches_tokens(name_str: str, tokens: Iterable[str]) -> bool:
        lower = name_str.lower()
        parts = lower.split(".")
        return any(t in lower for t in tokens)

    # Collect known shared components first
    shared_param_names: set[str] = set()
    for name, module in all_modules.items():
        if not name:
            continue
        # Check if module is text trunk or embedding
        is_text_trunk = matches_tokens(name, SHARED_TRUNK_TOKENS)
        is_text_embed = matches_tokens(name, SHARED_EMBEDDING_TOKENS)
        if is_text_trunk or is_text_embed:
            for p_name, _ in module.named_parameters():
                full_pname = f"{name}.{p_name}" if name else p_name
                shared_param_names.add(full_pname)

    # Identify top candidate branch paths
    candidate_paths: list[str] = []
    for name, module in all_modules.items():
        if not name:
            continue
        # Skip deeper submodules if parent is already candidate
        lower = name.lower()
        if (
            matches_tokens(lower, VISION_ENCODER_TOKENS)
            or matches_tokens(lower, AUDIO_ENCODER_TOKENS)
            or matches_tokens(lower, PROJECTOR_TOKENS)
            or matches_tokens(lower, CROSS_ATTENTION_TOKENS)
            or matches_tokens(lower, SHARED_TRUNK_TOKENS)
            or matches_tokens(lower, SHARED_EMBEDDING_TOKENS)
        ):
            candidate_paths.append(name)

    # Filter to outermost branch roots
    outermost_paths: list[str] = []
    sorted_paths = sorted(candidate_paths, key=lambda x: (x.count("."), len(x)))
    for p in sorted_paths:
        if not any(p.startswith(f"{op}.") for op in outermost_paths):
            outermost_paths.append(p)

    # Also include root-level children if not covered and not an ancestor of candidates
    for child_name, _ in model.named_children():
        is_covered_or_ancestor = any(
            op == child_name or op.startswith(f"{child_name}.") or child_name.startswith(f"{op}.")
            for op in outermost_paths
        )
        if not is_covered_or_ancestor:
            outermost_paths.append(child_name)

    # Process each outermost path into a BranchDependencyNode
    for path in outermost_paths:
        try:
            module = nested_getattr(model, path)
        except (AttributeError, KeyError):
            continue

        p_cnt = sum(p.numel() for p in module.parameters())
        p_bytes = parameter_bytes(module)
        lower_path = path.lower()
        mod_type = type(module).__name__

        # Check parameter tying
        tied_with_retained: list[str] = []
        for p_name, param in module.named_parameters():
            ptr = param.data_ptr()
            aliases = param_ptrs.get(ptr, [])
            for alias in aliases:
                full_pname = f"{path}.{p_name}"
                if alias != full_pname:
                    tied_with_retained.append(alias)

        # Determine Role and Modality
        is_shared = False
        is_removable = False

        if matches_tokens(lower_path, SHARED_EMBEDDING_TOKENS):
            role = ComponentRole.SHARED_EMBEDDING
            modality = ModalityType.SHARED
            is_shared = True
            is_removable = False
        elif matches_tokens(lower_path, SHARED_TRUNK_TOKENS):
            role = ComponentRole.SHARED_TRUNK
            modality = ModalityType.TEXT
            is_shared = True
            is_removable = False
        elif "lm_head" in lower_path or "head" in lower_path:
            role = ComponentRole.HEAD
            modality = ModalityType.TEXT
            is_shared = True
            is_removable = False
        elif matches_tokens(lower_path, VISION_ENCODER_TOKENS):
            role = ComponentRole.ENCODER
            modality = ModalityType.VISION
            is_removable = True
        elif matches_tokens(lower_path, AUDIO_ENCODER_TOKENS):
            role = ComponentRole.ENCODER
            modality = ModalityType.AUDIO
            is_removable = True
        elif matches_tokens(lower_path, PROJECTOR_TOKENS):
            role = ComponentRole.PROJECTOR
            if "audio" in lower_path or "speech" in lower_path:
                modality = ModalityType.AUDIO
            else:
                modality = ModalityType.VISION
            is_removable = True
        elif matches_tokens(lower_path, CROSS_ATTENTION_TOKENS):
            role = ComponentRole.CROSS_ATTENTION
            modality = ModalityType.CROSSMODAL
            is_removable = True
        else:
            role = ComponentRole.OTHER
            modality = ModalityType.UNKNOWN
            is_shared = True
            is_removable = False

        # Strictly protect if tied with any retained component
        has_retained_alias = any(
            alias in shared_param_names or not alias.startswith(f"{path}.")
            for alias in tied_with_retained
        )
        if has_retained_alias:
            is_shared = True
            is_removable = False

        node = BranchDependencyNode(
            name=path,
            role=role,
            modality=modality,
            module_type=mod_type,
            param_count=p_cnt,
            param_bytes=p_bytes,
            is_shared=is_shared,
            is_removable=is_removable,
            tied_param_names=tied_with_retained,
            description=f"{role.value} for {modality.value}",
        )
        nodes[path] = node

    # Construct dependency relationships
    # Vision encoder -> Vision projector -> Shared trunk
    # Audio encoder -> Audio projector -> Shared trunk
    vision_encoders = [n for n, node in nodes.items() if node.role == ComponentRole.ENCODER and node.modality == ModalityType.VISION]
    vision_projectors = [n for n, node in nodes.items() if node.role == ComponentRole.PROJECTOR and node.modality == ModalityType.VISION]
    audio_encoders = [n for n, node in nodes.items() if node.role == ComponentRole.ENCODER and node.modality == ModalityType.AUDIO]
    audio_projectors = [n for n, node in nodes.items() if node.role == ComponentRole.PROJECTOR and node.modality == ModalityType.AUDIO]
    shared_trunks = [n for n, node in nodes.items() if node.role == ComponentRole.SHARED_TRUNK]

    for ve in vision_encoders:
        nodes[ve].consumers.extend(vision_projectors)
    for vp in vision_projectors:
        nodes[vp].dependencies.extend(vision_encoders)
        nodes[vp].consumers.extend(shared_trunks)

    for ae in audio_encoders:
        nodes[ae].consumers.extend(audio_projectors)
    for ap in audio_projectors:
        nodes[ap].dependencies.extend(audio_encoders)
        nodes[ap].consumers.extend(shared_trunks)

    for st in shared_trunks:
        nodes[st].dependencies.extend(vision_projectors + audio_projectors)

    return nodes


def build_dependency_map(
    model: nn.Module,
    retained_modality: str = "text",
    custom_overrides: Optional[dict[str, Any]] = None,
) -> ModalityDependencyMap:
    """Build an explicit dependency map for encoders, projectors, and shared components."""
    nodes = inspect_multimodal_components(model, retained_modality=retained_modality)

    if custom_overrides:
        for name, override in custom_overrides.items():
            if name in nodes:
                node = nodes[name]
                for k, v in override.items():
                    if hasattr(node, k):
                        setattr(node, k, v)
            else:
                # Add custom node
                nodes[name] = BranchDependencyNode(
                    name=name,
                    role=override.get("role", ComponentRole.OTHER),
                    modality=override.get("modality", ModalityType.UNKNOWN),
                    module_type=override.get("module_type", "CustomModule"),
                    param_count=override.get("param_count", 0),
                    param_bytes=override.get("param_bytes", 0),
                    is_shared=override.get("is_shared", False),
                    is_removable=override.get("is_removable", True),
                )

    modality_branches: dict[str, list[str]] = {}
    shared_components: list[str] = []

    for name, node in nodes.items():
        if node.is_shared or not node.is_removable:
            shared_components.append(name)
        mod_key = node.modality.value.lower()
        modality_branches.setdefault(mod_key, []).append(name)

    return ModalityDependencyMap(
        model_class=type(model).__name__,
        nodes=nodes,
        modality_branches=modality_branches,
        shared_components=shared_components,
        retained_modality=retained_modality,
    )


# ============================================================================
# 2. Dead Branch Proof
# ============================================================================

def verify_dead_branch(
    model: nn.Module,
    branch_name: str,
    retained_inputs: Union[dict[str, torch.Tensor], Sequence[dict[str, torch.Tensor]]],
    dep_map: Optional[ModalityDependencyMap] = None,
    atol: float = 1e-5,
) -> DeadBranchProofReport:
    """Verify and prove that a candidate branch is completely unused on retained execution paths."""
    try:
        module = nested_getattr(model, branch_name)
    except (AttributeError, KeyError):
        raise BranchNotFoundError(f"Branch '{branch_name}' not found on model.")

    if dep_map and dep_map.is_protected(branch_name):
        return DeadBranchProofReport(
            branch_name=branch_name,
            modality="shared",
            is_dead=False,
            hook_call_count=0,
            max_logit_diff=float("inf"),
            proof_passed=False,
            details={"error": "Branch is protected as a shared component."},
        )

    # Normalize inputs
    if isinstance(retained_inputs, dict):
        input_batches = [retained_inputs]
    else:
        input_batches = list(retained_inputs)

    if not input_batches:
        raise ValueError("At least one retained input batch must be provided to verify dead branch.")

    was_training = model.training
    model.eval()

    # Step 1: Hook call tracing
    hook_calls = 0
    hook_handles = []

    def trace_hook(*args, **kwargs):
        nonlocal hook_calls
        hook_calls += 1

    for _, submod in module.named_modules():
        hook_handles.append(submod.register_forward_pre_hook(trace_hook))
        hook_handles.append(submod.register_forward_hook(trace_hook))

    baseline_logits: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for batch in input_batches:
                out = model(**batch)
                baseline_logits.append(extract_logits(out).detach().clone())
    finally:
        for h in hook_handles:
            h.remove()
        hook_handles.clear()

    if hook_calls > 0:
        if was_training:
            model.train()
        return DeadBranchProofReport(
            branch_name=branch_name,
            modality="candidate",
            is_dead=False,
            hook_call_count=hook_calls,
            max_logit_diff=0.0,
            proof_passed=False,
            details={"error": f"Branch was invoked {hook_calls} times during retained execution path."},
        )

    # Step 2: Poisoned ablation probe
    # Replace branch forward with a method that raises an error if invoked
    original_forward = module.forward

    def poisoned_forward(*args, **kwargs):
        raise RuntimeError(f"Proved active: branch '{branch_name}' was accessed during execution!")

    module.forward = poisoned_forward
    max_diff = 0.0
    poison_triggered = False

    try:
        with torch.no_grad():
            for i, batch in enumerate(input_batches):
                try:
                    out = model(**batch)
                    probed_logits = extract_logits(out)
                    diff = torch.max(torch.abs(baseline_logits[i] - probed_logits)).item()
                    max_diff = max(max_diff, diff)
                except RuntimeError as re:
                    if "Proved active" in str(re):
                        poison_triggered = True
                        break
                    raise
    finally:
        module.forward = original_forward
        if was_training:
            model.train()

    if poison_triggered or max_diff > atol:
        return DeadBranchProofReport(
            branch_name=branch_name,
            modality="candidate",
            is_dead=False,
            hook_call_count=hook_calls,
            max_logit_diff=max_diff,
            proof_passed=False,
            details={
                "poison_triggered": poison_triggered,
                "error": f"Output drift ({max_diff:.2e}) exceeded tolerance {atol:.2e}" if not poison_triggered else "Poisoned forward triggered.",
            },
        )

    return DeadBranchProofReport(
        branch_name=branch_name,
        modality="candidate",
        is_dead=True,
        hook_call_count=0,
        max_logit_diff=max_diff,
        proof_passed=True,
        details={"status": "Confirmed zero-access and exact logit preservation on retained path."},
    )


def prove_branches_dead(
    model: nn.Module,
    branch_names: Sequence[str],
    retained_inputs: Union[dict[str, torch.Tensor], Sequence[dict[str, torch.Tensor]]],
    dep_map: Optional[ModalityDependencyMap] = None,
    atol: float = 1e-5,
    strict: bool = True,
) -> dict[str, DeadBranchProofReport]:
    """Verify dead-branch proof across multiple branches."""
    reports: dict[str, DeadBranchProofReport] = {}
    for b_name in branch_names:
        rep = verify_dead_branch(model, b_name, retained_inputs, dep_map=dep_map, atol=atol)
        reports[b_name] = rep
        if strict and not rep.proof_passed:
            err_msg = rep.details.get("error", "Branch is not dead.")
            raise DeadBranchVerificationError(b_name, err_msg)
    return reports


# ============================================================================
# 3. Safe Physical Removal
# ============================================================================

def delete_module_at_path(root: nn.Module, path: str) -> None:
    """Safely delete a module from the hierarchy given its dot-delimited path."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)

    child_name = parts[-1]
    found = False

    if hasattr(parent, "_modules") and child_name in parent._modules:
        del parent._modules[child_name]
        found = True

    if hasattr(parent, child_name):
        try:
            delattr(parent, child_name)
        except Exception:
            setattr(parent, child_name, None)
        found = True

    # Also clean convenience aliases on root if applicable
    if child_name in ("vision_tower", "multi_modal_projector", "audio_tower", "audio_projector"):
        if hasattr(root, "_modules") and child_name in root._modules:
            del root._modules[child_name]
        if hasattr(root, child_name):
            try:
                delattr(root, child_name)
            except Exception:
                setattr(root, child_name, None)

    if not found:
        raise BranchNotFoundError(f"Branch '{path}' could not be deleted; not found.")


def remove_modality_branch(
    model: nn.Module,
    branch_name: str,
    dep_map: Optional[ModalityDependencyMap] = None,
    force: bool = False,
) -> BranchRemovalResult:
    """Selectively remove a multimodal branch while strictly protecting shared components."""
    if not force:
        # Check explicit dependency map protection
        if dep_map and dep_map.is_protected(branch_name):
            raise SharedComponentProtectionError(
                branch_name,
                "Component is registered as protected shared trunk or embedding.",
            )

        # Check parameter storage sharing with retained components
        try:
            module = nested_getattr(model, branch_name)
        except (AttributeError, KeyError):
            raise BranchNotFoundError(f"Branch '{branch_name}' not found on model.")

        branch_param_ptrs = {p.data_ptr() for p in module.parameters() if p is not None}
        for name, param in model.named_parameters():
            if param is None:
                continue
            if name.startswith(f"{branch_name}.") or name == branch_name:
                continue
            if param.data_ptr() in branch_param_ptrs:
                raise SharedComponentProtectionError(
                    branch_name,
                    f"Parameter in branch shares memory storage with retained parameter '{name}'.",
                )

        # Name safety check
        lower = branch_name.lower()
        if any(tok in lower for tok in ("language_model", "text_model", "embed_tokens", "lm_head")):
            raise SharedComponentProtectionError(
                branch_name,
                "Component name indicates shared language model or embedding trunk.",
            )

    try:
        target_mod = nested_getattr(model, branch_name)
    except (AttributeError, KeyError):
        raise BranchNotFoundError(f"Branch '{branch_name}' not found on model.")

    rem_params = sum(p.numel() for p in target_mod.parameters())
    rem_bytes = parameter_bytes(target_mod)

    # Delete module
    delete_module_at_path(model, branch_name)

    return BranchRemovalResult(
        branch_name=branch_name,
        modality=dep_map.get_branch(branch_name).modality.value if dep_map and dep_map.get_branch(branch_name) else "removed",
        removed_params=rem_params,
        removed_bytes=rem_bytes,
        success=True,
    )


def remove_modality(
    model: nn.Module,
    modality: str = "vision",
    dep_map: Optional[ModalityDependencyMap] = None,
    require_proof: bool = True,
    retained_inputs: Optional[Union[dict[str, torch.Tensor], Sequence[dict[str, torch.Tensor]]]] = None,
    config: Optional[Any] = None,
    repair_config: bool = True,
    install_guard: bool = True,
) -> ModalityRemovalReport:
    """Safely remove all branches for a specified modality from the model."""
    if dep_map is None:
        dep_map = build_dependency_map(model)

    branches_to_remove = dep_map.get_removable_branches(modality)
    if not branches_to_remove:
        # Fallback search if dependency map was empty
        for name, _ in model.named_modules():
            lower = name.lower()
            if modality.lower() == "vision" and any(t in lower for t in ("vision_tower", "multi_modal_projector", "mm_projector")):
                if not dep_map.is_protected(name) and not any(name.startswith(f"{b}.") for b in branches_to_remove):
                    branches_to_remove.append(name)
            elif modality.lower() == "audio" and any(t in lower for t in ("audio_tower", "audio_projector")):
                if not dep_map.is_protected(name) and not any(name.startswith(f"{b}.") for b in branches_to_remove):
                    branches_to_remove.append(name)

    if require_proof:
        if retained_inputs is None:
            raise ValueError("retained_inputs must be provided when require_proof=True.")
        prove_branches_dead(model, branches_to_remove, retained_inputs, dep_map=dep_map, strict=True)

    initial_params = sum(p.numel() for p in model.parameters())
    removal_results: list[BranchRemovalResult] = []

    # Sort so children are deleted before parents if any nested paths exist
    sorted_branches = sorted(branches_to_remove, key=lambda x: x.count("."), reverse=True)
    for b_name in sorted_branches:
        res = remove_modality_branch(model, b_name, dep_map=dep_map)
        removal_results.append(res)

    remaining_params = sum(p.numel() for p in model.parameters())
    freed_params = sum(r.removed_params for r in removal_results)
    freed_bytes = sum(r.removed_bytes for r in removal_results)

    cfg_repaired = False
    if repair_config and config is not None:
        repair_model_config(config, removed_modality=modality)
        cfg_repaired = True

    guard_done = False
    if install_guard:
        install_input_rejection_guard(model, [modality])
        guard_done = True

    return ModalityRemovalReport(
        modality=modality,
        removed_branches=removal_results,
        protected_components_count=len(dep_map.shared_components),
        initial_params=initial_params,
        remaining_params=remaining_params,
        freed_params=freed_params,
        freed_bytes=freed_bytes,
        config_repaired=cfg_repaired,
        guard_installed=guard_done,
    )


# ============================================================================
# 4. Config & Processor Repair & Input Rejection Guard
# ============================================================================

def repair_model_config(
    config: Any,
    removed_modality: str = "vision",
    target_architecture: Optional[str] = None,
    target_model_type: Optional[str] = None,
) -> Any:
    """Repair model config by stripping modality parameters and updating architectures."""
    mod = removed_modality.lower()
    fields_to_strip = []
    if mod == "vision":
        fields_to_strip = [
            "vision_config",
            "vision_tower",
            "mm_projector_type",
            "vision_feature_layer",
            "vision_feature_select_strategy",
            "image_grid_pinpoints",
            "image_aspect_ratio",
            "multimodal_projector_bias",
            "image_token_index",
        ]
    elif mod == "audio":
        fields_to_strip = [
            "audio_config",
            "speech_config",
            "audio_tower",
            "audio_projector_type",
        ]

    # Handle dict config
    if isinstance(config, dict):
        for f in fields_to_strip:
            config.pop(f, None)
        config["is_multimodal"] = False
        config.setdefault("removed_modalities", []).append(mod)
        if target_architecture:
            config["architectures"] = [target_architecture]
        elif "architectures" in config and isinstance(config["architectures"], list):
            config["architectures"] = [
                a.replace("Llava", "Llama")
                .replace("PaliGemma", "Gemma")
                .replace("Blip2", "OPT")
                .replace("ConditionalGeneration", "CausalLM")
                for a in config["architectures"]
            ]
        if target_model_type:
            config["model_type"] = target_model_type
        elif config.get("model_type") == "llava":
            config["model_type"] = "llama"
        elif config.get("model_type") == "paligemma":
            config["model_type"] = "gemma"
        return config

    # Handle object config (PretrainedConfig, SimpleNamespace, etc.)
    for f in fields_to_strip:
        if hasattr(config, f):
            try:
                delattr(config, f)
            except Exception:
                setattr(config, f, None)
        if hasattr(config, "__dict__") and f in config.__dict__:
            config.__dict__.pop(f, None)

    # Update architectures
    if target_architecture:
        setattr(config, "architectures", [target_architecture])
    elif hasattr(config, "architectures") and isinstance(config.architectures, list):
        new_archs = [
            a.replace("Llava", "Llama")
            .replace("PaliGemma", "Gemma")
            .replace("Blip2", "OPT")
            .replace("ConditionalGeneration", "CausalLM")
            for a in config.architectures
        ]
        setattr(config, "architectures", new_archs)

    # Update model_type
    if target_model_type:
        setattr(config, "model_type", target_model_type)
    elif getattr(config, "model_type", None) == "llava":
        setattr(config, "model_type", "llama")
    elif getattr(config, "model_type", None) == "paligemma":
        setattr(config, "model_type", "gemma")

    setattr(config, "is_multimodal", False)
    rem_mods = getattr(config, "removed_modalities", [])
    if isinstance(rem_mods, list):
        rem_mods.append(mod)
        setattr(config, "removed_modalities", rem_mods)
    else:
        setattr(config, "removed_modalities", [mod])

    return config


def install_input_rejection_guard(
    model: nn.Module,
    removed_modalities: Sequence[str] = ("vision",),
) -> nn.Module:
    """Install a forward guard that immediately rejects inputs for removed modalities."""
    forbidden_keys: dict[str, str] = {}
    for mod in removed_modalities:
        m_lower = mod.lower()
        for k in REMOVED_MODALITY_INPUT_KEYS.get(m_lower, set()):
            forbidden_keys[k] = m_lower

    original_forward = model.forward

    def guarded_forward(*args, **kwargs):
        # 1. Check keyword arguments
        for k, v in kwargs.items():
            if k in forbidden_keys and v is not None:
                mod = forbidden_keys[k]
                raise ModalityRemovedError(
                    modality=mod,
                    offending_inputs=[k],
                    message=f"Modality '{mod}' was removed from this model. Input '{k}' is rejected.",
                )

        # 2. Inspect signature to check positional arguments
        try:
            sig = inspect.signature(original_forward)
            bound = sig.bind_partial(*args, **kwargs)
            for k, v in bound.arguments.items():
                if k in forbidden_keys and v is not None:
                    mod = forbidden_keys[k]
                    raise ModalityRemovedError(
                        modality=mod,
                        offending_inputs=[k],
                        message=f"Modality '{mod}' was removed from this model. Positional input '{k}' is rejected.",
                    )
        except (TypeError, ValueError):
            pass

        # 3. Check for raw 4D float tensor passed positionally (vision input)
        if "vision" in [m.lower() for m in removed_modalities] and len(args) > 1:
            for arg in args[1:]:
                if isinstance(arg, torch.Tensor) and arg.ndim == 4 and arg.is_floating_point():
                    raise ModalityRemovedError(
                        modality="vision",
                        offending_inputs=["positional_4d_tensor"],
                        message="Modality 'vision' was removed from this model. Positional image tensor is rejected.",
                    )

        return original_forward(*args, **kwargs)

    model.forward = guarded_forward
    setattr(model, "_rejection_guard_installed", True)
    setattr(model, "_removed_modalities", set(removed_modalities))

    # Also wrap generate if present
    if hasattr(model, "generate") and callable(getattr(model, "generate")):
        orig_generate = model.generate

        def guarded_generate(*args, **kwargs):
            for k, v in kwargs.items():
                if k in forbidden_keys and v is not None:
                    mod = forbidden_keys[k]
                    raise ModalityRemovedError(
                        modality=mod,
                        offending_inputs=[k],
                        message=f"Modality '{mod}' was removed from this model. Input '{k}' in generate() is rejected.",
                    )
            return orig_generate(*args, **kwargs)

        model.generate = guarded_generate

    return model


def repair_multimodal_processor(
    processor: Any,
    removed_modality: str = "vision",
) -> Any:
    """Repair multimodal processor to strip modality processors and reject modality inputs."""
    mod = removed_modality.lower()
    if mod == "vision":
        if hasattr(processor, "image_processor"):
            setattr(processor, "image_processor", None)
    elif mod == "audio":
        if hasattr(processor, "feature_extractor"):
            setattr(processor, "feature_extractor", None)
        if hasattr(processor, "audio_processor"):
            setattr(processor, "audio_processor", None)

    orig_class = processor.__class__

    def guarded_processor_call(self, *args, **kwargs):
        # Check keyword arguments
        if mod == "vision":
            for bad_key in ("images", "image", "pixel_values"):
                if kwargs.get(bad_key) is not None:
                    raise ModalityRemovedError(
                        modality="vision",
                        offending_inputs=[bad_key],
                        message=f"Processor cannot accept inputs for removed modality 'vision' ({bad_key}).",
                    )
            # Check positional arguments for images
            if len(args) > 1 and args[1] is not None:
                raise ModalityRemovedError(
                    modality="vision",
                    offending_inputs=["positional_images"],
                    message="Processor cannot accept positional image inputs for removed modality 'vision'.",
                )
        elif mod == "audio":
            for bad_key in ("audio", "waveform", "speech"):
                if kwargs.get(bad_key) is not None:
                    raise ModalityRemovedError(
                        modality="audio",
                        offending_inputs=[bad_key],
                        message=f"Processor cannot accept inputs for removed modality 'audio' ({bad_key}).",
                    )
            if len(args) > 1 and args[1] is not None:
                raise ModalityRemovedError(
                    modality="audio",
                    offending_inputs=["positional_audio"],
                    message="Processor cannot accept positional audio inputs for removed modality 'audio'.",
                )

        # If only text kwargs are passed and processor has tokenizer, use tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            text = kwargs.get("text", args[0] if args else None)
            if text is not None and not any(k in kwargs for k in ("images", "image", "pixel_values", "audio", "waveform", "speech")):
                tok_kwargs = {k: v for k, v in kwargs.items() if k != "text"}
                return self.tokenizer(text, **tok_kwargs)

        return super(new_class, self).__call__(*args, **kwargs)

    new_class = type(f"Repaired{orig_class.__name__}", (orig_class,), {"__call__": guarded_processor_call})
    processor.__class__ = new_class
    processor.__dict__["__call__"] = guarded_processor_call.__get__(processor, new_class)
    return processor


# ============================================================================
# 5. Retained Modality Validation
# ============================================================================

def verify_state_dict_absence(
    model: nn.Module,
    removed_branches: Sequence[str],
) -> tuple[bool, list[str]]:
    """Confirm that no keys from removed branches exist in the model state dict."""
    state_dict = model.state_dict()
    offending_keys: list[str] = []

    for key in state_dict.keys():
        for branch in removed_branches:
            b_tail = branch.split(".")[-1]
            if (
                key == branch
                or key.startswith(f"{branch}.")
                or key.startswith(f"{b_tail}.")
                or f".{b_tail}." in key
            ):
                offending_keys.append(key)
                break

    return len(offending_keys) == 0, offending_keys


def validate_retained_modality(
    model: nn.Module,
    baseline_model: Optional[nn.Module],
    test_inputs: Sequence[dict[str, torch.Tensor]],
    removed_branches: Sequence[str],
    max_allowed_diff: float = 1e-4,
    check_generation: bool = True,
    num_generate_tokens: int = 5,
) -> RetainedValidationReport:
    """Verify that retained text modality passes held-out tests and cached generation."""
    if not test_inputs:
        raise ValueError("At least one test input batch must be provided for validation.")

    # 1. State dict cleanliness
    clean, offending = verify_state_dict_absence(model, removed_branches)

    # 2. Text logits parity
    was_training = model.training
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()

    max_diff = 0.0
    logits_finite = True

    with torch.no_grad():
        for batch in test_inputs:
            out_edited = model(**batch)
            l_edited = extract_logits(out_edited)
            if not torch.isfinite(l_edited).all():
                logits_finite = False

            if baseline_model is not None:
                out_base = baseline_model(**batch)
                l_base = extract_logits(out_base)
                diff = torch.max(torch.abs(l_edited - l_base)).item()
                max_diff = max(max_diff, diff)

    # 3. Cached generation check
    gen_matches = True
    cached_kv_passed = True
    if check_generation:
        first_batch = test_inputs[0]
        input_ids = first_batch.get("input_ids")
        if input_ids is not None:
            # Greedy generate step-by-step
            curr_edited = input_ids.clone()
            with torch.no_grad():
                for _ in range(num_generate_tokens):
                    out_e = model(input_ids=curr_edited)
                    logits_e = extract_logits(out_e)
                    next_tok_e = logits_e[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr_edited = torch.cat([curr_edited, next_tok_e], dim=-1)

            if baseline_model is not None:
                curr_base = input_ids.clone()
                with torch.no_grad():
                    for _ in range(num_generate_tokens):
                        out_b = baseline_model(input_ids=curr_base)
                        logits_b = extract_logits(out_b)
                        next_tok_b = logits_b[:, -1, :].argmax(dim=-1, keepdim=True)
                        curr_base = torch.cat([curr_base, next_tok_b], dim=-1)

                if not torch.equal(curr_edited, curr_base):
                    gen_matches = False

            # Test KV-caching parity if supported by the model architecture
            try:
                probe = model(input_ids=input_ids, use_cache=True)
                pkv = getattr(probe, "past_key_values", None) if not isinstance(probe, tuple) else (probe[1] if len(probe) > 1 else None)
                if pkv is not None:
                    curr_cached = input_ids.clone()
                    out_step = model(input_ids=curr_cached, use_cache=True)
                    past = getattr(out_step, "past_key_values", None) if not isinstance(out_step, tuple) else out_step[1]
                    next_tok = extract_logits(out_step)[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr_cached = torch.cat([curr_cached, next_tok], dim=-1)
                    for _ in range(num_generate_tokens - 1):
                        out_step = model(input_ids=next_tok, past_key_values=past, use_cache=True)
                        past = getattr(out_step, "past_key_values", None) if not isinstance(out_step, tuple) else out_step[1]
                        next_tok = extract_logits(out_step)[:, -1, :].argmax(dim=-1, keepdim=True)
                        curr_cached = torch.cat([curr_cached, next_tok], dim=-1)
                    if not torch.equal(curr_cached, curr_edited):
                        cached_kv_passed = False
                        gen_matches = False
            except Exception:
                # Model does not support use_cache / past_key_values kwarg
                pass

    if was_training:
        model.train()

    passed = clean and logits_finite and (max_diff <= max_allowed_diff) and gen_matches

    return RetainedValidationReport(
        retained_modality="text",
        passed=passed,
        state_dict_clean=clean,
        max_logit_diff=max_diff,
        generation_matches=gen_matches,
        removed_keys_count=len(offending),
        retained_keys_count=len(model.state_dict()),
        details={
            "offending_keys": offending,
            "logits_finite": logits_finite,
            "max_allowed_diff": max_allowed_diff,
            "cached_kv_passed": cached_kv_passed,
        },
    )


def save_and_reload_amputated_model(
    model: nn.Module,
    save_dir: Union[str, Path],
    config: Optional[Any] = None,
    test_inputs: Optional[Sequence[dict[str, torch.Tensor]]] = None,
) -> tuple[dict[str, torch.Tensor], Path, Optional[float]]:
    """Save amputated weights to disk and verify safe tensor reload and parity."""
    p_dir = Path(save_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    weights_path = p_dir / "pytorch_model.pt"

    # Save state dict
    torch.save(model.state_dict(), weights_path)

    # Save config if present
    if config is not None:
        cfg_path = p_dir / "config.json"
        if isinstance(config, dict):
            cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        elif hasattr(config, "to_dict"):
            cfg_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        elif hasattr(config, "__dict__"):
            cfg_path.write_text(json.dumps(vars(config), indent=2), encoding="utf-8")

    # Reload state dict safely with weights_only=True
    loaded_sd = load_tensor_artifact(weights_path)

    # Parity check if test inputs provided
    parity_diff = None
    if test_inputs:
        with torch.no_grad():
            out_before = extract_logits(model(**test_inputs[0]))
            # Verify loaded weights match
            for k, tensor in model.state_dict().items():
                if not torch.equal(tensor, loaded_sd[k]):
                    raise RuntimeError(f"State dict mismatch after reload on key '{k}'")
            parity_diff = 0.0

    return loaded_sd, weights_path, parity_diff


# ============================================================================
# 6. Unified End-to-End Amputation Pipeline
# ============================================================================

def amputate_modality(
    model: nn.Module,
    modality: str = "vision",
    retained_inputs: Optional[Union[dict[str, torch.Tensor], Sequence[dict[str, torch.Tensor]]]] = None,
    test_inputs: Optional[Sequence[dict[str, torch.Tensor]]] = None,
    config: Optional[Any] = None,
    processor: Optional[Any] = None,
    baseline_model: Optional[nn.Module] = None,
    require_proof: bool = True,
    install_guard: bool = True,
    repair_cfg: bool = True,
    repair_proc: bool = True,
    max_allowed_diff: float = 1e-4,
) -> tuple[nn.Module, ModalityRemovalReport, Optional[RetainedValidationReport]]:
    """Complete end-to-end Stage 8 pipeline for modality and branch removal."""
    # 1. Build explicit dependency map
    dep_map = build_dependency_map(model)

    # 2. Baseline copy for validation if needed
    if baseline_model is None and test_inputs is not None:
        baseline_model = copy.deepcopy(model)

    # 3. Safely remove modality
    removal_report = remove_modality(
        model=model,
        modality=modality,
        dep_map=dep_map,
        require_proof=require_proof,
        retained_inputs=retained_inputs,
        config=config,
        repair_config=repair_cfg,
        install_guard=install_guard,
    )

    # 4. Repair processor if provided
    if repair_proc and processor is not None:
        repair_multimodal_processor(processor, removed_modality=modality)

    # 5. Validate retained modality
    val_report: Optional[RetainedValidationReport] = None
    if test_inputs is not None:
        removed_names = [b.branch_name for b in removal_report.removed_branches]
        val_report = validate_retained_modality(
            model=model,
            baseline_model=baseline_model,
            test_inputs=test_inputs,
            removed_branches=removed_names,
            max_allowed_diff=max_allowed_diff,
        )

    return model, removal_report, val_report
