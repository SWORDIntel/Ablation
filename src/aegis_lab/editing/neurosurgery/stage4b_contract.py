from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import re
from typing import Any, Iterable, Optional, Union
import uuid

import torch
import torch.nn as nn
import yaml


# ============================================================================
# Enums and Exceptions
# ============================================================================

class OperationKind(str, Enum):
    MASK = "mask"
    SCALE = "scale"
    CLAMP = "clamp"
    PROJECT = "project"
    REPLACE = "replace"
    LOW_RANK_DELTA = "low_rank_delta"
    PHYSICAL_REMOVE = "physical_remove"


class ExecutionSemantics(str, Enum):
    RUNTIME_ONLY = "runtime_only"
    PERSISTENT = "persistent"


class ComponentType(str, Enum):
    MLP_CHANNEL = "mlp_channel"
    ATTENTION_HEAD = "attention_head"
    ATTENTION_GROUP = "attention_group"
    MOE_EXPERT = "moe_expert"
    FULL_LAYER = "full_layer"
    WEIGHT_TENSOR = "weight_tensor"


class ConflictSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ConflictType(str, Enum):
    TIED_WEIGHT_DESTRUCTION = "tied_weight_destruction"
    UNDEFINED_COMPOSITION_ORDER = "undefined_composition_order"
    INCOMPATIBLE_SEMANTICS = "incompatible_execution_semantics"
    OVERLAPPING_ALIAS_MUTATION = "overlapping_alias_mutation"


class NeurosurgeryError(Exception):
    """Base error for neurosurgery operations."""


class ValidationError(NeurosurgeryError):
    """Raised when an operation or plan fails schema or semantic validation."""


class ConflictingOperationsError(NeurosurgeryError):
    """Raised when conflicting operations or unsafe aliasing edits are detected."""

    def __init__(self, message: str, conflicts: Optional[list["ConflictReport"]] = None):
        super().__init__(message)
        self.conflicts = conflicts or []


class InvalidCompositionPathError(NeurosurgeryError):
    """Raised when composing operations along an invalid or unsupported path."""


class UnsupportedOperationError(NeurosurgeryError):
    """Raised when an operation is not supported by the adapter or architecture."""


# ============================================================================
# 1. Unified Operation Contract
# ============================================================================

def _infer_layer_index(target_path: str) -> Optional[int]:
    match = re.search(r'(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)', target_path)
    return int(match.group(1)) if match else None


def _remap_path_layer(target_path: str, new_layer_index: int) -> str:
    return re.sub(
        r'((?:^|\.)(?:layers|h)\.)(\d+)((?:\.|$))',
        lambda m: f"{m.group(1)}{new_layer_index}{m.group(3)}",
        target_path,
        count=1,
    )


@dataclass
class UnifiedOperation:
    kind: OperationKind
    target_path: str
    component_type: ComponentType
    execution_semantics: ExecutionSemantics
    export_supported: bool
    operation_id: Optional[str] = None
    layer_index: Optional[int] = None
    target_indices: Optional[list[int]] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    order: Optional[int] = None

    def __post_init__(self):
        # Coerce strings to enums if passed as raw strings
        if isinstance(self.kind, str):
            try:
                self.kind = OperationKind(self.kind)
            except ValueError as exc:
                raise ValidationError(f"Invalid operation kind: {self.kind}") from exc

        if isinstance(self.component_type, str):
            try:
                self.component_type = ComponentType(self.component_type)
            except ValueError as exc:
                raise ValidationError(f"Invalid component type: {self.component_type}") from exc

        if isinstance(self.execution_semantics, str):
            try:
                self.execution_semantics = ExecutionSemantics(self.execution_semantics)
            except ValueError as exc:
                raise ValidationError(f"Invalid execution semantics: {self.execution_semantics}") from exc

        if not isinstance(self.export_supported, bool):
            raise ValidationError("export_supported must be a boolean")

        if not self.target_path or not isinstance(self.target_path, str):
            raise ValidationError("target_path must be a non-empty string")

        if self.operation_id is None:
            self.operation_id = f"op_{uuid.uuid4().hex[:8]}"

        if self.layer_index is None:
            self.layer_index = _infer_layer_index(self.target_path)

        if self.target_indices is not None:
            if not isinstance(self.target_indices, list) or any(
                isinstance(x, bool) or not isinstance(x, int) for x in self.target_indices
            ):
                raise ValidationError("target_indices must be a list of integers")

        self.validate()

    def validate(self) -> None:
        """Validates semantic constraints of the operation schema."""
        # Runtime-only operations cannot declare static export support
        if self.execution_semantics == ExecutionSemantics.RUNTIME_ONLY and self.export_supported:
            raise ValidationError(
                f"Operation '{self.operation_id}' with execution_semantics='runtime_only' "
                f"cannot have export_supported=True; dynamic hooks cannot be exported to static checkpoint weights."
            )

        # Validation per operation kind
        if self.kind == OperationKind.SCALE:
            if "scale_factor" not in self.parameters:
                raise ValidationError(f"Scale operation '{self.operation_id}' requires 'scale_factor' parameter")
        elif self.kind == OperationKind.CLAMP:
            if "min_value" not in self.parameters and "max_value" not in self.parameters:
                raise ValidationError(
                    f"Clamp operation '{self.operation_id}' requires at least 'min_value' or 'max_value'"
                )
            if "min_value" in self.parameters and "max_value" in self.parameters:
                if self.parameters["min_value"] > self.parameters["max_value"]:
                    raise ValidationError(
                        f"Clamp operation '{self.operation_id}' has min_value > max_value: "
                        f"{self.parameters['min_value']} > {self.parameters['max_value']}"
                    )
        elif self.kind == OperationKind.PROJECT:
            if not any(k in self.parameters for k in ("direction", "subspace_basis", "basis")):
                raise ValidationError(
                    f"Project operation '{self.operation_id}' requires 'direction' or 'subspace_basis'"
                )
        elif self.kind == OperationKind.LOW_RANK_DELTA:
            has_rank = "rank" in self.parameters and isinstance(self.parameters["rank"], int) and self.parameters["rank"] > 0
            has_uv = "u" in self.parameters and "v" in self.parameters
            if not has_rank and not has_uv:
                raise ValidationError(
                    f"Low-rank delta operation '{self.operation_id}' requires positive 'rank' or 'u'/'v' factors"
                )
        elif self.kind == OperationKind.REPLACE:
            if not any(k in self.parameters for k in ("replacement_value", "replacement_tensor", "replacement_mode", "constant")):
                raise ValidationError(
                    f"Replace operation '{self.operation_id}' requires replacement value or mode"
                )
        elif self.kind == OperationKind.PHYSICAL_REMOVE:
            # Must declare what is removed or kept
            has_indices = (
                self.target_indices is not None
                or self.layer_index is not None
                or "kept_indices" in self.parameters
                or "removed_indices" in self.parameters
            )
            if not has_indices:
                raise ValidationError(
                    f"Physical remove operation '{self.operation_id}' must specify target_indices, layer_index, "
                    f"kept_indices, or removed_indices"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the operation into a JSON/YAML serializable dictionary."""
        def _clean(val: Any) -> Any:
            if isinstance(val, torch.Tensor):
                return val.tolist()
            if isinstance(val, dict):
                return {k: _clean(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_clean(x) for x in val]
            return val

        payload = {
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "target_path": self.target_path,
            "component_type": self.component_type.value,
            "execution_semantics": self.execution_semantics.value,
            "export_supported": self.export_supported,
            "layer_index": self.layer_index,
            "target_indices": self.target_indices,
            "parameters": _clean(self.parameters),
            "metadata": _clean(self.metadata),
            "order": self.order,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedOperation":
        """Deserializes a dictionary into a UnifiedOperation."""
        if not isinstance(data, dict):
            raise ValidationError("Operation data must be a dictionary")
        required = {"kind", "target_path", "component_type", "execution_semantics", "export_supported"}
        missing = required - set(data)
        if missing:
            raise ValidationError(f"Missing required operation fields: {missing}")

        return cls(
            operation_id=data.get("operation_id"),
            kind=OperationKind(data["kind"]),
            target_path=data["target_path"],
            component_type=ComponentType(data["component_type"]),
            execution_semantics=ExecutionSemantics(data["execution_semantics"]),
            export_supported=data["export_supported"],
            layer_index=data.get("layer_index"),
            target_indices=data.get("target_indices"),
            parameters=copy.deepcopy(data.get("parameters", {})),
            metadata=copy.deepcopy(data.get("metadata", {})),
            order=data.get("order"),
        )

    # Convenient Factory Constructors
    @classmethod
    def create_mask(
        cls,
        target_path: str,
        component_type: Union[ComponentType, str],
        target_indices: Optional[list[int]] = None,
        mask_value: float = 0.0,
        execution_semantics: Union[ExecutionSemantics, str] = ExecutionSemantics.RUNTIME_ONLY,
        export_supported: bool = False,
        **kwargs,
    ) -> "UnifiedOperation":
        params = kwargs.pop("parameters", {})
        params["mask_value"] = mask_value
        return cls(
            kind=OperationKind.MASK,
            target_path=target_path,
            component_type=ComponentType(component_type),
            execution_semantics=ExecutionSemantics(execution_semantics),
            export_supported=export_supported,
            target_indices=target_indices,
            parameters=params,
            **kwargs,
        )

    @classmethod
    def create_scale(
        cls,
        target_path: str,
        component_type: Union[ComponentType, str],
        scale_factor: Union[float, list[float]],
        target_indices: Optional[list[int]] = None,
        execution_semantics: Union[ExecutionSemantics, str] = ExecutionSemantics.PERSISTENT,
        export_supported: bool = True,
        **kwargs,
    ) -> "UnifiedOperation":
        params = kwargs.pop("parameters", {})
        params["scale_factor"] = scale_factor
        return cls(
            kind=OperationKind.SCALE,
            target_path=target_path,
            component_type=ComponentType(component_type),
            execution_semantics=ExecutionSemantics(execution_semantics),
            export_supported=export_supported,
            target_indices=target_indices,
            parameters=params,
            **kwargs,
        )

    @classmethod
    def create_clamp(
        cls,
        target_path: str,
        component_type: Union[ComponentType, str],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        target_indices: Optional[list[int]] = None,
        execution_semantics: Union[ExecutionSemantics, str] = ExecutionSemantics.RUNTIME_ONLY,
        export_supported: bool = False,
        **kwargs,
    ) -> "UnifiedOperation":
        params = kwargs.pop("parameters", {})
        if min_value is not None:
            params["min_value"] = min_value
        if max_value is not None:
            params["max_value"] = max_value
        return cls(
            kind=OperationKind.CLAMP,
            target_path=target_path,
            component_type=ComponentType(component_type),
            execution_semantics=ExecutionSemantics(execution_semantics),
            export_supported=export_supported,
            target_indices=target_indices,
            parameters=params,
            **kwargs,
        )

    @classmethod
    def create_project(
        cls,
        target_path: str,
        component_type: Union[ComponentType, str],
        direction: Optional[Any] = None,
        subspace_basis: Optional[Any] = None,
        preserve_subspace: bool = True,
        execution_semantics: Union[ExecutionSemantics, str] = ExecutionSemantics.PERSISTENT,
        export_supported: bool = True,
        **kwargs,
    ) -> "UnifiedOperation":
        params = kwargs.pop("parameters", {})
        if direction is not None:
            params["direction"] = direction
        if subspace_basis is not None:
            params["subspace_basis"] = subspace_basis
        params["preserve_subspace"] = preserve_subspace
        return cls(
            kind=OperationKind.PROJECT,
            target_path=target_path,
            component_type=ComponentType(component_type),
            execution_semantics=ExecutionSemantics(execution_semantics),
            export_supported=export_supported,
            parameters=params,
            **kwargs,
        )

    @classmethod
    def create_replace(
        cls,
        target_path: str,
        component_type: Union[ComponentType, str],
        replacement_value: Any,
        target_indices: Optional[list[int]] = None,
        execution_semantics: Union[ExecutionSemantics, str] = ExecutionSemantics.PERSISTENT,
        export_supported: bool = True,
        **kwargs,
    ) -> "UnifiedOperation":
        params = kwargs.pop("parameters", {})
        params["replacement_value"] = replacement_value
        return cls(
            kind=OperationKind.REPLACE,
            target_path=target_path,
            component_type=ComponentType(component_type),
            execution_semantics=ExecutionSemantics(execution_semantics),
            export_supported=export_supported,
            target_indices=target_indices,
            parameters=params,
            **kwargs,
        )

    @classmethod
    def create_low_rank_delta(
        cls,
        target_path: str,
        component_type: Union[ComponentType, str],
        rank: int,
        alpha: float = 1.0,
        u: Optional[Any] = None,
        v: Optional[Any] = None,
        execution_semantics: Union[ExecutionSemantics, str] = ExecutionSemantics.PERSISTENT,
        export_supported: bool = True,
        **kwargs,
    ) -> "UnifiedOperation":
        params = kwargs.pop("parameters", {})
        params["rank"] = rank
        params["alpha"] = alpha
        if u is not None:
            params["u"] = u
        if v is not None:
            params["v"] = v
        return cls(
            kind=OperationKind.LOW_RANK_DELTA,
            target_path=target_path,
            component_type=ComponentType(component_type),
            execution_semantics=ExecutionSemantics(execution_semantics),
            export_supported=export_supported,
            parameters=params,
            **kwargs,
        )

    @classmethod
    def create_physical_remove(
        cls,
        target_path: str,
        component_type: Union[ComponentType, str],
        target_indices: Optional[list[int]] = None,
        kept_indices: Optional[list[int]] = None,
        removed_indices: Optional[list[int]] = None,
        execution_semantics: Union[ExecutionSemantics, str] = ExecutionSemantics.PERSISTENT,
        export_supported: bool = True,
        **kwargs,
    ) -> "UnifiedOperation":
        params = kwargs.pop("parameters", {})
        if kept_indices is not None:
            params["kept_indices"] = kept_indices
        if removed_indices is not None:
            params["removed_indices"] = removed_indices
        return cls(
            kind=OperationKind.PHYSICAL_REMOVE,
            target_path=target_path,
            component_type=ComponentType(component_type),
            execution_semantics=ExecutionSemantics(execution_semantics),
            export_supported=export_supported,
            target_indices=target_indices,
            parameters=params,
            **kwargs,
        )


@dataclass
class UnifiedPlan:
    version: int = 4
    model_type: Optional[str] = None
    operations: list[UnifiedOperation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.version != 4:
            raise ValidationError(f"Unsupported plan version: {self.version} (expected 4)")
        if not isinstance(self.operations, list):
            raise ValidationError("Plan operations must be a list")
        for op in self.operations:
            op.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "operations": [op.to_dict() for op in self.operations],
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedPlan":
        if not isinstance(data, dict):
            raise ValidationError("Plan data must be a dictionary")
        ops = [UnifiedOperation.from_dict(d) for d in data.get("operations", [])]
        plan = cls(
            version=data.get("version", 4),
            model_type=data.get("model_type"),
            operations=ops,
            metadata=copy.deepcopy(data.get("metadata", {})),
        )
        plan.validate()
        return plan

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path_or_content: Union[str, Path]) -> "UnifiedPlan":
        if isinstance(path_or_content, Path):
            content = path_or_content.read_text(encoding="utf-8")
        elif isinstance(path_or_content, str):
            if "\n" not in path_or_content and len(path_or_content) < 4096:
                p = Path(path_or_content)
                if p.is_file():
                    content = p.read_text(encoding="utf-8")
                else:
                    content = path_or_content
            else:
                content = path_or_content
        else:
            raise ValidationError(f"Expected str or Path, got {type(path_or_content)}")

        data = yaml.safe_load(content)
        return cls.from_dict(data)


# ============================================================================
# 2. Shared-Weight & Aliasing Conflict Detection
# ============================================================================

def inspect_parameter_aliases(model: nn.Module) -> dict[str, list[str]]:
    """
    Inspects model parameter aliases (e.g. tied weights like lm_head.weight and
    model.embed_tokens.weight).

    Returns a mapping from every parameter path to the full list of parameter paths
    sharing the underlying parameter storage.
    """
    try:
        pairs = list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        pairs = list(model.named_parameters())

    storage_groups: dict[Any, list[str]] = {}
    for name, param in pairs:
        # Group by python object ID and data pointer if available
        key = (id(param), param.data_ptr() if param.numel() > 0 else 0)
        storage_groups.setdefault(key, []).append(name)

    aliases: dict[str, list[str]] = {}
    for group in storage_groups.values():
        unique_group = sorted(set(group))
        for name in unique_group:
            aliases[name] = unique_group
    return aliases


def get_alias_groups(model: nn.Module) -> list[set[str]]:
    """Returns list of tied parameter sets that share weights (2+ names)."""
    alias_map = inspect_parameter_aliases(model)
    unique_groups: set[tuple[str, ...]] = set()
    for names in alias_map.values():
        if len(names) > 1:
            unique_groups.add(tuple(sorted(names)))
    return [set(group) for group in unique_groups]


def inspect_module_aliases(model: nn.Module) -> dict[str, list[str]]:
    """Inspects shared/reused module instances across the model tree."""
    try:
        pairs = list(model.named_modules(remove_duplicate=False))
    except TypeError:
        pairs = list(model.named_modules())

    module_groups: dict[int, list[str]] = {}
    for name, module in pairs:
        path = name or "<root>"
        module_groups.setdefault(id(module), []).append(path)

    aliases: dict[str, list[str]] = {}
    for group in module_groups.values():
        unique_group = sorted(set(group))
        for path in unique_group:
            aliases[path] = unique_group
    return aliases


@dataclass
class ConflictReport:
    conflict_type: ConflictType
    severity: ConflictSeverity
    message: str
    target_paths: list[str]
    operation_ids: list[str]


class ConflictDetector:
    """Detects unsafe shared-weight edits, tied weight destruction, and composition conflicts."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        alias_map: Optional[dict[str, list[str]]] = None,
        module_alias_map: Optional[dict[str, list[str]]] = None,
    ):
        if alias_map is not None:
            self.alias_map = alias_map
        elif model is not None:
            self.alias_map = inspect_parameter_aliases(model)
        else:
            self.alias_map = {}

        if module_alias_map is not None:
            self.module_alias_map = module_alias_map
        elif model is not None:
            self.module_alias_map = inspect_module_aliases(model)
        else:
            self.module_alias_map = {}

    def _resolve_aliases_for_target(self, target_path: str) -> list[str]:
        # Direct match in parameter alias map
        if target_path in self.alias_map:
            return self.alias_map[target_path]
        # Match parameter within module
        prefix = f"{target_path}."
        matching = []
        for param_path, aliases in self.alias_map.items():
            if param_path.startswith(prefix) and len(aliases) > 1:
                matching.extend(aliases)
        if matching:
            return sorted(set(matching))
        # Match in module aliases
        if target_path in self.module_alias_map:
            return self.module_alias_map[target_path]
        return [target_path]

    def _targets_overlap(self, path1: str, path2: str, aliases1: list[str], aliases2: list[str]) -> bool:
        if set(aliases1) & set(aliases2):
            return True
        for a1 in aliases1:
            for a2 in aliases2:
                if a1 == a2 or a1.startswith(f"{a2}.") or a2.startswith(f"{a1}."):
                    return True
        return False

    def detect_conflicts(self, operations: list[UnifiedOperation]) -> list[ConflictReport]:
        conflicts: list[ConflictReport] = []

        # 1. Tied-Weight Destruction: physical removal on a parameter/module with aliases
        for op in operations:
            if op.kind == OperationKind.PHYSICAL_REMOVE:
                aliases = self._resolve_aliases_for_target(op.target_path)
                if len(aliases) > 1:
                    conflicts.append(
                        ConflictReport(
                            conflict_type=ConflictType.TIED_WEIGHT_DESTRUCTION,
                            severity=ConflictSeverity.ERROR,
                            message=(
                                f"Destructive operation '{op.operation_id}' (physical_remove) targets tied/shared "
                                f"component '{op.target_path}', which aliases: {aliases}. "
                                f"Modifying or slicing tied weights causes storage corruption or vocabulary mismatch."
                            ),
                            target_paths=aliases,
                            operation_ids=[op.operation_id],
                        )
                    )

        # 2. Overlapping and Aliasing Conflict Detection between operation pairs
        reported_pairs: set[tuple[str, str, str]] = set()

        for i, op1 in enumerate(operations):
            aliases1 = self._resolve_aliases_for_target(op1.target_path)
            for j in range(i + 1, len(operations)):
                op2 = operations[j]
                aliases2 = self._resolve_aliases_for_target(op2.target_path)

                if not self._targets_overlap(op1.target_path, op2.target_path, aliases1, aliases2):
                    continue

                common_targets = sorted(set(aliases1) | set(aliases2))

                # Check composition ordering: do they declare explicit unique order?
                if op1.order is None or op2.order is None or op1.order == op2.order:
                    pair_key = (op1.operation_id, op2.operation_id, ConflictType.UNDEFINED_COMPOSITION_ORDER.value)
                    if pair_key not in reported_pairs:
                        reported_pairs.add(pair_key)
                        conflicts.append(
                            ConflictReport(
                                conflict_type=ConflictType.UNDEFINED_COMPOSITION_ORDER,
                                severity=ConflictSeverity.ERROR,
                                message=(
                                    f"Operations '{op1.operation_id}' and '{op2.operation_id}' target overlapping "
                                    f"or aliased weights ({common_targets}) without explicit, unique declared "
                                    f"composition order (orders=[{op1.order}, {op2.order}])."
                                ),
                                target_paths=common_targets,
                                operation_ids=[op1.operation_id, op2.operation_id],
                            )
                        )

                # Check incompatible execution semantics
                if op1.execution_semantics != op2.execution_semantics:
                    pair_key = (op1.operation_id, op2.operation_id, ConflictType.INCOMPATIBLE_SEMANTICS.value)
                    if pair_key not in reported_pairs:
                        reported_pairs.add(pair_key)
                        conflicts.append(
                            ConflictReport(
                                conflict_type=ConflictType.INCOMPATIBLE_SEMANTICS,
                                severity=ConflictSeverity.ERROR,
                                message=(
                                    f"Incompatible execution semantics ['{op1.execution_semantics.value}', "
                                    f"'{op2.execution_semantics.value}'] on overlapping target ({common_targets}) "
                                    f"across operations '{op1.operation_id}' and '{op2.operation_id}'."
                                ),
                                target_paths=common_targets,
                                operation_ids=[op1.operation_id, op2.operation_id],
                            )
                        )

                # Check physical removal followed by downstream mutation
                first_op, second_op = (op1, op2) if (op1.order or 0) <= (op2.order or 0) else (op2, op1)
                if first_op.kind == OperationKind.PHYSICAL_REMOVE and second_op.kind != OperationKind.PHYSICAL_REMOVE:
                    pair_key = (first_op.operation_id, second_op.operation_id, ConflictType.OVERLAPPING_ALIAS_MUTATION.value)
                    if pair_key not in reported_pairs:
                        reported_pairs.add(pair_key)
                        conflicts.append(
                            ConflictReport(
                                conflict_type=ConflictType.OVERLAPPING_ALIAS_MUTATION,
                                severity=ConflictSeverity.ERROR,
                                message=(
                                    f"Operation '{second_op.operation_id}' targets component '{second_op.target_path}' "
                                    f"which overlaps with '{first_op.target_path}' physically removed by '{first_op.operation_id}'."
                                ),
                                target_paths=[first_op.target_path, second_op.target_path],
                                operation_ids=[first_op.operation_id, second_op.operation_id],
                            )
                        )

        return conflicts

    def validate(self, operations: list[UnifiedOperation]) -> None:
        conflicts = self.detect_conflicts(operations)
        errors = [c for c in conflicts if c.severity == ConflictSeverity.ERROR]
        if errors:
            msgs = "; ".join(f"[{c.conflict_type.value}] {c.message}" for c in errors)
            raise ConflictingOperationsError(f"Found {len(errors)} conflict(s): {msgs}", conflicts=errors)


def detect_conflicts(
    operations: list[UnifiedOperation],
    model: Optional[nn.Module] = None,
    alias_map: Optional[dict[str, list[str]]] = None,
) -> list[ConflictReport]:
    detector = ConflictDetector(model=model, alias_map=alias_map)
    return detector.detect_conflicts(operations)


def validate_no_conflicts(
    operations: list[UnifiedOperation],
    model: Optional[nn.Module] = None,
    alias_map: Optional[dict[str, list[str]]] = None,
) -> None:
    detector = ConflictDetector(model=model, alias_map=alias_map)
    detector.validate(operations)


# ============================================================================
# 3. Arbitrary Edit Composition & Index Remapping
# ============================================================================

class IndexRemapper:
    """
    Maintains original-to-current index remapping when layers, channels, heads,
    or experts are removed or sliced.
    """

    def __init__(self, initial_count: Optional[int] = None, domain: str = "index"):
        self.domain = domain
        self.initial_count = initial_count
        self._orig_to_curr: dict[int, Optional[int]] = {}
        self._curr_to_orig: dict[int, int] = {}
        self._removed_orig: set[int] = set()

        if initial_count is not None:
            if initial_count <= 0:
                raise InvalidCompositionPathError(f"Initial {domain} count must be positive, got {initial_count}")
            for i in range(initial_count):
                self._orig_to_curr[i] = i
                self._curr_to_orig[i] = i

    @property
    def current_count(self) -> int:
        return len(self._curr_to_orig)

    def is_removed(self, orig_idx: int) -> bool:
        return orig_idx in self._removed_orig

    def remap_index(self, orig_idx: int) -> int:
        if orig_idx in self._removed_orig:
            raise InvalidCompositionPathError(
                f"Target {self.domain} {orig_idx} has been removed by a preceding operation"
            )
        if orig_idx not in self._orig_to_curr:
            if self.initial_count is not None and (orig_idx < 0 or orig_idx >= self.initial_count):
                raise InvalidCompositionPathError(
                    f"Target {self.domain} {orig_idx} is out of bounds [0, {self.initial_count})"
                )
            # Dynamically register unseeded index if initial_count is open
            return orig_idx
        curr = self._orig_to_curr[orig_idx]
        if curr is None:
            raise InvalidCompositionPathError(
                f"Target {self.domain} {orig_idx} has been removed by a preceding operation"
            )
        return curr

    def remap_indices(self, orig_indices: Iterable[int]) -> list[int]:
        return [self.remap_index(i) for i in orig_indices]

    def remove_indices(self, orig_indices: Iterable[int]) -> None:
        remove_set = set(orig_indices)
        for idx in remove_set:
            if idx in self._removed_orig:
                raise InvalidCompositionPathError(
                    f"Target {self.domain} {idx} has already been removed"
                )
            if self.initial_count is not None and (idx < 0 or idx >= self.initial_count):
                raise InvalidCompositionPathError(
                    f"Target {self.domain} {idx} is out of bounds [0, {self.initial_count})"
                )
            self._removed_orig.add(idx)

        all_orig = list(range(self.initial_count)) if self.initial_count is not None else sorted(self._orig_to_curr.keys())
        surviving = [i for i in all_orig if i not in self._removed_orig]
        if not surviving:
            raise InvalidCompositionPathError(
                f"Cannot remove all {self.domain} components; remaining count cannot be 0"
            )

        self._curr_to_orig = {curr: orig for curr, orig in enumerate(surviving)}
        self._orig_to_curr = {orig: (curr if orig not in self._removed_orig else None) for curr, orig in enumerate(surviving)}
        for orig in self._removed_orig:
            self._orig_to_curr[orig] = None

    def keep_indices(self, orig_indices: Iterable[int]) -> None:
        keep_set = set(orig_indices)
        if not keep_set:
            raise InvalidCompositionPathError(
                f"Cannot keep 0 {self.domain} components; remaining count cannot be 0"
            )
        all_orig = set(range(self.initial_count)) if self.initial_count is not None else set(self._orig_to_curr.keys()) | keep_set

        for idx in keep_set:
            if idx in self._removed_orig:
                raise InvalidCompositionPathError(
                    f"Target {self.domain} {idx} has already been removed and cannot be kept"
                )
            if self.initial_count is not None and (idx < 0 or idx >= self.initial_count):
                raise InvalidCompositionPathError(
                    f"Target {self.domain} {idx} is out of bounds [0, {self.initial_count})"
                )

        to_remove = all_orig - keep_set
        self.remove_indices(to_remove)

    def get_mapping(self) -> dict[int, Optional[int]]:
        return dict(self._orig_to_curr)

    def get_inverse_mapping(self) -> dict[int, int]:
        return dict(self._curr_to_orig)


@dataclass
class ModelTopology:
    """Represents the dimensional structure of a model for index tracking."""

    num_layers: int
    mlp_intermediate_sizes: dict[int, int] = field(default_factory=dict)
    num_attention_heads: dict[int, int] = field(default_factory=dict)
    num_key_value_heads: dict[int, int] = field(default_factory=dict)
    num_experts: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelTopology":
        from .common import get_layers

        try:
            _, layers = get_layers(model)
        except Exception as exc:
            raise ValidationError(f"Could not extract transformer layers: {exc}") from exc

        num_layers = len(layers)
        mlp_sizes = {}
        attn_heads = {}
        kv_heads = {}
        expert_counts = {}

        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                gate = getattr(mlp, "gate_proj", None) or getattr(mlp, "gate", None)
                if gate is not None and hasattr(gate, "weight"):
                    mlp_sizes[i] = int(gate.weight.shape[0])

            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is not None:
                if hasattr(attn, "num_heads"):
                    attn_heads[i] = int(attn.num_heads)
                elif hasattr(attn, "num_attention_heads"):
                    attn_heads[i] = int(attn.num_attention_heads)

                if hasattr(attn, "num_key_value_heads"):
                    kv_heads[i] = int(attn.num_key_value_heads)

            moe = getattr(layer, "block_sparse_moe", None) or getattr(layer, "moe", None)
            if moe is not None and hasattr(moe, "experts"):
                expert_counts[i] = len(moe.experts)

        return cls(
            num_layers=num_layers,
            mlp_intermediate_sizes=mlp_sizes,
            num_attention_heads=attn_heads,
            num_key_value_heads=kv_heads,
            num_experts=expert_counts,
        )


@dataclass
class ComposedPipeline:
    """Result of arbitrary edit composition with remapped target paths and indices."""

    remapped_operations: list[UnifiedOperation]
    original_to_current_layer_map: dict[int, Optional[int]]
    removed_layers: set[int]
    component_remappers: dict[str, IndexRemapper]
    execution_order: list[str]


class OperationComposer:
    """
    Composes multiple operations sequentially with original-to-current index remapping.
    Validates execution ordering and rejects invalid composition paths.
    """

    def __init__(self, topology: Optional[ModelTopology] = None):
        self.topology = topology
        self.layer_remapper = IndexRemapper(
            initial_count=topology.num_layers if topology else None,
            domain="layer",
        )
        self.mlp_remappers: dict[int, IndexRemapper] = {}
        self.attention_remappers: dict[int, IndexRemapper] = {}
        self.moe_remappers: dict[int, IndexRemapper] = {}

    def _get_mlp_remapper(self, orig_layer: int) -> IndexRemapper:
        if orig_layer not in self.mlp_remappers:
            count = self.topology.mlp_intermediate_sizes.get(orig_layer) if self.topology else None
            self.mlp_remappers[orig_layer] = IndexRemapper(initial_count=count, domain=f"layer_{orig_layer}_mlp_channel")
        return self.mlp_remappers[orig_layer]

    def _get_attention_remapper(self, orig_layer: int) -> IndexRemapper:
        if orig_layer not in self.attention_remappers:
            count = self.topology.num_key_value_heads.get(orig_layer) or self.topology.num_attention_heads.get(orig_layer) if self.topology else None
            self.attention_remappers[orig_layer] = IndexRemapper(initial_count=count, domain=f"layer_{orig_layer}_attn_group")
        return self.attention_remappers[orig_layer]

    def _get_moe_remapper(self, orig_layer: int) -> IndexRemapper:
        if orig_layer not in self.moe_remappers:
            count = self.topology.num_experts.get(orig_layer) if self.topology else None
            self.moe_remappers[orig_layer] = IndexRemapper(initial_count=count, domain=f"layer_{orig_layer}_expert")
        return self.moe_remappers[orig_layer]

    def compose(self, operations: list[UnifiedOperation]) -> ComposedPipeline:
        if not operations:
            return ComposedPipeline([], {}, set(), {}, [])

        # Validate ordering declaration
        orders = [op.order for op in operations if op.order is not None]
        if orders:
            # If any operation specifies order, verify sorting
            sorted_ops = sorted(operations, key=lambda op: (op.order if op.order is not None else float("inf")))
            # Check for conflicting identical order
            explicit_orders = [op.order for op in operations if op.order is not None]
            if len(explicit_orders) != len(set(explicit_orders)):
                raise InvalidCompositionPathError(
                    f"Ambiguous composition order: duplicate order indices found: {explicit_orders}"
                )
        else:
            sorted_ops = list(operations)

        remapped_ops: list[UnifiedOperation] = []

        for op in sorted_ops:
            # Make a deep copy to mutate into remapped form
            new_op = copy.deepcopy(op)

            # 1. Layer index remapping
            if op.layer_index is not None:
                orig_layer = op.layer_index
                if self.layer_remapper.is_removed(orig_layer):
                    raise InvalidCompositionPathError(
                        f"Operation '{op.operation_id}' targets layer {orig_layer} which was already "
                        f"physically removed by an earlier operation."
                    )
                curr_layer = self.layer_remapper.remap_index(orig_layer)
                new_op.layer_index = curr_layer
                new_op.target_path = _remap_path_layer(op.target_path, curr_layer)
                new_op.metadata["original_layer_index"] = orig_layer

            # 2. Component slicing and remapping
            if op.component_type == ComponentType.FULL_LAYER:
                if op.kind == OperationKind.PHYSICAL_REMOVE:
                    # Layer deletion
                    target_layers = []
                    if op.target_indices:
                        target_layers.extend(op.target_indices)
                    elif op.layer_index is not None:
                        target_layers.append(op.layer_index)
                    if not target_layers:
                        raise InvalidCompositionPathError(
                            f"Full layer removal '{op.operation_id}' specifies no target layers"
                        )
                    # Remove layers
                    self.layer_remapper.remove_indices(target_layers)
                    new_op.metadata["removed_original_layers"] = target_layers

            elif op.component_type == ComponentType.MLP_CHANNEL:
                if op.layer_index is None:
                    raise InvalidCompositionPathError(
                        f"MLP channel operation '{op.operation_id}' must specify layer_index or a layer path"
                    )
                remapper = self._get_mlp_remapper(op.layer_index)
                if op.kind == OperationKind.PHYSICAL_REMOVE:
                    if "kept_indices" in op.parameters:
                        remapper.keep_indices(op.parameters["kept_indices"])
                        new_op.parameters["remapped_kept_indices"] = [
                            remapper.remap_index(i) for i in op.parameters["kept_indices"]
                        ]
                    elif "removed_indices" in op.parameters:
                        remapper.remove_indices(op.parameters["removed_indices"])
                    elif op.target_indices:
                        remapper.remove_indices(op.target_indices)
                else:
                    if op.target_indices:
                        new_op.target_indices = remapper.remap_indices(op.target_indices)
                        new_op.metadata["original_target_indices"] = op.target_indices

            elif op.component_type in (ComponentType.ATTENTION_HEAD, ComponentType.ATTENTION_GROUP):
                if op.layer_index is None:
                    raise InvalidCompositionPathError(
                        f"Attention operation '{op.operation_id}' must specify layer_index or a layer path"
                    )
                remapper = self._get_attention_remapper(op.layer_index)
                if op.kind == OperationKind.PHYSICAL_REMOVE:
                    if "kept_indices" in op.parameters:
                        remapper.keep_indices(op.parameters["kept_indices"])
                        new_op.parameters["remapped_kept_indices"] = [
                            remapper.remap_index(i) for i in op.parameters["kept_indices"]
                        ]
                    elif "removed_indices" in op.parameters:
                        remapper.remove_indices(op.parameters["removed_indices"])
                    elif op.target_indices:
                        remapper.remove_indices(op.target_indices)
                else:
                    if op.target_indices:
                        new_op.target_indices = remapper.remap_indices(op.target_indices)
                        new_op.metadata["original_target_indices"] = op.target_indices

            elif op.component_type == ComponentType.MOE_EXPERT:
                if op.layer_index is None:
                    raise InvalidCompositionPathError(
                        f"MoE expert operation '{op.operation_id}' must specify layer_index or a layer path"
                    )
                remapper = self._get_moe_remapper(op.layer_index)
                if op.kind == OperationKind.PHYSICAL_REMOVE:
                    if "kept_indices" in op.parameters:
                        remapper.keep_indices(op.parameters["kept_indices"])
                        new_op.parameters["remapped_kept_indices"] = [
                            remapper.remap_index(i) for i in op.parameters["kept_indices"]
                        ]
                    elif "removed_indices" in op.parameters:
                        remapper.remove_indices(op.parameters["removed_indices"])
                    elif op.target_indices:
                        remapper.remove_indices(op.target_indices)
                else:
                    if op.target_indices:
                        new_op.target_indices = remapper.remap_indices(op.target_indices)
                        new_op.metadata["original_target_indices"] = op.target_indices

            remapped_ops.append(new_op)

        all_remappers = {"layer": self.layer_remapper}
        for k, v in self.mlp_remappers.items():
            all_remappers[f"layer_{k}_mlp"] = v
        for k, v in self.attention_remappers.items():
            all_remappers[f"layer_{k}_attn"] = v
        for k, v in self.moe_remappers.items():
            all_remappers[f"layer_{k}_moe"] = v

        return ComposedPipeline(
            remapped_operations=remapped_ops,
            original_to_current_layer_map=self.layer_remapper.get_mapping(),
            removed_layers=set(self.layer_remapper._removed_orig),
            component_remappers=all_remappers,
            execution_order=[op.operation_id for op in remapped_ops],
        )


def compose_operations(
    operations: list[UnifiedOperation],
    model: Optional[nn.Module] = None,
    topology: Optional[ModelTopology] = None,
) -> ComposedPipeline:
    """Top-level helper to compose operations with index remapping and path validation."""
    if topology is None and model is not None:
        topology = ModelTopology.from_model(model)
    composer = OperationComposer(topology=topology)
    return composer.compose(operations)


# ============================================================================
# 4. Adapter Capability Matrix
# ============================================================================

@dataclass
class ArchitectureCapabilities:
    """Declares capabilities, supported operations, and constraints for an architecture/adapter."""

    architecture: str
    adapter_name: str
    supported_operations: set[OperationKind]
    supported_components: set[ComponentType]
    supports_runtime: bool = True
    supports_persistent: bool = True
    supports_export: bool = True
    unsupported_combinations: set[tuple[OperationKind, ComponentType]] = field(default_factory=set)
    notes: str = ""

    def is_operation_supported(self, op: UnifiedOperation) -> tuple[bool, Optional[str]]:
        if op.kind not in self.supported_operations:
            return False, f"Architecture '{self.architecture}' does not support operation kind '{op.kind.value}'"

        if op.component_type not in self.supported_components:
            return False, f"Architecture '{self.architecture}' does not support component type '{op.component_type.value}'"

        if (op.kind, op.component_type) in self.unsupported_combinations:
            return (
                False,
                f"Architecture '{self.architecture}' does not support combination ({op.kind.value}, {op.component_type.value})",
            )

        if op.execution_semantics == ExecutionSemantics.RUNTIME_ONLY and not self.supports_runtime:
            return False, f"Architecture '{self.architecture}' does not support runtime-only operations"

        if op.execution_semantics == ExecutionSemantics.PERSISTENT and not self.supports_persistent:
            return False, f"Architecture '{self.architecture}' does not support persistent operations"

        if op.export_supported and not self.supports_export:
            return False, f"Architecture '{self.architecture}' does not support exporting checkpoint artifacts"

        return True, None


@dataclass
class CapabilityCheckResult:
    """Report evaluating plan compatibility with target architecture."""

    supported: bool
    architecture: str
    adapter_name: str
    violations: list[str]
    supported_operations: list[UnifiedOperation]
    unsupported_operations: list[UnifiedOperation]


class AdapterCapabilityRegistry:
    """Registry maintaining supported operations and components per architecture adapter."""

    def __init__(self):
        self._capabilities: dict[str, ArchitectureCapabilities] = {}
        self._aliases: dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self):
        all_ops = {
            OperationKind.MASK,
            OperationKind.SCALE,
            OperationKind.CLAMP,
            OperationKind.PROJECT,
            OperationKind.REPLACE,
            OperationKind.LOW_RANK_DELTA,
            OperationKind.PHYSICAL_REMOVE,
        }

        # LLaMA / Mistral dense
        llama_comps = {
            ComponentType.MLP_CHANNEL,
            ComponentType.ATTENTION_HEAD,
            ComponentType.ATTENTION_GROUP,
            ComponentType.FULL_LAYER,
            ComponentType.WEIGHT_TENSOR,
        }
        self.register(
            ArchitectureCapabilities(
                architecture="llama",
                adapter_name="llama-like",
                supported_operations=all_ops,
                supported_components=llama_comps,
                supports_runtime=True,
                supports_persistent=True,
                supports_export=True,
                notes="Dense LLaMA-like architecture; does not support MoE expert operations.",
            )
        )
        self.register_alias("mistral", "llama")

        # Mixtral MoE
        mixtral_comps = {
            ComponentType.MOE_EXPERT,
            ComponentType.ATTENTION_HEAD,
            ComponentType.ATTENTION_GROUP,
            ComponentType.FULL_LAYER,
            ComponentType.WEIGHT_TENSOR,
            ComponentType.MLP_CHANNEL,
        }
        self.register(
            ArchitectureCapabilities(
                architecture="mixtral",
                adapter_name="moe-like",
                supported_operations=all_ops,
                supported_components=mixtral_comps,
                supports_runtime=True,
                supports_persistent=True,
                supports_export=True,
                notes="Sparse MoE architecture supporting router and expert pruning.",
            )
        )

        # Qwen dense
        qwen_comps = {
            ComponentType.MLP_CHANNEL,
            ComponentType.ATTENTION_HEAD,
            ComponentType.ATTENTION_GROUP,
            ComponentType.FULL_LAYER,
            ComponentType.WEIGHT_TENSOR,
        }
        self.register(
            ArchitectureCapabilities(
                architecture="qwen",
                adapter_name="llama-like",
                supported_operations=all_ops,
                supported_components=qwen_comps,
                supports_runtime=True,
                supports_persistent=True,
                supports_export=True,
                notes="Dense Qwen architecture; GQA attention and dense MLP.",
            )
        )
        self.register_alias("qwen2", "qwen")
        self.register_alias("qwen3", "qwen")

        # Qwen MoE
        self.register(
            ArchitectureCapabilities(
                architecture="qwen_moe",
                adapter_name="moe-like",
                supported_operations=all_ops,
                supported_components=mixtral_comps,
                supports_runtime=True,
                supports_persistent=True,
                supports_export=True,
                notes="MoE variant of Qwen architecture.",
            )
        )
        self.register_alias("qwen2_moe", "qwen_moe")
        self.register_alias("qwen3_moe", "qwen_moe")

        # Gemma dense
        gemma_comps = {
            ComponentType.MLP_CHANNEL,
            ComponentType.ATTENTION_HEAD,
            ComponentType.ATTENTION_GROUP,
            ComponentType.FULL_LAYER,
            ComponentType.WEIGHT_TENSOR,
        }
        self.register(
            ArchitectureCapabilities(
                architecture="gemma",
                adapter_name="llama-like",
                supported_operations=all_ops,
                supported_components=gemma_comps,
                supports_runtime=True,
                supports_persistent=True,
                supports_export=True,
                notes="Gemma architecture with tied embeddings and gated MLP.",
            )
        )
        self.register_alias("gemma2", "gemma")
        self.register_alias("gemma3_text", "gemma")

    def register(self, capabilities: ArchitectureCapabilities) -> None:
        self._capabilities[capabilities.architecture.lower()] = capabilities

    def register_alias(self, alias: str, canonical_architecture: str) -> None:
        canonical = canonical_architecture.lower()
        if canonical not in self._capabilities:
            raise KeyError(f"Canonical architecture '{canonical_architecture}' is not registered")
        self._aliases[alias.lower()] = canonical

    def resolve_architecture_name(self, architecture_or_model: Union[str, nn.Module]) -> str:
        if isinstance(architecture_or_model, nn.Module):
            cfg = getattr(architecture_or_model, "config", None)
            mtype = getattr(cfg, "model_type", None) or type(architecture_or_model).__name__
            name = str(mtype).lower()
        else:
            name = str(architecture_or_model).lower()

        canonical = self._aliases.get(name, name)
        return canonical

    def get(self, architecture_or_model: Union[str, nn.Module]) -> ArchitectureCapabilities:
        canonical = self.resolve_architecture_name(architecture_or_model)
        if canonical not in self._capabilities:
            raise KeyError(
                f"No capability entry registered for architecture '{canonical}' (available: {sorted(self._capabilities)})"
            )
        return self._capabilities[canonical]

    def check_plan_support(
        self,
        architecture_or_model: Union[str, nn.Module],
        operations: list[UnifiedOperation],
    ) -> CapabilityCheckResult:
        capabilities = self.get(architecture_or_model)
        violations: list[str] = []
        supported_ops: list[UnifiedOperation] = []
        unsupported_ops: list[UnifiedOperation] = []

        for op in operations:
            ok, reason = capabilities.is_operation_supported(op)
            if ok:
                supported_ops.append(op)
            else:
                unsupported_ops.append(op)
                violations.append(f"Op '{op.operation_id}' rejected: {reason}")

        return CapabilityCheckResult(
            supported=len(violations) == 0,
            architecture=capabilities.architecture,
            adapter_name=capabilities.adapter_name,
            violations=violations,
            supported_operations=supported_ops,
            unsupported_operations=unsupported_ops,
        )

    def validate_plan_support(
        self,
        architecture_or_model: Union[str, nn.Module],
        operations: list[UnifiedOperation],
    ) -> None:
        result = self.check_plan_support(architecture_or_model, operations)
        if not result.supported:
            msgs = "; ".join(result.violations)
            raise UnsupportedOperationError(
                f"Architecture '{result.architecture}' ({result.adapter_name}) cannot support planned operations: {msgs}"
            )


# Global capability matrix singleton
_GLOBAL_MATRIX = AdapterCapabilityRegistry()


def get_capability_matrix() -> AdapterCapabilityRegistry:
    return _GLOBAL_MATRIX


def check_adapter_capability(
    architecture_or_model: Union[str, nn.Module],
    operations: list[UnifiedOperation],
) -> CapabilityCheckResult:
    return _GLOBAL_MATRIX.check_plan_support(architecture_or_model, operations)


def validate_adapter_capability(
    architecture_or_model: Union[str, nn.Module],
    operations: list[UnifiedOperation],
) -> None:
    _GLOBAL_MATRIX.validate_plan_support(architecture_or_model, operations)


# ============================================================================
# 5. Unified Plan Executor / Applicator
# ============================================================================

def _resolve_model_target(
    model: nn.Module,
    path: str,
) -> tuple[Any, Optional[nn.Module], Optional[str]]:
    """
    Resolves a target path within model.
    Returns (target_obj, parent_module, attr_or_index).
    """
    named_params = dict(model.named_parameters(remove_duplicate=False) if hasattr(model, "named_parameters") else [])
    named_mods = dict(model.named_modules(remove_duplicate=False) if hasattr(model, "named_modules") else [])

    candidates = [path]
    if path.startswith("model."):
        candidates.append(path[6:])
    else:
        candidates.append(f"model.{path}")

    # 1. Direct named_parameters match
    for c in candidates:
        if c in named_params:
            param = named_params[c]
            parent_mod = model
            attr = c
            if "." in c:
                p_path, attr = c.rsplit(".", 1)
                parent_mod = named_mods.get(p_path, model)
            return param, parent_mod, attr

    # 2. Direct named_modules match
    for c in candidates:
        if c in named_mods:
            mod = named_mods[c]
            parent_mod = model
            attr = c
            if "." in c:
                p_path, attr = c.rsplit(".", 1)
                parent_mod = named_mods.get(p_path, model)
            return mod, parent_mod, attr

    # 3. Dynamic hierarchical traversal
    for c in candidates:
        parts = c.split(".")
        curr = model
        parent = None
        last_part = None
        found = True
        for part in parts:
            parent = curr
            last_part = part
            if isinstance(curr, (nn.ModuleList, list, tuple)) and part.isdigit():
                idx = int(part)
                if idx < 0 or idx >= len(curr):
                    found = False
                    break
                curr = curr[idx]
            elif hasattr(curr, part):
                curr = getattr(curr, part)
            else:
                found = False
                break
        if found:
            return curr, (parent if isinstance(parent, nn.Module) else None), str(last_part)

    raise ValidationError(f"Could not resolve target_path '{path}' in model of type {type(model).__name__}")


class PlanRuntimeContext:
    """Manages active runtime hooks installed by runtime-only operations."""

    def __init__(self, handles: Optional[list[Any]] = None):
        self.handles: list[Any] = handles or []
        self.is_active: bool = True

    def remove(self) -> None:
        """Removes all installed runtime hooks and deactivates context."""
        for h in self.handles:
            try:
                if hasattr(h, "remove"):
                    h.remove()
            except Exception:
                pass
        self.handles.clear()
        self.is_active = False

    def __enter__(self) -> "PlanRuntimeContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


@dataclass
class PlanExecutionResult:
    """Report and handle returned after executing a neurosurgery plan."""

    success: bool
    applied_operations: list[UnifiedOperation]
    modified_parameters: list[str] = field(default_factory=list)
    removed_components: dict[str, list[int]] = field(default_factory=dict)
    runtime_context: Optional[PlanRuntimeContext] = None
    details: list[dict[str, Any]] = field(default_factory=list)

    def __enter__(self) -> "PlanExecutionResult":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.runtime_context is not None:
            self.runtime_context.remove()


class PlanExecutor:
    """
    Unified plan executor and applicator.
    Applies composed operations (persistent and runtime-only) to a target model,
    enforcing adapter capabilities, conflict detection, and dynamic index remapping.
    """

    def __init__(
        self,
        capability_registry: Optional[AdapterCapabilityRegistry] = None,
        validate_capabilities: bool = True,
        validate_conflicts: bool = True,
        compose: bool = True,
    ):
        self.capability_registry = capability_registry or get_capability_matrix()
        self.validate_capabilities = validate_capabilities
        self.validate_conflicts = validate_conflicts
        self.compose = compose

    def execute(
        self,
        model: nn.Module,
        plan_or_ops: Union[UnifiedPlan, list[UnifiedOperation]],
    ) -> PlanExecutionResult:
        if isinstance(plan_or_ops, UnifiedPlan):
            plan_or_ops.validate()
            operations = list(plan_or_ops.operations)
        elif isinstance(plan_or_ops, list):
            operations = list(plan_or_ops)
            for op in operations:
                if not isinstance(op, UnifiedOperation):
                    raise ValidationError(f"Expected UnifiedOperation, got {type(op)}")
                op.validate()
        else:
            raise ValidationError(f"Expected UnifiedPlan or list[UnifiedOperation], got {type(plan_or_ops)}")

        if not operations:
            return PlanExecutionResult(
                success=True,
                applied_operations=[],
                modified_parameters=[],
                removed_components={},
                runtime_context=PlanRuntimeContext(),
                details=[],
            )

        # 1. Adapter capability check (pre-flight validation before any mutation)
        if self.validate_capabilities:
            try:
                self.capability_registry.validate_plan_support(model, operations)
            except KeyError:
                # If model architecture is not explicitly registered, allow fallback
                pass

        # 2. Shared-weight and parameter aliasing conflict detection (pre-flight validation)
        if self.validate_conflicts:
            validate_no_conflicts(operations, model=model)

        # 3. Arbitrary edit composition & index remapping
        if self.compose:
            topology = None
            try:
                topology = ModelTopology.from_model(model)
            except Exception:
                pass
            pipeline = compose_operations(operations, model=model, topology=topology)
            ops_to_apply = pipeline.remapped_operations
        else:
            ops_to_apply = operations

        # 4. Apply operations sequentially
        runtime_context = PlanRuntimeContext()
        result = PlanExecutionResult(
            success=False,
            applied_operations=[],
            modified_parameters=[],
            removed_components={},
            runtime_context=runtime_context,
            details=[],
        )

        for op in ops_to_apply:
            self._apply_single_operation(model, op, result)
            result.applied_operations.append(op)

        result.success = True
        return result

    def apply(
        self,
        model: nn.Module,
        plan_or_ops: Union[UnifiedPlan, list[UnifiedOperation]],
    ) -> PlanExecutionResult:
        """Alias for execute()."""
        return self.execute(model, plan_or_ops)

    def _apply_single_operation(
        self,
        model: nn.Module,
        op: UnifiedOperation,
        result: PlanExecutionResult,
    ) -> None:
        if op.kind == OperationKind.SCALE:
            self._apply_scale(model, op, result)
        elif op.kind == OperationKind.CLAMP:
            self._apply_clamp(model, op, result)
        elif op.kind == OperationKind.MASK:
            self._apply_mask(model, op, result)
        elif op.kind == OperationKind.PROJECT:
            self._apply_project(model, op, result)
        elif op.kind == OperationKind.REPLACE:
            self._apply_replace(model, op, result)
        elif op.kind == OperationKind.LOW_RANK_DELTA:
            self._apply_low_rank_delta(model, op, result)
        elif op.kind == OperationKind.PHYSICAL_REMOVE:
            self._apply_physical_remove(model, op, result)
        else:
            raise ValidationError(f"Unknown operation kind: {op.kind}")

    def _apply_scale(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        scale_factor = op.parameters["scale_factor"]
        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)

        if op.execution_semantics == ExecutionSemantics.RUNTIME_ONLY:
            module_to_hook = target_obj if isinstance(target_obj, nn.Module) else parent_mod
            if module_to_hook is None:
                raise ValidationError(f"Cannot attach runtime hook to non-module target: {op.target_path}")

            indices = op.target_indices

            def hook_fn(mod, inp, out):
                is_tuple = isinstance(out, tuple)
                val = out[0] if is_tuple else out
                if indices is not None:
                    c = val.clone()
                    c[..., indices] = c[..., indices] * scale_factor
                    return (c, *out[1:]) if is_tuple else c
                res = val * scale_factor
                return (res, *out[1:]) if is_tuple else res

            h = module_to_hook.register_forward_hook(hook_fn)
            result.runtime_context.handles.append(h)
            result.details.append({"op": op.operation_id, "type": "runtime_scale", "target": op.target_path})
        else:
            param = target_obj if isinstance(target_obj, (nn.Parameter, torch.Tensor)) else getattr(target_obj, "weight", None)
            if param is None:
                raise ValidationError(f"Target '{op.target_path}' is not a parameter and has no .weight attribute")

            with torch.no_grad():
                if op.target_indices is not None:
                    idx = torch.tensor(op.target_indices, dtype=torch.long, device=param.data.device)
                    param.data[idx] = param.data[idx] * scale_factor
                else:
                    param.data.mul_(scale_factor)
            result.modified_parameters.append(op.target_path)
            result.details.append({"op": op.operation_id, "type": "persistent_scale", "target": op.target_path})

    def _apply_clamp(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        min_val = op.parameters.get("min_value")
        max_val = op.parameters.get("max_value")
        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)

        if op.execution_semantics == ExecutionSemantics.RUNTIME_ONLY:
            module_to_hook = target_obj if isinstance(target_obj, nn.Module) else parent_mod
            if module_to_hook is None:
                raise ValidationError(f"Cannot attach runtime hook to non-module target: {op.target_path}")

            indices = op.target_indices

            def hook_fn(mod, inp, out):
                is_tuple = isinstance(out, tuple)
                val = out[0] if is_tuple else out
                if indices is not None:
                    c = val.clone()
                    c[..., indices] = torch.clamp(c[..., indices], min=min_val, max=max_val)
                    return (c, *out[1:]) if is_tuple else c
                res = torch.clamp(val, min=min_val, max=max_val)
                return (res, *out[1:]) if is_tuple else res

            h = module_to_hook.register_forward_hook(hook_fn)
            result.runtime_context.handles.append(h)
            result.details.append({"op": op.operation_id, "type": "runtime_clamp", "target": op.target_path})
        else:
            param = target_obj if isinstance(target_obj, (nn.Parameter, torch.Tensor)) else getattr(target_obj, "weight", None)
            if param is None:
                raise ValidationError(f"Target '{op.target_path}' is not a parameter and has no .weight attribute")

            with torch.no_grad():
                if op.target_indices is not None:
                    idx = torch.tensor(op.target_indices, dtype=torch.long, device=param.data.device)
                    param.data[idx] = torch.clamp(param.data[idx], min=min_val, max=max_val)
                else:
                    param.data.clamp_(min=min_val, max=max_val)
            result.modified_parameters.append(op.target_path)
            result.details.append({"op": op.operation_id, "type": "persistent_clamp", "target": op.target_path})

    def _apply_mask(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        mask_value = float(op.parameters.get("mask_value", 0.0))
        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)

        if op.execution_semantics == ExecutionSemantics.RUNTIME_ONLY:
            module_to_hook = target_obj if isinstance(target_obj, nn.Module) else parent_mod
            if module_to_hook is None:
                raise ValidationError(f"Cannot attach runtime hook to non-module target: {op.target_path}")

            indices = op.target_indices

            def hook_fn(mod, inp, out):
                is_tuple = isinstance(out, tuple)
                val = out[0].clone() if is_tuple else out.clone()
                if indices is not None:
                    val[..., indices] = mask_value
                else:
                    val.fill_(mask_value)
                return (val, *out[1:]) if is_tuple else val

            h = module_to_hook.register_forward_hook(hook_fn)
            result.runtime_context.handles.append(h)
            result.details.append({"op": op.operation_id, "type": "runtime_mask", "target": op.target_path})
        else:
            param = target_obj if isinstance(target_obj, (nn.Parameter, torch.Tensor)) else getattr(target_obj, "weight", None)
            if param is None:
                raise ValidationError(f"Target '{op.target_path}' is not a parameter and has no .weight attribute")

            with torch.no_grad():
                if op.target_indices is not None:
                    idx = torch.tensor(op.target_indices, dtype=torch.long, device=param.data.device)
                    param.data[idx] = mask_value
                else:
                    param.data.fill_(mask_value)
            result.modified_parameters.append(op.target_path)
            result.details.append({"op": op.operation_id, "type": "persistent_mask", "target": op.target_path})

    def _apply_project(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        from .math_ops import apply_constrained_directional_surgery

        direction = op.parameters.get("direction")
        subspace_basis = op.parameters.get("subspace_basis")
        strength = float(op.parameters.get("strength", 1.0))
        norm_preserve = bool(op.parameters.get("norm_preserve", False))
        preserve_subspace = bool(op.parameters.get("preserve_subspace", True))

        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)

        if op.execution_semantics == ExecutionSemantics.RUNTIME_ONLY:
            module_to_hook = target_obj if isinstance(target_obj, nn.Module) else parent_mod
            if module_to_hook is None:
                raise ValidationError(f"Cannot attach runtime hook to non-module target: {op.target_path}")

            dir_tensor = torch.as_tensor(direction, dtype=torch.float32) if direction is not None else None

            def hook_fn(mod, inp, out):
                is_tuple = isinstance(out, tuple)
                val = out[0] if is_tuple else out
                if dir_tensor is not None:
                    d = dir_tensor.to(device=val.device, dtype=val.dtype)
                    d_norm = d / (d.norm() + 1e-8)
                    dot = (val * d_norm).sum(dim=-1, keepdim=True)
                    proj = dot * d_norm
                    res = val - strength * proj
                else:
                    res = val
                return (res, *out[1:]) if is_tuple else res

            h = module_to_hook.register_forward_hook(hook_fn)
            result.runtime_context.handles.append(h)
            result.details.append({"op": op.operation_id, "type": "runtime_project", "target": op.target_path})
        else:
            param = target_obj if isinstance(target_obj, (nn.Parameter, torch.Tensor)) else getattr(target_obj, "weight", None)
            if param is None:
                raise ValidationError(f"Target '{op.target_path}' is not a parameter and has no .weight attribute")

            with torch.no_grad():
                dir_tensor = torch.as_tensor(direction, dtype=torch.float32, device=param.data.device) if direction is not None else None
                basis_tensor = torch.as_tensor(subspace_basis, dtype=torch.float32, device=param.data.device) if (subspace_basis is not None and preserve_subspace) else None

                if dir_tensor is not None:
                    new_w = apply_constrained_directional_surgery(
                        param.data,
                        dir_tensor,
                        strength=strength,
                        preserve_basis=basis_tensor,
                        norm_preserve=norm_preserve,
                    )
                    param.data.copy_(new_w.to(device=param.data.device, dtype=param.data.dtype))
                elif basis_tensor is not None:
                    w = param.data.float()
                    if basis_tensor.shape[0] == w.shape[1]:
                        delta = -strength * (w @ basis_tensor @ basis_tensor.T)
                    else:
                        delta = -strength * (basis_tensor @ (basis_tensor.T @ w))
                    param.data.copy_((w + delta).to(device=param.data.device, dtype=param.data.dtype))

            result.modified_parameters.append(op.target_path)
            result.details.append({"op": op.operation_id, "type": "persistent_project", "target": op.target_path})

    def _apply_replace(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        rep_val = op.parameters.get("replacement_value")
        if rep_val is None:
            rep_val = op.parameters.get("replacement_tensor")
        if rep_val is None:
            raise ValidationError(f"Replace operation '{op.operation_id}' has no replacement value or tensor")

        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)

        if op.execution_semantics == ExecutionSemantics.RUNTIME_ONLY:
            module_to_hook = target_obj if isinstance(target_obj, nn.Module) else parent_mod
            if module_to_hook is None:
                raise ValidationError(f"Cannot attach runtime hook to non-module target: {op.target_path}")

            indices = op.target_indices

            def hook_fn(mod, inp, out):
                is_tuple = isinstance(out, tuple)
                val = out[0] if is_tuple else out
                res = val.clone()
                if isinstance(rep_val, (int, float)):
                    if indices is not None:
                        res[..., indices] = rep_val
                    else:
                        res.fill_(rep_val)
                else:
                    rep_t = torch.as_tensor(rep_val, dtype=val.dtype, device=val.device)
                    if indices is not None:
                        res[..., indices] = rep_t
                    else:
                        res = rep_t.expand_as(res)
                return (res, *out[1:]) if is_tuple else res

            h = module_to_hook.register_forward_hook(hook_fn)
            result.runtime_context.handles.append(h)
            result.details.append({"op": op.operation_id, "type": "runtime_replace", "target": op.target_path})
        else:
            param = target_obj if isinstance(target_obj, (nn.Parameter, torch.Tensor)) else getattr(target_obj, "weight", None)
            if param is None:
                raise ValidationError(f"Target '{op.target_path}' is not a parameter and has no .weight attribute")

            with torch.no_grad():
                if isinstance(rep_val, (int, float)):
                    if op.target_indices is not None:
                        idx = torch.tensor(op.target_indices, dtype=torch.long, device=param.data.device)
                        param.data[idx] = rep_val
                    else:
                        param.data.fill_(rep_val)
                else:
                    rep_t = torch.as_tensor(rep_val, dtype=param.data.dtype, device=param.data.device)
                    if op.target_indices is not None:
                        idx = torch.tensor(op.target_indices, dtype=torch.long, device=param.data.device)
                        param.data[idx] = rep_t
                    else:
                        param.data.copy_(rep_t)
            result.modified_parameters.append(op.target_path)
            result.details.append({"op": op.operation_id, "type": "persistent_replace", "target": op.target_path})

    def _apply_low_rank_delta(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        rank = int(op.parameters.get("rank", 1))
        alpha = float(op.parameters.get("alpha", 1.0))
        u = op.parameters.get("u")
        v = op.parameters.get("v")

        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)
        param = target_obj if isinstance(target_obj, (nn.Parameter, torch.Tensor)) else getattr(target_obj, "weight", None)
        if param is None:
            raise ValidationError(f"Target '{op.target_path}' is not a parameter and has no .weight attribute")

        with torch.no_grad():
            if u is not None and v is not None:
                u_t = torch.as_tensor(u, dtype=torch.float32, device=param.data.device)
                v_t = torch.as_tensor(v, dtype=torch.float32, device=param.data.device)
                delta = (alpha / rank) * (u_t @ v_t)
            else:
                out_dim, in_dim = param.data.shape[0], param.data.shape[1]
                u_t = torch.ones((out_dim, rank), device=param.data.device, dtype=torch.float32) * (1.0 / math.sqrt(rank))
                v_t = torch.ones((rank, in_dim), device=param.data.device, dtype=torch.float32) * (1.0 / math.sqrt(rank))
                delta = (alpha / rank) * (u_t @ v_t)

            param.data.add_(delta.to(device=param.data.device, dtype=param.data.dtype))

        result.modified_parameters.append(op.target_path)
        result.details.append({"op": op.operation_id, "type": "persistent_low_rank_delta", "target": op.target_path})

    def _apply_physical_remove(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        if op.component_type == ComponentType.FULL_LAYER:
            self._remove_layer(model, op, result)
        elif op.component_type == ComponentType.MLP_CHANNEL:
            self._remove_mlp_channels(model, op, result)
        elif op.component_type in (ComponentType.ATTENTION_HEAD, ComponentType.ATTENTION_GROUP):
            self._remove_attention(model, op, result)
        elif op.component_type == ComponentType.MOE_EXPERT:
            self._remove_moe_experts(model, op, result)
        elif op.component_type == ComponentType.WEIGHT_TENSOR:
            self._remove_weight_tensor_slice(model, op, result)
        else:
            raise ValidationError(f"Unsupported component type for physical_remove: {op.component_type}")

    def _remove_layer(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        from .common import get_layers, set_layers

        target_layer = op.layer_index
        if target_layer is None and op.target_indices:
            target_layers = list(op.target_indices)
        elif target_layer is not None:
            target_layers = [target_layer]
        else:
            raise ValidationError(f"Full layer removal '{op.operation_id}' has no target layer")

        layer_path = None
        layers_list = None
        try:
            layer_path, layers = get_layers(model)
            layers_list = list(layers)
        except Exception:
            target_obj, parent, key = _resolve_model_target(model, op.target_path)
            if isinstance(parent, nn.ModuleList):
                layers_list = list(parent)
                layer_path = op.target_path.rsplit(".", 1)[0] if "." in op.target_path else "layers"
            else:
                raise ValidationError(f"Could not locate layers list to drop from {op.target_path}")

        drop_set = set(target_layers)
        for idx in drop_set:
            if idx < 0 or idx >= len(layers_list):
                raise IndexError(f"Layer index {idx} out of range [0, {len(layers_list)})")

        surviving = [layer for i, layer in enumerate(layers_list) if i not in drop_set]
        if not surviving:
            raise InvalidCompositionPathError("Cannot drop all layers in model; at least 1 must remain")

        if layer_path is not None:
            set_layers(model, layer_path, surviving)

        result.removed_components.setdefault("layers", []).extend(target_layers)
        result.details.append({
            "op": op.operation_id,
            "type": "layer_removal",
            "dropped_layers": target_layers,
            "remaining_layers": len(surviving),
        })

    def _remove_mlp_channels(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)
        mlp = target_obj if isinstance(target_obj, nn.Module) else parent_mod

        gate = getattr(mlp, "gate_proj", getattr(mlp, "gate", None))
        up = getattr(mlp, "up_proj", getattr(mlp, "up", None))
        down = getattr(mlp, "down_proj", getattr(mlp, "down", None))

        if gate is None or up is None or down is None:
            raise ValidationError(f"MLP module at '{op.target_path}' lacks gate_proj/up_proj/down_proj")

        old_intermediate = gate.weight.shape[0]
        kept = op.parameters.get("remapped_kept_indices") or op.parameters.get("kept_indices")
        if kept is None:
            removed = set(op.parameters.get("removed_indices") or op.target_indices or [])
            kept = [i for i in range(old_intermediate) if i not in removed]

        if not kept:
            raise InvalidCompositionPathError("Cannot remove all MLP channels; intermediate size cannot be 0")

        idx_t = torch.tensor(kept, dtype=torch.long, device=gate.weight.device)
        with torch.no_grad():
            gate.weight.data = gate.weight.data.index_select(0, idx_t)
            up.weight.data = up.weight.data.index_select(0, idx_t)
            down.weight.data = down.weight.data.index_select(1, idx_t)

            if getattr(gate, "bias", None) is not None:
                gate.bias.data = gate.bias.data.index_select(0, idx_t)
            if getattr(up, "bias", None) is not None:
                up.bias.data = up.bias.data.index_select(0, idx_t)

        new_intermediate = len(kept)
        if hasattr(gate, "out_features"):
            gate.out_features = new_intermediate
        if hasattr(up, "out_features"):
            up.out_features = new_intermediate
        if hasattr(down, "in_features"):
            down.in_features = new_intermediate
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = new_intermediate

        for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
            if cfg is not None and hasattr(cfg, "intermediate_size"):
                cfg.intermediate_size = new_intermediate

        p_names = [f"{op.target_path}.gate_proj.weight", f"{op.target_path}.up_proj.weight", f"{op.target_path}.down_proj.weight"]
        result.modified_parameters.extend(p_names)
        result.removed_components.setdefault("mlp_channels", []).extend(
            [i for i in range(old_intermediate) if i not in set(kept)]
        )
        result.details.append({
            "op": op.operation_id,
            "type": "mlp_channel_slice",
            "old_intermediate": old_intermediate,
            "new_intermediate": new_intermediate,
            "kept_channels": kept,
        })

    def _remove_attention(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)
        attn = target_obj if isinstance(target_obj, nn.Module) else parent_mod

        q = getattr(attn, "q_proj", getattr(attn, "q", None))
        k = getattr(attn, "k_proj", getattr(attn, "k", None))
        v = getattr(attn, "v_proj", getattr(attn, "v", None))
        o = getattr(attn, "o_proj", getattr(attn, "o", None))

        if q is None or k is None or v is None or o is None:
            raise ValidationError(f"Attention module at '{op.target_path}' lacks q/k/v/o projections")

        num_heads = getattr(attn, "num_heads", getattr(attn, "num_attention_heads", None))
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        head_dim = getattr(attn, "head_dim", None)

        if head_dim is None and num_heads is not None:
            head_dim = q.weight.shape[0] // num_heads
        if num_heads is None and head_dim is not None:
            num_heads = q.weight.shape[0] // head_dim
            num_kv_heads = k.weight.shape[0] // head_dim

        if num_heads is None or num_kv_heads is None or head_dim is None:
            raise ValidationError("Could not deduce attention head counts or head_dim")

        group_size = max(1, num_heads // num_kv_heads)

        if op.component_type == ComponentType.ATTENTION_GROUP:
            kept_groups = op.parameters.get("remapped_kept_indices") or op.parameters.get("kept_indices")
            if kept_groups is None:
                removed_g = set(op.parameters.get("removed_indices") or op.target_indices or [])
                kept_groups = [g for g in range(num_kv_heads) if g not in removed_g]

            if not kept_groups:
                raise InvalidCompositionPathError("Cannot remove all attention groups; remaining count cannot be 0")

            q_heads = [g * group_size + off for g in kept_groups for off in range(group_size)]
            q_rows = [h * head_dim + off for h in q_heads for off in range(head_dim)]
            kv_rows = [g * head_dim + off for g in kept_groups for off in range(head_dim)]
            new_kv_heads = len(kept_groups)
            new_heads = len(q_heads)
        else:
            kept_heads = op.parameters.get("remapped_kept_indices") or op.parameters.get("kept_indices")
            if kept_heads is None:
                removed_h = set(op.parameters.get("removed_indices") or op.target_indices or [])
                kept_heads = [h for h in range(num_heads) if h not in removed_h]

            if not kept_heads:
                raise InvalidCompositionPathError("Cannot remove all attention heads; remaining count cannot be 0")

            q_rows = [h * head_dim + off for h in kept_heads for off in range(head_dim)]
            kv_rows = q_rows
            new_heads = len(kept_heads)
            new_kv_heads = len(kept_heads)

        q_idx = torch.tensor(q_rows, dtype=torch.long, device=q.weight.device)
        kv_idx = torch.tensor(kv_rows, dtype=torch.long, device=k.weight.device)

        with torch.no_grad():
            q.weight.data = q.weight.data.index_select(0, q_idx)
            k.weight.data = k.weight.data.index_select(0, kv_idx)
            v.weight.data = v.weight.data.index_select(0, kv_idx)
            o.weight.data = o.weight.data.index_select(1, q_idx)

            if getattr(q, "bias", None) is not None:
                q.bias.data = q.bias.data.index_select(0, q_idx)
            if getattr(k, "bias", None) is not None:
                k.bias.data = k.bias.data.index_select(0, kv_idx)
            if getattr(v, "bias", None) is not None:
                v.bias.data = v.bias.data.index_select(0, kv_idx)

        if hasattr(attn, "num_heads"):
            attn.num_heads = new_heads
        if hasattr(attn, "num_attention_heads"):
            attn.num_attention_heads = new_heads
        if hasattr(attn, "num_key_value_heads"):
            attn.num_key_value_heads = new_kv_heads
        if hasattr(q, "out_features"):
            q.out_features = len(q_rows)
        if hasattr(k, "out_features"):
            k.out_features = len(kv_rows)
        if hasattr(v, "out_features"):
            v.out_features = len(kv_rows)
        if hasattr(o, "in_features"):
            o.in_features = len(q_rows)

        for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
            if cfg is not None:
                if hasattr(cfg, "num_attention_heads"):
                    cfg.num_attention_heads = new_heads
                if hasattr(cfg, "num_key_value_heads"):
                    cfg.num_key_value_heads = new_kv_heads

        p_names = [f"{op.target_path}.q_proj.weight", f"{op.target_path}.k_proj.weight", f"{op.target_path}.v_proj.weight", f"{op.target_path}.o_proj.weight"]
        result.modified_parameters.extend(p_names)
        result.removed_components.setdefault("attention_groups", []).append(op.layer_index or 0)
        result.details.append({
            "op": op.operation_id,
            "type": "attention_slice",
            "new_heads": new_heads,
            "new_kv_heads": new_kv_heads,
        })

    def _remove_moe_experts(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)
        moe = target_obj if isinstance(target_obj, nn.Module) else parent_mod

        router = getattr(moe, "router", getattr(moe, "gate", None))
        experts = getattr(moe, "experts", None)

        if router is None or experts is None:
            raise ValidationError(f"MoE module at '{op.target_path}' lacks router or experts")

        old_experts = len(experts)
        kept = op.parameters.get("remapped_kept_indices") or op.parameters.get("kept_indices")
        if kept is None:
            removed = set(op.parameters.get("removed_indices") or op.target_indices or [])
            kept = [i for i in range(old_experts) if i not in removed]

        if not kept:
            raise InvalidCompositionPathError("Cannot remove all MoE experts; remaining count cannot be 0")

        idx_t = torch.tensor(kept, dtype=torch.long, device=router.weight.device)
        with torch.no_grad():
            router.weight.data = router.weight.data.index_select(0, idx_t)
            if getattr(router, "bias", None) is not None:
                router.bias.data = router.bias.data.index_select(0, idx_t)

        if hasattr(router, "out_features"):
            router.out_features = len(kept)

        moe.experts = nn.ModuleList([experts[i] for i in kept])

        new_count = len(kept)
        for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
            if cfg is not None:
                if hasattr(cfg, "num_local_experts"):
                    cfg.num_local_experts = new_count
                if hasattr(cfg, "num_experts"):
                    cfg.num_experts = new_count

        result.modified_parameters.append(f"{op.target_path}.router.weight")
        result.removed_components.setdefault("moe_experts", []).extend(
            [i for i in range(old_experts) if i not in set(kept)]
        )
        result.details.append({
            "op": op.operation_id,
            "type": "moe_expert_prune",
            "old_expert_count": old_experts,
            "new_expert_count": new_count,
            "kept_experts": kept,
        })

    def _remove_weight_tensor_slice(self, model: nn.Module, op: UnifiedOperation, result: PlanExecutionResult) -> None:
        target_obj, parent_mod, attr = _resolve_model_target(model, op.target_path)
        param = target_obj if isinstance(target_obj, (nn.Parameter, torch.Tensor)) else getattr(target_obj, "weight", None)
        if param is None:
            raise ValidationError(f"Target '{op.target_path}' is not a parameter and has no .weight attribute")

        old_dim = param.shape[0]
        kept = op.parameters.get("remapped_kept_indices") or op.parameters.get("kept_indices")
        if kept is None:
            removed = set(op.parameters.get("removed_indices") or op.target_indices or [])
            kept = [i for i in range(old_dim) if i not in removed]

        if not kept:
            raise InvalidCompositionPathError("Cannot slice weight tensor to 0 rows")

        idx_t = torch.tensor(kept, dtype=torch.long, device=param.device)
        with torch.no_grad():
            param.data = param.data.index_select(0, idx_t)

        result.modified_parameters.append(op.target_path)
        result.details.append({
            "op": op.operation_id,
            "type": "weight_tensor_slice",
            "kept_indices": kept,
        })


PlanApplicator = PlanExecutor


def apply_plan(
    model: nn.Module,
    plan_or_ops: Union[UnifiedPlan, list[UnifiedOperation]],
    validate_capabilities: bool = True,
    validate_conflicts: bool = True,
    compose: bool = True,
) -> PlanExecutionResult:
    """Applies a neurosurgery plan or operation list to a model."""
    executor = PlanExecutor(
        validate_capabilities=validate_capabilities,
        validate_conflicts=validate_conflicts,
        compose=compose,
    )
    return executor.apply(model, plan_or_ops)


def execute_plan(
    model: nn.Module,
    plan_or_ops: Union[UnifiedPlan, list[UnifiedOperation]],
    validate_capabilities: bool = True,
    validate_conflicts: bool = True,
    compose: bool = True,
) -> PlanExecutionResult:
    """Alias for apply_plan()."""
    return apply_plan(
        model,
        plan_or_ops,
        validate_capabilities=validate_capabilities,
        validate_conflicts=validate_conflicts,
        compose=compose,
    )

