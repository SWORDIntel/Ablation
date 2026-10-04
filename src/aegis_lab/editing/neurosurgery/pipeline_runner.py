"""Declarative Campaign Pipeline Runner for AEGIS-LAB Model Neurosurgery.

Implements multi-stage neurosurgery campaigns specified via YAML/dict configurations:
1. Campaign configuration & schema validation (model, datasets, steps: profile -> select ->
   preview -> apply -> recover -> quantize -> export, objectives, thresholds, output_dir).
2. Preflight checks: model path, dataset files/splits disjointness, adapter compatibility,
   and disk space validation before execution.
3. Sequential step execution with intermediate metric recording, artifact hashing, and
   state checkpointing for robust resumption if interrupted.
4. Comprehensive campaign reports: operator-readable summary (campaign_report.txt),
   machine-readable JSON (campaign_report.json), and consolidated Stage 4A provenance manifest.
5. Error handling and rollback: failure diagnostics capture, temporary resource cleanup,
   and baseline preservation/restoration.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import datetime
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import yaml

from .common import (
    LOG,
    file_sha256,
    get_layers,
    load_prompts,
    load_tensor_artifact,
    parameter_bytes,
    resolve_device,
    set_layers,
    trust_remote_code_enabled,
)
from .math_ops import (
    apply_constrained_directional_surgery,
    contrast_direction,
    preservation_basis,
)
from .stage4a_provenance import (
    ALLOWED_SPLITS,
    DatasetFingerprint,
    ProvenanceManifest,
    RestorationManifest,
    compute_config_hash,
    compute_model_hash,
    compute_state_dict_hashes,
    compute_tensor_hash,
    compute_tokenizer_hash,
)
from .stage4b_contract import (
    AdapterCapabilityRegistry,
    ComponentType,
    OperationKind,
    get_capability_matrix,
)

SCHEMA_VERSION_CAMPAIGN = "1.0"
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION_CAMPAIGN, "1"}
STANDARD_STEP_ORDER = [
    "profile",
    "select",
    "preview",
    "apply",
    "recover",
    "quantize",
    "export",
]
VALID_DATASET_KINDS = {"keep", "drop", "change", "calibration"}


# ==============================================================================
# Exceptions
# ==============================================================================


class NeurosurgeryPipelineError(Exception):
    """Base exception for all neurosurgery campaign pipeline errors."""
    pass


class CampaignConfigError(NeurosurgeryPipelineError):
    """Raised when campaign configuration schema or values are invalid."""
    pass


class PreflightCheckError(NeurosurgeryPipelineError):
    """Raised when preflight validation fails before campaign execution."""
    pass


class StepExecutionError(NeurosurgeryPipelineError):
    """Raised when a campaign step execution encounters an unrecoverable failure."""
    pass


class ThresholdGateError(NeurosurgeryPipelineError):
    """Raised when an executed campaign fails defined quality/performance threshold gates."""
    pass


class RollbackError(NeurosurgeryPipelineError):
    """Raised when restoring or rolling back a failed campaign encounters an error."""
    pass


# ==============================================================================
# Enums and Statuses
# ==============================================================================


class CampaignStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


# ==============================================================================
# 1. Campaign Configuration Schema
# ==============================================================================


@dataclass
class ModelConfig:
    """Model configuration for the campaign."""
    path: str
    device: str = "cpu"
    dtype: str = "float32"
    architecture: Optional[str] = None
    trust_remote_code: bool = False
    model_instance: Optional[nn.Module] = None
    tokenizer_instance: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "device": self.device,
            "dtype": self.dtype,
            "architecture": self.architecture,
            "trust_remote_code": self.trust_remote_code,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        if not isinstance(data, dict):
            raise CampaignConfigError("model configuration must be a mapping")
        if "path" not in data and "model_instance" not in data:
            raise CampaignConfigError("model configuration requires 'path'")
        return cls(
            path=str(data.get("path", "")),
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
            architecture=data.get("architecture"),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            model_instance=data.get("model_instance"),
            tokenizer_instance=data.get("tokenizer_instance"),
        )


@dataclass
class DatasetSpec:
    """Specification for a dataset used in the campaign."""
    name: str
    path: str
    split: str = "discovery"
    kind: str = "keep"
    format: Optional[str] = None
    max_prompts: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "split": self.split,
            "kind": self.kind,
            "format": self.format,
            "max_prompts": self.max_prompts,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> DatasetSpec:
        if not isinstance(data, dict):
            raise CampaignConfigError(f"Dataset '{name}' config must be a mapping")
        if "path" not in data:
            raise CampaignConfigError(f"Dataset '{name}' is missing required field 'path'")
        split = data.get("split", "discovery")
        kind = data.get("kind", "keep")
        return cls(
            name=name,
            path=str(data["path"]),
            split=str(split),
            kind=str(kind),
            format=data.get("format"),
            max_prompts=data.get("max_prompts"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class StepConfig:
    """Configuration for an individual campaign pipeline step."""
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "params": copy.deepcopy(self.params),
        }

    @classmethod
    def from_dict(cls, data: Union[str, dict[str, Any], StepConfig]) -> StepConfig:
        if isinstance(data, StepConfig):
            return data
        if isinstance(data, str):
            return cls(name=data, enabled=True, params={})
        if not isinstance(data, dict):
            raise CampaignConfigError("Step config must be a string step name or mapping")
        if "name" not in data:
            raise CampaignConfigError("Step config missing required 'name' field")
        return cls(
            name=str(data["name"]),
            enabled=bool(data.get("enabled", True)),
            params=dict(data.get("params", {})),
        )


@dataclass
class CampaignObjectives:
    """Target objectives for the neurosurgery campaign."""
    primary: str = "keep_retention"
    metrics: list[str] = field(default_factory=list)
    targets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "metrics": list(self.metrics),
            "targets": dict(self.targets),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> CampaignObjectives:
        if not data:
            return cls()
        return cls(
            primary=str(data.get("primary", "keep_retention")),
            metrics=list(data.get("metrics", [])),
            targets=dict(data.get("targets", {})),
        )


@dataclass
class CampaignThresholds:
    """Quality and safety gating thresholds for the campaign."""
    max_mean_kl: Optional[float] = None
    min_top1_agreement: Optional[float] = None
    max_drop_rebound: Optional[float] = None
    max_loss_delta: Optional[float] = None
    min_parameter_reduction: Optional[int] = None
    max_latency_ms: Optional[float] = None
    custom: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in (
            "max_mean_kl",
            "min_top1_agreement",
            "max_drop_rebound",
            "max_loss_delta",
            "min_parameter_reduction",
            "max_latency_ms",
        ):
            val = getattr(self, k)
            if val is not None:
                d[k] = val
        if self.custom:
            d["custom"] = dict(self.custom)
        return d

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> CampaignThresholds:
        if not data:
            return cls()
        custom = dict(data.get("custom", {}))
        return cls(
            max_mean_kl=float(data["max_mean_kl"]) if "max_mean_kl" in data else None,
            min_top1_agreement=(
                float(data["min_top1_agreement"]) if "min_top1_agreement" in data else None
            ),
            max_drop_rebound=(
                float(data["max_drop_rebound"]) if "max_drop_rebound" in data else None
            ),
            max_loss_delta=(
                float(data["max_loss_delta"]) if "max_loss_delta" in data else None
            ),
            min_parameter_reduction=(
                int(data["min_parameter_reduction"])
                if "min_parameter_reduction" in data
                else None
            ),
            max_latency_ms=(
                float(data["max_latency_ms"]) if "max_latency_ms" in data else None
            ),
            custom=custom,
        )


@dataclass
class CampaignConfig:
    """Comprehensive Declarative Campaign Configuration."""
    campaign_id: str
    output_dir: str
    model: ModelConfig
    datasets: dict[str, DatasetSpec] = field(default_factory=dict)
    steps: list[StepConfig] = field(default_factory=list)
    objectives: CampaignObjectives = field(default_factory=CampaignObjectives)
    thresholds: CampaignThresholds = field(default_factory=CampaignThresholds)
    schema_version: str = SCHEMA_VERSION_CAMPAIGN
    seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate campaign configuration structure, ordering, and constraints."""
        errors: list[str] = []

        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            errors.append(
                f"Unsupported schema_version '{self.schema_version}', expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        if not self.campaign_id or not isinstance(self.campaign_id, str):
            errors.append("campaign_id must be a non-empty string")

        if not self.output_dir or not isinstance(self.output_dir, str):
            errors.append("output_dir must be a non-empty string path")

        if not isinstance(self.model, ModelConfig):
            errors.append("model must be an instance of ModelConfig")
        elif not self.model.path and self.model.model_instance is None:
            errors.append("model configuration requires either 'path' or 'model_instance'")

        # Validate datasets
        for name, spec in self.datasets.items():
            if spec.split not in ALLOWED_SPLITS:
                errors.append(
                    f"Dataset '{name}' split '{spec.split}' is invalid (allowed: {sorted(ALLOWED_SPLITS)})"
                )
            if spec.kind not in VALID_DATASET_KINDS:
                errors.append(
                    f"Dataset '{name}' kind '{spec.kind}' is invalid (allowed: {sorted(VALID_DATASET_KINDS)})"
                )

        # Validate steps
        step_names = [s.name for s in self.steps]
        if len(step_names) != len(set(step_names)):
            errors.append("Duplicate step names found in campaign steps list")

        unknown_steps = set(step_names) - set(STANDARD_STEP_ORDER)
        if unknown_steps:
            errors.append(f"Unrecognized step names in campaign: {sorted(unknown_steps)}")

        # Validate sequential step ordering dependencies
        # Standard: profile -> select -> preview -> apply -> recover -> quantize -> export
        indices = {name: i for i, name in enumerate(step_names)}
        if "preview" in indices and "select" in indices and indices["preview"] < indices["select"]:
            errors.append("Step 'preview' cannot precede step 'select'")
        if "apply" in indices and "select" in indices and indices["apply"] < indices["select"]:
            errors.append("Step 'apply' cannot precede step 'select'")
        if "apply" in indices and "preview" in indices and indices["apply"] < indices["preview"]:
            errors.append("Step 'apply' cannot precede step 'preview'")
        if "recover" in indices and "apply" in indices and indices["recover"] < indices["apply"]:
            errors.append("Step 'recover' cannot precede step 'apply'")
        if "quantize" in indices and "apply" in indices and indices["quantize"] < indices["apply"]:
            errors.append("Step 'quantize' cannot precede step 'apply'")
        if "export" in indices and "apply" in indices and indices["export"] < indices["apply"]:
            errors.append("Step 'export' cannot precede step 'apply'")

        if errors:
            raise CampaignConfigError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "model": self.model.to_dict(),
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "steps": [s.to_dict() for s in self.steps],
            "objectives": self.objectives.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "metadata": dict(self.metadata),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignConfig:
        if not isinstance(data, dict):
            raise CampaignConfigError("Campaign configuration payload must be a mapping")

        now = datetime.datetime.now(datetime.timezone.utc)
        cid = str(data.get("campaign_id") or f"camp_{int(now.timestamp())}")
        output_dir = str(data.get("output_dir", "./campaign_output"))
        schema_ver = str(data.get("schema_version", SCHEMA_VERSION_CAMPAIGN))
        seed = int(data.get("seed", 42))

        model_dict = data.get("model")
        if not model_dict:
            raise CampaignConfigError("Missing required 'model' section in campaign config")
        model = ModelConfig.from_dict(model_dict)

        datasets: dict[str, DatasetSpec] = {}
        raw_datasets = data.get("datasets", {})
        if isinstance(raw_datasets, list):
            for item in raw_datasets:
                spec = DatasetSpec.from_dict(item["name"], item)
                datasets[spec.name] = spec
        elif isinstance(raw_datasets, dict):
            for name, item in raw_datasets.items():
                datasets[name] = DatasetSpec.from_dict(name, item)

        steps: list[StepConfig] = []
        raw_steps = data.get("steps", [])
        if not raw_steps:
            # Default to full standard pipeline if none specified
            raw_steps = [StepConfig(name=s) for s in STANDARD_STEP_ORDER]
        for s in raw_steps:
            steps.append(StepConfig.from_dict(s))

        objectives = CampaignObjectives.from_dict(data.get("objectives"))
        thresholds = CampaignThresholds.from_dict(data.get("thresholds"))
        metadata = dict(data.get("metadata", {}))

        return cls(
            campaign_id=cid,
            output_dir=output_dir,
            model=model,
            datasets=datasets,
            steps=steps,
            objectives=objectives,
            thresholds=thresholds,
            schema_version=schema_ver,
            seed=seed,
            metadata=metadata,
        )

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> CampaignConfig:
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(path_or_str)
        data = yaml.safe_load(content)
        return cls.from_dict(data)


# ==============================================================================
# 2. Preflight Checks
# ==============================================================================


@dataclass
class PreflightCheckResult:
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass
class PreflightResult:
    passed: bool
    checks: list[PreflightCheckResult]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disk_space_available_bytes: int = 0
    disk_space_required_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "disk_space_available_bytes": self.disk_space_available_bytes,
            "disk_space_required_bytes": self.disk_space_required_bytes,
        }


def _check_model_path(config: CampaignConfig) -> PreflightCheckResult:
    if config.model.model_instance is not None:
        return PreflightCheckResult(
            name="model_path",
            passed=True,
            message="In-memory PyTorch model provided",
            details={"type": type(config.model.model_instance).__name__},
        )
    p = Path(config.model.path)
    if not p.exists():
        return PreflightCheckResult(
            name="model_path",
            passed=False,
            message=f"Model path does not exist: {p}",
        )
    # Check directory contents or file
    is_dir = p.is_dir()
    details: dict[str, Any] = {"is_dir": is_dir}
    if is_dir:
        has_config = (p / "config.json").exists()
        has_weights = any(
            p.glob("*.safetensors")
        ) or any(p.glob("*.bin")) or any(p.glob("*.pt"))
        details["has_config"] = has_config
        details["has_weights"] = has_weights
        if not has_config and not has_weights:
            return PreflightCheckResult(
                name="model_path",
                passed=False,
                message=f"Model directory '{p}' contains neither config.json nor weight files",
                details=details,
            )
    return PreflightCheckResult(
        name="model_path",
        passed=True,
        message=f"Model path verified: {p}",
        details=details,
    )


def _check_datasets(config: CampaignConfig) -> list[PreflightCheckResult]:
    results = []
    # 1. Existence and non-emptiness
    prompts_by_kind_split: dict[tuple[str, str], set[str]] = {}

    for name, spec in config.datasets.items():
        p = Path(spec.path)
        if not p.exists():
            results.append(
                PreflightCheckResult(
                    name=f"dataset_{name}_exists",
                    passed=False,
                    message=f"Dataset file does not exist: {p}",
                    details={"path": str(p)},
                )
            )
            continue

        if p.stat().st_size == 0:
            results.append(
                PreflightCheckResult(
                    name=f"dataset_{name}_non_empty",
                    passed=False,
                    message=f"Dataset file is empty: {p}",
                    details={"path": str(p), "size_bytes": 0},
                )
            )
            continue

        try:
            prompts = load_prompts(p, max_prompts=spec.max_prompts)
            results.append(
                PreflightCheckResult(
                    name=f"dataset_{name}_content",
                    passed=True,
                    message=f"Loaded {len(prompts)} prompts from {name}",
                    details={"count": len(prompts), "split": spec.split, "kind": spec.kind},
                )
            )
            prompts_by_kind_split[(spec.kind, spec.split)] = set(prompts)
        except Exception as exc:
            results.append(
                PreflightCheckResult(
                    name=f"dataset_{name}_load",
                    passed=False,
                    message=f"Failed loading dataset '{name}': {exc}",
                )
            )

    # 2. Check disjointness between splits for matching kinds (discovery, search, validation, test)
    # Stage 4A requires disjoint splits
    kinds = {spec.kind for spec in config.datasets.values()}
    for kind in kinds:
        splits = [
            (spec.split, prompts_by_kind_split.get((kind, spec.split)))
            for spec in config.datasets.values()
            if spec.kind == kind and (kind, spec.split) in prompts_by_kind_split
        ]
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1_name, s1_set = splits[i]
                s2_name, s2_set = splits[j]
                if s1_name != s2_name and s1_set and s2_set:
                    overlap = s1_set & s2_set
                    if overlap:
                        results.append(
                            PreflightCheckResult(
                                name=f"dataset_disjoint_{kind}_{s1_name}_{s2_name}",
                                passed=False,
                                message=(
                                    f"Dataset split overlap detected for kind '{kind}' between "
                                    f"'{s1_name}' and '{s2_name}': {len(overlap)} overlapping prompts"
                                ),
                                details={"overlap_count": len(overlap)},
                            )
                        )

    return results


def _check_adapter_compatibility(config: CampaignConfig) -> PreflightCheckResult:
    # Resolve architecture
    arch = config.model.architecture
    if not arch and config.model.model_instance is not None:
        cfg = getattr(config.model.model_instance, "config", None)
        if isinstance(cfg, dict):
            arch = cfg.get("model_type")
        elif cfg is not None:
            arch = getattr(cfg, "model_type", None) or type(config.model.model_instance).__name__

    if not arch:
        p = Path(config.model.path)
        if p.exists() and (p / "config.json").exists():
            try:
                raw_cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
                arch = raw_cfg.get("model_type")
            except Exception:
                pass

    if not arch:
        # Default or fallback architecture
        arch = "llama"

    matrix = get_capability_matrix()
    try:
        caps = matrix.get(arch)
    except KeyError:
        return PreflightCheckResult(
            name="adapter_compatibility",
            passed=False,
            message=f"Architecture '{arch}' not recognized in AdapterCapabilityRegistry",
            details={"architecture": arch},
        )

    # Check operations requested in steps
    violations = []
    for step in config.steps:
        if not step.enabled:
            continue
        params = step.params
        if "moe" in params and ComponentType.MOE_EXPERT not in caps.supported_components:
            violations.append(f"Step '{step.name}' requests MoE operations, but '{arch}' does not support MoE experts")

    if violations:
        return PreflightCheckResult(
            name="adapter_compatibility",
            passed=False,
            message="; ".join(violations),
            details={"architecture": arch, "violations": violations},
        )

    return PreflightCheckResult(
        name="adapter_compatibility",
        passed=True,
        message=f"Architecture '{arch}' adapter '{caps.adapter_name}' verified compatible",
        details={"architecture": arch, "adapter_name": caps.adapter_name},
    )


def _check_disk_space(config: CampaignConfig, min_disk_mb: float = 50.0) -> PreflightCheckResult:
    out_path = Path(config.output_dir)
    # Find closest existing parent path
    check_path = out_path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent

    try:
        usage = shutil.disk_usage(check_path)
        available_mb = usage.free / (1024 * 1024)
        required_bytes = int(min_disk_mb * 1024 * 1024)
        passed = usage.free >= required_bytes
        msg = (
            f"{available_mb:.1f} MB available on {check_path} (required: {min_disk_mb:.1f} MB)"
            if passed
            else f"Insufficient disk space: {available_mb:.1f} MB available, {min_disk_mb:.1f} MB required"
        )
        return PreflightCheckResult(
            name="disk_space",
            passed=passed,
            message=msg,
            details={
                "available_bytes": usage.free,
                "required_bytes": required_bytes,
                "available_mb": available_mb,
            },
        )
    except Exception as exc:
        return PreflightCheckResult(
            name="disk_space",
            passed=False,
            message=f"Failed checking disk usage: {exc}",
        )


def _check_output_dir_writable(config: CampaignConfig) -> PreflightCheckResult:
    out_path = Path(config.output_dir)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
        probe = out_path / f".write_probe_{os.getpid()}.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return PreflightCheckResult(
            name="output_dir_writable",
            passed=True,
            message=f"Output directory is writable: {out_path}",
        )
    except Exception as exc:
        return PreflightCheckResult(
            name="output_dir_writable",
            passed=False,
            message=f"Output directory not writable: {exc}",
        )


def run_preflight_checks(config: CampaignConfig, min_disk_mb: float = 50.0) -> PreflightResult:
    """Execute all preflight checks on campaign configuration."""
    checks: list[PreflightCheckResult] = []

    checks.append(_check_model_path(config))
    checks.extend(_check_datasets(config))
    checks.append(_check_adapter_compatibility(config))
    checks.append(_check_disk_space(config, min_disk_mb=min_disk_mb))
    checks.append(_check_output_dir_writable(config))

    errors = [c.message for c in checks if not c.passed]
    warnings: list[str] = []
    disk_check = next((c for c in checks if c.name == "disk_space"), None)
    avail_bytes = disk_check.details.get("available_bytes", 0) if disk_check else 0
    req_bytes = disk_check.details.get("required_bytes", 0) if disk_check else 0

    return PreflightResult(
        passed=len(errors) == 0,
        checks=checks,
        errors=errors,
        warnings=warnings,
        disk_space_available_bytes=avail_bytes,
        disk_space_required_bytes=req_bytes,
    )


# ==============================================================================
# 3. Step Execution & Artifact Tracking
# ==============================================================================


@dataclass
class CampaignArtifact:
    """Descriptor for an artifact produced by a campaign step."""
    name: str
    path: str
    step: str
    size_bytes: int
    sha256: str
    artifact_type: str = "generic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "step": self.step,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "artifact_type": self.artifact_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignArtifact:
        return cls(
            name=str(data["name"]),
            path=str(data["path"]),
            step=str(data["step"]),
            size_bytes=int(data["size_bytes"]),
            sha256=str(data["sha256"]),
            artifact_type=str(data.get("artifact_type", "generic")),
        )


@dataclass
class StepResult:
    """Result of executing a single campaign step."""
    step_name: str
    status: StepStatus
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, CampaignArtifact] = field(default_factory=dict)
    error: Optional[str] = None
    resumed_from_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "metrics": copy.deepcopy(self.metrics),
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "error": self.error,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepResult:
        arts = {
            k: CampaignArtifact.from_dict(v)
            for k, v in data.get("artifacts", {}).items()
        }
        return cls(
            step_name=str(data["step_name"]),
            status=StepStatus(data["status"]),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            metrics=dict(data.get("metrics", {})),
            artifacts=arts,
            error=data.get("error"),
            resumed_from_checkpoint=bool(data.get("resumed_from_checkpoint", False)),
        )


@dataclass
class ThresholdResult:
    """Evaluation result for an individual threshold constraint."""
    criterion: str
    threshold: float
    observed: Optional[float]
    comparison: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "threshold": self.threshold,
            "observed": self.observed,
            "comparison": self.comparison,
            "passed": self.passed,
        }


@dataclass
class RollbackResult:
    """Result of rolling back a campaign or step."""
    success: bool
    restored_files: list[str] = field(default_factory=list)
    cleaned_paths: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "restored_files": list(self.restored_files),
            "cleaned_paths": list(self.cleaned_paths),
            "details": dict(self.details),
            "error": self.error,
        }


@dataclass
class CampaignResult:
    """Comprehensive end-to-end campaign execution result."""
    campaign_id: str
    status: CampaignStatus
    passed: bool
    output_dir: Path
    started_at: str
    finished_at: str
    duration_seconds: float
    config: dict[str, Any]
    preflight: PreflightResult
    step_results: dict[str, StepResult]
    metrics: dict[str, Any]
    artifacts: list[CampaignArtifact]
    threshold_results: dict[str, ThresholdResult]
    report_txt_path: Optional[Path] = None
    report_json_path: Optional[Path] = None
    provenance_manifest_path: Optional[Path] = None
    failure_diagnostics_path: Optional[Path] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "status": self.status.value,
            "passed": self.passed,
            "output_dir": str(self.output_dir),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "config": copy.deepcopy(self.config),
            "preflight": self.preflight.to_dict(),
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
            "metrics": copy.deepcopy(self.metrics),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "threshold_results": {k: v.to_dict() for k, v in self.threshold_results.items()},
            "report_txt_path": str(self.report_txt_path) if self.report_txt_path else None,
            "report_json_path": str(self.report_json_path) if self.report_json_path else None,
            "provenance_manifest_path": (
                str(self.provenance_manifest_path) if self.provenance_manifest_path else None
            ),
            "failure_diagnostics_path": (
                str(self.failure_diagnostics_path) if self.failure_diagnostics_path else None
            ),
            "error": self.error,
        }


# ==============================================================================
# 4. Campaign Pipeline Runner
# ==============================================================================


class CampaignPipelineRunner:
    """Declarative runner executing multi-stage neurosurgery campaigns."""

    def __init__(
        self,
        config: Union[CampaignConfig, dict[str, Any], str, Path],
        custom_step_handlers: Optional[dict[str, Callable[..., dict[str, Any]]]] = None,
    ) -> None:
        if isinstance(config, CampaignConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = CampaignConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = CampaignConfig.from_dict(config)
        else:
            raise TypeError(f"Invalid config type: {type(config).__name__}")

        self.output_dir = Path(self.config.output_dir)
        self.custom_step_handlers = dict(custom_step_handlers or {})
        self.step_results: dict[str, StepResult] = {}
        self.metrics: dict[str, Any] = {}
        self.artifacts: list[CampaignArtifact] = []
        self.threshold_results: dict[str, ThresholdResult] = {}

        # Internal references
        self.model: Optional[nn.Module] = self.config.model.model_instance
        self.tokenizer: Optional[Any] = self.config.model.tokenizer_instance or getattr(self.model, "tokenizer", None)
        self._baseline_model: Optional[nn.Module] = (
            copy.deepcopy(self.model) if self.model is not None else None
        )

    def register_step_handler(
        self, step_name: str, handler: Callable[..., dict[str, Any]]
    ) -> None:
        """Register or override a custom step execution handler."""
        self.custom_step_handlers[step_name] = handler

    # --------------------------------------------------------------------------
    # Checkpointing and State Persistence
    # --------------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "campaign_checkpoint.json"

    def _save_checkpoint(self, last_step: str) -> None:
        state = {
            "campaign_id": self.config.campaign_id,
            "schema_version": self.config.schema_version,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "last_completed_step": last_step,
            "steps": {k: v.to_dict() for k, v in self.step_results.items()},
        }
        ckpt = self._checkpoint_path()
        tmp_ckpt = self.output_dir / f".tmp_{ckpt.name}"
        tmp_ckpt.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp_ckpt.replace(ckpt)

    def _load_checkpoint(self) -> Optional[dict[str, Any]]:
        ckpt = self._checkpoint_path()
        if not ckpt.exists():
            return None
        try:
            return json.loads(ckpt.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.warning("Failed loading checkpoint: %s", exc)
            return None

    def _validate_resumable_step(self, step_result: StepResult) -> bool:
        """Verify that all recorded artifacts for a step exist and match checksums."""
        for art in step_result.artifacts.values():
            p = Path(art.path)
            if not p.exists():
                LOG.warning("Artifact %s missing on disk; cannot resume step %s", p, step_result.step_name)
                return False
            actual_hash = file_sha256(p)
            if actual_hash != art.sha256:
                LOG.warning(
                    "Artifact %s hash mismatch (%s != %s); cannot resume step %s",
                    p, actual_hash, art.sha256, step_result.step_name
                )
                return False
        return True

    # --------------------------------------------------------------------------
    # Core Step Implementations
    # --------------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            if self.tokenizer is None:
                self.tokenizer = getattr(self.model, "tokenizer", None)
            return
        p = Path(self.config.model.path)
        if not p.exists():
            raise FileNotFoundError(f"Model path does not exist: {p}")

        device = resolve_device(self.config.model.device)
        # Attempt to load HF causal LM or standard PyTorch weights
        if (p / "config.json").exists():
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(
                    str(p), trust_remote_code=self.config.model.trust_remote_code
                )
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "left"
                self.tokenizer = tok
                kwargs: dict[str, Any] = {
                    "trust_remote_code": self.config.model.trust_remote_code,
                }
                if self.config.model.dtype:
                    dtype = getattr(torch, self.config.model.dtype, None)
                    if not isinstance(dtype, torch.dtype):
                        raise CampaignConfigError(f"Unsupported model dtype: {self.config.model.dtype}")
                    kwargs["dtype"] = dtype
                if device != "cpu":
                    kwargs["device_map"] = device
                model = AutoModelForCausalLM.from_pretrained(str(p), **kwargs)
                if device == "cpu":
                    model.to(device)
                model.eval()
                self.model = model
                return
            except Exception as exc:
                raise PreflightCheckError(f"HF model/tokenizer loading failed: {exc}") from exc

        raise PreflightCheckError("Measured campaigns require a local HF checkpoint/tokenizer or explicit model instances")

    def _step_profile(self, step_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Execute Stage 0 KEEP vs DROP residual profiling."""
        step_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model_loaded()

        keep_spec = next(
            (s for s in self.config.datasets.values() if s.kind == "keep" and s.split in {"discovery", "search"}), None
        )
        drop_spec = next(
            (s for s in self.config.datasets.values() if s.kind == "drop" and s.split in {"discovery", "search"}), None
        )
        if not keep_spec or not drop_spec:
            raise CampaignConfigError("Step 'profile' requires at least one 'keep' and one 'drop' dataset")

        keep_prompts = load_prompts(keep_spec.path, max_prompts=params.get("max_prompts", 32))
        drop_prompts = load_prompts(drop_spec.path, max_prompts=params.get("max_prompts", 32))

        from .measured import measured_profile
        meta = measured_profile(self.model, self.tokenizer, keep_prompts, drop_prompts,
                                step_dir, batch_size=params.get("batch_size", 2),
                                basis_rank=params.get("basis_rank", 4))
        return dict(num_layers=meta["num_layers"], keep_prompts=len(keep_prompts),
                    drop_prompts=len(drop_prompts), measurement="model_activations",
                    mean_contrast_separation=sum(m["contrast_separation"] for m in meta["layers"]) / meta["num_layers"])

    def _step_select(self, step_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Compile component selectors into an explicit surgery plan."""
        step_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model_loaded()

        drop_layers = [int(x) for x in params.get("drop_layers", [])]
        ablation = params.get(
            "ablation",
            {
                "layers": [],
                "targets": ["mlp.down_proj", "self_attn.o_proj"],
                "strength": 0.5,
                "norm_preserve": True,
                "preserve_subspace": True,
            },
        )
        structured = dict(params.get("structured", {}))

        _, layers = get_layers(self.model)
        total_layers = len(layers)
        for idx in drop_layers:
            if idx < 0 or idx >= total_layers:
                raise IndexError(f"drop_layer {idx} out of range [0, {total_layers})")

        plan = {
            "version": 3,
            "source_model": self.config.model.path,
            "drop_layers": drop_layers,
            "ablation": ablation,
            "structured": structured,
        }

        plan_path = step_dir / "surgery_plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

        total_ops = len(drop_layers) + len(ablation.get("layers", [])) + len(structured)
        return {
            "drop_layers_count": len(drop_layers),
            "ablation_layers_count": len(ablation.get("layers", [])),
            "total_operations": total_ops,
            "total_model_layers": total_layers,
        }

    def _step_preview(self, step_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Run read-only preview and conflict detection on surgery plan."""
        step_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model_loaded()

        select_dir = self.output_dir / "select"
        plan_path = select_dir / "surgery_plan.yaml"
        if not plan_path.exists():
            raise FileNotFoundError(f"Surgery plan not found at {plan_path}")

        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        params_before = sum(p.numel() for p in self.model.parameters())
        bytes_before = parameter_bytes(self.model)

        from .preview import build_preview
        profile = self.output_dir / "profile" / "profile.pt"
        geometry = build_preview(self.model, str(plan_path), str(profile) if profile.exists() else None)
        _, layers = get_layers(self.model)
        drop_layers = set(plan.get("drop_layers", []))
        removed_params = sum(sum(p.numel() for p in layers[i].parameters()) for i in drop_layers)
        for op in geometry["operations"]:
            if op.get("layer") not in drop_layers and "shape_before" in op:
                removed_params += math.prod(op["shape_before"]) - math.prod(op["shape_after"])
            elif op.get("layer") not in drop_layers and op.get("kind") == "moe_expert_remove":
                removed_params += sum(math.prod(param["shape"]) for param in op["parameters"])

        params_after = params_before - removed_params
        reduction_ratio = removed_params / max(params_before, 1)

        preview = {
            "model": self.config.model.path,
            "parameters_before": params_before,
            "parameters_after": params_after,
            "parameters_removed": removed_params,
            "parameter_reduction_ratio": reduction_ratio,
            "bytes_before": bytes_before,
            "conflicts": [],
            "geometry": geometry,
        }

        (step_dir / "preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")

        return {
            "parameters_before": params_before,
            "parameters_after": params_after,
            "parameters_removed": removed_params,
            "parameter_reduction_ratio": reduction_ratio,
            "conflicts_count": 0,
        }

    def _step_apply(self, step_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Backup baseline and execute surgery plan (directional, structured, layer drops)."""
        step_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model_loaded()

        backup_dir = self.output_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # 1. Baseline checkpoint backup
        backup_model_file = backup_dir / "baseline_model.pt"
        torch.save(self.model.state_dict(), backup_model_file)
        if self._baseline_model is None and self.model is not None:
            self._baseline_model = copy.deepcopy(self.model)
        source_hashes = compute_state_dict_hashes(self.model.state_dict())

        # 2. Load plan
        select_dir = self.output_dir / "select"
        plan_path = select_dir / "surgery_plan.yaml"
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))

        params_before = sum(p.numel() for p in self.model.parameters())

        from .apply import apply_to_model
        profile_path = self.output_dir / "profile" / "profile.pt"
        operations = apply_to_model(self.model, str(profile_path) if profile_path.exists() else None,
                                    str(plan_path))
        drop_layers = operations["dropped_layers"]

        params_after = sum(p.numel() for p in self.model.parameters())
        candidate_hashes = compute_state_dict_hashes(self.model.state_dict())
        edited_tensors = [
            k for k in sorted(set(candidate_hashes) | set(source_hashes))
            if candidate_hashes.get(k) != source_hashes.get(k)
        ]

        # 5. Save candidate model and manifests
        torch.save(self.model.state_dict(), step_dir / "candidate_model.pt")
        if hasattr(self.model, "save_pretrained") and getattr(self.model, "config", None) is not None:
            try:
                self.model.save_pretrained(step_dir)
                if self.tokenizer and hasattr(self.tokenizer, "save_pretrained"):
                    self.tokenizer.save_pretrained(step_dir)
            except Exception:
                pass

        restoration_manifest = RestorationManifest(
            source_checkpoint=str(backup_model_file),
            candidate_checkpoint=str(step_dir / "candidate_model.pt"),
            source_hashes=source_hashes,
            candidate_hashes=candidate_hashes,
            edited_tensors=edited_tensors,
            backup_location=str(backup_dir),
        )
        restoration_manifest.save(step_dir / "restoration_manifest.yaml")

        manifest_data = {
            "source_model": self.config.model.path,
            "parameters_before": params_before,
            "parameters_after": params_after,
            "parameters_removed": params_before - params_after,
            "reduction_ratio": (params_before - params_after) / max(params_before, 1),
            "dropped_layers": drop_layers,
            "edited_tensor_count": len(edited_tensors),
        }
        (step_dir / "neurosurgery_manifest.json").write_text(
            json.dumps(manifest_data, indent=2), encoding="utf-8"
        )

        return {
            "parameters_before": params_before,
            "parameters_after": params_after,
            "parameters_removed": params_before - params_after,
            "reduction_ratio": (params_before - params_after) / max(params_before, 1),
            "edited_tensors_count": len(edited_tensors),
        }

    def _step_recover(self, step_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Execute post-surgery recovery fine-tuning with DROP rebound monitoring."""
        step_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model_loaded()

        from .measured import recover_model, load_change_examples
        def prompts(kind):
            spec = next((s for s in self.config.datasets.values() if s.kind == kind
                         and s.split in {"discovery", "search"}), None)
            if spec and kind == "change":
                return load_change_examples(spec.path)
            return load_prompts(spec.path, spec.max_prompts) if spec else None
        self.model, report = recover_model(
            self.model, self.tokenizer, prompts("keep"), prompts("drop"), prompts("change"),
            steps=int(params.get("steps", 5)), lr=float(params.get("lr", 1e-4)),
            rank=int(params.get("lora_r", 8)), seed=self.config.seed,
            max_rebound=self.config.thresholds.max_drop_rebound)
        if not hasattr(self.model, "save_pretrained"):
            raise ValueError("Recovery requires a reloadable HF checkpoint")
        self.model.save_pretrained(step_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(step_dir)
        (step_dir / "recovery_report.json").write_text(json.dumps(report, indent=2))
        return report

    def _step_quantize(self, step_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Execute post-surgery quantization and validate against drift."""
        step_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model_loaded()

        from .stage6_quantization import (QuantizationConfig, quantize_model,
            validate_quantized_candidate, QuantizationValidationThresholds,
            export_quantized_checkpoint, bind_calibration_dataset, calculate_model_memory_bytes)
        spec = next((s for s in self.config.datasets.values() if s.kind == "calibration"), None)
        if spec is None or self.tokenizer is None:
            raise ValueError("Quantization requires real calibration data and a tokenizer")
        prompts = load_prompts(spec.path, spec.max_prompts)
        baseline = copy.deepcopy(self.model)
        cfg = QuantizationConfig(bits=int(params.get("bits", 8)))
        self._quantization_config = cfg
        before = calculate_model_memory_bytes(self.model)
        self.model = quantize_model(self.model, cfg)
        validation = validate_quantized_candidate(baseline, self.model, prompts,
            tokenizer=self.tokenizer, device=self.config.model.device,
            thresholds=QuantizationValidationThresholds(
                max_kl_drift=self.config.thresholds.max_mean_kl
                if self.config.thresholds.max_mean_kl is not None else 0.5))
        self._quantization_binding = bind_calibration_dataset(prompts, tokenizer=self.tokenizer)
        export_quantized_checkpoint(self.model, step_dir, cfg,
                                   calibration_binding=self._quantization_binding)
        after = calculate_model_memory_bytes(self.model)
        report = dict(quantization_bits=cfg.bits, memory_before_bytes=before,
                      memory_after_bytes=after, compression_ratio=before / max(after, 1),
                      mean_kl_drift=validation.kl_drift, passed=validation.passed)
        (step_dir / "quantization_report.json").write_text(json.dumps(report, indent=2))
        return report

    def _step_export(self, step_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Export runtime package and measure performance."""
        step_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model_loaded()

        from .stage7_export import benchmark_runtime_profile, BenchmarkProfile
        from .stage6_quantization import QuantizedLinear, export_quantized_checkpoint, QuantizationConfig
        package_dir = step_dir / "package"
        if any(isinstance(m, QuantizedLinear) for m in self.model.modules()):
            export_quantized_checkpoint(self.model, package_dir,
                                        self._quantization_config,
                                        calibration_binding=self._quantization_binding)
            runtime = "aegis torch dequantizing linear (custom loader required)"
        else:
            if not hasattr(self.model, "save_pretrained"):
                raise ValueError("Export requires a reloadable HF checkpoint")
            self.model.save_pretrained(package_dir, safe_serialization=params.get("format") != "pytorch")
            runtime = "transformers"
        if self.tokenizer is None:
            raise ValueError("Export requires tokenizer assets")
        self.tokenizer.save_pretrained(package_dir)
        benchmark = benchmark_runtime_profile(self.model, BenchmarkProfile(
            prompt_length=int(params.get("prompt_length", 8)), decode_steps=int(params.get("decode_steps", 2))),
            device=self.config.model.device, num_warmup=1, num_repeats=3).to_dict()
        pkg_bytes = sum(p.stat().st_size for p in package_dir.rglob("*") if p.is_file())
        report = dict(export_format=params.get("format", "safetensors"), package_dir=str(package_dir),
                      package_bytes=pkg_bytes, runtime=runtime, benchmark=benchmark)
        (step_dir / "export_report.json").write_text(json.dumps(report, indent=2))
        return dict(package_bytes=pkg_bytes, prefill_latency_ms=benchmark["prefill_latency_ms"],
                    decode_latency_ms=benchmark["decode_latency_per_token_ms"])

    # --------------------------------------------------------------------------
    # Execution Loop and Error Handling
    # --------------------------------------------------------------------------

    def run(
        self,
        resume: bool = False,
        strict_preflight: bool = True,
        rollback_on_failure: bool = True,
    ) -> CampaignResult:
        """Run campaign pipeline from start or resume from checkpoint."""
        now = datetime.datetime.now(datetime.timezone.utc)
        started_at = now.isoformat()
        t0 = now.timestamp()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Preflight checks
        preflight = run_preflight_checks(self.config)
        if not preflight.passed:
            if strict_preflight:
                diag_path = self._write_failure_diagnostics(
                    failed_step="preflight",
                    exc=PreflightCheckError(f"Preflight checks failed: {preflight.errors}"),
                )
                self._generate_reports(
                    status=CampaignStatus.FAILED,
                    started_at=started_at,
                    preflight=preflight,
                    error=f"Preflight checks failed: {preflight.errors}",
                    failure_diagnostics_path=diag_path,
                )
                raise PreflightCheckError(f"Preflight checks failed: {preflight.errors}")
            else:
                LOG.warning("Preflight checks failed, proceeding non-strictly: %s", preflight.errors)

        from .measured import bind_files, verify_binding
        input_paths = {"model": self.config.model.path,
                       **{f"dataset_{name}": spec.path for name, spec in self.config.datasets.items()}}
        input_binding = bind_files(input_paths)
        self._ensure_model_loaded()
        source_model_hash = compute_model_hash(self.model)
        source_tokenizer_hash = compute_tokenizer_hash(self.tokenizer)
        from .operator_audit import implementation_binding, callable_binding
        implementation = implementation_binding()
        handlers = {name: callable_binding(handler) for name, handler in self.custom_step_handlers.items()}
        binding_path = self.output_dir / "input_binding.json"
        order = [step.to_dict() for step in self.config.steps if step.enabled]
        if resume and binding_path.exists():
            previous = json.loads(binding_path.read_text())
            verify_binding(previous["binding"])
            if previous.get("implementation") != implementation or previous.get("handlers") != handlers:
                raise CampaignConfigError("Resume changed adapter/operator implementation or custom handlers")
            if previous.get("source_model_hash") != source_model_hash or previous.get("tokenizer_hash") != source_tokenizer_hash:
                raise CampaignConfigError("Resume changed the actual source model or tokenizer")
            prior_order = previous["edit_order"]
            if order[:len(prior_order)] != prior_order:
                raise CampaignConfigError("Resume changed existing edit order or step parameters")
        (binding_path).write_text(json.dumps(dict(binding=input_binding, edit_order=order,
                                                   implementation=implementation, handlers=handlers,
                                                   source_model_hash=source_model_hash, tokenizer_hash=source_tokenizer_hash), indent=2))

        # 2. Checkpoint resumption resolution
        resumed_step_names: set[str] = set()
        if resume:
            ckpt = self._load_checkpoint()
            if ckpt and "steps" in ckpt:
                all_valid = True
                for step_name, raw_res in ckpt["steps"].items():
                    step_res = StepResult.from_dict(raw_res)
                    if step_res.status == StepStatus.COMPLETED and self._validate_resumable_step(step_res):
                        step_res.resumed_from_checkpoint = True
                        self.step_results[step_name] = step_res
                        resumed_step_names.add(step_name)
                    else:
                        all_valid = False
                        break
                LOG.info("Resumed %d completed steps from checkpoint: %s", len(resumed_step_names), resumed_step_names)

        if "apply" in resumed_step_names:
            checkpoint_dir = self.output_dir / ("recover" if "recover" in resumed_step_names else "apply")
            if "apply" not in self.custom_step_handlers:
                from .validate import _load_hf
                self.model, self.tokenizer = _load_hf(str(checkpoint_dir), self.config.model.device)
            if "quantize" in resumed_step_names and "quantize" not in self.custom_step_handlers:
                from .stage6_quantization import load_quantized_checkpoint, QuantizationConfig, CalibrationBinding
                self.model, metadata = load_quantized_checkpoint(self.output_dir / "quantize", base_model=self.model)
                self._quantization_config = QuantizationConfig.from_dict(metadata["quantization_config"])
                self._quantization_binding = CalibrationBinding.from_dict(metadata["calibration_binding"])

        # 3. Sequential step execution
        enabled_steps = [s for s in self.config.steps if s.enabled]
        status = CampaignStatus.RUNNING

        step_dispatch = {
            "profile": self._step_profile,
            "select": self._step_select,
            "preview": self._step_preview,
            "apply": self._step_apply,
            "recover": self._step_recover,
            "quantize": self._step_quantize,
            "export": self._step_export,
        }

        for step in enabled_steps:
            if step.name in resumed_step_names:
                LOG.info("Skipping already completed step: %s", step.name)
                # Accumulate metrics and artifacts
                for art in self.step_results[step.name].artifacts.values():
                    self.artifacts.append(art)
                self.metrics[step.name] = self.step_results[step.name].metrics
                continue

            step_dir = self.output_dir / step.name
            step_dir.mkdir(parents=True, exist_ok=True)
            step_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
            st0 = datetime.datetime.now(datetime.timezone.utc).timestamp()
            LOG.info("Starting campaign step: %s", step.name)

            try:
                # Dispatch handler
                if step.name in self.custom_step_handlers:
                    step_metrics = self.custom_step_handlers[step.name](step_dir, step.params)
                elif step.name in step_dispatch:
                    step_metrics = step_dispatch[step.name](step_dir, step.params)
                else:
                    raise StepExecutionError(f"No execution handler registered for step '{step.name}'")

                verify_binding(input_binding)
                if compute_tokenizer_hash(self.tokenizer) != source_tokenizer_hash:
                    raise CampaignConfigError("Tokenizer changed during campaign")
                if implementation_binding() != implementation:
                    raise CampaignConfigError("Operator implementation changed during campaign")
                now_finish = datetime.datetime.now(datetime.timezone.utc)
                st1 = now_finish.timestamp()
                step_finish = now_finish.isoformat()
                step_dur = max(0.001, st1 - st0)

                # Collect step artifacts
                step_artifacts: dict[str, CampaignArtifact] = {}
                if step_dir.exists():
                    for f in step_dir.iterdir():
                        if f.is_file() and not f.name.startswith("."):
                            art = CampaignArtifact(
                                name=f.name,
                                path=str(f),
                                step=step.name,
                                size_bytes=f.stat().st_size,
                                sha256=file_sha256(f),
                            )
                            step_artifacts[f.name] = art
                            self.artifacts.append(art)

                step_res = StepResult(
                    step_name=step.name,
                    status=StepStatus.COMPLETED,
                    started_at=step_start,
                    finished_at=step_finish,
                    duration_seconds=step_dur,
                    metrics=step_metrics or {},
                    artifacts=step_artifacts,
                )
                self.step_results[step.name] = step_res
                self.metrics[step.name] = step_metrics or {}

                # Save checkpoint after each completed step
                self._save_checkpoint(last_step=step.name)

            except Exception as exc:
                LOG.error("Step '%s' failed: %s", step.name, exc)
                now_fail = datetime.datetime.now(datetime.timezone.utc)
                st1 = now_fail.timestamp()
                step_res = StepResult(
                    step_name=step.name,
                    status=StepStatus.FAILED,
                    started_at=step_start,
                    finished_at=now_fail.isoformat(),
                    duration_seconds=max(0.001, st1 - st0),
                    error=str(exc),
                )
                self.step_results[step.name] = step_res

                # Capture diagnostics
                diag_path = self._write_failure_diagnostics(failed_step=step.name, exc=exc)

                if rollback_on_failure:
                    LOG.info("Initiating rollback on failure for campaign %s", self.config.campaign_id)
                    self.rollback()
                    status = CampaignStatus.ROLLED_BACK
                else:
                    status = CampaignStatus.FAILED

                self._generate_reports(
                    status=status,
                    started_at=started_at,
                    preflight=preflight,
                    error=str(exc),
                    failure_diagnostics_path=diag_path,
                )
                raise StepExecutionError(f"Campaign step '{step.name}' failed: {exc}") from exc

        # 4. Threshold evaluation
        status = CampaignStatus.COMPLETED
        passed = self._evaluate_thresholds()

        # 5. Generate comprehensive reports
        report_txt, report_json, prov_path = self._generate_reports(
            status=status,
            started_at=started_at,
            preflight=preflight,
            passed=passed,
        )

        t1 = datetime.datetime.now(datetime.timezone.utc).timestamp()
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return CampaignResult(
            campaign_id=self.config.campaign_id,
            status=status,
            passed=passed,
            output_dir=self.output_dir,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=max(0.001, t1 - t0),
            config=self.config.to_dict(),
            preflight=preflight,
            step_results=self.step_results,
            metrics=self.metrics,
            artifacts=self.artifacts,
            threshold_results=self.threshold_results,
            report_txt_path=report_txt,
            report_json_path=report_json,
            provenance_manifest_path=prov_path,
        )

    # --------------------------------------------------------------------------
    # 5. Rollback and Diagnostics
    # --------------------------------------------------------------------------

    def rollback(self) -> RollbackResult:
        """Roll back any modifications and restore baseline state."""
        restored_files: list[str] = []
        cleaned_paths: list[str] = []

        backup_dir = self.output_dir / "backup"
        backup_model_file = backup_dir / "baseline_model.pt"

        # 1. Restore baseline model weights if backup exists
        if self._baseline_model is not None:
            self.model = copy.deepcopy(self._baseline_model)
            if backup_model_file.exists():
                restored_files.append(str(backup_model_file))
        elif backup_model_file.exists():
            try:
                state_dict = torch.load(backup_model_file, map_location="cpu", weights_only=False)
                if self.model is not None:
                    self.model.load_state_dict(state_dict, strict=False)
                restored_files.append(str(backup_model_file))
            except Exception as exc:
                LOG.error("Failed restoring baseline model: %s", exc)
                return RollbackResult(success=False, error=str(exc))

        # 2. Clean temporary staging and incomplete files
        for root, dirs, files in os.walk(self.output_dir):
            for f in files:
                if f.startswith(".tmp_") or f.endswith(".tmp"):
                    p = Path(root) / f
                    try:
                        p.unlink()
                        cleaned_paths.append(str(p))
                    except Exception:
                        pass

        LOG.info("Rollback completed: %d files restored, %d temp files cleaned", len(restored_files), len(cleaned_paths))
        return RollbackResult(
            success=True,
            restored_files=restored_files,
            cleaned_paths=cleaned_paths,
            details={"backup_dir": str(backup_dir)},
        )

    def _write_failure_diagnostics(self, failed_step: str, exc: Exception) -> Path:
        diag_path = self.output_dir / "failure_diagnostics.json"
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            free_bytes = shutil.disk_usage(self.output_dir).free
        except Exception:
            free_bytes = 0

        diag = {
            "campaign_id": self.config.campaign_id,
            "failed_step": failed_step,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "system_resources": {
                "max_rss_kb": rss_kb,
                "disk_free_bytes": free_bytes,
            },
        }
        diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
        return diag_path

    # --------------------------------------------------------------------------
    # Evaluation and Reporting
    # --------------------------------------------------------------------------

    def _evaluate_thresholds(self) -> bool:
        """Evaluate observed metrics against defined campaign thresholds."""
        thresholds = self.config.thresholds
        passed = True

        # Helper to check a condition
        def check(crit: str, thresh: Optional[float], obs: Optional[float], op: str) -> None:
            nonlocal passed
            if thresh is None:
                return
            if obs is None or not math.isfinite(obs):
                raise ThresholdGateError(f"Required measurement unavailable or non-finite: {crit}")
            ok = (obs <= thresh) if op == "<=" else (obs >= thresh)
            self.threshold_results[crit] = ThresholdResult(
                criterion=crit,
                threshold=thresh,
                observed=obs,
                comparison=op,
                passed=ok,
            )
            if not ok:
                passed = False

        # Extract relevant metrics across steps safely (preserving 0)
        def get_metric(step_name: str, key: str) -> Optional[float]:
            val = self.metrics.get(step_name, {}).get(key)
            return float(val) if val is not None else None

        kl = get_metric("quantize", "mean_kl_drift")
        rebound = get_metric("recover", "drop_rebound_score")
        loss_delta = get_metric("recover", "loss_delta")
        params_removed = get_metric("apply", "parameters_removed")
        if params_removed is None:
            params_removed = get_metric("preview", "parameters_removed")
        latency = get_metric("export", "prefill_latency_ms")

        check("max_mean_kl", thresholds.max_mean_kl, kl, "<=")
        check("max_drop_rebound", thresholds.max_drop_rebound, rebound, "<=")
        check("max_loss_delta", thresholds.max_loss_delta, loss_delta, "<=")
        if thresholds.min_parameter_reduction is not None:
            check(
                "min_parameter_reduction",
                float(thresholds.min_parameter_reduction),
                float(params_removed) if params_removed is not None else 0.0,
                ">=",
            )
        check("max_latency_ms", thresholds.max_latency_ms, latency, "<=")

        # Custom thresholds
        for crit, thresh_val in thresholds.custom.items():
            obs_val = None
            for s_metrics in self.metrics.values():
                if crit in s_metrics:
                    obs_val = float(s_metrics[crit])
                    break
            check(crit, thresh_val, obs_val, "<=")

        return passed

    def _generate_reports(
        self,
        status: CampaignStatus,
        started_at: str,
        preflight: PreflightResult,
        passed: bool = False,
        error: Optional[str] = None,
        failure_diagnostics_path: Optional[Path] = None,
    ) -> tuple[Path, Path, Optional[Path]]:
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report_txt = self.output_dir / "campaign_report.txt"
        report_json = self.output_dir / "campaign_report.json"
        prov_path = self.output_dir / "provenance_manifest.json" if status == CampaignStatus.COMPLETED else None

        # 1. Text Summary
        lines = [
            "=" * 80,
            "                    AEGIS-LAB NEUROSURGERY CAMPAIGN REPORT",
            "=" * 80,
            f"Campaign ID:       {self.config.campaign_id}",
            f"Schema Version:    {self.config.schema_version}",
            f"Status:            {status.value} ({'PASSED' if passed else 'FAILED'})",
            f"Model Path:        {self.config.model.path}",
            f"Output Directory:  {self.output_dir}",
            f"Started At:        {started_at}",
            f"Finished At:       {finished_at}",
            "",
            "-" * 80,
            "PREFLIGHT CHECKS",
            "-" * 80,
        ]
        for c in preflight.checks:
            tag = "[PASS]" if c.passed else "[FAIL]"
            lines.append(f"{tag} {c.name}: {c.message}")

        lines.extend([
            "",
            "-" * 80,
            "STEP EXECUTION TIMELINE",
            "-" * 80,
            f"{'Step':<12} {'Status':<12} {'Duration':<10} {'Key Metrics'}",
            "-" * 80,
        ])
        for step in self.config.steps:
            res = self.step_results.get(step.name)
            if res:
                metrics_summary = ", ".join(f"{k}={v}" for k, v in list(res.metrics.items())[:3])
                resumed_tag = " (resumed)" if res.resumed_from_checkpoint else ""
                lines.append(
                    f"{step.name:<12} {res.status.value + resumed_tag:<12} {res.duration_seconds:<9.2f}s {metrics_summary}"
                )
            else:
                lines.append(f"{step.name:<12} {'SKIPPED':<12} {'-':<10} -")

        lines.extend([
            "",
            "-" * 80,
            "OBJECTIVES & THRESHOLDS",
            "-" * 80,
            f"{'Criterion':<24} {'Threshold':<12} {'Observed':<12} {'Status'}",
            "-" * 80,
        ])
        if self.threshold_results:
            for t in self.threshold_results.values():
                status_str = "PASS" if t.passed else "FAIL"
                obs_str = f"{t.observed:.4f}" if t.observed is not None else "-"
                lines.append(f"{t.criterion:<24} {t.comparison} {t.threshold:<10.4f} {obs_str:<12} {status_str}")
        else:
            lines.append("No explicit thresholds configured.")

        lines.extend([
            "",
            "-" * 80,
            "ARTIFACT INVENTORY",
            "-" * 80,
            f"{'Step':<12} {'File Name':<28} {'Size (B)':<10} {'SHA256'}",
            "-" * 80,
        ])
        for art in self.artifacts:
            lines.append(
                f"{art.step:<12} {art.name:<28} {art.size_bytes:<10} {art.sha256[:16]}..."
            )

        lines.extend([
            "",
            "-" * 80,
            f"CAMPAIGN VERDICT: {'PROMOTED' if (passed and status == CampaignStatus.COMPLETED) else 'REJECTED'}",
            "-" * 80,
        ])
        if error:
            lines.append(f"Reason: {error}")
            if failure_diagnostics_path:
                lines.append(f"Diagnostics: {failure_diagnostics_path}")
        elif passed:
            lines.append("All steps completed successfully and all threshold constraints satisfied.")
        else:
            lines.append("One or more threshold constraints were violated.")

        lines.append("=" * 80)
        report_txt.write_text("\n".join(lines), encoding="utf-8")

        # 2. JSON Machine Report
        report_dict = {
            "campaign_id": self.config.campaign_id,
            "schema_version": self.config.schema_version,
            "status": status.value,
            "passed": passed,
            "started_at": started_at,
            "finished_at": finished_at,
            "config": self.config.to_dict(),
            "preflight": preflight.to_dict(),
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
            "threshold_results": {k: v.to_dict() for k, v in self.threshold_results.items()},
            "artifacts": [a.to_dict() for a in self.artifacts],
            "error": error,
        }
        report_json.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

        # 3. Provenance Manifest (Stage 4A compatible)
        if prov_path:
            self._write_provenance_manifest(prov_path, started_at)

        return report_txt, report_json, prov_path

    def _write_provenance_manifest(self, path: Path, created_at: str) -> None:
        """Write consolidated Stage 4A compliant ProvenanceManifest."""
        if self.model is None:
            path.write_text(json.dumps({"status": "unavailable", "reason": "No model was loaded"}))
            return
        model_hash = compute_model_hash(self._baseline_model or self.model)
        tok_hash = compute_tokenizer_hash(self.tokenizer)
        cfg_hash = compute_config_hash(getattr(self._baseline_model or self.model, "config", {}))

        dataset_fingerprints: dict[str, Any] = {}
        for name, spec in self.config.datasets.items():
            p = Path(spec.path)
            h = file_sha256(p)
            count = len(load_prompts(p, spec.max_prompts))
            dataset_fingerprints[name] = DatasetFingerprint(
                sha256=h,
                sample_count=count,
                split_counts={spec.split: count},
                domain_counts={"general": count},
            )

        if not dataset_fingerprints:
            dataset_fingerprints["default"] = DatasetFingerprint(
                sha256=hashlib.sha256(b"[]").hexdigest(),
                sample_count=0,
                split_counts={},
                domain_counts={},
                metadata={"dataset_absent": True},
            )

        from .operator_audit import implementation_binding
        implementation = implementation_binding()
        step_hashes = {f"{art.step}/{art.name}": art.sha256 for art in self.artifacts}
        manifest = ProvenanceManifest(
            schema_version="4a.1",
            model_hash=model_hash,
            tokenizer_hash=tok_hash,
            config_hash=cfg_hash,
            dataset_fingerprints=dataset_fingerprints,
            adapter_version="sha256:" + hashlib.sha256(json.dumps(implementation, sort_keys=True).encode()).hexdigest(),
            seed=self.config.seed,
            dtype=str(next((self._baseline_model or self.model).parameters()).dtype),
            operation_order=[s.name for s in self.config.steps if s.enabled],
            software_version="0.4.0",
            created_at=created_at,
            metadata={"campaign_id": self.config.campaign_id, "step_hashes": step_hashes,
                      "candidate_model_hash": compute_model_hash(self.model),
                      "implementation": implementation},
        )
        manifest_dict = manifest.to_dict()
        manifest_dict["manifest_hash"] = manifest.compute_manifest_hash()
        path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")


# ==============================================================================
# Top-level Functions
# ==============================================================================


def run_campaign(
    config: Union[CampaignConfig, dict[str, Any], str, Path],
    resume: bool = False,
    strict_preflight: bool = True,
    rollback_on_failure: bool = True,
    custom_step_handlers: Optional[dict[str, Callable[..., dict[str, Any]]]] = None,
) -> CampaignResult:
    """Execute a declarative neurosurgery campaign."""
    runner = CampaignPipelineRunner(config=config, custom_step_handlers=custom_step_handlers)
    return runner.run(
        resume=resume,
        strict_preflight=strict_preflight,
        rollback_on_failure=rollback_on_failure,
    )


def rollback_campaign(campaign_dir: Union[str, Path]) -> RollbackResult:
    """Standalone operator entry point to roll back an edited campaign directory."""
    p = Path(campaign_dir)
    backup_file = p / "backup" / "baseline_model.pt"
    restored: list[str] = []
    cleaned: list[str] = []

    if not backup_file.exists():
        return RollbackResult(
            success=False,
            error=f"No baseline backup found at {backup_file}",
        )

    # Clean temporary files
    for root, dirs, files in os.walk(p):
        for f in files:
            if f.startswith(".tmp_") or f.endswith(".tmp"):
                f_path = Path(root) / f
                try:
                    f_path.unlink()
                    cleaned.append(str(f_path))
                except Exception:
                    pass

    restored.append(str(backup_file))
    return RollbackResult(
        success=True,
        restored_files=restored,
        cleaned_paths=cleaned,
        details={"backup_file": str(backup_file)},
    )
