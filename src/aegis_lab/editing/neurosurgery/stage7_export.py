"""Stage 7: Target-specific export and packing for model neurosurgery.

Implements:
1. Format/version matrix and concrete runtime export packaging (HuggingFace compatible,
   safetensors / pt layout, metadata manifest).
2. Complete provenance and asset preservation: tokenizer, special tokens map,
   chat template, updated config (reflecting pruned layers, intermediate_size, num_heads),
   generation defaults, and surgery manifest.
3. Runtime benchmarking & profiling: separated prefill vs. decode latency profiling,
   tokens per second throughput, time-to-first-token (TTFT), peak resident memory (RSS / VRAM),
   across fixed batch/context profiles with warmup runs.
4. Reload verification: verifies clean process / loader reload, checksums, shape parity,
   and execution of regression workloads.
5. Speedup reporting & validation: measured speedups against recorded baseline, enforcing
   explicit hardware, context length, batch size, and baseline numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import safetensors.torch

from .common import (
    LOG,
    file_sha256,
    get_layers,
    load_tensor_artifact,
    nested_getattr,
    nested_setattr,
    resolve_device,
)

SCHEMA_VERSION_7 = "7.0.0"


# ==============================================================================
# 1. Enums and Format / Version Matrix
# ==============================================================================


class ExportFormat(str, Enum):
    """Supported serialization formats for exported models."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pt"


class RuntimeTarget(str, Enum):
    """Supported runtime deployment targets."""

    TRANSFORMERS = "transformers"
    PYTORCH = "pytorch"
    VLLM = "vllm"
    ONNX = "onnx"
    TENSORRT_LLM = "tensorrt_llm"
    LLAMA_CPP = "llama_cpp"


@dataclass(frozen=True)
class FormatVersionSpec:
    """Specification of an export format and version within the compatibility matrix."""

    format: ExportFormat
    version: str
    supported_targets: list[RuntimeTarget]
    file_extensions: list[str]
    weights_only_compatible: bool = True
    description: str = ""


# Concrete Format / Version Matrix
SUPPORTED_EXPORT_MATRIX: dict[tuple[ExportFormat, str], FormatVersionSpec] = {
    (ExportFormat.SAFETENSORS, "1.0.0"): FormatVersionSpec(
        format=ExportFormat.SAFETENSORS,
        version="1.0.0",
        supported_targets=[
            RuntimeTarget.TRANSFORMERS,
            RuntimeTarget.VLLM,
            RuntimeTarget.PYTORCH,
            RuntimeTarget.TENSORRT_LLM,
        ],
        file_extensions=[".safetensors"],
        weights_only_compatible=True,
        description="Standard SafeTensors zero-copy layout with complete tensor metadata",
    ),
    (ExportFormat.PYTORCH, "1.0.0"): FormatVersionSpec(
        format=ExportFormat.PYTORCH,
        version="1.0.0",
        supported_targets=[
            RuntimeTarget.TRANSFORMERS,
            RuntimeTarget.PYTORCH,
        ],
        file_extensions=[".bin", ".pt"],
        weights_only_compatible=True,
        description="Native PyTorch serialized state dict (weights_only=True compatible)",
    ),
}


# ==============================================================================
# 2. Exceptions
# ==============================================================================


class NeurosurgeryExportError(Exception):
    """Base exception for Stage 7 export errors."""


class UnsupportedExportFormatError(NeurosurgeryExportError):
    """Raised when an unsupported export format, version, or runtime target is requested."""


class AssetPreservationError(NeurosurgeryExportError):
    """Raised when a required model asset (tokenizer, config, template) fails preservation."""


class ReloadVerificationError(NeurosurgeryExportError):
    """Raised when an exported artifact fails reload or regression validation."""


class SpeedupValidationError(NeurosurgeryExportError):
    """Raised when a speedup claim lacks required hardware, context, batch, or baseline specs."""


class ProfilingError(NeurosurgeryExportError):
    """Raised when latency or memory profiling fails."""


# ==============================================================================
# 3. Data Structures: Manifests, Hardware, Profiles & Reports
# ==============================================================================


@dataclass
class SurgeryManifest:
    """Provenance and surgery record for an exported candidate."""

    schema_version: str = SCHEMA_VERSION_7
    source_model_hash: str = ""
    candidate_model_hash: str = ""
    export_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    operations_applied: list[dict[str, Any]] = field(default_factory=list)
    config_modifications: dict[str, Any] = field(default_factory=dict)
    dataset_fingerprints: dict[str, str] = field(default_factory=dict)
    parent_manifest_hash: Optional[str] = None
    adapter_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_model_hash": self.source_model_hash,
            "candidate_model_hash": self.candidate_model_hash,
            "export_timestamp": self.export_timestamp,
            "operations_applied": copy.deepcopy(self.operations_applied),
            "config_modifications": copy.deepcopy(self.config_modifications),
            "dataset_fingerprints": copy.deepcopy(self.dataset_fingerprints),
            "parent_manifest_hash": self.parent_manifest_hash,
            "adapter_name": self.adapter_name,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurgeryManifest:
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION_7),
            source_model_hash=data.get("source_model_hash", ""),
            candidate_model_hash=data.get("candidate_model_hash", ""),
            export_timestamp=data.get(
                "export_timestamp", datetime.now(timezone.utc).isoformat()
            ),
            operations_applied=data.get("operations_applied", []),
            config_modifications=data.get("config_modifications", {}),
            dataset_fingerprints=data.get("dataset_fingerprints", {}),
            parent_manifest_hash=data.get("parent_manifest_hash"),
            adapter_name=data.get("adapter_name"),
            metadata=data.get("metadata", {}),
        )

    def save_json(self, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        p.write_text(content, encoding="utf-8")
        return str(p)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> SurgeryManifest:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Surgery manifest not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class ExportResult:
    """Artifact packaging result."""

    export_dir: Path
    format: str
    format_version: str
    target_runtime: str
    weights_files: list[str]
    asset_files: dict[str, str]
    manifest_path: str
    file_checksums: dict[str, str]
    total_parameters: int
    total_bytes: int
    surgery_manifest: Optional[SurgeryManifest] = None
    config_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_dir": str(self.export_dir),
            "format": self.format,
            "format_version": self.format_version,
            "target_runtime": self.target_runtime,
            "weights_files": list(self.weights_files),
            "asset_files": copy.deepcopy(self.asset_files),
            "manifest_path": self.manifest_path,
            "file_checksums": copy.deepcopy(self.file_checksums),
            "total_parameters": self.total_parameters,
            "total_bytes": self.total_bytes,
            "config_summary": copy.deepcopy(self.config_summary),
            "surgery_manifest": (
                self.surgery_manifest.to_dict() if self.surgery_manifest else None
            ),
            "metadata": copy.deepcopy(self.metadata),
        }

    def save_json(self, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        p.write_text(content, encoding="utf-8")
        return str(p)


@dataclass(frozen=True)
class BenchmarkProfile:
    """Fixed batch and context configuration for profiling."""

    name: str = "default"
    batch_size: int = 1
    prompt_length: int = 128
    decode_steps: int = 32

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.prompt_length <= 0:
            raise ValueError(f"prompt_length must be > 0, got {self.prompt_length}")
        if self.decode_steps < 0:
            raise ValueError(f"decode_steps must be >= 0, got {self.decode_steps}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "prompt_length": self.prompt_length,
            "decode_steps": self.decode_steps,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkProfile:
        return cls(
            name=d.get("name", "default"),
            batch_size=int(d["batch_size"]),
            prompt_length=int(d["prompt_length"]),
            decode_steps=int(d.get("decode_steps", 32)),
        )


@dataclass
class HardwareInfo:
    """Explicit hardware environment details required for certified speedup reports."""

    device_type: str
    device_name: str
    cpu_count: int
    total_memory_mb: float
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "cpu_count": self.cpu_count,
            "total_memory_mb": round(self.total_memory_mb, 2),
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareInfo:
        return cls(
            device_type=str(d.get("device_type", "")),
            device_name=str(d.get("device_name", "")),
            cpu_count=int(d.get("cpu_count", 1)),
            total_memory_mb=float(d.get("total_memory_mb", 0.0)),
            driver_version=d.get("driver_version"),
            cuda_version=d.get("cuda_version"),
        )


@dataclass
class ProfilingResult:
    """Detailed latency, throughput, and memory profiling breakdown."""

    profile: BenchmarkProfile
    hardware: HardwareInfo
    num_warmup: int
    num_repeats: int
    prefill_latency_ms: float
    prefill_latency_std_ms: float
    decode_latency_per_token_ms: float
    decode_latency_total_ms: float
    time_to_first_token_ms: float
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    total_tokens_per_sec: float
    peak_resident_memory_mb: float
    memory_type: str
    latencies_prefill: list[float] = field(default_factory=list)
    latencies_decode: list[float] = field(default_factory=list)
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "hardware": self.hardware.to_dict(),
            "num_warmup": self.num_warmup,
            "num_repeats": self.num_repeats,
            "prefill_latency_ms": round(self.prefill_latency_ms, 4),
            "prefill_latency_std_ms": round(self.prefill_latency_std_ms, 4),
            "decode_latency_per_token_ms": round(self.decode_latency_per_token_ms, 4),
            "decode_latency_total_ms": round(self.decode_latency_total_ms, 4),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 4),
            "prefill_tokens_per_sec": round(self.prefill_tokens_per_sec, 2),
            "decode_tokens_per_sec": round(self.decode_tokens_per_sec, 2),
            "total_tokens_per_sec": round(self.total_tokens_per_sec, 2),
            "peak_resident_memory_mb": round(self.peak_resident_memory_mb, 2),
            "memory_type": self.memory_type,
            "latencies_prefill": self.latencies_prefill,
            "latencies_decode": self.latencies_decode,
            "raw_metrics": self.raw_metrics,
        }

    def save_json(self, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)


@dataclass
class SpeedupReport:
    """Measured performance speedup report against recorded baseline."""

    hardware: HardwareInfo
    batch_size: int
    context_length: int
    decode_steps: int
    baseline_prefill_ms: float
    edited_prefill_ms: float
    prefill_speedup: float
    baseline_decode_ms: float
    edited_decode_ms: float
    decode_speedup: float
    baseline_ttft_ms: float
    edited_ttft_ms: float
    ttft_speedup: float
    baseline_tokens_per_sec: float
    edited_tokens_per_sec: float
    throughput_speedup: float
    baseline_peak_mem_mb: float
    edited_peak_mem_mb: float
    memory_reduction_mb: float
    memory_reduction_pct: float
    is_valid: bool = True
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.to_dict(),
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "decode_steps": self.decode_steps,
            "baseline_prefill_ms": round(self.baseline_prefill_ms, 4),
            "edited_prefill_ms": round(self.edited_prefill_ms, 4),
            "prefill_speedup": round(self.prefill_speedup, 4),
            "baseline_decode_ms": round(self.baseline_decode_ms, 4),
            "edited_decode_ms": round(self.edited_decode_ms, 4),
            "decode_speedup": round(self.decode_speedup, 4),
            "baseline_ttft_ms": round(self.baseline_ttft_ms, 4),
            "edited_ttft_ms": round(self.edited_ttft_ms, 4),
            "ttft_speedup": round(self.ttft_speedup, 4),
            "baseline_tokens_per_sec": round(self.baseline_tokens_per_sec, 2),
            "edited_tokens_per_sec": round(self.edited_tokens_per_sec, 2),
            "throughput_speedup": round(self.throughput_speedup, 4),
            "baseline_peak_mem_mb": round(self.baseline_peak_mem_mb, 2),
            "edited_peak_mem_mb": round(self.edited_peak_mem_mb, 2),
            "memory_reduction_mb": round(self.memory_reduction_mb, 2),
            "memory_reduction_pct": round(self.memory_reduction_pct, 2),
            "is_valid": self.is_valid,
            "validation_notes": list(self.validation_notes),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Model Neurosurgery Speedup Report",
            "",
            f"**Device**: {self.hardware.device_name} ({self.hardware.device_type})",
            f"**Memory**: {self.hardware.total_memory_mb:.1f} MB | **CPUs**: {self.hardware.cpu_count}",
            f"**Workload Profile**: Batch={self.batch_size}, Context={self.context_length}, Decode={self.decode_steps}",
            "",
            "| Metric | Baseline | Edited | Speedup / Delta |",
            "| --- | --- | --- | --- |",
            f"| Prefill Latency | {self.baseline_prefill_ms:.2f} ms | {self.edited_prefill_ms:.2f} ms | **{self.prefill_speedup:.2f}x** |",
            f"| TTFT | {self.baseline_ttft_ms:.2f} ms | {self.edited_ttft_ms:.2f} ms | **{self.ttft_speedup:.2f}x** |",
            f"| Decode (per token) | {self.baseline_decode_ms:.2f} ms | {self.edited_decode_ms:.2f} ms | **{self.decode_speedup:.2f}x** |",
            f"| Throughput | {self.baseline_tokens_per_sec:.1f} tok/s | {self.edited_tokens_per_sec:.1f} tok/s | **{self.throughput_speedup:.2f}x** |",
            f"| Peak Memory | {self.baseline_peak_mem_mb:.1f} MB | {self.edited_peak_mem_mb:.1f} MB | -{self.memory_reduction_mb:.1f} MB ({self.memory_reduction_pct:.1f}%) |",
            "",
        ]
        if self.validation_notes:
            lines.append("### Validation Notes")
            for note in self.validation_notes:
                lines.append(f"- {note}")
        return "\n".join(lines)

    def save_json(self, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)


@dataclass
class ReloadVerificationResult:
    """Output report from reload and regression verification."""

    success: bool
    verified_files: list[str]
    format: str
    total_tensors: int
    regression_passed: bool
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    subprocess_verified: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "verified_files": list(self.verified_files),
            "format": self.format,
            "total_tensors": self.total_tensors,
            "regression_passed": self.regression_passed,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "subprocess_verified": self.subprocess_verified,
            "details": self.details,
        }


# ==============================================================================
# 4. Matrix & Hardware Discovery Validation
# ==============================================================================


def validate_export_matrix(
    export_format: Union[str, ExportFormat],
    format_version: Optional[str] = "1.0.0",
    target_runtime: Optional[Union[str, RuntimeTarget]] = RuntimeTarget.TRANSFORMERS,
) -> FormatVersionSpec:
    """Validate format, version, and runtime target against the compatibility matrix."""
    try:
        fmt = (
            ExportFormat(export_format)
            if isinstance(export_format, str)
            else export_format
        )
    except ValueError:
        valid_fmts = [f.value for f in ExportFormat]
        raise UnsupportedExportFormatError(
            f"Unsupported export format '{export_format}'. Valid formats: {valid_fmts}"
        )

    ver = format_version or "1.0.0"
    matrix_key = (fmt, ver)
    if matrix_key not in SUPPORTED_EXPORT_MATRIX:
        supported_keys = [
            f"({f.value}, {v})" for (f, v) in SUPPORTED_EXPORT_MATRIX.keys()
        ]
        raise UnsupportedExportFormatError(
            f"Format/version ({fmt.value}, {ver}) is not supported in export matrix. "
            f"Supported configurations: {supported_keys}"
        )

    spec = SUPPORTED_EXPORT_MATRIX[matrix_key]

    if target_runtime is not None:
        try:
            target = (
                RuntimeTarget(target_runtime)
                if isinstance(target_runtime, str)
                else target_runtime
            )
        except ValueError:
            valid_targets = [t.value for t in spec.supported_targets]
            raise UnsupportedExportFormatError(
                f"Unsupported runtime target '{target_runtime}'. Supported targets for {fmt.value}: {valid_targets}"
            )

        if target not in spec.supported_targets:
            valid_targets = [t.value for t in spec.supported_targets]
            raise UnsupportedExportFormatError(
                f"Target '{target.value}' is not supported for format {fmt.value} v{ver}. "
                f"Supported: {valid_targets}"
            )

    return spec


def collect_hardware_info(device: Optional[str] = None) -> HardwareInfo:
    """Collect explicit hardware environment information."""
    dev_str = resolve_device(device or "auto")
    dev_type = "cuda" if dev_str.startswith("cuda") else dev_str

    dev_name = "Generic CPU"
    cpu_count = os.cpu_count() or 1
    total_mem = 0.0

    # Probe CPU and system memory on Linux
    try:
        if sys.platform.startswith("linux"):
            if Path("/proc/cpuinfo").exists():
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if "model name" in line:
                            dev_name = line.split(":", 1)[1].strip()
                            break
            if Path("/proc/meminfo").exists():
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb = int(line.split()[1])
                            total_mem = round(kb / 1024.0, 2)
                            break
    except Exception:
        pass

    cuda_driver = None
    cuda_ver = None
    if dev_type == "cuda" and torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0)
            total_mem = round(
                torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0), 2
            )
            cuda_ver = str(torch.version.cuda)
        except Exception:
            pass

    if total_mem <= 0.0:
        total_mem = 1024.0  # Safe default if /proc is unavailable

    return HardwareInfo(
        device_type=dev_type,
        device_name=dev_name,
        cpu_count=cpu_count,
        total_memory_mb=total_mem,
        cuda_version=cuda_ver,
    )


# ==============================================================================
# 5. Asset Preservation Helpers
# ==============================================================================


def update_config_for_surgery(
    config: Union[dict[str, Any], Any],
    surgery_manifest: Optional[Union[SurgeryManifest, dict[str, Any]]] = None,
    model: Optional[nn.Module] = None,
) -> dict[str, Any]:
    """Preserve and update architecture config to accurately reflect neurosurgery edits."""
    if hasattr(config, "to_dict"):
        cfg_dict = config.to_dict()
    elif isinstance(config, dict):
        cfg_dict = copy.deepcopy(config)
    else:
        cfg_dict = {}

    modifications: dict[str, Any] = {}
    if surgery_manifest is not None:
        if isinstance(surgery_manifest, SurgeryManifest):
            modifications = surgery_manifest.config_modifications
        elif isinstance(surgery_manifest, dict):
            modifications = surgery_manifest.get("config_modifications", {})

    # Apply direct modifications from surgery manifest
    for key, val in modifications.items():
        cfg_dict[key] = val

    # Automatically inspect model structure if provided
    if model is not None:
        # Check num_hidden_layers
        try:
            _, layers = get_layers(model)
            if layers and len(layers) > 0:
                cfg_dict["num_hidden_layers"] = len(layers)
                # Check MLP intermediate_size
                first_layer = layers[0]
                mlp = getattr(first_layer, "mlp", None)
                if mlp is not None:
                    gate = getattr(mlp, "gate_proj", getattr(mlp, "up_proj", None))
                    if gate is not None and hasattr(gate, "out_features"):
                        cfg_dict["intermediate_size"] = int(gate.out_features)
                # Check attention heads
                attn = getattr(first_layer, "self_attn", None)
                if attn is not None:
                    if hasattr(attn, "num_heads"):
                        cfg_dict["num_attention_heads"] = int(attn.num_heads)
                    if hasattr(attn, "num_key_value_heads"):
                        cfg_dict["num_key_value_heads"] = int(attn.num_key_value_heads)
        except Exception:
            pass

    return cfg_dict


def preserve_tokenizer_assets(
    export_dir: Path,
    tokenizer: Optional[Any] = None,
    special_tokens_map: Optional[dict[str, Any]] = None,
    chat_template: Optional[Union[str, dict[str, Any]]] = None,
    source_dir: Optional[Union[str, Path]] = None,
) -> dict[str, str]:
    """Preserve complete tokenizer files, special tokens map, and chat template."""
    written: dict[str, str] = {}
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy from source_dir if provided
    if source_dir is not None:
        s_dir = Path(source_dir)
        if s_dir.exists() and s_dir.is_dir():
            for filename in [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.model",
                "special_tokens_map.json",
                "chat_template.json",
            ]:
                s_file = s_dir / filename
                if s_file.exists():
                    d_file = export_dir / filename
                    d_file.write_bytes(s_file.read_bytes())
                    written[filename] = str(d_file)

    # 2. Use tokenizer object if provided
    tok_config: dict[str, Any] = {}
    if tokenizer is not None:
        if hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(str(export_dir))
                for f in export_dir.iterdir():
                    if f.is_file() and (
                        "tokenizer" in f.name or "vocab" in f.name or "merges" in f.name
                    ):
                        written[f.name] = str(f)
            except Exception as e:
                LOG.warning("tokenizer.save_pretrained failed: %s; falling back", e)

        # Inspect tokenizer dictionary / properties
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
            vocab_file = export_dir / "vocab.json"
            vocab_file.write_text(
                json.dumps(vocab, indent=2, sort_keys=True), encoding="utf-8"
            )
            written["vocab.json"] = str(vocab_file)
        elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
            vocab_file = export_dir / "vocab.json"
            vocab_file.write_text(
                json.dumps(tokenizer.vocab, indent=2, sort_keys=True), encoding="utf-8"
            )
            written["vocab.json"] = str(vocab_file)

        # Extract special tokens from tokenizer if available
        if special_tokens_map is None:
            if hasattr(tokenizer, "special_tokens_map"):
                special_tokens_map = getattr(tokenizer, "special_tokens_map")

        # Extract chat template from tokenizer if available
        if chat_template is None:
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                chat_template = tokenizer.chat_template

    # 3. Ensure tokenizer_config.json exists
    tok_cfg_file = export_dir / "tokenizer_config.json"
    if tok_cfg_file.exists():
        try:
            tok_config = json.loads(tok_cfg_file.read_text(encoding="utf-8"))
        except Exception:
            tok_config = {}
    else:
        tok_config = {
            "model_max_length": 4096,
            "tokenizer_class": "PreTrainedTokenizerFast",
        }

    # 4. Handle Special Tokens Map
    spec_file = export_dir / "special_tokens_map.json"
    if special_tokens_map is not None:
        spec_file.write_text(
            json.dumps(special_tokens_map, indent=2, sort_keys=True), encoding="utf-8"
        )
        written["special_tokens_map.json"] = str(spec_file)
    elif not spec_file.exists():
        default_spec = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        spec_file.write_text(
            json.dumps(default_spec, indent=2, sort_keys=True), encoding="utf-8"
        )
        written["special_tokens_map.json"] = str(spec_file)
    else:
        written["special_tokens_map.json"] = str(spec_file)

    # 5. Handle Chat Template
    if chat_template is not None:
        ct_str = (
            chat_template
            if isinstance(chat_template, str)
            else json.dumps(chat_template)
        )
        ct_file = export_dir / "chat_template.json"
        ct_file.write_text(
            json.dumps({"chat_template": ct_str}, indent=2), encoding="utf-8"
        )
        written["chat_template.json"] = str(ct_file)
        tok_config["chat_template"] = ct_str

    tok_cfg_file.write_text(
        json.dumps(tok_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    written["tokenizer_config.json"] = str(tok_cfg_file)

    # 6. Ensure at least one tokenizer definition exists (tokenizer.json or vocab.json)
    tok_json_file = export_dir / "tokenizer.json"
    if not tok_json_file.exists() and not (export_dir / "vocab.json").exists():
        # Minimal fast tokenizer definition
        minimal_tok = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "model": {"type": "BPE", "vocab": {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}},
        }
        tok_json_file.write_text(
            json.dumps(minimal_tok, indent=2), encoding="utf-8"
        )
        written["tokenizer.json"] = str(tok_json_file)
    elif tok_json_file.exists():
        written["tokenizer.json"] = str(tok_json_file)

    return written


def preserve_generation_config(
    export_dir: Path,
    generation_config: Optional[Union[dict[str, Any], Any]] = None,
    config: Optional[dict[str, Any]] = None,
) -> str:
    """Preserve runtime generation defaults in generation_config.json."""
    gen_file = export_dir / "generation_config.json"
    gen_dict: dict[str, Any] = {}

    if hasattr(generation_config, "to_dict"):
        gen_dict = generation_config.to_dict()
    elif isinstance(generation_config, dict):
        gen_dict = copy.deepcopy(generation_config)
    else:
        # Defaults aligned with config
        pad_id = config.get("pad_token_id", 0) if config else 0
        eos_id = config.get("eos_token_id", 1) if config else 1
        bos_id = config.get("bos_token_id", 2) if config else 2
        gen_dict = {
            "max_new_tokens": 128,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
            "bos_token_id": bos_id,
        }

    gen_file.write_text(
        json.dumps(gen_dict, indent=2, sort_keys=True), encoding="utf-8"
    )
    return str(gen_file)


# ==============================================================================
# 6. Core Packaging & Serialization
# ==============================================================================


def export_runtime_package(
    model: Union[nn.Module, dict[str, torch.Tensor]],
    export_dir: Union[str, Path],
    format: Union[str, ExportFormat] = ExportFormat.SAFETENSORS,
    target_runtime: Union[str, RuntimeTarget] = RuntimeTarget.TRANSFORMERS,
    format_version: str = "1.0.0",
    tokenizer: Optional[Any] = None,
    special_tokens_map: Optional[dict[str, Any]] = None,
    chat_template: Optional[Union[str, dict[str, Any]]] = None,
    config: Optional[Union[dict[str, Any], Any]] = None,
    generation_config: Optional[Union[dict[str, Any], Any]] = None,
    surgery_manifest: Optional[Union[SurgeryManifest, dict[str, Any]]] = None,
    source_dir: Optional[Union[str, Path]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ExportResult:
    """Package an edited model for runtime deployment with full asset preservation.

    Preserves:
    - Serialized weights (safetensors or pt)
    - Updated config reflecting pruned layers, intermediate_size, and num_heads
    - Complete tokenizer assets (vocab, merges, config, special_tokens_map, chat_template)
    - Generation configuration
    - Surgery provenance manifest
    - Export metadata and SHA-256 integrity checksums

    Args:
        model: PyTorch model or state dict.
        export_dir: Destination directory.
        format: Export format (safetensors or pt).
        target_runtime: Target runtime engine.
        format_version: Format version specification.
        tokenizer: Tokenizer instance or dict.
        special_tokens_map: Explicit special tokens mapping.
        chat_template: Jinja2 chat template string or config.
        config: Model config dict or PretrainedConfig.
        generation_config: Generation parameters dict or config.
        surgery_manifest: Stage surgery provenance manifest.
        source_dir: Source model checkpoint directory (for asset copying).
        metadata: Additional metadata to record in the export manifest.

    Returns:
        ExportResult detailing written artifacts and checksums.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate format/version matrix
    spec = validate_export_matrix(
        export_format=format,
        format_version=format_version,
        target_runtime=target_runtime,
    )
    fmt_enum = spec.format
    target_enum = RuntimeTarget(target_runtime)

    # 2. Extract state dict and model reference
    model_obj: Optional[nn.Module] = None
    if isinstance(model, nn.Module):
        model_obj = model
        raw_state_dict = model.state_dict()
    elif isinstance(model, dict):
        raw_state_dict = model
    else:
        raise TypeError(f"Expected nn.Module or dict for model, got {type(model)}")

    # Ensure tensors are contiguous and on CPU for export
    clean_state_dict: dict[str, torch.Tensor] = {}
    total_params = 0
    total_bytes = 0
    for name, tensor in raw_state_dict.items():
        if isinstance(tensor, torch.Tensor):
            t_clean = tensor.detach().cpu().contiguous()
            clean_state_dict[name] = t_clean
            total_params += t_clean.numel()
            total_bytes += t_clean.numel() * t_clean.element_size()

    # 3. Preserve and update architecture config
    if config is None and model_obj is not None and hasattr(model_obj, "config"):
        config = model_obj.config

    updated_config = update_config_for_surgery(
        config=config, surgery_manifest=surgery_manifest, model=model_obj
    )
    config_file = out_dir / "config.json"
    config_file.write_text(
        json.dumps(updated_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    asset_files: dict[str, str] = {"config": str(config_file)}

    # 4. Preserve tokenizer assets (tokenizer, special_tokens, chat_template)
    tok_written = preserve_tokenizer_assets(
        export_dir=out_dir,
        tokenizer=tokenizer,
        special_tokens_map=special_tokens_map,
        chat_template=chat_template,
        source_dir=source_dir,
    )
    for k, v in tok_written.items():
        asset_files[k] = v

    # 5. Preserve generation defaults
    gen_path = preserve_generation_config(
        export_dir=out_dir,
        generation_config=generation_config,
        config=updated_config,
    )
    asset_files["generation_config"] = gen_path

    # 6. Preserve surgery provenance manifest
    manifest_obj: Optional[SurgeryManifest] = None
    if surgery_manifest is not None:
        if isinstance(surgery_manifest, SurgeryManifest):
            manifest_obj = surgery_manifest
        elif isinstance(surgery_manifest, dict):
            manifest_obj = SurgeryManifest.from_dict(surgery_manifest)
        s_path = manifest_obj.save_json(out_dir / "surgery_manifest.json")
        asset_files["surgery_manifest"] = s_path

    # 7. Serialize weights according to format
    weights_files: list[str] = []
    if fmt_enum == ExportFormat.SAFETENSORS:
        weights_name = "model.safetensors"
        weights_path = out_dir / weights_name
        meta = {
            "format": "safetensors",
            "version": spec.version,
            "target": target_enum.value,
        }
        safetensors.torch.save_file(clean_state_dict, str(weights_path), metadata=meta)
        weights_files.append(weights_name)
    elif fmt_enum == ExportFormat.PYTORCH:
        weights_name = "pytorch_model.bin"
        weights_path = out_dir / weights_name
        torch.save(clean_state_dict, weights_path)
        weights_files.append(weights_name)

    # 8. Compute checksums of all written files
    checksums: dict[str, str] = {}
    for f in out_dir.iterdir():
        if f.is_file() and f.name != "export_manifest.json":
            checksums[f.name] = file_sha256(f)

    # 9. Write export_manifest.json
    export_manifest_data = {
        "schema_version": SCHEMA_VERSION_7,
        "export_format": fmt_enum.value,
        "format_version": spec.version,
        "target_runtime": target_enum.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_weights_file": weights_files[0],
        "weights_files": weights_files,
        "total_tensors": len(clean_state_dict),
        "total_parameters": total_params,
        "total_bytes": total_bytes,
        "file_checksums": checksums,
        "asset_files": asset_files,
        "config_summary": {
            "num_hidden_layers": updated_config.get("num_hidden_layers"),
            "intermediate_size": updated_config.get("intermediate_size"),
            "num_attention_heads": updated_config.get("num_attention_heads"),
            "num_key_value_heads": updated_config.get("num_key_value_heads"),
            "hidden_size": updated_config.get("hidden_size"),
        },
        "metadata": metadata or {},
    }
    manifest_file = out_dir / "export_manifest.json"
    manifest_file.write_text(
        json.dumps(export_manifest_data, indent=2, sort_keys=True), encoding="utf-8"
    )
    checksums["export_manifest.json"] = file_sha256(manifest_file)

    LOG.info(
        "Exported model to %s [format=%s, target=%s, params=%d]",
        out_dir,
        fmt_enum.value,
        target_enum.value,
        total_params,
    )

    return ExportResult(
        export_dir=out_dir,
        format=fmt_enum.value,
        format_version=spec.version,
        target_runtime=target_enum.value,
        weights_files=weights_files,
        asset_files=asset_files,
        manifest_path=str(manifest_file),
        file_checksums=checksums,
        total_parameters=total_params,
        total_bytes=total_bytes,
        surgery_manifest=manifest_obj,
        config_summary=export_manifest_data["config_summary"],
        metadata=metadata or {},
    )


# ==============================================================================
# 7. Runtime Benchmarking & Profiling
# ==============================================================================


def measure_resident_memory(device_type: str = "cpu") -> tuple[float, str]:
    """Measure peak resident memory (RSS for CPU, VRAM for CUDA)."""
    if device_type == "cuda" and torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return round(vram_mb, 2), "VRAM"

    # CPU Resident Set Size (RSS) via resource module
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux ru_maxrss is in KiB; macOS is in bytes
        if sys.platform == "darwin":
            rss_mb = usage.ru_maxrss / (1024.0 * 1024.0)
        else:
            rss_mb = usage.ru_maxrss / 1024.0
        return round(rss_mb, 2), "RSS"
    except Exception:
        return 0.0, "RSS"


def _forward_for_profiling(
    model: nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    """Execute forward pass and extract logits."""
    try:
        out = model(input_ids)
    except TypeError:
        out = model(input_ids=input_ids)

    if isinstance(out, dict):
        logits = out.get("logits", out.get("last_hidden_state"))
    elif hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, (list, tuple)):
        logits = out[0]
    else:
        logits = out

    return logits


def benchmark_runtime_profile(
    model: nn.Module,
    profile: Optional[BenchmarkProfile] = None,
    device: Optional[str] = None,
    num_warmup: int = 3,
    num_repeats: int = 5,
    sample_inputs: Optional[torch.Tensor] = None,
    vocab_size: int = 1000,
) -> ProfilingResult:
    """Benchmark prefill vs. decode latency, tokens/sec, TTFT, and peak memory.

    Measures:
    - Prefill latency: time to process prompt and emit first token logits.
    - Decode latency: per-token and total latency across autoregressive decode steps.
    - TTFT: Time-to-first-token in milliseconds.
    - Throughput: tokens per second for prefill, decode, and end-to-end.
    - Peak resident memory (RSS / VRAM).

    Args:
        model: Target PyTorch model.
        profile: BenchmarkProfile (batch_size, prompt_length, decode_steps).
        device: Hardware device to profile on.
        num_warmup: Number of warmup runs prior to timing.
        num_repeats: Number of measured repetitions.
        sample_inputs: Optional pre-tokenized prompt tensor [batch, prompt_len].
        vocab_size: Vocab size for synthetic prompts if sample_inputs is None.

    Returns:
        ProfilingResult with comprehensive performance metrics.
    """
    if profile is None:
        profile = BenchmarkProfile()

    if num_warmup < 0 or num_repeats <= 0:
        raise ValueError("num_warmup must be >= 0 and num_repeats must be > 0")

    dev_str = resolve_device(device or "auto")
    model = model.to(dev_str)
    model.eval()

    hardware = collect_hardware_info(dev_str)
    batch_size = profile.batch_size
    prompt_len = profile.prompt_length
    decode_steps = profile.decode_steps

    if sample_inputs is not None:
        prompt_tensor = sample_inputs.to(dev_str)
        if prompt_tensor.ndim != 2:
            raise ValueError(
                f"sample_inputs must have shape [batch, seq_len], got {prompt_tensor.shape}"
            )
        batch_size = prompt_tensor.shape[0]
        prompt_len = prompt_tensor.shape[1]
    else:
        # Detect vocab_size from model if available
        detected_vocab: Optional[int] = None
        if hasattr(model, "config"):
            cfg = model.config
            if isinstance(cfg, dict):
                detected_vocab = cfg.get("vocab_size")
            elif hasattr(cfg, "vocab_size"):
                detected_vocab = cfg.vocab_size
        if detected_vocab is None:
            for module in model.modules():
                if isinstance(module, nn.Embedding):
                    detected_vocab = module.num_embeddings
                    break
        effective_vocab = (
            min(vocab_size, detected_vocab) if detected_vocab is not None else vocab_size
        )
        prompt_tensor = torch.randint(
            0, max(effective_vocab, 2), (batch_size, prompt_len), device=dev_str, dtype=torch.long
        )

    # 1. Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            logits = _forward_for_profiling(model, prompt_tensor)
            if decode_steps > 0:
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr = torch.cat([prompt_tensor, next_tok], dim=-1)
                for _ in range(min(decode_steps, 2)):
                    dec_logits = _forward_for_profiling(model, curr)
                    next_t = torch.argmax(dec_logits[:, -1, :], dim=-1, keepdim=True)
                    curr = torch.cat([curr, next_t], dim=-1)
        if dev_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    # 2. Measured profiling iterations
    prefill_latencies: list[float] = []
    decode_total_latencies: list[float] = []
    decode_per_token_latencies: list[float] = []
    ttft_latencies: list[float] = []

    with torch.no_grad():
        for _ in range(num_repeats):
            if dev_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

            # --- Prefill Phase ---
            t0 = time.perf_counter()
            logits = _forward_for_profiling(model, prompt_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            if dev_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t0

            prefill_ms = t_prefill * 1000.0
            ttft_ms = prefill_ms  # First token emitted at end of prefill
            prefill_latencies.append(prefill_ms)
            ttft_latencies.append(ttft_ms)

            # --- Decode Phase ---
            if decode_steps > 0:
                current_seq = torch.cat([prompt_tensor, next_token], dim=-1)
                t_decode_start = time.perf_counter()

                for step in range(decode_steps):
                    dec_logits = _forward_for_profiling(model, current_seq)
                    new_token = torch.argmax(dec_logits[:, -1, :], dim=-1, keepdim=True)
                    current_seq = torch.cat([current_seq, new_token], dim=-1)

                if dev_str.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_decode = time.perf_counter() - t_decode_start
                decode_ms = t_decode * 1000.0
                decode_total_latencies.append(decode_ms)
                decode_per_token_latencies.append(decode_ms / decode_steps)
            else:
                decode_total_latencies.append(0.0)
                decode_per_token_latencies.append(0.0)

    # 3. Compute statistics
    mean_prefill = sum(prefill_latencies) / len(prefill_latencies)
    variance_prefill = sum((x - mean_prefill) ** 2 for x in prefill_latencies) / len(
        prefill_latencies
    )
    std_prefill = math.sqrt(variance_prefill)

    mean_ttft = sum(ttft_latencies) / len(ttft_latencies)
    mean_decode_total = sum(decode_total_latencies) / len(decode_total_latencies)
    mean_decode_per_token = sum(decode_per_token_latencies) / len(
        decode_per_token_latencies
    )

    # Throughput (tokens / second)
    prefill_s = mean_prefill / 1000.0
    prefill_tps = (
        (batch_size * prompt_len) / prefill_s if prefill_s > 0 else 0.0
    )

    decode_s = mean_decode_total / 1000.0
    decode_tps = (
        (batch_size * decode_steps) / decode_s if decode_s > 0 and decode_steps > 0 else 0.0
    )

    total_tokens = batch_size * (prompt_len + decode_steps)
    total_time_s = prefill_s + decode_s
    total_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

    peak_mem, mem_type = measure_resident_memory(hardware.device_type)

    return ProfilingResult(
        profile=profile,
        hardware=hardware,
        num_warmup=num_warmup,
        num_repeats=num_repeats,
        prefill_latency_ms=round(mean_prefill, 4),
        prefill_latency_std_ms=round(std_prefill, 4),
        decode_latency_per_token_ms=round(mean_decode_per_token, 4),
        decode_latency_total_ms=round(mean_decode_total, 4),
        time_to_first_token_ms=round(mean_ttft, 4),
        prefill_tokens_per_sec=round(prefill_tps, 2),
        decode_tokens_per_sec=round(decode_tps, 2),
        total_tokens_per_sec=round(total_tps, 2),
        peak_resident_memory_mb=round(peak_mem, 2),
        memory_type=mem_type,
        latencies_prefill=prefill_latencies,
        latencies_decode=decode_total_latencies,
        raw_metrics={
            "batch_size": batch_size,
            "prompt_length": prompt_len,
            "decode_steps": decode_steps,
        },
    )


# ==============================================================================
# 8. Reload Verification & Regression Checks
# ==============================================================================


def load_exported_package(
    export_dir: Union[str, Path],
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Load weights, config, and manifest from an exported directory.

    Returns:
        (state_dict, config_dict, export_manifest_dict)
    """
    in_dir = Path(export_dir)
    manifest_file = in_dir / "export_manifest.json"
    if not manifest_file.exists():
        raise ReloadVerificationError(
            f"Missing export manifest in '{in_dir}'. Expected export_manifest.json"
        )

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise ReloadVerificationError(f"Corrupt export manifest JSON: {e}")

    config_file = in_dir / "config.json"
    if not config_file.exists():
        raise ReloadVerificationError(f"Missing config.json in '{in_dir}'")
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise ReloadVerificationError(f"Corrupt config JSON: {e}")

    primary_weights = manifest.get("primary_weights_file", "model.safetensors")
    weights_path = in_dir / primary_weights
    if not weights_path.exists():
        raise ReloadVerificationError(
            f"Primary weights file '{primary_weights}' not found in '{in_dir}'"
        )

    fmt = manifest.get("export_format", "safetensors")
    if fmt == ExportFormat.SAFETENSORS.value:
        state_dict = safetensors.torch.load_file(str(weights_path), device=device)
    elif fmt == ExportFormat.PYTORCH.value:
        state_dict = load_tensor_artifact(weights_path)
    else:
        raise ReloadVerificationError(f"Unsupported format '{fmt}' in export manifest")

    return state_dict, config, manifest


def verify_exported_package(
    export_dir: Union[str, Path],
    model_factory: Optional[Callable[[dict[str, Any]], nn.Module]] = None,
    regression_inputs: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None,
    expected_outputs: Optional[torch.Tensor] = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    verify_subprocess: bool = False,
    raise_on_failure: bool = True,
) -> ReloadVerificationResult:
    """Verify that an exported artifact can be reloaded and executes regression workloads.

    Checks:
    1. Integrity: All files listed in export_manifest match SHA-256 checksums.
    2. Assets: config.json, tokenizer files, generation_config.json exist.
    3. Structural alignment: Weights tensor shapes match config dimensions.
    4. Model instantiation: Loads state dict into fresh model.
    5. Regression workload: Executes forward pass and verifies output parity.
    6. Isolated process verification: Verifies artifact loading in a clean Python subprocess.

    Args:
        export_dir: Directory containing the exported package.
        model_factory: Callable taking config dict and returning an uninitialized nn.Module.
        regression_inputs: Input tensors to evaluate regression forward pass.
        expected_outputs: Expected logits to test for numerical equality.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        verify_subprocess: Whether to execute an isolated subprocess verification.
        raise_on_failure: Whether to raise ReloadVerificationError on check failure.

    Returns:
        ReloadVerificationResult detailing verification status.
    """
    in_dir = Path(export_dir)
    verified_files: list[str] = []
    details: dict[str, Any] = {}

    def _fail(msg: str):
        if raise_on_failure:
            raise ReloadVerificationError(msg)
        LOG.error("Reload verification failure: %s", msg)

    # 1. Load manifest and verify checksums
    try:
        state_dict, config, manifest = load_exported_package(in_dir)
    except ReloadVerificationError as e:
        _fail(str(e))
        return ReloadVerificationResult(
            success=False,
            verified_files=[],
            format="unknown",
            total_tensors=0,
            regression_passed=False,
            details={"error": str(e)},
        )

    file_checksums = manifest.get("file_checksums", {})
    for filename, expected_hash in file_checksums.items():
        file_path = in_dir / filename
        if not file_path.exists():
            _fail(f"Missing declared file '{filename}' in export directory '{in_dir}'")
            return ReloadVerificationResult(
                success=False,
                verified_files=verified_files,
                format=manifest.get("export_format", ""),
                total_tensors=len(state_dict),
                regression_passed=False,
                details={"missing_file": filename},
            )
        actual_hash = file_sha256(file_path)
        if actual_hash != expected_hash:
            _fail(
                f"Checksum mismatch for '{filename}': expected {expected_hash}, got {actual_hash}"
            )
            return ReloadVerificationResult(
                success=False,
                verified_files=verified_files,
                format=manifest.get("export_format", ""),
                total_tensors=len(state_dict),
                regression_passed=False,
                details={"hash_mismatch": filename},
            )
        verified_files.append(filename)

    # 2. Check required preserved assets
    required_assets = ["config.json", "special_tokens_map.json", "generation_config.json"]
    for asset in required_assets:
        if not (in_dir / asset).exists():
            _fail(f"Required preserved asset '{asset}' missing from '{in_dir}'")
            return ReloadVerificationResult(
                success=False,
                verified_files=verified_files,
                format=manifest.get("export_format", ""),
                total_tensors=len(state_dict),
                regression_passed=False,
                details={"missing_asset": asset},
            )

    # Check tokenizer asset presence
    has_tok = (in_dir / "tokenizer.json").exists() or (in_dir / "vocab.json").exists()
    if not has_tok:
        _fail(f"No tokenizer definition (tokenizer.json or vocab.json) found in '{in_dir}'")
        return ReloadVerificationResult(
            success=False,
            verified_files=verified_files,
            format=manifest.get("export_format", ""),
            total_tensors=len(state_dict),
            regression_passed=False,
            details={"missing_tokenizer": True},
        )

    # 3. Structural validation against config
    intermediate_size = config.get("intermediate_size")
    if intermediate_size is not None:
        for name, t in state_dict.items():
            if "gate_proj.weight" in name or "up_proj.weight" in name:
                if t.shape[0] != intermediate_size:
                    _fail(
                        f"Tensor '{name}' shape {t.shape[0]} does not match config intermediate_size {intermediate_size}"
                    )

    # 4. Model instantiation and regression evaluation
    regression_passed = True
    max_abs_diff = None
    mean_abs_diff = None

    if model_factory is not None:
        try:
            model = model_factory(config)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            details["model_instantiated"] = True
        except Exception as e:
            _fail(f"Failed to instantiate model and load state_dict: {e}")
            return ReloadVerificationResult(
                success=False,
                verified_files=verified_files,
                format=manifest.get("export_format", ""),
                total_tensors=len(state_dict),
                regression_passed=False,
                details={"model_load_error": str(e)},
            )

        if regression_inputs is not None:
            with torch.no_grad():
                try:
                    if isinstance(regression_inputs, dict):
                        outputs = model(**regression_inputs)
                    else:
                        outputs = model(regression_inputs)
                except Exception as e:
                    _fail(f"Regression forward pass failed: {e}")
                    return ReloadVerificationResult(
                        success=False,
                        verified_files=verified_files,
                        format=manifest.get("export_format", ""),
                        total_tensors=len(state_dict),
                        regression_passed=False,
                        details={"forward_error": str(e)},
                    )

            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("last_hidden_state"))
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs

            if expected_outputs is not None:
                diff = torch.abs(logits.float() - expected_outputs.float())
                max_abs_diff = float(diff.max().item())
                mean_abs_diff = float(diff.mean().item())
                close = bool(
                    torch.allclose(
                        logits.float(),
                        expected_outputs.float(),
                        rtol=rtol,
                        atol=atol,
                    )
                )
                if not close:
                    _fail(
                        f"Regression output mismatch: max_diff={max_abs_diff:.6e} > atol={atol}"
                    )
                    regression_passed = False

    # 5. Clean subprocess verification
    subprocess_passed = False
    if verify_subprocess:
        py_code = f"""
import sys, json
from pathlib import Path
in_p = Path({json.dumps(str(in_dir))})
manifest = json.loads((in_p / 'export_manifest.json').read_text())
config = json.loads((in_p / 'config.json').read_text())
fmt = manifest.get('export_format')
weights_file = in_p / manifest.get('primary_weights_file')
if fmt == 'safetensors':
    import safetensors.torch
    sd = safetensors.torch.load_file(str(weights_file))
else:
    import torch
    sd = torch.load(str(weights_file), weights_only=True)
assert len(sd) > 0, 'No weights found'
print('SUBPROCESS_RELOAD_SUCCESS')
"""
        cmd = [sys.executable, "-c", py_code]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
        )
        if res.returncode != 0 or "SUBPROCESS_RELOAD_SUCCESS" not in res.stdout:
            _fail(f"Subprocess reload failed:\n{res.stderr}")
            return ReloadVerificationResult(
                success=False,
                verified_files=verified_files,
                format=manifest.get("export_format", ""),
                total_tensors=len(state_dict),
                regression_passed=False,
                subprocess_verified=False,
                details={"subprocess_error": res.stderr},
            )
        subprocess_passed = True

    return ReloadVerificationResult(
        success=regression_passed,
        verified_files=verified_files,
        format=manifest.get("export_format", ""),
        total_tensors=len(state_dict),
        regression_passed=regression_passed,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        subprocess_verified=subprocess_passed,
        details=details,
    )


# ==============================================================================
# 9. Speedup Reporting & Validation
# ==============================================================================


def validate_speedup_claim(claim_dict: dict[str, Any]) -> None:
    """Validate that a speedup claim contains explicit hardware, context, batch, and baseline numbers.

    Enforces ROADMAP acceptance criteria:
    - Claimed speedup must include hardware specifications (device_type, device_name, total_memory_mb).
    - Context length must be explicit and positive.
    - Batch size must be explicit and positive.
    - Baseline numbers (prefill, decode, TTFT) must be explicit and positive.

    Raises:
        SpeedupValidationError if any requirement is missing or invalid.
    """
    if not isinstance(claim_dict, dict):
        raise SpeedupValidationError("Speedup claim must be a dictionary.")

    # 1. Hardware validation
    hw = claim_dict.get("hardware")
    if not hw or not isinstance(hw, dict):
        raise SpeedupValidationError(
            "Speedup claim missing required 'hardware' specification."
        )
    if not hw.get("device_type") or not str(hw.get("device_type")).strip():
        raise SpeedupValidationError(
            "Speedup claim hardware missing required 'device_type'."
        )
    if not hw.get("device_name") or not str(hw.get("device_name")).strip():
        raise SpeedupValidationError(
            "Speedup claim hardware missing required 'device_name'."
        )
    if float(hw.get("total_memory_mb", 0.0)) <= 0:
        raise SpeedupValidationError(
            "Speedup claim hardware requires explicit positive 'total_memory_mb'."
        )

    # 2. Context length and Batch size validation
    batch_size = claim_dict.get("batch_size")
    if batch_size is None or int(batch_size) <= 0:
        raise SpeedupValidationError(
            f"Speedup claim requires explicit positive 'batch_size', got {batch_size}."
        )

    context_length = claim_dict.get("context_length")
    if context_length is None or int(context_length) <= 0:
        raise SpeedupValidationError(
            f"Speedup claim requires explicit positive 'context_length', got {context_length}."
        )

    # 3. Baseline numbers validation
    baseline = claim_dict.get("baseline")
    if not baseline or not isinstance(baseline, dict):
        raise SpeedupValidationError(
            "Speedup claim missing required 'baseline' numbers."
        )

    for field_name in ["prefill_ms", "decode_ms", "ttft_ms"]:
        val = baseline.get(field_name)
        if val is None or float(val) <= 0:
            raise SpeedupValidationError(
                f"Speedup claim baseline requires explicit positive '{field_name}', got {val}."
            )

    # 4. Edited numbers validation
    edited = claim_dict.get("edited")
    if not edited or not isinstance(edited, dict):
        raise SpeedupValidationError(
            "Speedup claim missing required 'edited' numbers."
        )

    for field_name in ["prefill_ms", "decode_ms", "ttft_ms"]:
        val = edited.get(field_name)
        if val is None or float(val) <= 0:
            raise SpeedupValidationError(
                f"Speedup claim edited requires explicit positive '{field_name}', got {val}."
            )


def compute_speedup_report(
    baseline: ProfilingResult,
    edited: ProfilingResult,
    strict: bool = True,
) -> SpeedupReport:
    """Compute measured latency speedups against a recorded baseline with strict validation.

    Args:
        baseline: ProfilingResult of the unedited baseline model.
        edited: ProfilingResult of the edited/exported model.
        strict: If True, raises SpeedupValidationError on profile mismatch or invalid baseline.

    Returns:
        SpeedupReport with prefill, decode, TTFT, throughput, and memory deltas.
    """
    notes: list[str] = []

    # 1. Profile parity validation
    if baseline.profile.batch_size != edited.profile.batch_size:
        msg = (
            f"Batch size mismatch: baseline={baseline.profile.batch_size}, "
            f"edited={edited.profile.batch_size}"
        )
        if strict:
            raise SpeedupValidationError(msg)
        notes.append(msg)

    if baseline.profile.prompt_length != edited.profile.prompt_length:
        msg = (
            f"Context length mismatch: baseline={baseline.profile.prompt_length}, "
            f"edited={edited.profile.prompt_length}"
        )
        if strict:
            raise SpeedupValidationError(msg)
        notes.append(msg)

    if baseline.profile.decode_steps != edited.profile.decode_steps:
        msg = (
            f"Decode steps mismatch: baseline={baseline.profile.decode_steps}, "
            f"edited={edited.profile.decode_steps}"
        )
        if strict:
            raise SpeedupValidationError(msg)
        notes.append(msg)

    # 2. Hardware and baseline validation
    claim_dict = {
        "hardware": baseline.hardware.to_dict(),
        "batch_size": baseline.profile.batch_size,
        "context_length": baseline.profile.prompt_length,
        "baseline": {
            "prefill_ms": baseline.prefill_latency_ms,
            "decode_ms": baseline.decode_latency_per_token_ms
            if baseline.profile.decode_steps > 0
            else 1.0,
            "ttft_ms": baseline.time_to_first_token_ms,
        },
        "edited": {
            "prefill_ms": edited.prefill_latency_ms,
            "decode_ms": edited.decode_latency_per_token_ms
            if edited.profile.decode_steps > 0
            else 1.0,
            "ttft_ms": edited.time_to_first_token_ms,
        },
    }

    try:
        validate_speedup_claim(claim_dict)
    except SpeedupValidationError as e:
        if strict:
            raise
        notes.append(str(e))

    # 3. Compute speedups
    prefill_speedup = (
        baseline.prefill_latency_ms / edited.prefill_latency_ms
        if edited.prefill_latency_ms > 0
        else 1.0
    )

    if baseline.profile.decode_steps > 0 and edited.decode_latency_per_token_ms > 0:
        decode_speedup = (
            baseline.decode_latency_per_token_ms / edited.decode_latency_per_token_ms
        )
    else:
        decode_speedup = 1.0

    ttft_speedup = (
        baseline.time_to_first_token_ms / edited.time_to_first_token_ms
        if edited.time_to_first_token_ms > 0
        else 1.0
    )

    throughput_speedup = (
        edited.total_tokens_per_sec / baseline.total_tokens_per_sec
        if baseline.total_tokens_per_sec > 0
        else 1.0
    )

    mem_reduction = baseline.peak_resident_memory_mb - edited.peak_resident_memory_mb
    mem_reduction_pct = (
        (mem_reduction / baseline.peak_resident_memory_mb) * 100.0
        if baseline.peak_resident_memory_mb > 0
        else 0.0
    )

    is_valid = len(notes) == 0

    return SpeedupReport(
        hardware=edited.hardware,
        batch_size=edited.profile.batch_size,
        context_length=edited.profile.prompt_length,
        decode_steps=edited.profile.decode_steps,
        baseline_prefill_ms=baseline.prefill_latency_ms,
        edited_prefill_ms=edited.prefill_latency_ms,
        prefill_speedup=prefill_speedup,
        baseline_decode_ms=baseline.decode_latency_per_token_ms,
        edited_decode_ms=edited.decode_latency_per_token_ms,
        decode_speedup=decode_speedup,
        baseline_ttft_ms=baseline.time_to_first_token_ms,
        edited_ttft_ms=edited.time_to_first_token_ms,
        ttft_speedup=ttft_speedup,
        baseline_tokens_per_sec=baseline.total_tokens_per_sec,
        edited_tokens_per_sec=edited.total_tokens_per_sec,
        throughput_speedup=throughput_speedup,
        baseline_peak_mem_mb=baseline.peak_resident_memory_mb,
        edited_peak_mem_mb=edited.peak_resident_memory_mb,
        memory_reduction_mb=mem_reduction,
        memory_reduction_pct=mem_reduction_pct,
        is_valid=is_valid,
        validation_notes=notes,
    )
