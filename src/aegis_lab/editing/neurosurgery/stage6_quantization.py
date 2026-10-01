"""Stage 6: Quantization after surgery in model neurosurgery.

Implements:
1. Calibration dataset binding with deterministic SHA-256 checksum and sample tracking.
2. Uniform affine quantization (INT8 and INT4, symmetric and asymmetric, per-tensor/per-channel)
   for weights and optional activations, with INT4 bit-packing.
3. Per-layer sensitivity profiling via KL divergence and MSE on calibration data to assign
   mixed-precision strategies (FP16/FP32 for sensitive layers, INT8/INT4 for robust layers).
4. Physical slicing protection: explicitly rejects physical slicing/surgery of packed
   quantized tensors unless unpacked, edited, and repacked (PackedQuantizationSurgeryError).
5. Independent validation of quantized candidate against declared drift limits (KEEP retention,
   DROP rebound suppression, memory reduction, generation correctness).
6. Export and reload parity: export checkpoint with quantization metadata and provenance;
   verify reloading and generation equivalence within numerical tolerance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    LOG,
    load_tensor_artifact,
    nested_getattr,
    nested_setattr,
)


QUANTIZATION_SCHEMA_VERSION = "stage6_quantization_v1"
SUPPORTED_BITS = (4, 8)
SUPPORTED_GRANULARITIES = ("per_tensor", "per_channel")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class Stage6QuantizationError(Exception):
    """Base exception for Stage 6 quantization operations."""
    pass


class CalibrationMismatchError(Stage6QuantizationError):
    """Raised when calibration dataset fails checksum or sample verification."""
    pass


class PackedQuantizationSurgeryError(Stage6QuantizationError, RuntimeError):
    """Raised when physical slicing or surgery is attempted on packed quantized tensors without unpacking."""
    pass


class QuantizationDriftExceededError(Stage6QuantizationError, ValueError):
    """Raised when a candidate quantized model exceeds declared drift limits during validation."""
    pass


class ReloadParityError(Stage6QuantizationError):
    """Raised when an exported and reloaded quantized checkpoint deviates beyond numerical tolerance."""
    pass


class InvalidQuantizationConfigError(Stage6QuantizationError, ValueError):
    """Raised when quantization configuration parameters are invalid."""
    pass


# ---------------------------------------------------------------------------
# Helpers for nested attribute manipulation and model inspection
# ---------------------------------------------------------------------------

def safe_nested_getattr(root: Any, path: str) -> Any:
    """Safely retrieves a nested attribute or ModuleList index along a dotted path."""
    obj = root
    for part in path.split("."):
        if part.isdigit() and isinstance(obj, (list, tuple, nn.ModuleList)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def safe_nested_setattr(root: Any, path: str, value: Any) -> None:
    """Safely sets a nested attribute or ModuleList index along a dotted path."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (list, tuple, nn.ModuleList)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit() and isinstance(parent, (list, tuple, nn.ModuleList)):
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def calculate_model_memory_bytes(model: nn.Module) -> int:
    """Computes accurate memory byte footprint of a model, accounting for parameters and buffers.
    
    Avoids double-counting tied weights by tracking unique data pointers.
    """
    total_bytes = 0
    visited_ptrs = set()

    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in visited_ptrs:
            visited_ptrs.add(ptr)
            total_bytes += p.numel() * p.element_size()

    for b in model.buffers():
        ptr = b.data_ptr()
        if ptr not in visited_ptrs:
            visited_ptrs.add(ptr)
            total_bytes += b.numel() * b.element_size()

    return total_bytes


def extract_logits(output: Any) -> torch.Tensor:
    """Extracts logits tensor from diverse model outputs (tensors, tuples, SimpleNamespace, HF objects)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        return output.logits
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError(f"Cannot extract logits from model output of type {type(output)}")


# ---------------------------------------------------------------------------
# Calibration Dataset Binding
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBinding:
    """Manifest binding calibration data to a quantization run with SHA-256 and sample tracking."""

    dataset_name: str
    sha256: str
    sample_count: int
    sample_ids: list[str]
    sample_hashes: list[str]
    domain_counts: dict[str, int]
    token_count: Optional[int] = None
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QUANTIZATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "sha256": self.sha256,
            "sample_count": self.sample_count,
            "sample_ids": list(self.sample_ids),
            "sample_hashes": list(self.sample_hashes),
            "domain_counts": dict(self.domain_counts),
            "token_count": self.token_count,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationBinding:
        schema = data.get("schema_version")
        if schema != QUANTIZATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {schema}, expected {QUANTIZATION_SCHEMA_VERSION}")
        return cls(
            dataset_name=str(data["dataset_name"]),
            sha256=str(data["sha256"]),
            sample_count=int(data["sample_count"]),
            sample_ids=[str(x) for x in data.get("sample_ids", [])],
            sample_hashes=[str(x) for x in data.get("sample_hashes", [])],
            domain_counts={str(k): int(v) for k, v in data.get("domain_counts", {}).items()},
            token_count=int(data["token_count"]) if data.get("token_count") is not None else None,
            created_at=str(data.get("created_at", "")),
            metadata=dict(data.get("metadata", {})),
            schema_version=schema,
        )

    def verify(self, samples: Sequence[Any]) -> bool:
        """Verifies that the supplied samples match this binding's deterministic SHA-256 and sample count."""
        recomputed = bind_calibration_dataset(
            samples=samples,
            dataset_name=self.dataset_name,
            metadata=self.metadata,
        )
        if recomputed.sample_count != self.sample_count:
            raise CalibrationMismatchError(
                f"Sample count mismatch: expected {self.sample_count}, got {recomputed.sample_count}"
            )
        if recomputed.sha256 != self.sha256:
            raise CalibrationMismatchError(
                f"Calibration SHA-256 mismatch: expected {self.sha256}, got {recomputed.sha256}"
            )
        return True


def _canonicalize_sample(sample: Any, index: int) -> dict[str, Any]:
    """Canonicalizes an arbitrary sample (str, Sample, dict, or tensor) into a deterministic record."""
    if isinstance(sample, str):
        text = sample.strip()
        if not text:
            raise ValueError(f"Calibration sample at index {index} is empty string")
        prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {
            "index": index,
            "sample_id": prompt_hash[:16],
            "prompt": text,
            "prompt_hash": prompt_hash,
            "domain": "default",
        }
    if hasattr(sample, "prompt"):
        text = str(sample.prompt).strip()
        if not text:
            raise ValueError(f"Calibration sample at index {index} has empty prompt")
        prompt_hash = getattr(sample, "prompt_hash", hashlib.sha256(text.encode("utf-8")).hexdigest())
        sample_id = getattr(sample, "sample_id", None) or prompt_hash[:16]
        domain = getattr(sample, "domain", "default") or "default"
        return {
            "index": index,
            "sample_id": str(sample_id),
            "prompt": text,
            "prompt_hash": str(prompt_hash),
            "domain": str(domain),
        }
    if isinstance(sample, dict):
        text = str(sample.get("prompt") or sample.get("text") or "").strip()
        if not text and "input_ids" not in sample:
            raise ValueError(f"Calibration sample at index {index} missing text or input_ids")
        if text:
            prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        else:
            prompt_hash = hashlib.sha256(str(sample["input_ids"]).encode("utf-8")).hexdigest()
        sample_id = str(sample.get("sample_id") or prompt_hash[:16])
        domain = str(sample.get("domain") or "default")
        return {
            "index": index,
            "sample_id": sample_id,
            "prompt": text,
            "prompt_hash": prompt_hash,
            "domain": domain,
        }
    if isinstance(sample, torch.Tensor):
        prompt_hash = hashlib.sha256(sample.cpu().numpy().tobytes()).hexdigest()
        return {
            "index": index,
            "sample_id": prompt_hash[:16],
            "prompt": f"<tensor_{tuple(sample.shape)}>",
            "prompt_hash": prompt_hash,
            "domain": "default",
        }
    text = str(sample).strip()
    prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "index": index,
        "sample_id": prompt_hash[:16],
        "prompt": text,
        "prompt_hash": prompt_hash,
        "domain": "default",
    }


def bind_calibration_dataset(
    samples: Sequence[Any],
    dataset_name: str = "calibration",
    tokenizer: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> CalibrationBinding:
    """Binds calibration data to a quantization run with deterministic SHA-256 and sample tracking."""
    if not samples:
        raise ValueError("Cannot bind empty calibration dataset")

    canonical_records = []
    sample_ids: list[str] = []
    sample_hashes: list[str] = []
    domain_counts: dict[str, int] = {}
    total_tokens = 0

    for i, s in enumerate(samples):
        rec = _canonicalize_sample(s, i)
        canonical_records.append(rec)
        sample_ids.append(rec["sample_id"])
        sample_hashes.append(rec["prompt_hash"])
        domain = rec["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if tokenizer is not None and rec["prompt"] and not rec["prompt"].startswith("<tensor_"):
            try:
                tokens = tokenizer(rec["prompt"], return_tensors="pt")
                if "input_ids" in tokens:
                    total_tokens += tokens["input_ids"].numel()
            except Exception:
                pass

    payload = {
        "dataset_name": dataset_name,
        "sample_count": len(samples),
        "records": canonical_records,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return CalibrationBinding(
        dataset_name=dataset_name,
        sha256=sha256,
        sample_count=len(samples),
        sample_ids=sample_ids,
        sample_hashes=sample_hashes,
        domain_counts=domain_counts,
        token_count=total_tokens if total_tokens > 0 else None,
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata=dict(metadata or {}),
        schema_version=QUANTIZATION_SCHEMA_VERSION,
    )


def verify_calibration_binding(binding: CalibrationBinding, samples: Sequence[Any]) -> bool:
    """Verifies that the calibration dataset strictly matches its binding manifest."""
    return binding.verify(samples)


# ---------------------------------------------------------------------------
# Quantization Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    """Configuration for uniform affine quantization schemes."""

    bits: int = 8
    symmetric: bool = True
    granularity: str = "per_channel"
    quantize_activations: bool = False
    act_bits: int = 8
    act_symmetric: bool = False
    act_granularity: str = "per_tensor"
    pack: bool = True

    def __post_init__(self) -> None:
        if self.bits not in SUPPORTED_BITS:
            raise InvalidQuantizationConfigError(
                f"Unsupported bits: {self.bits}. Supported bits are {SUPPORTED_BITS}"
            )
        if self.granularity not in SUPPORTED_GRANULARITIES:
            raise InvalidQuantizationConfigError(
                f"Unsupported granularity: {self.granularity}. Supported are {SUPPORTED_GRANULARITIES}"
            )
        if self.act_bits not in SUPPORTED_BITS:
            raise InvalidQuantizationConfigError(
                f"Unsupported act_bits: {self.act_bits}. Supported are {SUPPORTED_BITS}"
            )
        if self.act_granularity not in SUPPORTED_GRANULARITIES:
            raise InvalidQuantizationConfigError(
                f"Unsupported act_granularity: {self.act_granularity}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "symmetric": self.symmetric,
            "granularity": self.granularity,
            "quantize_activations": self.quantize_activations,
            "act_bits": self.act_bits,
            "act_symmetric": self.act_symmetric,
            "act_granularity": self.act_granularity,
            "pack": self.pack,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationConfig:
        return cls(
            bits=int(data.get("bits", 8)),
            symmetric=bool(data.get("symmetric", True)),
            granularity=str(data.get("granularity", "per_channel")),
            quantize_activations=bool(data.get("quantize_activations", False)),
            act_bits=int(data.get("act_bits", 8)),
            act_symmetric=bool(data.get("act_symmetric", False)),
            act_granularity=str(data.get("act_granularity", "per_tensor")),
            pack=bool(data.get("pack", True)),
        )


# ---------------------------------------------------------------------------
# INT4 Bit-Packing Primitives
# ---------------------------------------------------------------------------

def pack_int4(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Packs 4-bit integer values into a compact uint8 tensor (2 nibbles per byte).
    
    Supports arbitrary shapes and odd element counts via flattening.
    """
    shape = tuple(tensor.shape)
    flat = tensor.reshape(-1)
    orig_len = flat.numel()
    if orig_len % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])

    # Store 4-bit nibbles using low nibble for even indices, high nibble for odd indices
    v0 = (flat[0::2] & 0x0F).to(torch.uint8)
    v1 = (flat[1::2] & 0x0F).to(torch.uint8)
    packed = v0 | (v1 << 4)
    return packed, shape


def unpack_int4(
    packed: torch.Tensor,
    original_shape: tuple[int, ...],
    is_signed: bool = True,
) -> torch.Tensor:
    """Unpacks a uint8 tensor containing 4-bit values back to integer tensor of original_shape."""
    v0 = (packed & 0x0F).to(torch.int32)
    v1 = ((packed >> 4) & 0x0F).to(torch.int32)

    if is_signed:
        # 4-bit two's complement decoding: values in [8, 15] decode to [8-16, 15-16] = [-8, -1]
        v0 = torch.where(v0 >= 8, v0 - 16, v0)
        v1 = torch.where(v1 >= 8, v1 - 16, v1)

    flat = torch.empty(packed.numel() * 2, dtype=torch.int32, device=packed.device)
    flat[0::2] = v0
    flat[1::2] = v1

    total_elements = 1
    for s in original_shape:
        total_elements *= s

    return flat[:total_elements].reshape(original_shape)


# ---------------------------------------------------------------------------
# Uniform Affine Quantization Primitives
# ---------------------------------------------------------------------------

def quantize_affine(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    granularity: str = "per_channel",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies uniform affine quantization to a tensor.
    
    Returns:
        (quantized_tensor, scale, zero_point)
    """
    if bits not in SUPPORTED_BITS:
        raise InvalidQuantizationConfigError(f"Unsupported bits: {bits}")

    if granularity == "per_channel":
        dim = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)
    elif granularity == "per_tensor":
        x_min = x.amin()
        x_max = x.amax()
    else:
        raise InvalidQuantizationConfigError(f"Unsupported granularity: {granularity}")

    if symmetric:
        q_max = (1 << (bits - 1)) - 1
        q_min = -q_max
        abs_max = torch.maximum(x_min.abs(), x_max.abs())
        scale = torch.clamp(abs_max / float(q_max), min=1e-8)
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        q = torch.clamp(torch.round(x / scale), q_min, q_max).to(torch.int32)
    else:
        q_min = 0
        q_max = (1 << bits) - 1
        scale = torch.clamp((x_max - x_min) / float(q_max - q_min), min=1e-8)
        zp = torch.round(-x_min / scale)
        zero_point = torch.clamp(zp, q_min, q_max).to(torch.int32)
        q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max).to(torch.int32)

    return q, scale, zero_point


def dequantize_affine(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantizes an integer tensor given scale and zero_point."""
    return (q.to(dtype) - zero_point.to(dtype)) * scale.to(dtype)


# ---------------------------------------------------------------------------
# QuantizedTensor Structure
# ---------------------------------------------------------------------------

class QuantizedTensor:
    """Encapsulates a quantized tensor with scale, zero_point, packing, and slicing protection."""

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        original_shape: tuple[int, ...],
        bits: int = 8,
        symmetric: bool = True,
        granularity: str = "per_channel",
        is_packed: bool = False,
    ) -> None:
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.original_shape = tuple(original_shape)
        self.bits = bits
        self.symmetric = symmetric
        self.granularity = granularity
        self.is_packed = is_packed

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantizes to a floating-point tensor."""
        if self.is_packed:
            unpacked = unpack_int4(self.data, self.original_shape, is_signed=self.symmetric)
        else:
            unpacked = self.data
        return dequantize_affine(unpacked, self.scale, self.zero_point, dtype=dtype)

    def unpack(self) -> QuantizedTensor:
        """Unpacks packed representations into explicit unpacked integers."""
        if not self.is_packed:
            return self
        unpacked = unpack_int4(self.data, self.original_shape, is_signed=self.symmetric)
        dtype = torch.int8 if self.symmetric else torch.uint8
        return QuantizedTensor(
            data=unpacked.to(dtype),
            scale=self.scale.clone(),
            zero_point=self.zero_point.clone(),
            original_shape=self.original_shape,
            bits=self.bits,
            symmetric=self.symmetric,
            granularity=self.granularity,
            is_packed=False,
        )

    def pack(self) -> QuantizedTensor:
        """Packs an unpacked 4-bit tensor into uint8 nibbles."""
        if self.is_packed:
            return self
        if self.bits != 4:
            raise ValueError(f"Bit-packing is only supported for 4-bit tensors, got {self.bits}")
        packed_data, shape = pack_int4(self.data)
        return QuantizedTensor(
            data=packed_data,
            scale=self.scale.clone(),
            zero_point=self.zero_point.clone(),
            original_shape=self.original_shape,
            bits=self.bits,
            symmetric=self.symmetric,
            granularity=self.granularity,
            is_packed=True,
        )

    def byte_size(self) -> int:
        """Returns the actual resident memory bytes of the quantized data and metadata."""
        return (
            self.data.numel() * self.data.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero_point.numel() * self.zero_point.element_size()
        )

    def slice(self, dim: int, start: int, end: int) -> QuantizedTensor:
        """Safely slices an unpacked quantized tensor.
        
        Explicitly raises PackedQuantizationSurgeryError if the tensor is packed.
        """
        if self.is_packed:
            raise PackedQuantizationSurgeryError(
                f"Cannot physically slice packed quantized tensor along dimension {dim} (shape {self.original_shape}). "
                "Packed tensors encode multiple weights per byte and cannot be sliced without bit corruption. "
                "Unpack first with unpack(), perform surgery, and repack."
            )

        indices = list(range(start, end))
        return self.select_indices(dim=dim, indices=indices)

    def select_indices(self, dim: int, indices: Sequence[int]) -> QuantizedTensor:
        """Selects specific channel indices along dim."""
        if self.is_packed:
            raise PackedQuantizationSurgeryError(
                f"Cannot select indices on packed quantized tensor along dimension {dim}. "
                "Unpack the tensor first with unpack(), perform surgery, and repack."
            )

        idx_t = torch.tensor(indices, dtype=torch.long, device=self.data.device)
        new_data = torch.index_select(self.data, dim=dim, index=idx_t)

        new_scale = self.scale
        new_zp = self.zero_point

        if self.granularity == "per_channel" and dim == 0:
            new_scale = torch.index_select(self.scale, dim=0, index=idx_t)
            new_zp = torch.index_select(self.zero_point, dim=0, index=idx_t)

        new_shape = list(self.original_shape)
        new_shape[dim] = len(indices)

        return QuantizedTensor(
            data=new_data,
            scale=new_scale,
            zero_point=new_zp,
            original_shape=tuple(new_shape),
            bits=self.bits,
            symmetric=self.symmetric,
            granularity=self.granularity,
            is_packed=False,
        )

    def __getitem__(self, item: Any) -> Any:
        """Prohibits direct indexing/slicing on packed quantized tensors."""
        if self.is_packed:
            raise PackedQuantizationSurgeryError(
                "Direct indexing or slicing of a packed quantized tensor is prohibited. "
                "Packed representations must be unpacked before slicing or editing."
            )
        return self.data[item]


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    granularity: str = "per_channel",
    pack: bool = False,
) -> QuantizedTensor:
    """Helper to quantize a tensor and wrap it in a QuantizedTensor."""
    q, scale, zp = quantize_affine(tensor, bits=bits, symmetric=symmetric, granularity=granularity)
    dtype = torch.int8 if symmetric else torch.uint8
    qt = QuantizedTensor(
        data=q.to(dtype),
        scale=scale,
        zero_point=zp,
        original_shape=tuple(tensor.shape),
        bits=bits,
        symmetric=symmetric,
        granularity=granularity,
        is_packed=False,
    )
    if pack and bits == 4:
        qt = qt.pack()
    return qt


# ---------------------------------------------------------------------------
# QuantizedLinear Module
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """Linear layer using uniform affine quantized weights with optional activation quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: QuantizedTensor,
        bias: Optional[torch.Tensor] = None,
        quantize_activations: bool = False,
        act_bits: int = 8,
        act_symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = qweight.bits
        self.symmetric = qweight.symmetric
        self.granularity = qweight.granularity
        self.is_packed = qweight.is_packed
        self.original_shape = qweight.original_shape

        self.quantize_activations = quantize_activations
        self.act_bits = act_bits
        self.act_symmetric = act_symmetric

        # Register quantized weights and parameters as buffers for transparent state_dict loading
        self.register_buffer("weight_data", qweight.data)
        self.register_buffer("weight_scale", qweight.scale)
        self.register_buffer("weight_zero_point", qweight.zero_point)

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)

    @property
    def qweight(self) -> QuantizedTensor:
        """Constructs a QuantizedTensor view from internal registered buffers."""
        return QuantizedTensor(
            data=self.weight_data,
            scale=self.weight_scale,
            zero_point=self.weight_zero_point,
            original_shape=self.original_shape,
            bits=self.bits,
            symmetric=self.symmetric,
            granularity=self.granularity,
            is_packed=self.is_packed,
        )

    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dequantizes the weight tensor to floating-point representation."""
        return self.qweight.dequantize(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional activation quantization
        if self.quantize_activations:
            x_q, x_scale, x_zp = quantize_affine(
                x,
                bits=self.act_bits,
                symmetric=self.act_symmetric,
                granularity="per_tensor",
            )
            x = dequantize_affine(x_q, x_scale, x_zp, dtype=x.dtype)

        # Dequantize weight on the fly
        w = self.dequantize_weight(dtype=x.dtype).to(device=x.device)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, w, bias)

    def unpack(self) -> None:
        """Unpacks weight_data buffer if currently packed."""
        if not self.is_packed:
            return
        unpacked_qt = self.qweight.unpack()
        self.weight_data = unpacked_qt.data
        self.is_packed = False

    def pack(self) -> None:
        """Packs weight_data buffer if 4-bit and unpacked."""
        if self.is_packed or self.bits != 4:
            return
        packed_qt = self.qweight.pack()
        self.weight_data = packed_qt.data
        self.is_packed = True

    def byte_size(self) -> int:
        """Returns total resident memory bytes of this module."""
        total = self.qweight.byte_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def slice_channels(self, indices: Sequence[int], axis: int = 0) -> QuantizedLinear:
        """Channel slicing method with strict physical slicing protection for packed tensors."""
        if self.is_packed:
            raise PackedQuantizationSurgeryError(
                f"Cannot slice channels of packed QuantizedLinear along axis {axis}. "
                "Physical slicing of packed quantized layers is rejected. "
                "Use unpack() first, perform physical surgery, and repack, or use unpack_edit_repack()."
            )

        new_qweight = self.qweight.select_indices(dim=axis, indices=indices)
        new_bias = None
        if self.bias is not None:
            if axis == 0:
                idx_t = torch.tensor(indices, dtype=torch.long, device=self.bias.device)
                new_bias = torch.index_select(self.bias.data, dim=0, index=idx_t)
            else:
                new_bias = self.bias.data.clone()

        new_out = len(indices) if axis == 0 else self.out_features
        new_in = len(indices) if axis == 1 else self.in_features

        return QuantizedLinear(
            in_features=new_in,
            out_features=new_out,
            qweight=new_qweight,
            bias=new_bias,
            quantize_activations=self.quantize_activations,
            act_bits=self.act_bits,
            act_symmetric=self.act_symmetric,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QuantizationConfig) -> QuantizedLinear:
        """Converts an existing nn.Linear module into a QuantizedLinear module."""
        w = linear.weight.data
        qweight = quantize_tensor(
            w,
            bits=config.bits,
            symmetric=config.symmetric,
            granularity=config.granularity,
            pack=config.pack,
        )
        bias = linear.bias.data if linear.bias is not None else None
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            qweight=qweight,
            bias=bias,
            quantize_activations=config.quantize_activations,
            act_bits=config.act_bits,
            act_symmetric=config.act_symmetric,
        )

    @classmethod
    def create_empty(
        cls,
        in_features: int,
        out_features: int,
        config: QuantizationConfig,
        is_packed: bool = False,
        has_bias: bool = True,
        device: Optional[torch.device] = None,
    ) -> QuantizedLinear:
        """Creates an empty QuantizedLinear shell ready for state_dict loading."""
        mod = cls.__new__(cls)
        nn.Module.__init__(mod)
        mod.in_features = in_features
        mod.out_features = out_features
        mod.bits = config.bits
        mod.symmetric = config.symmetric
        mod.granularity = config.granularity
        mod.is_packed = is_packed
        mod.original_shape = (out_features, in_features)
        mod.quantize_activations = config.quantize_activations
        mod.act_bits = config.act_bits
        mod.act_symmetric = config.act_symmetric

        if is_packed and config.bits == 4:
            packed_len = math.ceil(in_features * out_features / 2)
            mod.register_buffer("weight_data", torch.zeros(packed_len, dtype=torch.uint8, device=device))
        else:
            dtype = torch.int8 if config.symmetric else torch.uint8
            mod.register_buffer("weight_data", torch.zeros((out_features, in_features), dtype=dtype, device=device))

        scale_shape = (out_features, 1) if config.granularity == "per_channel" else (1, 1)
        mod.register_buffer("weight_scale", torch.zeros(scale_shape, dtype=torch.float32, device=device))
        mod.register_buffer("weight_zero_point", torch.zeros(scale_shape, dtype=torch.int32, device=device))

        if has_bias:
            mod.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, device=device))
        else:
            mod.register_parameter("bias", None)

        return mod


# ---------------------------------------------------------------------------
# Physical Slicing Protection and Safe Surgery Adapter
# ---------------------------------------------------------------------------

def reject_packed_slicing_surgery(module: nn.Module) -> None:
    """Explicitly checks if module or any submodule is a packed quantized tensor and raises error."""
    for name, sub in module.named_modules():
        if isinstance(sub, QuantizedLinear) and sub.is_packed:
            raise PackedQuantizationSurgeryError(
                f"Module '{name}' is a packed quantized layer. "
                "Physical slicing of packed quantized models is rejected unless unpacked, edited, and repacked."
            )


def unpack_edit_repack(
    quant_linear: QuantizedLinear,
    edit_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    indices: Optional[Sequence[int]] = None,
    axis: int = 0,
) -> QuantizedLinear:
    """Safely performs structural surgery on a QuantizedLinear module.
    
    1. Unpacks and dequantizes weights to full precision.
    2. Applies the structural surgery transformation (slicing / editing).
    3. Re-quantizes and repairs scale and zero-point metadata.
    4. Repacks into the original packed format if originally packed.
    """
    was_packed = quant_linear.is_packed

    # Dequantize to float representation
    w_float = quant_linear.dequantize_weight()

    if indices is not None:
        idx_t = torch.tensor(indices, dtype=torch.long, device=w_float.device)
        w_sliced = torch.index_select(w_float, dim=axis, index=idx_t)
        if quant_linear.bias is not None:
            if axis == 0:
                bias_sliced = torch.index_select(quant_linear.bias.data, dim=0, index=idx_t)
            else:
                bias_sliced = quant_linear.bias.data.clone()
        else:
            bias_sliced = None
    elif edit_fn is not None:
        w_sliced = edit_fn(w_float)
        bias_sliced = quant_linear.bias.data.clone() if quant_linear.bias is not None else None
    else:
        raise ValueError("Must provide either edit_fn or indices to unpack_edit_repack")

    new_out = w_sliced.shape[0]
    new_in = w_sliced.shape[1]

    # Re-quantize with original configuration
    qweight = quantize_tensor(
        w_sliced,
        bits=quant_linear.bits,
        symmetric=quant_linear.symmetric,
        granularity=quant_linear.granularity,
        pack=was_packed,
    )

    return QuantizedLinear(
        in_features=new_in,
        out_features=new_out,
        qweight=qweight,
        bias=bias_sliced,
        quantize_activations=quant_linear.quantize_activations,
        act_bits=quant_linear.act_bits,
        act_symmetric=quant_linear.act_symmetric,
    )


def apply_slicing_surgery(
    module: nn.Module,
    indices: Sequence[int],
    axis: int = 0,
    allow_unpack_repack: bool = False,
) -> nn.Module:
    """Applies channel slicing to a module with strict packed quantization rejection."""
    if isinstance(module, QuantizedLinear):
        if module.is_packed and not allow_unpack_repack:
            raise PackedQuantizationSurgeryError(
                f"Cannot physically slice packed quantized module {module}. "
                "Physical slicing of packed quantized tensors is rejected unless unpacked, edited, and repacked."
            )
        if module.is_packed and allow_unpack_repack:
            return unpack_edit_repack(module, indices=indices, axis=axis)
        return module.slice_channels(indices=indices, axis=axis)

    if isinstance(module, nn.Linear):
        idx_t = torch.tensor(indices, dtype=torch.long, device=module.weight.device)
        w_sliced = torch.index_select(module.weight.data, dim=axis, index=idx_t)
        b_sliced = None
        if module.bias is not None:
            if axis == 0:
                b_sliced = torch.index_select(module.bias.data, dim=0, index=idx_t)
            else:
                b_sliced = module.bias.data.clone()
        new_lin = nn.Linear(w_sliced.shape[1], w_sliced.shape[0], bias=module.bias is not None)
        new_lin.weight.data = w_sliced
        if b_sliced is not None:
            new_lin.bias.data = b_sliced
        return new_lin

    raise TypeError(f"Unsupported module type for slicing surgery: {type(module)}")


# ---------------------------------------------------------------------------
# Per-Layer Sensitivity Profiling and Mixed Precision
# ---------------------------------------------------------------------------

@dataclass
class LayerSensitivity:
    """Sensitivity measurement for a single layer on calibration data."""

    layer_name: str
    kl_divergence: float
    mse: float
    sensitivity_score: float
    candidate_bits: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "kl_divergence": self.kl_divergence,
            "mse": self.mse,
            "sensitivity_score": self.sensitivity_score,
            "candidate_bits": self.candidate_bits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerSensitivity:
        return cls(
            layer_name=str(data["layer_name"]),
            kl_divergence=float(data["kl_divergence"]),
            mse=float(data["mse"]),
            sensitivity_score=float(data["sensitivity_score"]),
            candidate_bits=int(data.get("candidate_bits", 4)),
        )


@dataclass
class SensitivityProfile:
    """Collection of per-layer sensitivity measurements."""

    layers: dict[str, LayerSensitivity]
    calibration_binding: CalibrationBinding
    candidate_bits: int = 4

    def sorted_by_sensitivity(self, metric: str = "kl") -> list[LayerSensitivity]:
        """Returns layer measurements sorted from most sensitive to least sensitive."""
        items = list(self.layers.values())
        if metric == "mse":
            return sorted(items, key=lambda s: s.mse, reverse=True)
        if metric == "score":
            return sorted(items, key=lambda s: s.sensitivity_score, reverse=True)
        return sorted(items, key=lambda s: s.kl_divergence, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_bits": self.candidate_bits,
            "calibration_binding": self.calibration_binding.to_dict(),
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensitivityProfile:
        return cls(
            candidate_bits=int(data.get("candidate_bits", 4)),
            calibration_binding=CalibrationBinding.from_dict(data["calibration_binding"]),
            layers={k: LayerSensitivity.from_dict(v) for k, v in data.get("layers", {}).items()},
        )


@dataclass
class MixedPrecisionStrategy:
    """Strategy configuration for mapping sensitivity profiles to layer precision levels."""

    strategy_type: str = "threshold"  # "threshold", "top_k", or "percentile"
    fp_threshold_kl: float = 0.05
    int4_threshold_kl: float = 0.01
    top_k_fp: int = 1
    top_k_int8: int = 1
    fp_percentile: float = 0.25
    int8_percentile: float = 0.35
    fp_precision: str = "fp16"
    moderate_precision: str = "int8"
    robust_precision: str = "int4"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_type": self.strategy_type,
            "fp_threshold_kl": self.fp_threshold_kl,
            "int4_threshold_kl": self.int4_threshold_kl,
            "top_k_fp": self.top_k_fp,
            "top_k_int8": self.top_k_int8,
            "fp_percentile": self.fp_percentile,
            "int8_percentile": self.int8_percentile,
            "fp_precision": self.fp_precision,
            "moderate_precision": self.moderate_precision,
            "robust_precision": self.robust_precision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MixedPrecisionStrategy:
        return cls(
            strategy_type=str(data.get("strategy_type", "threshold")),
            fp_threshold_kl=float(data.get("fp_threshold_kl", 0.05)),
            int4_threshold_kl=float(data.get("int4_threshold_kl", 0.01)),
            top_k_fp=int(data.get("top_k_fp", 1)),
            top_k_int8=int(data.get("top_k_int8", 1)),
            fp_percentile=float(data.get("fp_percentile", 0.25)),
            int8_percentile=float(data.get("int8_percentile", 0.35)),
            fp_precision=str(data.get("fp_precision", "fp16")),
            moderate_precision=str(data.get("moderate_precision", "int8")),
            robust_precision=str(data.get("robust_precision", "int4")),
        )


@dataclass
class MixedPrecisionPlan:
    """Assignment of precision levels to model layers."""

    layer_precisions: dict[str, str]
    profile: SensitivityProfile
    strategy: MixedPrecisionStrategy
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_precisions": dict(self.layer_precisions),
            "profile": self.profile.to_dict(),
            "strategy": self.strategy.to_dict(),
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MixedPrecisionPlan:
        return cls(
            layer_precisions=dict(data.get("layer_precisions", {})),
            profile=SensitivityProfile.from_dict(data["profile"]),
            strategy=MixedPrecisionStrategy.from_dict(data["strategy"]),
            summary=dict(data.get("summary", {})),
        )


def _prepare_calibration_inputs(
    calibration_data: Sequence[Any],
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> list[Any]:
    """Converts diverse calibration sample representations into batches for model forward passes."""
    batches = []
    for sample in calibration_data:
        if isinstance(sample, str):
            if tokenizer is not None:
                encoded = tokenizer([sample], return_tensors="pt")
                if device is not None:
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                batches.append(encoded)
            else:
                raise ValueError("Tokenizer required to process string calibration samples")
        elif hasattr(sample, "prompt"):
            if tokenizer is not None:
                encoded = tokenizer([sample.prompt], return_tensors="pt")
                if device is not None:
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                batches.append(encoded)
            else:
                raise ValueError("Tokenizer required to process Sample calibration samples")
        elif isinstance(sample, dict):
            if "input_ids" in sample:
                b = {k: (v.to(device) if isinstance(v, torch.Tensor) and device else v) for k, v in sample.items()}
                batches.append(b)
            elif "prompt" in sample and tokenizer is not None:
                encoded = tokenizer([sample["prompt"]], return_tensors="pt")
                if device is not None:
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                batches.append(encoded)
            else:
                batches.append(sample)
        elif isinstance(sample, torch.Tensor):
            batches.append(sample.to(device) if device else sample)
        else:
            batches.append(sample)
    return batches


def _forward_model_logits(model: nn.Module, batch: Any) -> torch.Tensor:
    """Executes a forward pass on model and extracts logits tensor."""
    if isinstance(batch, dict):
        out = model(**batch)
    elif isinstance(batch, (tuple, list)):
        out = model(*batch)
    else:
        out = model(batch)
    return extract_logits(out)


def profile_layer_sensitivity(
    model: nn.Module,
    calibration_data: Sequence[Any],
    tokenizer: Optional[Any] = None,
    candidate_config: Optional[QuantizationConfig] = None,
    candidate_bits: int = 4,
    layers_to_profile: Optional[list[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> SensitivityProfile:
    """Profiles per-layer sensitivity using KL divergence and MSE on calibration data.
    
    Temporarily quantizes each candidate layer to candidate_bits (default INT4) to assess
    its impact on baseline output distribution.
    """
    if candidate_config is None:
        candidate_config = QuantizationConfig(
            bits=candidate_bits,
            symmetric=True,
            granularity="per_channel",
            pack=True,
        )

    binding = bind_calibration_dataset(
        samples=calibration_data,
        dataset_name="sensitivity_calibration",
        tokenizer=tokenizer,
    )

    if device is not None:
        target_device = torch.device(device)
    else:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")

    batches = _prepare_calibration_inputs(calibration_data, tokenizer=tokenizer, device=target_device)
    model.eval()

    # Step 1: Collect baseline logits
    base_logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for b in batches:
            logits = _forward_model_logits(model, b)
            base_logits_list.append(logits.detach())

    # Step 2: Determine candidate layers
    if layers_to_profile is None:
        layers_to_profile = [
            name for name, mod in model.named_modules()
            if isinstance(mod, (nn.Linear, QuantizedLinear))
        ]

    layer_results: dict[str, LayerSensitivity] = {}

    for name in layers_to_profile:
        orig_mod = safe_nested_getattr(model, name)
        if not isinstance(orig_mod, nn.Linear):
            continue

        # Temporarily quantize layer
        quant_mod = QuantizedLinear.from_linear(orig_mod, candidate_config)
        safe_nested_setattr(model, name, quant_mod)

        # Run forward passes and compute KL and MSE
        kl_scores: list[float] = []
        mse_scores: list[float] = []

        with torch.no_grad():
            for b, base_logits in zip(batches, base_logits_list):
                cand_logits = _forward_model_logits(model, b)

                p_base = F.softmax(base_logits.float(), dim=-1)
                log_p_cand = F.log_softmax(cand_logits.float(), dim=-1)

                kl = F.kl_div(log_p_cand, p_base, log_target=False, reduction="batchmean").item()
                mse = F.mse_loss(cand_logits.float(), base_logits.float()).item()

                if not math.isfinite(kl):
                    kl = 1e6
                if not math.isfinite(mse):
                    mse = 1e6

                kl_scores.append(kl)
                mse_scores.append(mse)

        # Restore original module
        safe_nested_setattr(model, name, orig_mod)

        avg_kl = float(sum(kl_scores) / max(len(kl_scores), 1))
        avg_mse = float(sum(mse_scores) / max(len(mse_scores), 1))
        score = avg_kl + 0.1 * avg_mse

        layer_results[name] = LayerSensitivity(
            layer_name=name,
            kl_divergence=avg_kl,
            mse=avg_mse,
            sensitivity_score=score,
            candidate_bits=candidate_bits,
        )

    return SensitivityProfile(
        layers=layer_results,
        calibration_binding=binding,
        candidate_bits=candidate_bits,
    )


def assign_mixed_precision(
    profile: SensitivityProfile,
    strategy: Optional[MixedPrecisionStrategy] = None,
) -> MixedPrecisionPlan:
    """Assigns mixed-precision strategies (FP16/FP32, INT8, INT4) based on sensitivity profile."""
    if strategy is None:
        strategy = MixedPrecisionStrategy()

    sorted_layers = profile.sorted_by_sensitivity(metric="kl")
    total_layers = len(sorted_layers)
    precisions: dict[str, str] = {}

    if strategy.strategy_type == "threshold":
        for l in sorted_layers:
            if l.kl_divergence > strategy.fp_threshold_kl:
                precisions[l.layer_name] = strategy.fp_precision
            elif l.kl_divergence <= strategy.int4_threshold_kl:
                precisions[l.layer_name] = strategy.robust_precision
            else:
                precisions[l.layer_name] = strategy.moderate_precision

    elif strategy.strategy_type == "top_k":
        top_fp = set(l.layer_name for l in sorted_layers[:strategy.top_k_fp])
        mid_int8 = set(
            l.layer_name for l in sorted_layers[strategy.top_k_fp : strategy.top_k_fp + strategy.top_k_int8]
        )
        for l in sorted_layers:
            if l.layer_name in top_fp:
                precisions[l.layer_name] = strategy.fp_precision
            elif l.layer_name in mid_int8:
                precisions[l.layer_name] = strategy.moderate_precision
            else:
                precisions[l.layer_name] = strategy.robust_precision

    elif strategy.strategy_type == "percentile":
        k_fp = int(math.ceil(total_layers * strategy.fp_percentile))
        k_int8 = int(math.ceil(total_layers * strategy.int8_percentile))
        top_fp = set(l.layer_name for l in sorted_layers[:k_fp])
        mid_int8 = set(l.layer_name for l in sorted_layers[k_fp : k_fp + k_int8])
        for l in sorted_layers:
            if l.layer_name in top_fp:
                precisions[l.layer_name] = strategy.fp_precision
            elif l.layer_name in mid_int8:
                precisions[l.layer_name] = strategy.moderate_precision
            else:
                precisions[l.layer_name] = strategy.robust_precision
    else:
        raise ValueError(f"Unknown strategy_type: {strategy.strategy_type}")

    counts = {}
    for p in precisions.values():
        counts[p] = counts.get(p, 0) + 1

    summary = {
        "total_layers": total_layers,
        "counts_by_precision": counts,
    }

    return MixedPrecisionPlan(
        layer_precisions=precisions,
        profile=profile,
        strategy=strategy,
        summary=summary,
    )


def apply_mixed_precision_plan(
    model: nn.Module,
    plan: MixedPrecisionPlan,
    inplace: bool = False,
) -> nn.Module:
    """Applies a mixed-precision assignment plan to a model."""
    if not inplace:
        model = copy.deepcopy(model)

    for name, precision in plan.layer_precisions.items():
        try:
            mod = safe_nested_getattr(model, name)
        except (AttributeError, KeyError, IndexError):
            continue

        if not isinstance(mod, nn.Linear):
            continue

        norm_prec = precision.lower().strip()
        if norm_prec in ("int4", "4bit", "4-bit"):
            cfg = QuantizationConfig(bits=4, symmetric=True, granularity="per_channel", pack=True)
            qmod = QuantizedLinear.from_linear(mod, cfg)
            safe_nested_setattr(model, name, qmod)
        elif norm_prec in ("int8", "8bit", "8-bit"):
            cfg = QuantizationConfig(bits=8, symmetric=True, granularity="per_channel", pack=False)
            qmod = QuantizedLinear.from_linear(mod, cfg)
            safe_nested_setattr(model, name, qmod)
        elif norm_prec in ("fp16", "float16"):
            mod.to(torch.float16)
        elif norm_prec in ("fp32", "float32"):
            mod.to(torch.float32)

    return model


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    inplace: bool = False,
) -> nn.Module:
    """Quantizes all nn.Linear modules in model according to config."""
    if not inplace:
        model = copy.deepcopy(model)

    linear_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and not isinstance(mod, QuantizedLinear)
    ]
    for name in linear_names:
        mod = safe_nested_getattr(model, name)
        qmod = QuantizedLinear.from_linear(mod, config)
        safe_nested_setattr(model, name, qmod)

    return model


# ---------------------------------------------------------------------------
# Independent Validation of Quantized Candidate
# ---------------------------------------------------------------------------

@dataclass
class QuantizationValidationThresholds:
    """Declared drift limits and criteria for validating quantized candidates."""

    max_kl_drift: float = 0.50
    max_mse_drift: float = 1.00
    min_keep_retention: float = 0.80
    max_drop_rebound: float = 0.20
    min_memory_reduction_ratio: float = 0.15
    max_generation_nll_drift: Optional[float] = 1.00

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_kl_drift": self.max_kl_drift,
            "max_mse_drift": self.max_mse_drift,
            "min_keep_retention": self.min_keep_retention,
            "max_drop_rebound": self.max_drop_rebound,
            "min_memory_reduction_ratio": self.min_memory_reduction_ratio,
            "max_generation_nll_drift": self.max_generation_nll_drift,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationValidationThresholds:
        return cls(
            max_kl_drift=float(data.get("max_kl_drift", 0.50)),
            max_mse_drift=float(data.get("max_mse_drift", 1.00)),
            min_keep_retention=float(data.get("min_keep_retention", 0.80)),
            max_drop_rebound=float(data.get("max_drop_rebound", 0.20)),
            min_memory_reduction_ratio=float(data.get("min_memory_reduction_ratio", 0.15)),
            max_generation_nll_drift=float(data["max_generation_nll_drift"]) if data.get("max_generation_nll_drift") is not None else None,
        )


@dataclass
class QuantizationValidationResult:
    """Outcome and detailed breakdown of quantized candidate validation."""

    passed: bool
    keep_retention: float
    drop_leak: float
    kl_drift: float
    mse_drift: float
    memory_reduction_ratio: float
    baseline_memory_bytes: int
    quantized_memory_bytes: int
    generation_nll_drift: Optional[float] = None
    failure_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "keep_retention": self.keep_retention,
            "drop_leak": self.drop_leak,
            "kl_drift": self.kl_drift,
            "mse_drift": self.mse_drift,
            "generation_nll_drift": self.generation_nll_drift,
            "memory_reduction_ratio": self.memory_reduction_ratio,
            "baseline_memory_bytes": self.baseline_memory_bytes,
            "quantized_memory_bytes": self.quantized_memory_bytes,
            "failure_reasons": list(self.failure_reasons),
            "metrics": dict(self.metrics),
        }


def validate_quantized_candidate(
    baseline_model: nn.Module,
    quantized_model: nn.Module,
    keep_samples: Sequence[Any],
    drop_samples: Optional[Sequence[Any]] = None,
    tokenizer: Optional[Any] = None,
    thresholds: Optional[QuantizationValidationThresholds] = None,
    device: Optional[Union[str, torch.device]] = None,
    raise_on_failure: bool = True,
) -> QuantizationValidationResult:
    """Validates a quantized candidate against declared drift limits and objectives.
    
    Evaluates:
    - KEEP task retention and KL/MSE drift.
    - DROP suppression maintenance (rebound prevention).
    - Memory footprint reduction.
    - Generation correctness (NLL drift).
    
    Rejects candidates exceeding thresholds by raising QuantizationDriftExceededError
    when raise_on_failure=True.
    """
    if thresholds is None:
        thresholds = QuantizationValidationThresholds()

    if device is not None:
        target_device = torch.device(device)
    else:
        try:
            target_device = next(quantized_model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")

    baseline_model.eval()
    quantized_model.eval()

    failure_reasons: list[str] = []

    # 1. Memory reduction check
    base_bytes = calculate_model_memory_bytes(baseline_model)
    quant_bytes = calculate_model_memory_bytes(quantized_model)
    if base_bytes > 0:
        mem_reduction = max(0.0, (base_bytes - quant_bytes) / base_bytes)
    else:
        mem_reduction = 0.0

    if mem_reduction < thresholds.min_memory_reduction_ratio:
        failure_reasons.append(
            f"Memory reduction {mem_reduction:.4f} < min threshold {thresholds.min_memory_reduction_ratio:.4f} "
            f"(baseline={base_bytes} B, quantized={quant_bytes} B)"
        )

    # 2. KEEP retention & drift check
    keep_batches = _prepare_calibration_inputs(keep_samples, tokenizer=tokenizer, device=target_device)
    kl_scores: list[float] = []
    mse_scores: list[float] = []
    match_scores: list[float] = []
    nll_drifts: list[float] = []

    with torch.no_grad():
        for b in keep_batches:
            base_logits = _forward_model_logits(baseline_model, b)
            quant_logits = _forward_model_logits(quantized_model, b)

            # Top-1 token prediction match
            base_pred = base_logits.argmax(dim=-1)
            quant_pred = quant_logits.argmax(dim=-1)
            match = (base_pred == quant_pred).float().mean().item()
            match_scores.append(match)

            # KL divergence and MSE
            p_base = F.softmax(base_logits.float(), dim=-1)
            log_p_quant = F.log_softmax(quant_logits.float(), dim=-1)
            kl = F.kl_div(log_p_quant, p_base, log_target=False, reduction="batchmean").item()
            mse = F.mse_loss(quant_logits.float(), base_logits.float()).item()

            kl_scores.append(kl)
            mse_scores.append(mse)

            # Sequence NLL if targets exist
            if isinstance(b, dict) and "input_ids" in b:
                ids = b["input_ids"]
                if ids.shape[-1] > 1:
                    t_in = ids[:, :-1]
                    t_target = ids[:, 1:]
                    base_loss = F.cross_entropy(base_logits[:, :-1, :].reshape(-1, base_logits.shape[-1]), t_target.reshape(-1)).item()
                    quant_loss = F.cross_entropy(quant_logits[:, :-1, :].reshape(-1, quant_logits.shape[-1]), t_target.reshape(-1)).item()
                    nll_drifts.append(abs(quant_loss - base_loss))

    avg_keep_retention = float(sum(match_scores) / max(len(match_scores), 1))
    avg_kl = float(sum(kl_scores) / max(len(kl_scores), 1))
    avg_mse = float(sum(mse_scores) / max(len(mse_scores), 1))
    avg_nll_drift = float(sum(nll_drifts) / max(len(nll_drifts), 1)) if nll_drifts else None

    # Check for non-finite values
    if not math.isfinite(avg_keep_retention) or not math.isfinite(avg_kl) or not math.isfinite(avg_mse):
        failure_reasons.append("Non-finite values encountered in KEEP retention / drift metrics")

    if avg_keep_retention < thresholds.min_keep_retention:
        failure_reasons.append(
            f"KEEP retention {avg_keep_retention:.4f} < min threshold {thresholds.min_keep_retention:.4f}"
        )
    if avg_kl > thresholds.max_kl_drift:
        failure_reasons.append(
            f"KL drift {avg_kl:.4f} > max threshold {thresholds.max_kl_drift:.4f}"
        )
    if avg_mse > thresholds.max_mse_drift:
        failure_reasons.append(
            f"MSE drift {avg_mse:.4f} > max threshold {thresholds.max_mse_drift:.4f}"
        )
    if (
        thresholds.max_generation_nll_drift is not None
        and avg_nll_drift is not None
        and avg_nll_drift > thresholds.max_generation_nll_drift
    ):
        failure_reasons.append(
            f"Generation NLL drift {avg_nll_drift:.4f} > max threshold {thresholds.max_generation_nll_drift:.4f}"
        )

    # 3. DROP objectives & rebound check
    drop_leak_score = 0.0
    if drop_samples:
        drop_batches = _prepare_calibration_inputs(drop_samples, tokenizer=tokenizer, device=target_device)
        rebound_scores: list[float] = []

        with torch.no_grad():
            for b in drop_batches:
                base_logits = _forward_model_logits(baseline_model, b)
                quant_logits = _forward_model_logits(quantized_model, b)

                # Rebound measured as top-1 probability shift towards original unsuppressed behavior
                # or divergence from post-surgery suppression state
                base_prob = F.softmax(base_logits.float(), dim=-1)
                quant_prob = F.softmax(quant_logits.float(), dim=-1)
                prob_diff = torch.clamp(quant_prob - base_prob, min=0.0).amax(dim=-1).mean().item()
                rebound_scores.append(prob_diff)

        drop_leak_score = float(sum(rebound_scores) / max(len(rebound_scores), 1))
        if not math.isfinite(drop_leak_score):
            failure_reasons.append("Non-finite value encountered in DROP rebound metric")

        if drop_leak_score > thresholds.max_drop_rebound:
            failure_reasons.append(
                f"DROP rebound {drop_leak_score:.4f} > max threshold {thresholds.max_drop_rebound:.4f}"
            )

    passed = len(failure_reasons) == 0
    metrics = {
        "keep_retention": avg_keep_retention,
        "kl_drift": avg_kl,
        "mse_drift": avg_mse,
        "drop_leak": drop_leak_score,
        "memory_reduction_ratio": mem_reduction,
        "baseline_bytes": float(base_bytes),
        "quantized_bytes": float(quant_bytes),
    }
    if avg_nll_drift is not None:
        metrics["generation_nll_drift"] = avg_nll_drift

    result = QuantizationValidationResult(
        passed=passed,
        keep_retention=avg_keep_retention,
        drop_leak=drop_leak_score,
        kl_drift=avg_kl,
        mse_drift=avg_mse,
        generation_nll_drift=avg_nll_drift,
        memory_reduction_ratio=mem_reduction,
        baseline_memory_bytes=base_bytes,
        quantized_memory_bytes=quant_bytes,
        failure_reasons=failure_reasons,
        metrics=metrics,
    )

    if not passed and raise_on_failure:
        raise QuantizationDriftExceededError(
            f"Quantized candidate failed validation drift limits: {'; '.join(failure_reasons)}"
        )

    return result


# ---------------------------------------------------------------------------
# Export, Reload, and Parity Verification
# ---------------------------------------------------------------------------

def export_quantized_checkpoint(
    model: nn.Module,
    export_dir: Union[str, Path],
    config_or_plan: Union[QuantizationConfig, MixedPrecisionPlan],
    calibration_binding: CalibrationBinding,
    provenance: Optional[Any] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Exports a quantized model checkpoint with weights, quantization metadata, and provenance.
    
    Weights are saved using torch.save for strict weights_only=True compatibility.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_file = out_dir / "quantized_model.pt"
    metadata_file = out_dir / "quantization_metadata.json"

    # Save state dict
    torch.save(model.state_dict(), weights_file)

    # Collect module schemas
    modules_meta: dict[str, Any] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QuantizedLinear):
            modules_meta[name] = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bits": mod.bits,
                "symmetric": mod.symmetric,
                "granularity": mod.granularity,
                "is_packed": mod.is_packed,
                "original_shape": list(mod.original_shape),
                "has_bias": mod.bias is not None,
                "quantize_activations": mod.quantize_activations,
                "act_bits": mod.act_bits,
                "act_symmetric": mod.act_symmetric,
            }

    plan_meta: Optional[dict[str, Any]] = None
    cfg_meta: Optional[dict[str, Any]] = None
    if isinstance(config_or_plan, MixedPrecisionPlan):
        plan_meta = config_or_plan.to_dict()
    elif isinstance(config_or_plan, QuantizationConfig):
        cfg_meta = config_or_plan.to_dict()

    meta_payload = {
        "schema_version": QUANTIZATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_binding": calibration_binding.to_dict(),
        "quantization_config": cfg_meta,
        "mixed_precision_plan": plan_meta,
        "modules": modules_meta,
        "memory_bytes": calculate_model_memory_bytes(model),
        "extra_metadata": dict(metadata or {}),
    }

    metadata_file.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")

    written = {
        "weights": str(weights_file),
        "metadata": str(metadata_file),
    }

    if provenance is not None:
        prov_file = out_dir / "provenance.yaml"
        if hasattr(provenance, "save"):
            provenance.save(prov_file)
        elif hasattr(provenance, "to_dict"):
            import yaml
            prov_file.write_text(yaml.safe_dump(provenance.to_dict()), encoding="utf-8")
        written["provenance"] = str(prov_file)

    return written


def load_quantized_checkpoint(
    export_dir: Union[str, Path],
    base_model: Optional[nn.Module] = None,
) -> tuple[Union[nn.Module, dict[str, torch.Tensor]], dict[str, Any]]:
    """Loads an exported quantized checkpoint with weights_only=True and re-materializes modules."""
    in_dir = Path(export_dir)
    weights_file = in_dir / "quantized_model.pt"
    metadata_file = in_dir / "quantization_metadata.json"

    if not weights_file.exists():
        raise FileNotFoundError(f"Quantized weights not found: {weights_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Quantization metadata not found: {metadata_file}")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != QUANTIZATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported metadata schema_version: {metadata.get('schema_version')}, expected {QUANTIZATION_SCHEMA_VERSION}"
        )

    # Strictly load tensor artifacts with weights_only=True
    state_dict = load_tensor_artifact(weights_file)

    if base_model is None:
        return state_dict, metadata

    # Re-materialize QuantizedLinear modules into base_model architecture
    modules_meta = metadata.get("modules", {})
    for mod_name, mod_info in modules_meta.items():
        q_cfg = QuantizationConfig(
            bits=mod_info["bits"],
            symmetric=mod_info["symmetric"],
            granularity=mod_info["granularity"],
            quantize_activations=mod_info.get("quantize_activations", False),
            act_bits=mod_info.get("act_bits", 8),
            act_symmetric=mod_info.get("act_symmetric", False),
            pack=mod_info.get("is_packed", False),
        )
        qmod = QuantizedLinear.create_empty(
            in_features=mod_info["in_features"],
            out_features=mod_info["out_features"],
            config=q_cfg,
            is_packed=mod_info["is_packed"],
            has_bias=mod_info.get("has_bias", True),
        )
        safe_nested_setattr(base_model, mod_name, qmod)

    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()
    return base_model, metadata


def verify_export_reload_parity(
    original_model: nn.Module,
    reloaded_model: nn.Module,
    test_inputs: Sequence[Any],
    tokenizer: Optional[Any] = None,
    tolerance: float = 1e-4,
) -> bool:
    """Verifies that an exported and reloaded quantized model exhibits numerical and generation parity."""
    original_model.eval()
    reloaded_model.eval()

    device = None
    try:
        device = next(original_model.parameters()).device
    except StopIteration:
        pass

    batches = _prepare_calibration_inputs(test_inputs, tokenizer=tokenizer, device=device)

    with torch.no_grad():
        for i, b in enumerate(batches):
            orig_logits = _forward_model_logits(original_model, b)
            reload_logits = _forward_model_logits(reloaded_model, b)

            max_diff = torch.max(torch.abs(orig_logits - reload_logits)).item()
            if max_diff > tolerance:
                raise ReloadParityError(
                    f"Export/reload parity verification failed on batch {i}: "
                    f"maximum logit difference {max_diff:.6e} > tolerance {tolerance:.6e}"
                )

    return True
