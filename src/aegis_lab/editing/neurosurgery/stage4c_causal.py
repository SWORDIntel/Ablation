from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import get_attention_adapter, get_mlp_adapter
from .common import LOG, get_layers, nested_getattr


class NeurosurgeryCausalError(Exception):
    """Base exception for causal neurosurgery operations."""
    pass


class MemoryLimitExceededError(NeurosurgeryCausalError):
    """Raised when activation cache exceeds its configured memory ceiling."""
    pass


class HookResolutionError(NeurosurgeryCausalError):
    """Raised when a designated component or hook target cannot be resolved."""
    pass


class FeatureAdapterCompatibilityError(NeurosurgeryCausalError):
    """Raised when a feature/SAE adapter is incompatible with model or activation dimensions."""
    pass


class InteractionRegressionError(NeurosurgeryCausalError):
    """Raised or reported when combining edits induces severe interaction regressions."""
    pass


@dataclass(frozen=True)
class ComponentTarget:
    """Designates a model component for capture, patching, or intervention."""

    layer_idx: int = 0
    component_type: str = "layer"
    sub_idx: Optional[int] = None
    module_path: Optional[str] = None
    head_dim: Optional[int] = None
    hook_type: Optional[str] = None

    def __str__(self) -> str:
        if self.module_path:
            return f"path:{self.module_path}"
        if self.sub_idx is not None:
            return f"layer.{self.layer_idx}.{self.component_type}.{self.sub_idx}"
        return f"layer.{self.layer_idx}.{self.component_type}"


def parse_component_target(spec: Union[str, ComponentTarget]) -> ComponentTarget:
    """Parses a component specifier string or returns an existing ComponentTarget."""
    if isinstance(spec, ComponentTarget):
        return spec
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"Invalid component target specifier: {spec!r}")

    s = spec.strip()
    if s.startswith("path:"):
        return ComponentTarget(module_path=s[5:], component_type="custom")

    parts = s.split(".")
    if len(parts) >= 2 and parts[0] in ("layer", "layers"):
        try:
            layer_idx = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Could not parse layer index from {s!r}") from exc

        if len(parts) == 2:
            return ComponentTarget(layer_idx=layer_idx, component_type="layer")
        elif len(parts) == 3:
            comp = parts[2].lower()
            return ComponentTarget(layer_idx=layer_idx, component_type=comp)
        elif len(parts) == 4:
            comp = parts[2].lower()
            try:
                sub_idx = int(parts[3])
            except ValueError as exc:
                raise ValueError(f"Could not parse sub-index from {s!r}") from exc
            if comp in ("head", "attention_head"):
                return ComponentTarget(layer_idx=layer_idx, component_type="attention_head", sub_idx=sub_idx)
            elif comp in ("channel", "mlp_channel"):
                return ComponentTarget(layer_idx=layer_idx, component_type="mlp_channel", sub_idx=sub_idx)
            return ComponentTarget(layer_idx=layer_idx, component_type=comp, sub_idx=sub_idx)

    return ComponentTarget(module_path=s, component_type="custom")


def resolve_component_module(
    model: nn.Module,
    target: Union[str, ComponentTarget],
    head_dim: Optional[int] = None,
) -> tuple[nn.Module, str, dict[str, Any]]:
    """Resolves a component target to (module, hook_type, metadata).

    hook_type is 'pre' for forward_pre_hook or 'post' for forward_hook.
    """
    comp_target = parse_component_target(target)
    if comp_target.module_path:
        try:
            mod = nested_getattr(model, comp_target.module_path)
            hook_t = comp_target.hook_type or "post"
            return mod, hook_t, {"component_type": "custom", "target": comp_target}
        except (AttributeError, KeyError) as e:
            raise HookResolutionError(f"Cannot resolve module path {comp_target.module_path!r}") from e

    layers = None
    try:
        _, layers = get_layers(model)
    except Exception:
        for attr in ("layers", "model.layers", "transformer.h", "model.transformer.h"):
            try:
                val = nested_getattr(model, attr)
                if isinstance(val, (nn.ModuleList, list, tuple)):
                    layers = val
                    break
            except Exception:
                pass

    if layers is None:
        raise HookResolutionError(f"Could not locate transformer layers in model of type {type(model).__name__}")
    if comp_target.layer_idx < 0 or comp_target.layer_idx >= len(layers):
        raise HookResolutionError(
            f"Layer index {comp_target.layer_idx} out of range [0, {len(layers)})"
        )

    layer = layers[comp_target.layer_idx]
    ctype = comp_target.component_type.lower()

    if ctype in ("layer", "residual_output", "layer_output", "output"):
        return layer, "post", {"target": comp_target}
    elif ctype in ("residual_input", "layer_input", "input"):
        return layer, "pre", {"target": comp_target}
    elif ctype in ("attention", "attn"):
        attn = getattr(layer, "self_attn", getattr(layer, "attention", getattr(layer, "attn", None)))
        if attn is None:
            raise HookResolutionError(f"Layer {comp_target.layer_idx} has no self_attn / attention submodule")
        return attn, "post", {"target": comp_target}
    elif ctype == "attention_head":
        try:
            adapter = get_attention_adapter(model)
            packs = adapter.attention_packs(model)
            if comp_target.layer_idx < len(packs):
                pack = packs[comp_target.layer_idx]
                h_dim = comp_target.head_dim or head_dim or pack.head_dim
                return pack.o, "pre", {"head_idx": comp_target.sub_idx, "head_dim": h_dim, "target": comp_target}
        except Exception:
            pass

        attn = getattr(layer, "self_attn", getattr(layer, "attention", getattr(layer, "attn", None)))
        if attn is not None and hasattr(attn, "o_proj"):
            h_dim = comp_target.head_dim or head_dim or getattr(getattr(model, "config", None), "head_dim", None)
            if h_dim is None and hasattr(attn.o_proj, "weight"):
                num_heads = getattr(getattr(model, "config", None), "num_attention_heads", None) or getattr(attn, "num_heads", None)
                if num_heads:
                    h_dim = attn.o_proj.weight.shape[1] // num_heads
            return attn.o_proj, "pre", {"head_idx": comp_target.sub_idx, "head_dim": h_dim, "target": comp_target}
        elif attn is not None:
            return attn, "post", {"head_idx": comp_target.sub_idx, "target": comp_target}
        raise HookResolutionError(f"Layer {comp_target.layer_idx} attention head cannot be resolved")

    elif ctype in ("mlp", "feedforward"):
        mlp = getattr(layer, "mlp", getattr(layer, "feedforward", None))
        if mlp is None:
            raise HookResolutionError(f"Layer {comp_target.layer_idx} has no mlp submodule")
        return mlp, "post", {"target": comp_target}
    elif ctype in ("mlp_channel", "mlp_intermediate"):
        try:
            adapter = get_mlp_adapter(model)
            triplets = adapter.mlp_triplets(model)
            if comp_target.layer_idx < len(triplets):
                return triplets[comp_target.layer_idx].down, "pre", {
                    "channel_idx": comp_target.sub_idx,
                    "target": comp_target,
                }
        except Exception:
            pass

        mlp = getattr(layer, "mlp", getattr(layer, "feedforward", None))
        if mlp is not None and hasattr(mlp, "down_proj"):
            return mlp.down_proj, "pre", {"channel_idx": comp_target.sub_idx, "target": comp_target}
        elif mlp is not None:
            return mlp, "post", {"channel_idx": comp_target.sub_idx, "target": comp_target}
        raise HookResolutionError(f"Layer {comp_target.layer_idx} MLP channel cannot be resolved")

    raise HookResolutionError(f"Unsupported component type {comp_target.component_type!r}")


# ---------------------------------------------------------------------------
# 1. Activation Capture & Hook Framework
# ---------------------------------------------------------------------------


class HookHandle:
    """Reversible hook handle wrapper with context manager lifecycle."""

    def __init__(self, raw_handles: Union[Any, list[Any]]):
        if isinstance(raw_handles, (list, tuple)):
            self._raw = list(raw_handles)
        else:
            self._raw = [raw_handles]
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    def remove(self) -> None:
        if self._active:
            for h in self._raw:
                if hasattr(h, "remove"):
                    h.remove()
            self._active = False

    def __enter__(self) -> "HookHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.remove()
        return False


class ActivationCache:
    """Bounded-memory container for captured model activations."""

    def __init__(
        self,
        max_bytes: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        detach: bool = True,
        to_dtype: Optional[torch.dtype] = None,
        token_indices: Optional[Union[int, Sequence[int]]] = None,
        clone: bool = True,
    ):
        self.max_bytes = max_bytes
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detach = detach
        self.to_dtype = to_dtype
        self.token_indices = token_indices
        self.clone = clone
        self._storage: dict[str, torch.Tensor] = {}
        self._current_bytes: int = 0

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    def store(
        self,
        key: str,
        tensor: torch.Tensor,
        token_indices: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        t = tensor
        if self.detach:
            t = t.detach()
        if self.clone:
            t = t.clone()

        tokens = token_indices if token_indices is not None else self.token_indices
        if tokens is not None and t.ndim >= 3:
            if isinstance(tokens, int):
                t = t[:, -1:, :] if tokens == -1 else t[:, tokens : tokens + 1, :]
            elif isinstance(tokens, (list, tuple)):
                t = t[:, list(tokens), :]

        if self.device is not None:
            t = t.to(device=self.device)
        if self.to_dtype is not None:
            t = t.to(dtype=self.to_dtype)

        new_bytes = t.numel() * t.element_size()
        old_bytes = 0
        if key in self._storage:
            old_bytes = self._storage[key].numel() * self._storage[key].element_size()

        projected = self._current_bytes - old_bytes + new_bytes
        if self.max_bytes is not None and projected > self.max_bytes:
            raise MemoryLimitExceededError(
                f"Storing activation {key!r} ({new_bytes} bytes) exceeds memory ceiling of "
                f"{self.max_bytes} bytes (current={self._current_bytes} bytes, projected={projected} bytes)"
            )

        self._storage[key] = t
        self._current_bytes = projected

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._storage[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.store(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._storage

    def get(self, key: str, default: Any = None) -> Any:
        return self._storage.get(key, default)

    def clear(self) -> None:
        self._storage.clear()
        self._current_bytes = 0

    def keys(self):
        return self._storage.keys()

    def values(self):
        return self._storage.values()

    def items(self):
        return self._storage.items()

    def __len__(self) -> int:
        return len(self._storage)


def make_capture_hook(
    cache: ActivationCache,
    key: str,
    token_indices: Optional[Union[int, Sequence[int]]] = None,
    is_pre: bool = False,
) -> Callable:
    """Constructs a forward hook that saves activations into an ActivationCache."""
    if is_pre:
        def pre_hook(module, args):
            tensor = args[0] if isinstance(args, tuple) else args
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            cache.store(key, tensor, token_indices=token_indices)
            return None
        return pre_hook
    else:
        def post_hook(module, inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            cache.store(key, tensor, token_indices=token_indices)
            return None
        return post_hook


def attach_capture_hooks(
    model: nn.Module,
    targets: Sequence[Union[str, ComponentTarget]],
    cache: ActivationCache,
    token_indices: Optional[Union[int, Sequence[int]]] = None,
) -> HookHandle:
    """Attaches capture hooks to a model for all designated targets."""
    handles = []
    for target in targets:
        comp_target = parse_component_target(target) if isinstance(target, str) else target
        module, hook_type, _ = resolve_component_module(model, comp_target)
        key = str(target)
        hook_fn = make_capture_hook(cache, key, token_indices=token_indices, is_pre=(hook_type == "pre"))
        if hook_type == "pre":
            h = module.register_forward_pre_hook(hook_fn)
        else:
            h = module.register_forward_hook(hook_fn)
        handles.append(h)
    return HookHandle(handles)


@dataclass
class WorkloadPair:
    """A paired clean and corrupted input with associated target tokens."""

    clean_input: Union[str, dict[str, Any], torch.Tensor]
    corrupted_input: Union[str, dict[str, Any], torch.Tensor]
    clean_target: Optional[Union[int, str]] = None
    corrupted_target: Optional[Union[int, str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PairedWorkload:
    """Ordered collection of clean and corrupted workload pairs."""

    def __init__(self, pairs: Sequence[WorkloadPair]):
        if not pairs:
            raise ValueError("Workload cannot be empty")
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> WorkloadPair:
        return self.pairs[idx]

    def __iter__(self) -> Iterable[WorkloadPair]:
        return iter(self.pairs)

    @classmethod
    def from_prompts(
        cls,
        clean_prompts: Sequence[str],
        corrupted_prompts: Sequence[str],
        clean_targets: Optional[Sequence[Union[int, str]]] = None,
        corrupted_targets: Optional[Sequence[Union[int, str]]] = None,
        metadata: Optional[Sequence[dict[str, Any]]] = None,
    ) -> "PairedWorkload":
        if len(clean_prompts) != len(corrupted_prompts):
            raise ValueError(
                f"Mismatch: {len(clean_prompts)} clean prompts vs {len(corrupted_prompts)} corrupted prompts"
            )
        pairs = []
        for i in range(len(clean_prompts)):
            c_tar = clean_targets[i] if clean_targets is not None else None
            cr_tar = corrupted_targets[i] if corrupted_targets is not None else None
            meta = metadata[i] if metadata is not None else {}
            pairs.append(WorkloadPair(clean_prompts[i], corrupted_prompts[i], c_tar, cr_tar, meta))
        return cls(pairs)


def _forward_model(model: nn.Module, inputs: Any, tokenizer: Any = None, device: Any = None) -> torch.Tensor:
    """Executes model and returns logits tensor."""
    if isinstance(inputs, str):
        if tokenizer is None:
            raise ValueError("Tokenizer required when input is text string")
        enc = tokenizer(inputs, return_tensors="pt")
        target_device = device if device is not None else getattr(model, "device", None)
        if target_device is not None:
            enc = {k: v.to(target_device) for k, v in enc.items()}
        out = model(**enc)
    elif isinstance(inputs, dict):
        target_device = device if device is not None else getattr(model, "device", None)
        if target_device is not None:
            inputs = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        out = model(**inputs)
    elif isinstance(inputs, torch.Tensor):
        target_device = device if device is not None else getattr(model, "device", None)
        if target_device is not None:
            inputs = inputs.to(target_device)
        out = model(inputs)
    else:
        out = model(inputs)

    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class PairedWorkloadRunner:
    """Executes paired workloads and captures activations."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def resolve_token_id(self, target: Optional[Union[int, str]]) -> Optional[int]:
        if target is None:
            return None
        if isinstance(target, int):
            return target
        if isinstance(target, str):
            if self.tokenizer is None:
                raise ValueError(f"Cannot resolve token string {target!r} without tokenizer")
            enc = self.tokenizer.encode(target, add_special_tokens=False)
            if not enc:
                raise ValueError(f"Target string {target!r} tokenized to empty list")
            return enc[-1]
        raise ValueError(f"Invalid target specifier {target!r}")

    def run_clean(
        self,
        workload: PairedWorkload,
        capture_components: Optional[Sequence[Union[str, ComponentTarget]]] = None,
        max_bytes: Optional[int] = None,
        token_indices: Optional[Union[int, Sequence[int]]] = None,
    ) -> tuple[list[torch.Tensor], ActivationCache]:
        cache = ActivationCache(max_bytes=max_bytes, token_indices=token_indices)
        handles = attach_capture_hooks(self.model, capture_components or [], cache, token_indices=token_indices)
        logits_list = []
        try:
            with torch.inference_mode():
                for pair in workload:
                    logits = _forward_model(self.model, pair.clean_input, self.tokenizer, self.device)
                    logits_list.append(logits.detach().cpu())
        finally:
            handles.remove()
        return logits_list, cache

    def run_corrupted(
        self,
        workload: PairedWorkload,
        capture_components: Optional[Sequence[Union[str, ComponentTarget]]] = None,
        max_bytes: Optional[int] = None,
        token_indices: Optional[Union[int, Sequence[int]]] = None,
    ) -> tuple[list[torch.Tensor], ActivationCache]:
        cache = ActivationCache(max_bytes=max_bytes, token_indices=token_indices)
        handles = attach_capture_hooks(self.model, capture_components or [], cache, token_indices=token_indices)
        logits_list = []
        try:
            with torch.inference_mode():
                for pair in workload:
                    logits = _forward_model(self.model, pair.corrupted_input, self.tokenizer, self.device)
                    logits_list.append(logits.detach().cpu())
        finally:
            handles.remove()
        return logits_list, cache


# ---------------------------------------------------------------------------
# 4. Runtime Interventions (Defined early for patching integration)
# ---------------------------------------------------------------------------


def apply_subselected_transform(
    tensor: torch.Tensor,
    transform_fn: Callable[[torch.Tensor], torch.Tensor],
    token_indices: Optional[Union[int, Sequence[int]]] = None,
    channel_indices: Optional[Union[int, Sequence[int]]] = None,
    head_indices: Optional[Union[int, Sequence[int]]] = None,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Applies a transformation function to designated tokens, channels, or heads."""
    feature_slice: Optional[list[int]] = None
    if head_indices is not None and head_dim is not None:
        heads = [head_indices] if isinstance(head_indices, int) else list(head_indices)
        dim_indices = []
        for h in heads:
            dim_indices.extend(range(h * head_dim, (h + 1) * head_dim))
        feature_slice = dim_indices
    elif channel_indices is not None:
        feature_slice = [channel_indices] if isinstance(channel_indices, int) else list(channel_indices)

    if token_indices is None:
        if feature_slice is None:
            return transform_fn(tensor)
        res = tensor.clone()
        res[..., feature_slice] = transform_fn(res[..., feature_slice])
        return res
    else:
        if tensor.ndim < 3:
            res = tensor.clone()
            if feature_slice is None:
                return transform_fn(res)
            res[..., feature_slice] = transform_fn(res[..., feature_slice])
            return res

        res = tensor.clone()
        t_list = [token_indices] if isinstance(token_indices, int) else list(token_indices)
        for t_idx in t_list:
            if feature_slice is None:
                res[:, t_idx, ...] = transform_fn(res[:, t_idx, ...])
            else:
                res[:, t_idx, feature_slice] = transform_fn(res[:, t_idx, feature_slice])
        return res


class RuntimeIntervention:
    """Base class for reversible runtime interventions."""

    target: Union[str, ComponentTarget]
    token_indices: Optional[Union[int, Sequence[int]]] = None
    channel_indices: Optional[Union[int, Sequence[int]]] = None
    head_indices: Optional[Union[int, Sequence[int]]] = None
    head_dim: Optional[int] = None

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class ScaleIntervention(RuntimeIntervention):
    """Multiplies targeted activation by a scale factor (e.g. 0.0 for ablation)."""

    target: Union[str, ComponentTarget]
    scale: float = 0.0
    token_indices: Optional[Union[int, Sequence[int]]] = None
    channel_indices: Optional[Union[int, Sequence[int]]] = None
    head_indices: Optional[Union[int, Sequence[int]]] = None
    head_dim: Optional[int] = None

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return apply_subselected_transform(
            tensor,
            lambda x: x * self.scale,
            token_indices=self.token_indices,
            channel_indices=self.channel_indices,
            head_indices=self.head_indices,
            head_dim=self.head_dim,
        )


@dataclass
class ClampIntervention(RuntimeIntervention):
    """Clamps activations elementwise and/or caps L2 norm."""

    target: Union[str, ComponentTarget]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    max_norm: Optional[float] = None
    token_indices: Optional[Union[int, Sequence[int]]] = None
    channel_indices: Optional[Union[int, Sequence[int]]] = None
    head_indices: Optional[Union[int, Sequence[int]]] = None
    head_dim: Optional[int] = None

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        def fn(x: torch.Tensor) -> torch.Tensor:
            res = x
            if self.min_val is not None or self.max_val is not None:
                res = torch.clamp(res, min=self.min_val, max=self.max_val)
            if self.max_norm is not None:
                norm = res.norm(dim=-1, keepdim=True)
                scale = torch.clamp(self.max_norm / norm.clamp_min(1e-8), max=1.0)
                res = res * scale
            return res

        return apply_subselected_transform(
            tensor,
            fn,
            token_indices=self.token_indices,
            channel_indices=self.channel_indices,
            head_indices=self.head_indices,
            head_dim=self.head_dim,
        )


@dataclass
class ProjectionIntervention(RuntimeIntervention):
    """Projects activations along or away from a direction vector or subspace."""

    target: Union[str, ComponentTarget]
    direction: torch.Tensor
    strength: float = 1.0
    mode: str = "remove_projection"
    token_indices: Optional[Union[int, Sequence[int]]] = None
    channel_indices: Optional[Union[int, Sequence[int]]] = None
    head_indices: Optional[Union[int, Sequence[int]]] = None
    head_dim: Optional[int] = None

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        def fn(x: torch.Tensor) -> torch.Tensor:
            d = self.direction.to(device=x.device, dtype=x.dtype)
            if self.mode == "subspace" or d.ndim == 2:
                proj = (x @ d) @ d.T
                return x - self.strength * proj
            elif self.mode == "remove_projection":
                norm = torch.norm(d, p=2)
                if norm < 1e-8:
                    return x
                d_unit = d / norm
                dot = (x * d_unit).sum(dim=-1, keepdim=True)
                proj = dot * d_unit
                return x - self.strength * proj
            elif self.mode == "steer":
                norm = torch.norm(d, p=2)
                d_unit = d / norm if norm > 1e-8 else d
                return x + self.strength * d_unit
            raise ValueError(f"Unknown projection mode {self.mode!r}")

        return apply_subselected_transform(
            tensor,
            fn,
            token_indices=self.token_indices,
            channel_indices=self.channel_indices,
            head_indices=self.head_indices,
            head_dim=self.head_dim,
        )


@dataclass
class ReplacementIntervention(RuntimeIntervention):
    """Replaces targeted activation slices with pre-cached tensors."""

    target: Union[str, ComponentTarget]
    replacement: torch.Tensor
    token_indices: Optional[Union[int, Sequence[int]]] = None
    channel_indices: Optional[Union[int, Sequence[int]]] = None
    head_indices: Optional[Union[int, Sequence[int]]] = None
    head_dim: Optional[int] = None

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        rep = self.replacement.to(device=tensor.device, dtype=tensor.dtype)
        res = tensor.clone()

        feature_slice: Optional[list[int]] = None
        if self.head_indices is not None and self.head_dim is not None:
            heads = [self.head_indices] if isinstance(self.head_indices, int) else list(self.head_indices)
            dim_indices = []
            for h in heads:
                dim_indices.extend(range(h * self.head_dim, (h + 1) * self.head_dim))
            feature_slice = dim_indices
        elif self.channel_indices is not None:
            feature_slice = [self.channel_indices] if isinstance(self.channel_indices, int) else list(self.channel_indices)

        def get_rep_features(r: torch.Tensor) -> torch.Tensor:
            if feature_slice is not None and r.shape[-1] >= max(feature_slice) + 1:
                return r[..., feature_slice]
            return r

        if self.token_indices is None:
            rep_feat = get_rep_features(rep)
            if feature_slice is None:
                if rep_feat.shape == res.shape:
                    return rep_feat
                return rep_feat.expand_as(res)
            else:
                if rep_feat.ndim == res.ndim and rep_feat.shape[1] == res.shape[1]:
                    res[..., feature_slice] = rep_feat
                else:
                    res[..., feature_slice] = rep_feat.expand_as(res[..., feature_slice])
                return res
        else:
            t_list = [self.token_indices] if isinstance(self.token_indices, int) else list(self.token_indices)
            if res.ndim < 3:
                rep_feat = get_rep_features(rep)
                if feature_slice is None:
                    return rep_feat.expand_as(res)
                res[..., feature_slice] = rep_feat.expand_as(res[..., feature_slice])
                return res

            for t_idx in t_list:
                if rep.ndim >= 3:
                    if rep.shape[1] == 1:
                        rep_t = rep[:, 0, ...]
                    elif t_idx < rep.shape[1] or t_idx == -1:
                        rep_t = rep[:, t_idx, ...]
                    else:
                        rep_t = rep[:, -1, ...]
                else:
                    rep_t = rep

                rep_t_feat = get_rep_features(rep_t)
                if feature_slice is None:
                    res[:, t_idx, ...] = rep_t_feat.expand_as(res[:, t_idx, ...])
                else:
                    res[:, t_idx, feature_slice] = rep_t_feat.expand_as(res[:, t_idx, feature_slice])
            return res


class RuntimeInterventionContext:
    """Context manager for reversible runtime interventions without modifying base checkpoint weights."""

    def __init__(
        self,
        model: nn.Module,
        interventions: Sequence[RuntimeIntervention],
    ):
        self.model = model
        self.interventions = list(interventions)
        self.handles: list[HookHandle] = []
        self._param_signatures: dict[str, tuple[torch.Size, float]] = {}
        self.is_active = False

    def _record_param_signatures(self) -> None:
        self._param_signatures = {}
        for name, p in self.model.named_parameters():
            with torch.no_grad():
                self._param_signatures[name] = (p.shape, float(p.norm().item()))

    def verify_weights_unmodified(self) -> bool:
        if not self._param_signatures:
            return True
        for name, p in self.model.named_parameters():
            if name not in self._param_signatures:
                continue
            shape, norm = self._param_signatures[name]
            if p.shape != shape:
                raise RuntimeError(f"Weight {name!r} shape altered from {shape} to {p.shape}")
            current_norm = float(p.norm().item())
            if not math.isclose(current_norm, norm, abs_tol=1e-6, rel_tol=1e-5):
                raise RuntimeError(f"Weight {name!r} modified: initial norm {norm}, current {current_norm}")
        return True

    def attach(self) -> None:
        if self.is_active:
            return
        self._record_param_signatures()
        self.handles = []
        for interv in self.interventions:
            module, hook_type, meta = resolve_component_module(self.model, interv.target, head_dim=interv.head_dim)
            if interv.head_dim is None and meta.get("head_dim") is not None:
                interv.head_dim = meta["head_dim"]

            def make_hook(itv, htype):
                if htype == "pre":
                    def pre_hook(mod, inp):
                        x = inp[0]
                        transformed = itv.transform(x)
                        return (transformed, *inp[1:])
                    return pre_hook
                else:
                    def post_hook(mod, inp, out):
                        if isinstance(out, tuple):
                            transformed = itv.transform(out[0])
                            return (transformed, *out[1:])
                        return itv.transform(out)
                    return post_hook

            hook_fn = make_hook(interv, hook_type)
            if hook_type == "pre":
                h = module.register_forward_pre_hook(hook_fn)
            else:
                h = module.register_forward_hook(hook_fn)
            self.handles.append(HookHandle(h))
        self.is_active = True

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.is_active = False
        self.verify_weights_unmodified()

    def __enter__(self) -> "RuntimeInterventionContext":
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.remove()
        return False


def attach_runtime_intervention(model: nn.Module, intervention: RuntimeIntervention) -> HookHandle:
    """Attaches a single runtime intervention and returns its removable handle."""
    module, hook_type, meta = resolve_component_module(model, intervention.target, head_dim=intervention.head_dim)
    if intervention.head_dim is None and meta.get("head_dim") is not None:
        intervention.head_dim = meta["head_dim"]
    if hook_type == "pre":
        def pre_hook(mod, inp):
            x = inp[0]
            transformed = intervention.transform(x)
            return (transformed, *inp[1:])
        raw = module.register_forward_pre_hook(pre_hook)
    else:
        def post_hook(mod, inp, out):
            if isinstance(out, tuple):
                transformed = intervention.transform(out[0])
                return (transformed, *out[1:])
            return intervention.transform(out)
        raw = module.register_forward_hook(post_hook)
    return HookHandle(raw)


# ---------------------------------------------------------------------------
# 2. Clean/Corrupted Activation Patching
# ---------------------------------------------------------------------------


def compute_logit_diff(
    logits: torch.Tensor,
    clean_target: int,
    corrupted_target: Optional[int] = None,
    token_pos: int = -1,
) -> float:
    """Computes logit difference or target logit at the specified token position."""
    if logits.ndim == 3:
        l = logits[:, token_pos, :]
    else:
        l = logits
    clean_val = float(l[:, clean_target].mean().item())
    if corrupted_target is not None:
        corr_val = float(l[:, corrupted_target].mean().item())
        return clean_val - corr_val
    return clean_val


def compute_recovery_ratio(
    patched_val: float,
    clean_val: float,
    corrupted_val: float,
    eps: float = 1e-8,
) -> float:
    """Measures recovery ratio: (patched - corrupted) / (clean - corrupted + eps)."""
    denom = clean_val - corrupted_val
    if abs(denom) < eps:
        denom = eps if denom >= 0 else -eps
    return (patched_val - corrupted_val) / denom


def compute_target_prob(
    logits: torch.Tensor,
    target: int,
    token_pos: int = -1,
) -> float:
    """Computes softmax probability for a target token."""
    if logits.ndim == 3:
        l = logits[:, token_pos, :]
    else:
        l = logits
    probs = F.softmax(l.float(), dim=-1)
    return float(probs[:, target].mean().item())


@dataclass
class PatchingResult:
    """Measurement outcome from an activation patching intervention."""

    component: str
    component_type: str
    layer_idx: Optional[int]
    sub_idx: Optional[int]
    token_pos: Optional[Union[int, list[int]]]
    clean_logit_diff: float
    corrupted_logit_diff: float
    patched_logit_diff: float
    recovery_ratio: float
    clean_target_prob: float
    corrupted_target_prob: float
    patched_target_prob: float
    target_prob_diff: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ActivationPatcher:
    """Performs clean/corrupted activation patching across components, tokens, and layers."""

    def __init__(
        self,
        model: nn.Module,
        runner: Optional[PairedWorkloadRunner] = None,
        tokenizer: Any = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.runner = runner or PairedWorkloadRunner(model, tokenizer=tokenizer, device=device)

    def run_patching(
        self,
        workload: PairedWorkload,
        components: Sequence[Union[str, ComponentTarget]],
        token_indices: Optional[Union[int, Sequence[int]]] = None,
        mode: str = "restoration",
        max_bytes: Optional[int] = None,
        head_dim: Optional[int] = None,
    ) -> list[PatchingResult]:
        """Executes activation patching across candidate components."""
        parsed_components = [parse_component_target(c) for c in components]

        # Identify unique modules for clean capture
        capture_targets = []
        seen = set()
        for c in parsed_components:
            base_spec = ComponentTarget(
                layer_idx=c.layer_idx,
                component_type=c.component_type,
                module_path=c.module_path,
                head_dim=c.head_dim or head_dim,
            )
            key = str(base_spec)
            if key not in seen:
                seen.add(key)
                capture_targets.append(base_spec)

        clean_cache = ActivationCache(max_bytes=max_bytes)
        corrupted_cache = ActivationCache(max_bytes=max_bytes)

        if mode == "restoration":
            clean_logits_list, clean_cache = self.runner.run_clean(
                workload, capture_components=capture_targets, max_bytes=max_bytes
            )
            corrupted_logits_list, _ = self.runner.run_corrupted(workload, max_bytes=max_bytes)
        else:
            corrupted_logits_list, corrupted_cache = self.runner.run_corrupted(
                workload, capture_components=capture_targets, max_bytes=max_bytes
            )
            clean_logits_list, _ = self.runner.run_clean(workload, max_bytes=max_bytes)

        results: list[PatchingResult] = []

        for comp in parsed_components:
            _, _, comp_meta = resolve_component_module(self.model, comp, head_dim=head_dim)
            h_dim = comp.head_dim or head_dim or comp_meta.get("head_dim")

            base_spec = ComponentTarget(
                layer_idx=comp.layer_idx,
                component_type=comp.component_type,
                module_path=comp.module_path,
                head_dim=h_dim,
            )
            source_cache = clean_cache if mode == "restoration" else corrupted_cache
            raw_act = source_cache[str(base_spec)]

            ch_idx = comp.sub_idx if comp.component_type == "mlp_channel" else None
            hd_idx = comp.sub_idx if comp.component_type == "attention_head" else None

            interv = ReplacementIntervention(
                target=base_spec,
                replacement=raw_act,
                token_indices=token_indices,
                channel_indices=ch_idx,
                head_indices=hd_idx,
                head_dim=h_dim,
            )

            patched_logits_list = []
            with RuntimeInterventionContext(self.model, [interv]):
                with torch.inference_mode():
                    for pair in workload:
                        inp = pair.corrupted_input if mode == "restoration" else pair.clean_input
                        out = _forward_model(self.model, inp, self.runner.tokenizer, self.runner.device)
                        patched_logits_list.append(out.detach().cpu())

            c_lds, cr_lds, p_lds = [], [], []
            c_probs, cr_probs, p_probs = [], [], []

            for i, pair in enumerate(workload):
                c_tar = self.runner.resolve_token_id(pair.clean_target)
                cr_tar = self.runner.resolve_token_id(pair.corrupted_target)
                if c_tar is None:
                    continue

                c_ld = compute_logit_diff(clean_logits_list[i], c_tar, cr_tar)
                cr_ld = compute_logit_diff(corrupted_logits_list[i], c_tar, cr_tar)
                p_ld = compute_logit_diff(patched_logits_list[i], c_tar, cr_tar)

                c_lds.append(c_ld)
                cr_lds.append(cr_ld)
                p_lds.append(p_ld)

                c_probs.append(compute_target_prob(clean_logits_list[i], c_tar))
                cr_probs.append(compute_target_prob(corrupted_logits_list[i], c_tar))
                p_probs.append(compute_target_prob(patched_logits_list[i], c_tar))

            mean_c_ld = sum(c_lds) / max(1, len(c_lds))
            mean_cr_ld = sum(cr_lds) / max(1, len(cr_lds))
            mean_p_ld = sum(p_lds) / max(1, len(p_lds))
            rec_ratio = compute_recovery_ratio(mean_p_ld, mean_c_ld, mean_cr_ld)

            mean_c_p = sum(c_probs) / max(1, len(c_probs))
            mean_cr_p = sum(cr_probs) / max(1, len(cr_probs))
            mean_p_p = sum(p_probs) / max(1, len(p_probs))

            results.append(
                PatchingResult(
                    component=str(comp),
                    component_type=comp.component_type,
                    layer_idx=comp.layer_idx,
                    sub_idx=comp.sub_idx,
                    token_pos=token_indices if isinstance(token_indices, (int, list)) else None,
                    clean_logit_diff=mean_c_ld,
                    corrupted_logit_diff=mean_cr_ld,
                    patched_logit_diff=mean_p_ld,
                    recovery_ratio=rec_ratio,
                    clean_target_prob=mean_c_p,
                    corrupted_target_prob=mean_cr_p,
                    patched_target_prob=mean_p_p,
                    target_prob_diff=mean_p_p - mean_cr_p,
                    metadata={"mode": mode},
                )
            )

        return results


# ---------------------------------------------------------------------------
# 3. Component Causal Ranking & Controls
# ---------------------------------------------------------------------------


@dataclass
class CausalRankEntry:
    """Ranked component entry by causal effect."""

    rank: int
    component: str
    component_type: str
    layer_idx: Optional[int]
    sub_idx: Optional[int]
    causal_effect: float
    result: PatchingResult


def rank_components_by_causal_effect(
    results: Sequence[PatchingResult],
    metric: str = "recovery_ratio",
    descending: bool = True,
) -> list[CausalRankEntry]:
    """Ranks components by observed causal impact."""
    valid_metrics = ("recovery_ratio", "target_prob_diff", "patched_logit_diff")
    if metric not in valid_metrics:
        raise ValueError(f"Unknown ranking metric {metric!r}; choices are {valid_metrics}")

    def score_fn(res: PatchingResult) -> float:
        return getattr(res, metric)

    sorted_res = sorted(results, key=score_fn, reverse=descending)
    ranked = []
    for i, r in enumerate(sorted_res):
        ranked.append(
            CausalRankEntry(
                rank=i + 1,
                component=r.component,
                component_type=r.component_type,
                layer_idx=r.layer_idx,
                sub_idx=r.sub_idx,
                causal_effect=score_fn(r),
                result=r,
            )
        )
    return ranked


@dataclass
class RandomBaselineResult:
    """Outcome of random-component baseline control."""

    k_count: int
    num_trials: int
    mean_causal_effect: float
    std_causal_effect: float
    trial_effects: list[float]
    seed: int


def evaluate_random_baseline(
    patcher: ActivationPatcher,
    workload: PairedWorkload,
    all_available_components: Sequence[Union[str, ComponentTarget]],
    k_count: int,
    num_trials: int = 5,
    seed: int = 42,
    metric: str = "recovery_ratio",
    token_indices: Optional[Union[int, Sequence[int]]] = None,
    head_dim: Optional[int] = None,
) -> RandomBaselineResult:
    """Evaluates causal effect for randomly chosen components of identical cardinality."""
    rng = random.Random(seed)
    pool = list(all_available_components)
    if k_count > len(pool):
        raise ValueError(f"k_count {k_count} exceeds pool size {len(pool)}")

    trial_effects = []
    for _ in range(num_trials):
        sampled = rng.sample(pool, k_count)
        res = patcher.run_patching(workload, sampled, token_indices=token_indices, head_dim=head_dim)
        mean_eff = sum(getattr(r, metric) for r in res) / len(res)
        trial_effects.append(mean_eff)

    mean_eff = sum(trial_effects) / len(trial_effects)
    variance = sum((x - mean_eff) ** 2 for x in trial_effects) / max(1, len(trial_effects) - 1)
    std_eff = math.sqrt(variance)

    return RandomBaselineResult(
        k_count=k_count,
        num_trials=num_trials,
        mean_causal_effect=mean_eff,
        std_causal_effect=std_eff,
        trial_effects=trial_effects,
        seed=seed,
    )


@dataclass
class MagnitudeBaselineResult:
    """Outcome of magnitude-only baseline control."""

    k_count: int
    top_magnitude_components: list[str]
    magnitudes: list[float]
    causal_effect: float


def evaluate_magnitude_baseline(
    patcher: ActivationPatcher,
    workload: PairedWorkload,
    all_available_components: Sequence[Union[str, ComponentTarget]],
    k_count: int,
    clean_cache: Optional[ActivationCache] = None,
    metric: str = "recovery_ratio",
    token_indices: Optional[Union[int, Sequence[int]]] = None,
    head_dim: Optional[int] = None,
) -> MagnitudeBaselineResult:
    """Evaluates causal impact of components selected purely by activation magnitude."""
    parsed_pool = [parse_component_target(c) for c in all_available_components]

    if clean_cache is None:
        capture_targets = []
        seen = set()
        for pt in parsed_pool:
            base_spec = ComponentTarget(
                layer_idx=pt.layer_idx,
                component_type=pt.component_type,
                module_path=pt.module_path,
                head_dim=pt.head_dim or head_dim,
            )
            key = str(base_spec)
            if key not in seen:
                seen.add(key)
                capture_targets.append(base_spec)
        _, clean_cache = patcher.runner.run_clean(
            workload, capture_components=capture_targets, token_indices=token_indices
        )

    mags = []
    for pt in parsed_pool:
        base_spec = ComponentTarget(
            layer_idx=pt.layer_idx,
            component_type=pt.component_type,
            module_path=pt.module_path,
            head_dim=pt.head_dim or head_dim,
        )
        candidates_keys = [str(base_spec), str(pt)]
        if pt.component_type == "mlp_channel":
            candidates_keys.extend([f"layer.{pt.layer_idx}.mlp_channel", f"layer.{pt.layer_idx}.mlp", f"layer.{pt.layer_idx}"])
        elif pt.component_type == "attention_head":
            candidates_keys.extend([f"layer.{pt.layer_idx}.attention_head", f"layer.{pt.layer_idx}.attention", f"layer.{pt.layer_idx}"])

        found_key = None
        for k in candidates_keys:
            if k in clean_cache:
                found_key = k
                break

        if found_key is not None:
            act = clean_cache[found_key].float()
            if pt.component_type == "mlp_channel" and pt.sub_idx is not None and act.shape[-1] > pt.sub_idx:
                mag = float(act[..., pt.sub_idx].norm().item())
            elif pt.component_type == "attention_head" and pt.sub_idx is not None:
                hd = pt.head_dim or head_dim
                if hd is not None and act.shape[-1] >= (pt.sub_idx + 1) * hd:
                    mag = float(act[..., pt.sub_idx * hd : (pt.sub_idx + 1) * hd].norm().item())
                else:
                    mag = float(act.norm().item())
            else:
                mag = float(act.norm().item())
            mags.append((pt, mag))

    if not mags:
        raise ValueError("No components found in clean_cache for magnitude calculation")

    mags.sort(key=lambda x: x[1], reverse=True)
    top_k = mags[:k_count]
    top_comps = [x[0] for x in top_k]

    res = patcher.run_patching(workload, top_comps, token_indices=token_indices, head_dim=head_dim)
    mean_eff = sum(getattr(r, metric) for r in res) / len(res)

    return MagnitudeBaselineResult(
        k_count=k_count,
        top_magnitude_components=[str(c) for c in top_comps],
        magnitudes=[x[1] for x in top_k],
        causal_effect=mean_eff,
    )


@dataclass
class KeepDamageResult:
    """Damage assessment on retained (KEEP) distribution."""

    mean_kl: float
    max_kl: float
    top1_agreement: float
    samples: int
    acceptable: bool


def evaluate_keep_damage(
    model: nn.Module,
    keep_inputs: Sequence[Union[str, dict[str, Any], torch.Tensor]],
    interventions: Sequence[RuntimeIntervention],
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
    max_allowed_kl: float = 0.5,
) -> KeepDamageResult:
    """Evaluates KL divergence and top-1 agreement on KEEP data under interventions."""
    if not keep_inputs:
        raise ValueError("keep_inputs cannot be empty")

    base_logprobs_list = []
    with torch.inference_mode():
        for inp in keep_inputs:
            logits = _forward_model(model, inp, tokenizer, device)
            lp = F.log_softmax(logits[:, -1, :].float(), dim=-1).cpu()
            base_logprobs_list.append(lp)

    interv_logprobs_list = []
    with RuntimeInterventionContext(model, interventions):
        with torch.inference_mode():
            for inp in keep_inputs:
                logits = _forward_model(model, inp, tokenizer, device)
                lp = F.log_softmax(logits[:, -1, :].float(), dim=-1).cpu()
                interv_logprobs_list.append(lp)

    kls = []
    agree = 0
    for p_log, q_log in zip(base_logprobs_list, interv_logprobs_list):
        p = p_log.exp()
        kl = torch.sum(p * (p_log - q_log), dim=-1).item()
        kls.append(kl)
        agree += int(int(p_log.argmax(dim=-1).item()) == int(q_log.argmax(dim=-1).item()))

    mean_kl = sum(kls) / len(kls)
    max_kl = max(kls)
    agreement = agree / len(kls)

    return KeepDamageResult(
        mean_kl=mean_kl,
        max_kl=max_kl,
        top1_agreement=agreement,
        samples=len(kls),
        acceptable=mean_kl <= max_allowed_kl,
    )


@dataclass
class CausalComparisonReport:
    """Comprehensive causal validation comparing candidate against baselines and controls."""

    candidate_components: list[str]
    candidate_causal_effect: float
    random_baseline: RandomBaselineResult
    magnitude_baseline: MagnitudeBaselineResult
    keep_damage: Optional[KeepDamageResult]
    causal_gain_over_random: float
    causal_gain_over_magnitude: float
    z_score: float
    specificity_ratio: float
    is_causally_distinct: bool

    def summary_dict(self) -> dict[str, Any]:
        return {
            "candidate_components": self.candidate_components,
            "candidate_causal_effect": self.candidate_causal_effect,
            "causal_gain_over_random": self.causal_gain_over_random,
            "causal_gain_over_magnitude": self.causal_gain_over_magnitude,
            "z_score": self.z_score,
            "random_mean_effect": self.random_baseline.mean_causal_effect,
            "magnitude_effect": self.magnitude_baseline.causal_effect,
            "keep_damage_mean_kl": self.keep_damage.mean_kl if self.keep_damage else None,
            "specificity_ratio": self.specificity_ratio,
            "is_causally_distinct": self.is_causally_distinct,
        }


def compare_causal_controls(
    candidate_components: Sequence[Union[str, ComponentTarget]],
    candidate_causal_effect: float,
    random_baseline: RandomBaselineResult,
    magnitude_baseline: MagnitudeBaselineResult,
    keep_damage: Optional[KeepDamageResult] = None,
) -> CausalComparisonReport:
    """Builds a comparison report testing causal specificity against all baselines."""
    gain_rand = candidate_causal_effect - random_baseline.mean_causal_effect
    z = gain_rand / (random_baseline.std_causal_effect + 1e-8)
    gain_mag = candidate_causal_effect - magnitude_baseline.causal_effect

    spec_ratio = candidate_causal_effect
    if keep_damage is not None:
        spec_ratio = candidate_causal_effect / (keep_damage.mean_kl + 1e-8)

    distinct = gain_rand > 0 and gain_mag > 0 and (keep_damage is None or keep_damage.acceptable)

    return CausalComparisonReport(
        candidate_components=[str(c) for c in candidate_components],
        candidate_causal_effect=candidate_causal_effect,
        random_baseline=random_baseline,
        magnitude_baseline=magnitude_baseline,
        keep_damage=keep_damage,
        causal_gain_over_random=gain_rand,
        causal_gain_over_magnitude=gain_mag,
        z_score=z,
        specificity_ratio=spec_ratio,
        is_causally_distinct=distinct,
    )


# ---------------------------------------------------------------------------
# High-Level Conveniences
# ---------------------------------------------------------------------------


def rank_layers(
    model: nn.Module,
    workload: PairedWorkload,
    token_indices: Optional[Union[int, Sequence[int]]] = -1,
    metric: str = "recovery_ratio",
    max_bytes: Optional[int] = None,
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
) -> list[CausalRankEntry]:
    """Causally ranks all transformer layers by observed patching impact."""
    _, layers = get_layers(model)
    patcher = ActivationPatcher(model, tokenizer=tokenizer, device=device)
    targets = [ComponentTarget(layer_idx=i, component_type="layer") for i in range(len(layers))]
    res = patcher.run_patching(workload, targets, token_indices=token_indices, max_bytes=max_bytes)
    return rank_components_by_causal_effect(res, metric=metric)


def rank_attention_heads(
    model: nn.Module,
    workload: PairedWorkload,
    layer_indices: Optional[Sequence[int]] = None,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    token_indices: Optional[Union[int, Sequence[int]]] = -1,
    metric: str = "recovery_ratio",
    max_bytes: Optional[int] = None,
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
) -> list[CausalRankEntry]:
    """Causally ranks attention heads by observed patching impact."""
    _, layers = get_layers(model)
    target_layers = list(range(len(layers))) if layer_indices is None else list(layer_indices)

    cfg = getattr(model, "config", None)
    n_heads = num_heads or getattr(cfg, "num_attention_heads", None)
    h_dim = head_dim or getattr(cfg, "head_dim", None)

    if n_heads is None:
        first_layer = layers[target_layers[0]]
        attn = getattr(first_layer, "self_attn", getattr(first_layer, "attention", None))
        n_heads = getattr(attn, "num_heads", None) or 8

    targets = []
    for l_idx in target_layers:
        for h_idx in range(n_heads):
            targets.append(
                ComponentTarget(
                    layer_idx=l_idx,
                    component_type="attention_head",
                    sub_idx=h_idx,
                    head_dim=h_dim,
                )
            )

    patcher = ActivationPatcher(model, tokenizer=tokenizer, device=device)
    res = patcher.run_patching(
        workload, targets, token_indices=token_indices, head_dim=h_dim, max_bytes=max_bytes
    )
    return rank_components_by_causal_effect(res, metric=metric)


def rank_mlp_channels(
    model: nn.Module,
    workload: PairedWorkload,
    layer_indices: Optional[Sequence[int]] = None,
    channel_indices: Optional[Sequence[int]] = None,
    token_indices: Optional[Union[int, Sequence[int]]] = -1,
    metric: str = "recovery_ratio",
    max_bytes: Optional[int] = None,
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
) -> list[CausalRankEntry]:
    """Causally ranks MLP intermediate channels by observed patching impact."""
    _, layers = get_layers(model)
    target_layers = list(range(len(layers))) if layer_indices is None else list(layer_indices)

    cfg = getattr(model, "config", None)
    interm_size = getattr(cfg, "intermediate_size", None)
    if interm_size is None and channel_indices is None:
        first_layer = layers[target_layers[0]]
        mlp = getattr(first_layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj") and hasattr(mlp.down_proj, "weight"):
            interm_size = mlp.down_proj.weight.shape[1]
        else:
            interm_size = 16

    channels = list(range(interm_size)) if channel_indices is None else list(channel_indices)

    targets = []
    for l_idx in target_layers:
        for c_idx in channels:
            targets.append(
                ComponentTarget(layer_idx=l_idx, component_type="mlp_channel", sub_idx=c_idx)
            )

    patcher = ActivationPatcher(model, tokenizer=tokenizer, device=device)
    res = patcher.run_patching(workload, targets, token_indices=token_indices, max_bytes=max_bytes)
    return rank_components_by_causal_effect(res, metric=metric)


# ---------------------------------------------------------------------------
# 5. Feature / SAE Adapter Framework
# ---------------------------------------------------------------------------


def _detect_module_activation_dim(
    model: nn.Module,
    target: ComponentTarget,
    module: nn.Module,
) -> Optional[int]:
    """Infers the expected activation feature dimension for a target component."""
    cfg = getattr(model, "config", None)
    ctype = target.component_type.lower()

    if ctype in ("layer", "residual_output", "residual_input", "output", "input"):
        if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
            return cfg.hidden_size
        if hasattr(cfg, "d_model") and cfg.d_model is not None:
            return cfg.d_model
        if hasattr(cfg, "num_attention_heads") and hasattr(cfg, "head_dim"):
            if cfg.num_attention_heads and cfg.head_dim:
                return cfg.num_attention_heads * cfg.head_dim
        embed = getattr(model, "embed", getattr(model, "embed_tokens", None))
        if embed is not None and hasattr(embed, "embedding_dim"):
            return embed.embedding_dim
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "in_features"):
            return lm_head.in_features
        if hasattr(module, "self_attn") and hasattr(module.self_attn, "o_proj"):
            return getattr(module.self_attn.o_proj, "out_features", None)
    elif ctype in ("mlp", "feedforward"):
        if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
            return cfg.hidden_size
        if hasattr(module, "down_proj") and hasattr(module.down_proj, "out_features"):
            return module.down_proj.out_features
    elif ctype in ("mlp_channel", "mlp_intermediate"):
        if hasattr(cfg, "intermediate_size") and cfg.intermediate_size is not None:
            return cfg.intermediate_size
        if hasattr(module, "in_features"):
            return module.in_features
    elif ctype in ("attention", "attn"):
        if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
            return cfg.hidden_size
        if hasattr(module, "o_proj") and hasattr(module.o_proj, "out_features"):
            return module.o_proj.out_features
    elif ctype == "attention_head":
        h_dim = target.head_dim or getattr(cfg, "head_dim", None)
        if h_dim is not None:
            return h_dim

    # Generic module weight inspection
    if hasattr(module, "out_features"):
        return module.out_features
    if hasattr(module, "in_features"):
        return module.in_features
    if hasattr(module, "weight") and hasattr(module.weight, "shape"):
        if len(module.weight.shape) >= 2:
            return module.weight.shape[0]

    return None


class FeatureAdapter(nn.Module):
    """Adapter for projecting activations into sparse feature / SAE latent space.

    Supports model-, layer-, and activation-space compatibility validation,
    sparse encoding, decoding, and reconstruction quality measurement.
    """

    def __init__(
        self,
        activation_dim: int,
        feature_dim: int,
        target: Optional[Union[str, ComponentTarget]] = None,
        layer_idx: Optional[int] = None,
        expected_model_type: Optional[str] = None,
        encoder_weight: Optional[torch.Tensor] = None,
        decoder_weight: Optional[torch.Tensor] = None,
        encoder_bias: Optional[torch.Tensor] = None,
        decoder_bias: Optional[torch.Tensor] = None,
        activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.feature_dim = feature_dim
        self.target = parse_component_target(target) if isinstance(target, str) else target
        self.layer_idx = layer_idx if layer_idx is not None else (self.target.layer_idx if self.target else None)
        self.expected_model_type = expected_model_type
        self.activation_fn = activation_fn or F.relu
        self.top_k = top_k

        # Encoder weight: (activation_dim, feature_dim)
        if encoder_weight is not None:
            if encoder_weight.shape == (feature_dim, activation_dim):
                encoder_weight = encoder_weight.T
            if encoder_weight.shape != (activation_dim, feature_dim):
                raise ValueError(
                    f"encoder_weight shape {encoder_weight.shape} incompatible with "
                    f"activation_dim={activation_dim}, feature_dim={feature_dim}"
                )
            self.encoder_weight = nn.Parameter(encoder_weight.clone())
        else:
            w = torch.randn(activation_dim, feature_dim) / math.sqrt(activation_dim)
            self.encoder_weight = nn.Parameter(w)

        # Decoder weight: (feature_dim, activation_dim)
        if decoder_weight is not None:
            if decoder_weight.shape == (activation_dim, feature_dim):
                decoder_weight = decoder_weight.T
            if decoder_weight.shape != (feature_dim, activation_dim):
                raise ValueError(
                    f"decoder_weight shape {decoder_weight.shape} incompatible with "
                    f"feature_dim={feature_dim}, activation_dim={activation_dim}"
                )
            self.decoder_weight = nn.Parameter(decoder_weight.clone())
        else:
            w = torch.randn(feature_dim, activation_dim)
            w = F.normalize(w, dim=-1)
            self.decoder_weight = nn.Parameter(w)

        # Biases
        if encoder_bias is not None:
            self.encoder_bias = nn.Parameter(encoder_bias.clone())
        else:
            self.encoder_bias = nn.Parameter(torch.zeros(feature_dim))

        if decoder_bias is not None:
            self.decoder_bias = nn.Parameter(decoder_bias.clone())
        else:
            self.decoder_bias = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Projects activations into latent feature space."""
        if x.shape[-1] != self.activation_dim:
            raise FeatureAdapterCompatibilityError(
                f"Input activation feature dimension {x.shape[-1]} does not match "
                f"adapter activation_dim={self.activation_dim}"
            )
        enc_w = self.encoder_weight.to(device=x.device, dtype=x.dtype)
        enc_b = self.encoder_bias.to(device=x.device, dtype=x.dtype)

        pre_act = x @ enc_w + enc_b
        f = self.activation_fn(pre_act)
        if self.top_k is not None and self.top_k < self.feature_dim:
            top_vals, top_idx = torch.topk(f, k=self.top_k, dim=-1)
            f_sparse = torch.zeros_like(f)
            f_sparse.scatter_(-1, top_idx, top_vals)
            return f_sparse
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Projects latent features back into activation space."""
        if f.shape[-1] != self.feature_dim:
            raise FeatureAdapterCompatibilityError(
                f"Feature dimension {f.shape[-1]} does not match adapter feature_dim={self.feature_dim}"
            )
        dec_w = self.decoder_weight.to(device=f.device, dtype=f.dtype)
        dec_b = self.decoder_bias.to(device=f.device, dtype=f.dtype)
        return f @ dec_w + dec_b

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs activations through encode -> decode."""
        return self.decode(self.encode(x))

    def check_compatibility(
        self,
        model: nn.Module,
        target: Optional[Union[str, ComponentTarget]] = None,
    ) -> bool:
        """Validates model-, layer-, and activation-space compatibility."""
        target_spec = target or self.target
        if target_spec is None:
            raise FeatureAdapterCompatibilityError(
                "No target component specified for feature adapter compatibility check"
            )

        comp_target = parse_component_target(target_spec)

        # 1. Model architecture type check
        if self.expected_model_type is not None:
            cfg = getattr(model, "config", None)
            model_type = getattr(cfg, "model_type", None) or getattr(cfg, "architectures", [None])[0]
            if model_type is not None and self.expected_model_type.lower() not in str(model_type).lower():
                raise FeatureAdapterCompatibilityError(
                    f"Model type mismatch: adapter expects {self.expected_model_type!r}, "
                    f"got model with type {model_type!r}"
                )

        # 2. Layer index check
        layers = None
        try:
            _, layers = get_layers(model)
        except Exception:
            pass
        if layers is not None:
            if comp_target.layer_idx < 0 or comp_target.layer_idx >= len(layers):
                raise FeatureAdapterCompatibilityError(
                    f"Layer index {comp_target.layer_idx} out of range [0, {len(layers)})"
                )
            if self.layer_idx is not None and self.layer_idx != comp_target.layer_idx:
                raise FeatureAdapterCompatibilityError(
                    f"Adapter configured for layer {self.layer_idx}, but target specifies layer {comp_target.layer_idx}"
                )

        # 3. Activation space dimensionality check
        module, _, _ = resolve_component_module(model, comp_target)
        detected_dim = _detect_module_activation_dim(model, comp_target, module)
        if detected_dim is not None and detected_dim != self.activation_dim:
            raise FeatureAdapterCompatibilityError(
                f"Activation dimension mismatch for {comp_target}: "
                f"adapter expects activation_dim={self.activation_dim}, "
                f"but module has activation dimension {detected_dim}"
            )

        return True


@dataclass
class FeatureReconstructionResult:
    """Evaluation of SAE / FeatureAdapter reconstruction fidelity."""

    mse: float
    nmse: float
    cosine_similarity: float
    explained_variance: float
    l0_sparsity: float
    l0_fraction: float
    samples: int
    acceptable: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "mse": self.mse,
            "nmse": self.nmse,
            "cosine_similarity": self.cosine_similarity,
            "explained_variance": self.explained_variance,
            "l0_sparsity": self.l0_sparsity,
            "l0_fraction": self.l0_fraction,
            "samples": self.samples,
            "acceptable": self.acceptable,
        }


def measure_reconstruction_quality(
    adapter: FeatureAdapter,
    activations: torch.Tensor,
    max_allowed_nmse: float = 0.25,
    min_explained_variance: float = 0.70,
) -> FeatureReconstructionResult:
    """Evaluates reconstruction quality and sparsity of activations through a FeatureAdapter."""
    with torch.no_grad():
        x = activations.float()
        if x.shape[-1] != adapter.activation_dim:
            raise FeatureAdapterCompatibilityError(
                f"Activations dim {x.shape[-1]} incompatible with adapter activation_dim={adapter.activation_dim}"
            )
        x_flat = x.reshape(-1, adapter.activation_dim)
        f_flat = adapter.encode(x_flat)
        x_rec_flat = adapter.decode(f_flat)

        residual = x_flat - x_rec_flat
        mse = float(torch.mean(residual ** 2).item())
        var_x = float(torch.var(x_flat, unbiased=False).item())
        var_res = float(torch.var(residual, unbiased=False).item())

        denom_nmse = float(torch.sum(x_flat ** 2).item())
        nmse = float(torch.sum(residual ** 2).item()) / (denom_nmse + 1e-8)

        explained_var = 1.0 - (var_res / (var_x + 1e-8))
        cos_sim = float(F.cosine_similarity(x_flat, x_rec_flat, dim=-1).mean().item())

        l0_sparsity = float((f_flat.abs() > 1e-6).sum(dim=-1).float().mean().item())
        l0_fraction = l0_sparsity / max(1, adapter.feature_dim)

        acceptable = (nmse <= max_allowed_nmse) and (explained_var >= min_explained_variance)

        return FeatureReconstructionResult(
            mse=mse,
            nmse=nmse,
            cosine_similarity=cos_sim,
            explained_variance=explained_var,
            l0_sparsity=l0_sparsity,
            l0_fraction=l0_fraction,
            samples=x_flat.shape[0],
            acceptable=acceptable,
        )


@dataclass
class FeatureIntervention(RuntimeIntervention):
    """Intervention operating in sparse feature / SAE latent space."""

    target: Union[str, ComponentTarget]
    adapter: FeatureAdapter
    feature_indices: Union[int, Sequence[int]]
    action: str = "zero"  # "zero", "scale", "clamp", "set", "steer"
    scale: float = 0.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    set_value: Optional[float] = None
    steer_strength: Optional[float] = None
    error_preserving: bool = True
    token_indices: Optional[Union[int, Sequence[int]]] = None
    channel_indices: Optional[Union[int, Sequence[int]]] = None
    head_indices: Optional[Union[int, Sequence[int]]] = None
    head_dim: Optional[int] = None

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        indices = [self.feature_indices] if isinstance(self.feature_indices, int) else list(self.feature_indices)

        def transform_slice(x: torch.Tensor) -> torch.Tensor:
            if x.shape[-1] != self.adapter.activation_dim:
                raise FeatureAdapterCompatibilityError(
                    f"Activation dimension {x.shape[-1]} does not match adapter activation_dim={self.adapter.activation_dim}"
                )
            f = self.adapter.encode(x)
            f_mod = f.clone()

            if self.action == "zero":
                f_mod[..., indices] = 0.0
            elif self.action == "scale":
                f_mod[..., indices] = f_mod[..., indices] * self.scale
            elif self.action == "clamp":
                f_mod[..., indices] = torch.clamp(f_mod[..., indices], min=self.min_val, max=self.max_val)
            elif self.action == "set":
                val = 0.0 if self.set_value is None else self.set_value
                f_mod[..., indices] = val
            elif self.action == "steer":
                s_val = self.steer_strength if self.steer_strength is not None else self.scale
                f_mod[..., indices] = f_mod[..., indices] + s_val
            else:
                raise ValueError(f"Unknown feature intervention action {self.action!r}")

            if self.error_preserving:
                x_rec_orig = self.adapter.decode(f)
                x_rec_mod = self.adapter.decode(f_mod)
                delta = x_rec_mod - x_rec_orig
                return (x + delta).to(device=tensor.device, dtype=tensor.dtype)
            else:
                return self.adapter.decode(f_mod).to(device=tensor.device, dtype=tensor.dtype)

        if self.token_indices is None:
            return transform_slice(tensor)
        else:
            if tensor.ndim < 3:
                return transform_slice(tensor)
            res = tensor.clone()
            t_list = [self.token_indices] if isinstance(self.token_indices, int) else list(self.token_indices)
            for t_idx in t_list:
                res[:, t_idx, :] = transform_slice(res[:, t_idx, :])
            return res


class FeatureAdapterHook:
    """Context manager for attaching a FeatureAdapter intervention to a model."""

    def __init__(
        self,
        model: nn.Module,
        intervention: FeatureIntervention,
    ):
        self.model = model
        self.intervention = intervention
        self.context = RuntimeInterventionContext(model, [intervention])

    @property
    def is_active(self) -> bool:
        return self.context.is_active

    def attach(self) -> None:
        self.intervention.adapter.check_compatibility(self.model, self.intervention.target)
        self.context.attach()

    def remove(self) -> None:
        self.context.remove()

    def __enter__(self) -> "FeatureAdapterHook":
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.remove()
        return False


# ---------------------------------------------------------------------------
# 6. Interacting Components & Edit Composition Order
# ---------------------------------------------------------------------------


def _component_to_intervention(
    target: Union[str, ComponentTarget, RuntimeIntervention],
    intervention_type: str = "scale",
    strength: float = 0.0,
    token_indices: Optional[Union[int, Sequence[int]]] = None,
    head_dim: Optional[int] = None,
    direction: Optional[torch.Tensor] = None,
) -> RuntimeIntervention:
    """Converts a component specifier or target into a concrete RuntimeIntervention."""
    if isinstance(target, RuntimeIntervention):
        return target
    comp = parse_component_target(target) if isinstance(target, str) else target
    ch_idx = comp.sub_idx if comp.component_type == "mlp_channel" else None
    hd_idx = comp.sub_idx if comp.component_type == "attention_head" else None
    h_dim = comp.head_dim or head_dim

    base_spec = ComponentTarget(
        layer_idx=comp.layer_idx,
        component_type=comp.component_type,
        sub_idx=comp.sub_idx,
        module_path=comp.module_path,
        head_dim=h_dim,
    )

    if intervention_type == "scale":
        return ScaleIntervention(
            target=base_spec,
            scale=strength,
            token_indices=token_indices,
            channel_indices=ch_idx,
            head_indices=hd_idx,
            head_dim=h_dim,
        )
    elif intervention_type == "clamp":
        return ClampIntervention(
            target=base_spec,
            max_norm=strength,
            token_indices=token_indices,
            channel_indices=ch_idx,
            head_indices=hd_idx,
            head_dim=h_dim,
        )
    elif intervention_type == "project":
        if direction is None:
            raise ValueError(f"Direction required for project intervention on {comp}")
        return ProjectionIntervention(
            target=base_spec,
            direction=direction,
            strength=strength,
            token_indices=token_indices,
            channel_indices=ch_idx,
            head_indices=hd_idx,
            head_dim=h_dim,
        )
    raise ValueError(f"Unknown intervention type {intervention_type!r}")


def _evaluate_workload_under_interventions(
    model: nn.Module,
    interventions: Sequence[RuntimeIntervention],
    workload: PairedWorkload,
    clean_logits_list: list[torch.Tensor],
    corrupted_logits_list: list[torch.Tensor],
    runner: PairedWorkloadRunner,
    metric: str = "recovery_ratio",
    mode: str = "restoration",
) -> float:
    """Executes model under given interventions and computes workload causal metric."""
    patched_logits_list = []
    with RuntimeInterventionContext(model, interventions):
        with torch.inference_mode():
            for pair in workload:
                inp = pair.corrupted_input if mode == "restoration" else pair.clean_input
                out = _forward_model(model, inp, runner.tokenizer, runner.device)
                patched_logits_list.append(out.detach().cpu())

    p_scores = []
    for i, pair in enumerate(workload):
        c_tar = runner.resolve_token_id(pair.clean_target)
        cr_tar = runner.resolve_token_id(pair.corrupted_target)
        if c_tar is None:
            continue

        c_ld = compute_logit_diff(clean_logits_list[i], c_tar, cr_tar)
        cr_ld = compute_logit_diff(corrupted_logits_list[i], c_tar, cr_tar)
        p_ld = compute_logit_diff(patched_logits_list[i], c_tar, cr_tar)

        if metric == "recovery_ratio":
            p_scores.append(compute_recovery_ratio(p_ld, c_ld, cr_ld))
        elif metric == "target_prob_diff":
            p_prob = compute_target_prob(patched_logits_list[i], c_tar)
            cr_prob = compute_target_prob(corrupted_logits_list[i], c_tar)
            p_scores.append(p_prob - cr_prob)
        elif metric == "patched_logit_diff":
            p_scores.append(p_ld)
        else:
            raise ValueError(f"Unknown metric {metric!r}")

    return sum(p_scores) / max(1, len(p_scores))


@dataclass
class InteractionResult:
    """Analysis of joint vs individual causal effects and interaction regressions."""

    components: list[str]
    individual_effects: dict[str, float]
    individual_keep_damages: dict[str, Optional[float]]
    sum_individual_effects: float
    max_individual_effect: float
    joint_effect: float
    joint_keep_damage: Optional[KeepDamageResult]
    interaction_effect: float
    synergy_type: str  # "synergistic", "additive", "subadditive", "antagonistic"
    has_target_regression: bool
    has_keep_regression: bool
    has_regression: bool
    regression_details: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "components": self.components,
            "individual_effects": self.individual_effects,
            "sum_individual_effects": self.sum_individual_effects,
            "max_individual_effect": self.max_individual_effect,
            "joint_effect": self.joint_effect,
            "interaction_effect": self.interaction_effect,
            "synergy_type": self.synergy_type,
            "has_target_regression": self.has_target_regression,
            "has_keep_regression": self.has_keep_regression,
            "has_regression": self.has_regression,
            "regression_details": self.regression_details,
            "joint_keep_mean_kl": self.joint_keep_damage.mean_kl if self.joint_keep_damage else None,
        }


def evaluate_interacting_components(
    model: nn.Module,
    components: Sequence[Union[str, ComponentTarget, RuntimeIntervention]],
    workload: PairedWorkload,
    keep_inputs: Optional[Sequence[Union[str, dict[str, Any], torch.Tensor]]] = None,
    intervention_type: str = "scale",
    strength: float = 0.0,
    metric: str = "recovery_ratio",
    max_allowed_kl: float = 0.5,
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
    head_dim: Optional[int] = None,
    token_indices: Optional[Union[int, Sequence[int]]] = -1,
    raise_on_regression: bool = False,
) -> InteractionResult:
    """Evaluates causal impact and interaction effects of combining multiple components."""
    runner = PairedWorkloadRunner(model, tokenizer=tokenizer, device=device)
    clean_logits_list, _ = runner.run_clean(workload)
    corrupted_logits_list, _ = runner.run_corrupted(workload)

    # Convert components to RuntimeInterventions
    interventions = [
        _component_to_intervention(
            c,
            intervention_type=intervention_type,
            strength=strength,
            token_indices=token_indices,
            head_dim=head_dim,
        )
        for c in components
    ]

    indiv_effects: dict[str, float] = {}
    indiv_keep_damages: dict[str, Optional[float]] = {}
    indiv_keep_objs: list[Optional[KeepDamageResult]] = []

    for itv in interventions:
        key = str(itv.target)
        eff = _evaluate_workload_under_interventions(
            model, [itv], workload, clean_logits_list, corrupted_logits_list, runner, metric=metric
        )
        indiv_effects[key] = eff

        if keep_inputs:
            kd = evaluate_keep_damage(
                model, keep_inputs, [itv], tokenizer=tokenizer, device=device, max_allowed_kl=max_allowed_kl
            )
            indiv_keep_damages[key] = kd.mean_kl
            indiv_keep_objs.append(kd)
        else:
            indiv_keep_damages[key] = None

    # Joint evaluation
    joint_eff = _evaluate_workload_under_interventions(
        model, interventions, workload, clean_logits_list, corrupted_logits_list, runner, metric=metric
    )
    joint_kd = None
    if keep_inputs:
        joint_kd = evaluate_keep_damage(
            model, keep_inputs, interventions, tokenizer=tokenizer, device=device, max_allowed_kl=max_allowed_kl
        )

    sum_indiv = sum(indiv_effects.values())
    max_indiv = max(indiv_effects.values()) if indiv_effects else 0.0
    interaction_eff = joint_eff - sum_indiv

    # Synergy classification
    if joint_eff > sum_indiv + 0.05:
        synergy_type = "synergistic"
    elif abs(joint_eff - sum_indiv) <= 0.05:
        synergy_type = "additive"
    elif joint_eff > max_indiv + 1e-4:
        synergy_type = "subadditive"
    else:
        synergy_type = "antagonistic"

    # Regression detection
    reg_details = []
    has_target_reg = joint_eff < max_indiv - 1e-4
    if has_target_reg:
        reg_details.append(
            f"Target causal effect regressed: joint ({joint_eff:.4f}) is lower than best single ({max_indiv:.4f})"
        )

    has_keep_reg = False
    if joint_kd is not None and keep_inputs:
        max_indiv_kl = max(
            (kd.mean_kl for kd in indiv_keep_objs if kd is not None),
            default=0.0,
        )
        if joint_kd.mean_kl > max(max_indiv_kl * 1.5, max_indiv_kl + 0.05) or not joint_kd.acceptable:
            has_keep_reg = True
            reg_details.append(
                f"KEEP damage regressed: joint KL ({joint_kd.mean_kl:.4f}) exceeds acceptable threshold or "
                f"amplifies single component KL ({max_indiv_kl:.4f})"
            )

    has_reg = has_target_reg or has_keep_reg

    res = InteractionResult(
        components=[str(itv.target) for itv in interventions],
        individual_effects=indiv_effects,
        individual_keep_damages=indiv_keep_damages,
        sum_individual_effects=sum_indiv,
        max_individual_effect=max_indiv,
        joint_effect=joint_eff,
        joint_keep_damage=joint_kd,
        interaction_effect=interaction_eff,
        synergy_type=synergy_type,
        has_target_regression=has_target_reg,
        has_keep_regression=has_keep_reg,
        has_regression=has_reg,
        regression_details=reg_details,
    )

    if raise_on_regression and has_reg:
        raise InteractionRegressionError(
            f"Interaction regression detected for {res.components}: {'; '.join(reg_details)}"
        )

    return res


@dataclass
class InteractingSetsReport:
    """Comprehensive report on interacting component candidate sets."""

    evaluated_sets: list[InteractionResult]
    synergistic_sets: list[InteractionResult]
    regressive_sets: list[InteractionResult]
    best_set: Optional[InteractionResult]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "total_sets_evaluated": len(self.evaluated_sets),
            "synergistic_count": len(self.synergistic_sets),
            "regressive_count": len(self.regressive_sets),
            "best_set": self.best_set.summary_dict() if self.best_set else None,
        }


def search_interacting_component_sets(
    model: nn.Module,
    candidate_components: Sequence[Union[str, ComponentTarget]],
    workload: PairedWorkload,
    set_sizes: Sequence[int] = (2,),
    max_combinations: int = 20,
    keep_inputs: Optional[Sequence[Union[str, dict[str, Any], torch.Tensor]]] = None,
    intervention_type: str = "scale",
    strength: float = 0.0,
    metric: str = "recovery_ratio",
    max_allowed_kl: float = 0.5,
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
    head_dim: Optional[int] = None,
    token_indices: Optional[Union[int, Sequence[int]]] = -1,
) -> InteractingSetsReport:
    """Searches small interacting component sets and identifies synergies and regressions."""
    evaluated: list[InteractionResult] = []
    combo_count = 0

    pool = list(candidate_components)
    for k in set_sizes:
        if k > len(pool):
            continue
        for combo in itertools.combinations(pool, k):
            if combo_count >= max_combinations:
                break
            res = evaluate_interacting_components(
                model=model,
                components=combo,
                workload=workload,
                keep_inputs=keep_inputs,
                intervention_type=intervention_type,
                strength=strength,
                metric=metric,
                max_allowed_kl=max_allowed_kl,
                tokenizer=tokenizer,
                device=device,
                head_dim=head_dim,
                token_indices=token_indices,
            )
            evaluated.append(res)
            combo_count += 1
        if combo_count >= max_combinations:
            break

    synergistic = [r for r in evaluated if r.synergy_type == "synergistic"]
    regressive = [r for r in evaluated if r.has_regression]

    non_reg = [r for r in evaluated if not r.has_regression]
    candidates_for_best = non_reg if non_reg else evaluated
    best_set = max(candidates_for_best, key=lambda r: r.joint_effect) if candidates_for_best else None

    return InteractingSetsReport(
        evaluated_sets=evaluated,
        synergistic_sets=synergistic,
        regressive_sets=regressive,
        best_set=best_set,
    )


@dataclass
class CompositionOrderResult:
    """Evaluation of edit composition order sensitivity and permutations."""

    interventions: list[str]
    order_permutations: list[tuple[str, ...]]
    permutation_effects: dict[tuple[str, ...], float]
    permutation_keep_damages: dict[tuple[str, ...], Optional[float]]
    is_order_invariant: bool
    max_order_delta: float
    best_order: tuple[str, ...]
    worst_order: tuple[str, ...]
    best_effect: float
    worst_effect: float
    regressed_orders: list[tuple[str, ...]]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "interventions": self.interventions,
            "is_order_invariant": self.is_order_invariant,
            "max_order_delta": self.max_order_delta,
            "best_order": list(self.best_order),
            "worst_order": list(self.worst_order),
            "best_effect": self.best_effect,
            "worst_effect": self.worst_effect,
            "regressed_orders_count": len(self.regressed_orders),
        }


def _intervention_id(it: RuntimeIntervention) -> str:
    """Returns a unique descriptive identifier for an intervention instance."""
    target_str = str(it.target)
    cname = type(it).__name__
    if isinstance(it, ScaleIntervention):
        return f"{target_str}[scale={it.scale}]"
    elif isinstance(it, ClampIntervention):
        parts = []
        if it.min_val is not None:
            parts.append(f"min={it.min_val}")
        if it.max_val is not None:
            parts.append(f"max={it.max_val}")
        if it.max_norm is not None:
            parts.append(f"max_norm={it.max_norm}")
        return f"{target_str}[clamp({','.join(parts)})]"
    elif isinstance(it, ProjectionIntervention):
        return f"{target_str}[project(mode={it.mode},strength={it.strength})]"
    elif isinstance(it, FeatureIntervention):
        return f"{target_str}[feat({it.action}:{it.feature_indices})]"
    elif isinstance(it, ReplacementIntervention):
        return f"{target_str}[replace]"
    return f"{target_str}[{cname}]"


def evaluate_composition_order(
    model: nn.Module,
    interventions: Sequence[RuntimeIntervention],
    workload: PairedWorkload,
    keep_inputs: Optional[Sequence[Union[str, dict[str, Any], torch.Tensor]]] = None,
    metric: str = "recovery_ratio",
    tolerance: float = 1e-3,
    max_permutations: int = 12,
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
) -> CompositionOrderResult:
    """Evaluates how intervention execution order affects causal impact and KEEP retention."""
    if not interventions:
        raise ValueError("Interventions list cannot be empty")

    runner = PairedWorkloadRunner(model, tokenizer=tokenizer, device=device)
    clean_logits_list, _ = runner.run_clean(workload)
    corrupted_logits_list, _ = runner.run_corrupted(workload)

    itv_list = list(interventions)
    perms = list(itertools.permutations(itv_list))
    if len(perms) > max_permutations:
        perms = perms[:max_permutations]

    order_names_list: list[tuple[str, ...]] = []
    effects: dict[tuple[str, ...], float] = {}
    keep_damages: dict[tuple[str, ...], Optional[float]] = {}

    for perm in perms:
        order_key = tuple(_intervention_id(it) for it in perm)
        order_names_list.append(order_key)

        eff = _evaluate_workload_under_interventions(
            model, perm, workload, clean_logits_list, corrupted_logits_list, runner, metric=metric
        )
        effects[order_key] = eff

        if keep_inputs:
            kd = evaluate_keep_damage(model, keep_inputs, perm, tokenizer=tokenizer, device=device)
            keep_damages[order_key] = kd.mean_kl
        else:
            keep_damages[order_key] = None

    best_order = max(effects, key=lambda k: effects[k])
    worst_order = min(effects, key=lambda k: effects[k])
    best_eff = effects[best_order]
    worst_eff = effects[worst_order]
    max_delta = best_eff - worst_eff

    is_invariant = max_delta <= tolerance
    regressed_orders = [o for o, eff in effects.items() if (best_eff - eff) > tolerance]

    return CompositionOrderResult(
        interventions=[_intervention_id(it) for it in itv_list],
        order_permutations=order_names_list,
        permutation_effects=effects,
        permutation_keep_damages=keep_damages,
        is_order_invariant=is_invariant,
        max_order_delta=max_delta,
        best_order=best_order,
        worst_order=worst_order,
        best_effect=best_eff,
        worst_effect=worst_eff,
        regressed_orders=regressed_orders,
    )


# ---------------------------------------------------------------------------
# 7. Hyperparameter & Causal Search
# ---------------------------------------------------------------------------


@dataclass
class CausalSearchSpace:
    """Configurable grid for causal intervention search across locations, strengths, ranks."""

    locations: Sequence[Union[str, ComponentTarget, Sequence[Union[str, ComponentTarget]]]]
    strengths: Sequence[float]
    ranks: Optional[Sequence[int]] = None
    intervention_type: str = "scale"  # "scale", "clamp", "project"
    projection_directions: Optional[dict[str, torch.Tensor]] = None
    token_indices: Optional[Union[int, Sequence[int]]] = -1
    head_dim: Optional[int] = None
    max_allowed_kl: float = 0.5
    keep_penalty_weight: float = 1.0


@dataclass
class CausalCandidateEvaluation:
    """Outcome for a single hyperparameter candidate evaluation."""

    location: Union[str, tuple[str, ...]]
    strength: float
    rank: Optional[int]
    causal_effect: float
    keep_damage: Optional[KeepDamageResult]
    objective_score: float
    is_pareto_optimal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalSearchResult:
    """Result of causal search across locations, strengths, and ranks."""

    evaluations: list[CausalCandidateEvaluation]
    best_candidate: CausalCandidateEvaluation
    pareto_candidates: list[CausalCandidateEvaluation]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated": len(self.evaluations),
            "best_candidate": {
                "location": self.best_candidate.location,
                "strength": self.best_candidate.strength,
                "rank": self.best_candidate.rank,
                "causal_effect": self.best_candidate.causal_effect,
                "objective_score": self.best_candidate.objective_score,
                "keep_mean_kl": self.best_candidate.keep_damage.mean_kl if self.best_candidate.keep_damage else None,
            },
            "pareto_count": len(self.pareto_candidates),
        }


def search_causal_interventions(
    model: nn.Module,
    search_space: CausalSearchSpace,
    workload: PairedWorkload,
    keep_inputs: Optional[Sequence[Union[str, dict[str, Any], torch.Tensor]]] = None,
    metric: str = "recovery_ratio",
    tokenizer: Any = None,
    device: Optional[Union[str, torch.device]] = None,
) -> CausalSearchResult:
    """Performs budgeted hyperparameter search over edit locations, strengths, and ranks."""
    runner = PairedWorkloadRunner(model, tokenizer=tokenizer, device=device)
    clean_logits_list, _ = runner.run_clean(workload)
    corrupted_logits_list, _ = runner.run_corrupted(workload)

    evaluations: list[CausalCandidateEvaluation] = []
    rank_options = list(search_space.ranks) if search_space.ranks else [None]

    for loc in search_space.locations:
        loc_components = [loc] if not isinstance(loc, (list, tuple)) else list(loc)
        loc_key = str(loc_components[0]) if len(loc_components) == 1 else tuple(str(c) for c in loc_components)

        for strength in search_space.strengths:
            for rank_val in rank_options:
                intervs = []
                for comp in loc_components:
                    direction = None
                    if search_space.intervention_type == "project" and search_space.projection_directions:
                        comp_str = str(comp)
                        direction = search_space.projection_directions.get(comp_str)
                        if direction is not None and rank_val is not None and direction.ndim == 2:
                            direction = direction[:, :rank_val]

                    itv = _component_to_intervention(
                        comp,
                        intervention_type=search_space.intervention_type,
                        strength=strength,
                        token_indices=search_space.token_indices,
                        head_dim=search_space.head_dim,
                        direction=direction,
                    )
                    intervs.append(itv)

                eff = _evaluate_workload_under_interventions(
                    model, intervs, workload, clean_logits_list, corrupted_logits_list, runner, metric=metric
                )

                kd = None
                kl_val = 0.0
                if keep_inputs:
                    kd = evaluate_keep_damage(
                        model,
                        keep_inputs,
                        intervs,
                        tokenizer=tokenizer,
                        device=device,
                        max_allowed_kl=search_space.max_allowed_kl,
                    )
                    kl_val = kd.mean_kl

                obj_score = eff - (search_space.keep_penalty_weight * kl_val)

                evaluations.append(
                    CausalCandidateEvaluation(
                        location=loc_key,
                        strength=strength,
                        rank=rank_val,
                        causal_effect=eff,
                        keep_damage=kd,
                        objective_score=obj_score,
                    )
                )

    if not evaluations:
        raise ValueError("Search space produced no candidate evaluations")

    # Pareto optimality: maximize causal_effect, minimize keep_damage.mean_kl
    for i, cand_a in enumerate(evaluations):
        dominated = False
        kl_a = cand_a.keep_damage.mean_kl if cand_a.keep_damage else 0.0
        eff_a = cand_a.causal_effect

        for j, cand_b in enumerate(evaluations):
            if i == j:
                continue
            kl_b = cand_b.keep_damage.mean_kl if cand_b.keep_damage else 0.0
            eff_b = cand_b.causal_effect

            # cand_b dominates cand_a if eff_b >= eff_a and kl_b <= kl_a with at least one strict inequality
            if eff_b >= eff_a and kl_b <= kl_a and (eff_b > eff_a or kl_b < kl_a):
                dominated = True
                break

        cand_a.is_pareto_optimal = not dominated

    pareto_candidates = [c for c in evaluations if c.is_pareto_optimal]
    best_candidate = max(evaluations, key=lambda c: c.objective_score)

    return CausalSearchResult(
        evaluations=evaluations,
        best_candidate=best_candidate,
        pareto_candidates=pareto_candidates,
    )


__all__ = [
    "NeurosurgeryCausalError",
    "MemoryLimitExceededError",
    "HookResolutionError",
    "FeatureAdapterCompatibilityError",
    "InteractionRegressionError",
    "ComponentTarget",
    "parse_component_target",
    "resolve_component_module",
    "HookHandle",
    "ActivationCache",
    "make_capture_hook",
    "attach_capture_hooks",
    "WorkloadPair",
    "PairedWorkload",
    "PairedWorkloadRunner",
    "compute_logit_diff",
    "compute_recovery_ratio",
    "compute_target_prob",
    "PatchingResult",
    "ActivationPatcher",
    "CausalRankEntry",
    "rank_components_by_causal_effect",
    "RandomBaselineResult",
    "evaluate_random_baseline",
    "MagnitudeBaselineResult",
    "evaluate_magnitude_baseline",
    "KeepDamageResult",
    "evaluate_keep_damage",
    "CausalComparisonReport",
    "compare_causal_controls",
    "RuntimeIntervention",
    "ScaleIntervention",
    "ClampIntervention",
    "ProjectionIntervention",
    "ReplacementIntervention",
    "RuntimeInterventionContext",
    "attach_runtime_intervention",
    "rank_layers",
    "rank_attention_heads",
    "rank_mlp_channels",
    "FeatureAdapter",
    "FeatureReconstructionResult",
    "measure_reconstruction_quality",
    "FeatureIntervention",
    "FeatureAdapterHook",
    "InteractionResult",
    "InteractingSetsReport",
    "evaluate_interacting_components",
    "search_interacting_component_sets",
    "CompositionOrderResult",
    "evaluate_composition_order",
    "CausalSearchSpace",
    "CausalCandidateEvaluation",
    "CausalSearchResult",
    "search_causal_interventions",
]
