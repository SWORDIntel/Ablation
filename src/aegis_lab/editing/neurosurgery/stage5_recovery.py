"""Stage 5: Recovery, distillation, and targeted fine-tuning for model neurosurgery.

Implements:
1. Targeted LoRA injection and explicit parameter freeze masks with tied-weight safety.
2. Multi-objective recovery loss (KEEP replay, CHANGE target loss, logit distillation).
3. Training & recovery loop with AdamW, gradient accumulation, step/time/VRAM budgets,
   and DROP behavioral monitoring (penalizing/failing on rebound).
4. Adapter merge, state export, checkpointing, and parity verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    LOG,
    file_sha256,
    load_tensor_artifact,
    nested_getattr,
    nested_setattr,
    resolve_device,
)


@dataclass
class TrainableParamsReport:
    """Audit report of model trainable parameters and tied-weight safety."""

    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_ratio: float
    trainable_param_names: list[str]
    tied_param_groups: list[list[str]]
    tied_safe: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryLossConfig:
    """Configuration for multi-objective recovery loss."""

    w_keep: float = 1.0
    w_change: float = 1.0
    w_distill: float = 0.0
    distill_temperature: float = 2.0
    ignore_index: int = -100


@dataclass
class RecoveryLossOutput:
    """Breakdown of computed recovery losses."""

    total_loss: torch.Tensor
    keep_loss: float = 0.0
    change_loss: float = 0.0
    distill_loss: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class RecoveryConfig:
    """Hyperparameters and budget configuration for Stage 5 recovery training."""

    max_steps: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    seed: int = 42
    loss_config: RecoveryLossConfig = field(default_factory=RecoveryLossConfig)
    mixed_precision: Optional[str] = None
    max_time_seconds: Optional[float] = None
    max_vram_gb: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-4
    eval_drop_every: int = 10
    max_drop_threshold: Optional[float] = None
    max_drop_rebound: Optional[float] = None
    stop_on_drop_rebound: bool = True
    checkpoint_dir: Optional[str] = None
    checkpoint_every: Optional[int] = None


@dataclass
class RecoveryResult:
    """Result of Stage 5 recovery training run."""

    success: bool
    status: str
    steps_completed: int
    final_loss: float
    initial_drop_score: Optional[float] = None
    final_drop_score: Optional[float] = None
    drop_violation: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    trainable_params_report: Optional[TrainableParamsReport] = None
    checkpoint_path: Optional[str] = None


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around an nn.Linear module.

    Forward computes: h = W_base x + (alpha / r) * (x @ A.T @ B.T).
    Initialized with A ~ Kaiming uniform and B = 0, so initial delta is 0.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"base_layer must be an instance of torch.nn.Linear, got {type(base_layer)}"
            )
        if r < 0:
            raise ValueError(f"LoRA rank r must be non-negative, got {r}")
        if r > 0 and lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / r) if r > 0 else 1.0

        # Base layer parameters are frozen by default
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(
                    r,
                    self.in_features,
                    dtype=base_layer.weight.dtype,
                    device=base_layer.weight.device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.empty(
                    self.out_features,
                    r,
                    dtype=base_layer.weight.dtype,
                    device=base_layer.weight.device,
                )
            )
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.merged = False

    def reset_parameters(self) -> None:
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.r == 0 or self.merged:
            return base_out

        dropped = self.lora_dropout(x)
        if dropped.dtype != self.lora_A.dtype:
            dropped = dropped.to(self.lora_A.dtype)
        lora_out = F.linear(F.linear(dropped, self.lora_A), self.lora_B) * self.scaling
        if lora_out.dtype != base_out.dtype:
            lora_out = lora_out.to(base_out.dtype)
        return base_out + lora_out

    def merge(self) -> None:
        """Merge low-rank weights directly into base linear layer."""
        if self.merged:
            return
        if self.r > 0:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.base_layer.weight.data.add_(delta_w.to(self.base_layer.weight.dtype))
        self.merged = True

    def unmerge(self) -> None:
        """Unmerge low-rank weights from base linear layer."""
        if not self.merged:
            return
        if self.r > 0:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.base_layer.weight.data.sub_(delta_w.to(self.base_layer.weight.dtype))
        self.merged = False


def _matches_module_target(name: str, targets: Sequence[str]) -> bool:
    for target in targets:
        if name == target:
            return True
        if name.endswith("." + target):
            return True
        if target in name.split("."):
            return True
    return False


def inject_lora(
    model: nn.Module,
    target_modules: Optional[Sequence[str]] = None,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> dict[str, LoRALinear]:
    """Inject LoRALinear adapters into selected linear modules of the model.

    Args:
        model: Target PyTorch model.
        target_modules: Names or sub-paths of modules to target (e.g. ['down_proj', 'o_proj']).
            Defaults to ('down_proj', 'o_proj').
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability applied to LoRA input.

    Returns:
        Dictionary mapping module name paths to injected LoRALinear instances.
    """
    targets = tuple(target_modules) if target_modules is not None else ("down_proj", "o_proj")
    to_replace: list[tuple[str, nn.Linear]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            if _matches_module_target(name, targets):
                to_replace.append((name, module))

    injected: dict[str, LoRALinear] = {}
    for name, module in to_replace:
        lora_mod = LoRALinear(
            base_layer=module,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        if "." in name:
            parent_path, child_name = name.rsplit(".", 1)
            parent = nested_getattr(model, parent_path)
        else:
            parent = model
            child_name = name

        setattr(parent, child_name, lora_mod)
        injected[name] = lora_mod

    LOG.info("Injected %d LoRA adapters (targets: %s, r=%d, alpha=%.1f)", len(injected), targets, r, lora_alpha)
    return injected


def apply_freeze_mask(
    model: nn.Module,
    trainable_patterns: Optional[Sequence[str]] = None,
    allow_lora_only: bool = True,
) -> TrainableParamsReport:
    """Freeze base model weights and enable gradients only for LoRA or specified submodules.

    Args:
        model: Target model.
        trainable_patterns: Explicit parameter name substrings/patterns allowed to be trainable.
        allow_lora_only: If True, enables gradients for LoRALinear lora_A/lora_B parameters.

    Returns:
        TrainableParamsReport detailing parameter counts and tied parameter safety.
    """
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Enable LoRA parameters if requested
    if allow_lora_only:
        for module in model.modules():
            if isinstance(module, LoRALinear):
                if module.lora_A is not None:
                    module.lora_A.requires_grad = True
                if module.lora_B is not None:
                    module.lora_B.requires_grad = True

    # Enable explicit trainable patterns if provided
    if trainable_patterns:
        for name, param in model.named_parameters():
            if any(pat in name for pat in trainable_patterns):
                param.requires_grad = True

    return verify_trainable_parameters(
        model,
        allowed_patterns=(
            (["lora_A", "lora_B"] if allow_lora_only else [])
            + (list(trainable_patterns) if trainable_patterns else [])
        ),
        strict=False,
    )


def verify_trainable_parameters(
    model: nn.Module,
    allowed_patterns: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> TrainableParamsReport:
    """Verify that only authorized parameters have requires_grad=True and verify tied parameter safety.

    Args:
        model: Target model.
        allowed_patterns: Optional list of substrings that authorized trainable parameters must match.
        strict: If True, raises ValueError if any unauthorized parameter has requires_grad=True.

    Returns:
        TrainableParamsReport.
    """
    trainable_names: list[str] = []
    unauthorized_names: list[str] = []

    for name, param in model.named_parameters(remove_duplicate=False):
        if param.requires_grad:
            trainable_names.append(name)
            if allowed_patterns is not None:
                if not any(pattern in name for pattern in allowed_patterns):
                    unauthorized_names.append(name)

    if strict and unauthorized_names:
        raise ValueError(
            f"Unauthorized trainable parameters found ({len(unauthorized_names)}): {unauthorized_names[:5]}"
        )

    # Analyze tied / shared parameters by storage pointer
    ptr_map: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        ptr_map.setdefault(param.data_ptr(), []).append(name)

    tied_groups = [names for names in ptr_map.values() if len(names) > 1]

    # Tied parameter safety check:
    # If a tied group contains trainable parameters, ensure all aliases were authorized.
    tied_safe = True
    details: dict[str, Any] = {"unauthorized_parameters": unauthorized_names}

    for group in tied_groups:
        group_trainable = [name for name in group if name in trainable_names]
        if group_trainable:
            # Check if all names in this tied group are expected to be trainable
            if len(group_trainable) != len(group):
                tied_safe = False
                details.setdefault("tied_asymmetric_groups", []).append(
                    {"group": group, "trainable_subset": group_trainable}
                )
            if allowed_patterns is not None:
                for name in group_trainable:
                    if not any(pat in name for pat in allowed_patterns):
                        tied_safe = False

    # Unique parameter counts (avoid double-counting tied weights)
    unique_params = {p.data_ptr(): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params.values())

    unique_trainable = {p.data_ptr(): p for p in model.parameters() if p.requires_grad}
    trainable_params = sum(p.numel() for p in unique_trainable.values())
    frozen_params = total_params - trainable_params
    trainable_ratio = (trainable_params / total_params) if total_params > 0 else 0.0

    return TrainableParamsReport(
        total_params=total_params,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        trainable_ratio=trainable_ratio,
        trainable_param_names=trainable_names,
        tied_param_groups=tied_groups,
        tied_safe=tied_safe,
        details=details,
    )


def forward_for_logits(
    model: nn.Module, batch: Union[dict[str, Any], torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Execute model forward pass and extract logits, labels, and attention mask.

    Handles HF model outputs, raw tensors, and dictionary batches.
    """
    labels = None
    attention_mask = None

    if isinstance(batch, dict):
        inputs = {k: v for k, v in batch.items() if k not in ("labels",)}
        labels = batch.get("labels", None)
        attention_mask = batch.get("attention_mask", None)
        out = model(**inputs)
    else:
        out = model(batch)

    if hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    elif isinstance(out, torch.Tensor):
        logits = out
    else:
        raise TypeError(f"Unsupported model output type: {type(out)}")

    return logits, labels, attention_mask


def compute_lm_cross_entropy(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute causal autoregressive cross-entropy loss with padding support."""
    if logits.ndim == 2:
        # Direct classification or single-step logits
        if labels is None and input_ids is not None:
            labels = input_ids
        if labels is None:
            raise ValueError("labels or input_ids must be provided")
        return F.cross_entropy(logits, labels.view(-1), ignore_index=ignore_index)

    if labels is None:
        if input_ids is None:
            raise ValueError("Either labels or input_ids must be provided")
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].clone().contiguous()
        if attention_mask is not None:
            valid = (attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool())
            shift_labels.masked_fill_(~valid, ignore_index)
    else:
        if labels.shape[1] == logits.shape[1] and labels.shape[1] > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].clone().contiguous()
            if attention_mask is not None:
                valid = attention_mask[:, 1:].bool()
                shift_labels.masked_fill_(~valid, ignore_index)
        else:
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()

    # Guard against all tokens being ignored (returns 0 loss rather than NaN)
    valid_tokens = shift_labels != ignore_index
    if not valid_tokens.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    vocab_size = shift_logits.shape[-1]
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute teacher-student logit distillation loss via KL divergence with temperature scaling."""
    if temperature <= 0.0:
        raise ValueError(f"Distillation temperature must be positive, got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student logits shape {student_logits.shape} does not match teacher shape {teacher_logits.shape}"
        )

    s_log_p = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_p = F.softmax(teacher_logits.float() / temperature, dim=-1)

    kl = F.kl_div(s_log_p, t_p, reduction="none").sum(dim=-1)

    if attention_mask is not None:
        mask = attention_mask.bool()
        if not mask.any():
            return torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype, requires_grad=True)
        loss = (kl * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss = kl.mean()

    return loss * (temperature ** 2)


def compute_recovery_loss(
    model: nn.Module,
    keep_batch: Optional[Union[dict[str, Any], torch.Tensor]] = None,
    change_batch: Optional[Union[dict[str, Any], torch.Tensor]] = None,
    distill_batch: Optional[Union[dict[str, Any], torch.Tensor]] = None,
    teacher_model: Optional[nn.Module] = None,
    loss_config: Optional[RecoveryLossConfig] = None,
) -> RecoveryLossOutput:
    """Compute multi-objective recovery loss (KEEP replay, CHANGE targets, distillation).

    Args:
        model: Student model being recovered.
        keep_batch: Batch of data for retained tasks (KEEP).
        change_batch: Batch of data with desired target outputs (CHANGE).
        distill_batch: Batch for teacher logit distillation (defaults to keep_batch).
        teacher_model: Optional untouched baseline or stronger teacher model.
        loss_config: Loss weights and temperature configuration.

    Returns:
        RecoveryLossOutput.
    """
    if loss_config is None:
        loss_config = RecoveryLossConfig()

    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    keep_loss_val = 0.0
    change_loss_val = 0.0
    distill_loss_val = 0.0

    # 1. Supervised KEEP replay loss
    if keep_batch is not None and loss_config.w_keep > 0.0:
        logits, labels, mask = forward_for_logits(model, keep_batch)
        in_ids = keep_batch.get("input_ids") if isinstance(keep_batch, dict) else keep_batch
        loss_k = compute_lm_cross_entropy(
            logits,
            labels=labels,
            input_ids=in_ids,
            attention_mask=mask,
            ignore_index=loss_config.ignore_index,
        )
        total_loss = total_loss + (loss_config.w_keep * loss_k)
        keep_loss_val = float(loss_k.detach().item())

    # 2. Supervised CHANGE loss
    if change_batch is not None and loss_config.w_change > 0.0:
        logits, labels, mask = forward_for_logits(model, change_batch)
        in_ids = change_batch.get("input_ids") if isinstance(change_batch, dict) else change_batch
        loss_c = compute_lm_cross_entropy(
            logits,
            labels=labels,
            input_ids=in_ids,
            attention_mask=mask,
            ignore_index=loss_config.ignore_index,
        )
        total_loss = total_loss + (loss_config.w_change * loss_c)
        change_loss_val = float(loss_c.detach().item())

    # 3. Teacher-student logit distillation loss
    if teacher_model is not None and loss_config.w_distill > 0.0:
        t_batch = distill_batch if distill_batch is not None else keep_batch
        if t_batch is not None:
            with torch.no_grad():
                t_logits, _, _ = forward_for_logits(teacher_model, t_batch)
            s_logits, _, s_mask = forward_for_logits(model, t_batch)
            loss_d = compute_distillation_loss(
                student_logits=s_logits,
                teacher_logits=t_logits,
                temperature=loss_config.distill_temperature,
                attention_mask=s_mask,
            )
            total_loss = total_loss + (loss_config.w_distill * loss_d)
            distill_loss_val = float(loss_d.detach().item())

    return RecoveryLossOutput(
        total_loss=total_loss,
        keep_loss=keep_loss_val,
        change_loss=change_loss_val,
        distill_loss=distill_loss_val,
        metrics={
            "total_loss": float(total_loss.detach().item()),
            "keep_loss": keep_loss_val,
            "change_loss": change_loss_val,
            "distill_loss": distill_loss_val,
        },
    )


def seed_everything(seed: int = 42) -> None:
    """Seed python random, torch CPU, and torch CUDA RNG states."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_rng_states() -> dict[str, Any]:
    """Capture current RNG states across random and torch."""
    states: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python_random": random.getstate(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        np_st = np.random.get_state()
        # np_st[1] is a numpy ndarray of uint32 keys. Convert to torch tensor for safe unpickling with weights_only=True
        states["numpy"] = (
            np_st[0],
            torch.from_numpy(np_st[1].copy()),
            np_st[2],
            np_st[3],
            np_st[4],
        )
    except ImportError:
        pass
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """Restore saved RNG states."""
    if "torch_cpu" in states and states["torch_cpu"] is not None:
        torch.set_rng_state(states["torch_cpu"])
    if "python_random" in states and states["python_random"] is not None:
        random.setstate(states["python_random"])
    if (
        "torch_cuda" in states
        and states["torch_cuda"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    if "numpy" in states and states["numpy"] is not None:
        try:
            import numpy as np

            np_raw = states["numpy"]
            arr = (
                np_raw[1].cpu().numpy()
                if isinstance(np_raw[1], torch.Tensor)
                else np_raw[1]
            )
            np.random.set_state((np_raw[0], arr, np_raw[2], np_raw[3], np_raw[4]))
        except ImportError:
            pass


def save_recovery_checkpoint(
    path: Union[str, Path],
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_states: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[Path, str]:
    """Save recovery training checkpoint containing model parameters, optimizer, and RNG state."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": rng_states,
        "metrics": metrics,
    }
    torch.save(payload, p)
    return p, file_sha256(p)


def load_recovery_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    """Restore model weights, optimizer state, and RNG states from checkpoint."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    checkpoint = load_tensor_artifact(p)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "rng_states" in checkpoint:
        set_rng_states(checkpoint["rng_states"])

    return {
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def _make_cycle_iterator(data: Any) -> Optional[Iterator[Any]]:
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return None

        def _list_gen():
            while True:
                for item in data:
                    yield item

        return _list_gen()

    def _loader_gen():
        while True:
            for item in data:
                yield item

    return _loader_gen()


def run_recovery_training(
    model: nn.Module,
    keep_data: Optional[Any] = None,
    change_data: Optional[Any] = None,
    distill_data: Optional[Any] = None,
    teacher_model: Optional[nn.Module] = None,
    config: Optional[RecoveryConfig] = None,
    drop_evaluator: Optional[Callable[[nn.Module], float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> RecoveryResult:
    """Execute Stage 5 recovery training loop with DROP behavioral monitoring.

    Args:
        model: Target model (with injected LoRA adapters).
        keep_data: Dataset/loader for retained KEEP tasks.
        change_data: Dataset/loader for desired CHANGE target outputs.
        distill_data: Dataset/loader for teacher logit distillation (optional).
        teacher_model: Teacher model for distillation (optional).
        config: Training hyperparameters and budget constraints.
        drop_evaluator: Callable returning scalar DROP score. Higher indicates rebound.
        optimizer: Optional custom optimizer. Defaults to AdamW on trainable parameters.

    Returns:
        RecoveryResult.
    """
    if config is None:
        config = RecoveryConfig()

    seed_everything(config.seed)

    # Verify trainable parameters
    trainable_report = verify_trainable_parameters(model, strict=False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for recovery fine-tuning.")

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=config.lr, weight_decay=config.weight_decay
        )

    # Initial DROP behavioral evaluation
    initial_drop_score: Optional[float] = None
    if drop_evaluator is not None:
        initial_drop_score = float(drop_evaluator(model))
        LOG.info("Initial DROP evaluation score: %.4f", initial_drop_score)

    keep_iter = _make_cycle_iterator(keep_data)
    change_iter = _make_cycle_iterator(change_data)
    distill_iter = _make_cycle_iterator(distill_data)

    if keep_iter is None and change_iter is None and distill_iter is None:
        raise ValueError(
            "At least one data source (keep_data, change_data, or distill_data) must be provided."
        )

    start_time = time.time()
    steps_completed = 0
    final_loss = 0.0
    history: list[dict[str, Any]] = []
    status = "SUCCESS"
    drop_violation = False
    best_loss = float("inf")
    patience_counter = 0
    last_checkpoint_path: Optional[str] = None

    model.train()
    optimizer.zero_grad()

    for step in range(config.max_steps):
        # Budget check: time
        if config.max_time_seconds is not None:
            if (time.time() - start_time) >= config.max_time_seconds:
                LOG.warning("Time budget exceeded (%.2fs)", config.max_time_seconds)
                status = "BUDGET_EXCEEDED"
                break

        # Budget check: VRAM
        if config.max_vram_gb is not None and torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
            if vram_gb >= config.max_vram_gb:
                LOG.warning("VRAM budget exceeded (%.2f GB)", vram_gb)
                status = "BUDGET_EXCEEDED"
                break

        k_batch = next(keep_iter) if keep_iter is not None else None
        c_batch = next(change_iter) if change_iter is not None else None
        d_batch = next(distill_iter) if distill_iter is not None else None

        loss_out = compute_recovery_loss(
            model=model,
            keep_batch=k_batch,
            change_batch=c_batch,
            distill_batch=d_batch,
            teacher_model=teacher_model,
            loss_config=config.loss_config,
        )

        loss_accum = loss_out.total_loss / config.gradient_accumulation_steps
        loss_accum.backward()

        final_loss = float(loss_out.total_loss.detach().item())
        steps_completed += 1

        is_accum_step = ((step + 1) % config.gradient_accumulation_steps == 0) or (
            step + 1 == config.max_steps
        )
        if is_accum_step:
            optimizer.step()
            optimizer.zero_grad()

        # Step metrics recording
        step_record: dict[str, Any] = {
            "step": step + 1,
            "total_loss": final_loss,
            "keep_loss": loss_out.keep_loss,
            "change_loss": loss_out.change_loss,
            "distill_loss": loss_out.distill_loss,
        }

        # Periodic DROP behavioral monitoring
        if drop_evaluator is not None and (
            (step + 1) % config.eval_drop_every == 0 or (step + 1) == config.max_steps
        ):
            current_drop = float(drop_evaluator(model))
            step_record["drop_score"] = current_drop
            LOG.debug("Step %d DROP score: %.4f", step + 1, current_drop)

            # Rebound threshold check
            if config.max_drop_threshold is not None and current_drop > config.max_drop_threshold:
                LOG.warning(
                    "DROP score rebounded above threshold: %.4f > %.4f",
                    current_drop,
                    config.max_drop_threshold,
                )
                drop_violation = True

            # Relative rebound delta check
            if config.max_drop_rebound is not None and initial_drop_score is not None:
                rebound_delta = current_drop - initial_drop_score
                if rebound_delta > config.max_drop_rebound:
                    LOG.warning(
                        "DROP score rebounded by delta: %.4f > %.4f",
                        rebound_delta,
                        config.max_drop_rebound,
                    )
                    drop_violation = True

            if drop_violation and config.stop_on_drop_rebound:
                status = "DROP_REBOUND_FAILED"
                history.append(step_record)
                break

        history.append(step_record)

        # Early stopping check
        if config.early_stopping_patience is not None:
            if final_loss < best_loss - config.early_stopping_min_delta:
                best_loss = final_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    LOG.info("Early stopping triggered at step %d", step + 1)
                    status = "EARLY_STOPPED"
                    break

        # Periodic checkpointing
        if (
            config.checkpoint_dir is not None
            and config.checkpoint_every is not None
            and (step + 1) % config.checkpoint_every == 0
        ):
            ckpt_path = Path(config.checkpoint_dir) / f"checkpoint_step_{step + 1}.pt"
            saved_p, _ = save_recovery_checkpoint(
                path=ckpt_path,
                step=step + 1,
                model=model,
                optimizer=optimizer,
                rng_states=get_rng_states(),
                metrics={"loss": final_loss},
            )
            last_checkpoint_path = str(saved_p)

    # Final DROP evaluation
    final_drop_score: Optional[float] = None
    if drop_evaluator is not None:
        final_drop_score = float(drop_evaluator(model))
        if config.max_drop_threshold is not None and final_drop_score > config.max_drop_threshold:
            drop_violation = True
        if config.max_drop_rebound is not None and initial_drop_score is not None:
            if (final_drop_score - initial_drop_score) > config.max_drop_rebound:
                drop_violation = True

        if drop_violation:
            status = "DROP_REBOUND_FAILED"

    # Save final checkpoint if requested
    if config.checkpoint_dir is not None:
        final_ckpt = Path(config.checkpoint_dir) / "checkpoint_final.pt"
        saved_p, _ = save_recovery_checkpoint(
            path=final_ckpt,
            step=steps_completed,
            model=model,
            optimizer=optimizer,
            rng_states=get_rng_states(),
            metrics={"final_loss": final_loss, "status": status},
        )
        last_checkpoint_path = str(saved_p)

    success = (status in ("SUCCESS", "EARLY_STOPPED")) and not drop_violation

    return RecoveryResult(
        success=success,
        status=status,
        steps_completed=steps_completed,
        final_loss=final_loss,
        initial_drop_score=initial_drop_score,
        final_drop_score=final_drop_score,
        drop_violation=drop_violation,
        history=history,
        trainable_params_report=trainable_report,
        checkpoint_path=last_checkpoint_path,
    )


def merge_lora(model: nn.Module, in_place: bool = True) -> nn.Module:
    """Merge all injected LoRALinear adapters back into base Linear layers.

    Replaces LoRALinear modules with the underlying base linear layer containing
    the folded delta weights.
    """
    target_model = model if in_place else copy.deepcopy(model)
    lora_modules: list[tuple[str, LoRALinear]] = []

    for name, module in target_model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules.append((name, module))

    for name, lora_mod in lora_modules:
        lora_mod.merge()
        base_layer = lora_mod.base_layer
        if "." in name:
            parent_path, child_name = name.rsplit(".", 1)
            parent = nested_getattr(target_model, parent_path)
        else:
            parent = target_model
            child_name = name
        setattr(parent, child_name, base_layer)

    LOG.info("Merged %d LoRA adapters into base model", len(lora_modules))
    return target_model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """Unmerge weights across all LoRALinear modules in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
    return model


def export_lora_state(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Export parameters of all injected LoRA adapters without base model weights."""
    adapters: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapters[name] = {
                "r": module.r,
                "lora_alpha": module.lora_alpha,
                "scaling": module.scaling,
                "lora_A": module.lora_A.detach().clone().cpu() if module.lora_A is not None else None,
                "lora_B": module.lora_B.detach().clone().cpu() if module.lora_B is not None else None,
            }
    return adapters


def save_lora_adapter(model: nn.Module, path: Union[str, Path]) -> tuple[Path, str]:
    """Save separate LoRA adapter state to disk with sha256 checksum."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "format": "aegis_lora_adapter",
        "adapters": export_lora_state(model),
    }
    torch.save(payload, p)
    return p, file_sha256(p)


def load_lora_adapter(model: nn.Module, path: Union[str, Path]) -> None:
    """Load saved LoRA adapter state into matching modules of the model."""
    payload = load_tensor_artifact(path)
    adapters = payload.get("adapters", {})

    for name, state in adapters.items():
        try:
            module = nested_getattr(model, name)
        except (AttributeError, KeyError):
            raise KeyError(f"Target module '{name}' not found in model hierarchy.")

        if not isinstance(module, LoRALinear):
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Target module '{name}' must be nn.Linear or LoRALinear, got {type(module)}")
            # Wrap with LoRALinear and replace
            lora_mod = LoRALinear(
                base_layer=module,
                r=state["r"],
                lora_alpha=state["lora_alpha"],
            )
            if "." in name:
                parent_path, child_name = name.rsplit(".", 1)
                parent = nested_getattr(model, parent_path)
            else:
                parent = model
                child_name = name
            setattr(parent, child_name, lora_mod)
            module = lora_mod

        if state["lora_A"] is not None and module.lora_A is not None:
            module.lora_A.data.copy_(state["lora_A"].to(module.lora_A.device))
        if state["lora_B"] is not None and module.lora_B is not None:
            module.lora_B.data.copy_(state["lora_B"].to(module.lora_B.device))


def verify_merge_parity(
    adapter_model: nn.Module,
    merged_model: Optional[nn.Module] = None,
    sample_inputs: Any = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Verify numeric parity between adapter-attached model and merged model.

    Args:
        adapter_model: Model with active LoRALinear adapters.
        merged_model: Optional merged model. If None, a copy of adapter_model is merged.
        sample_inputs: Inputs passed to both models.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        raise_on_failure: Whether to raise ValueError if parity check fails.

    Returns:
        Dict with parity status and error magnitudes.
    """
    if sample_inputs is None:
        raise ValueError("sample_inputs must be provided to verify merge parity.")

    if merged_model is None:
        merged_model = merge_lora(adapter_model, in_place=False)

    adapter_model.eval()
    merged_model.eval()

    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            out_adapter, _, _ = forward_for_logits(adapter_model, sample_inputs)
            out_merged, _, _ = forward_for_logits(merged_model, sample_inputs)
        else:
            out_adapter, _, _ = forward_for_logits(adapter_model, sample_inputs)
            out_merged, _, _ = forward_for_logits(merged_model, sample_inputs)

    diff = torch.abs(out_adapter.float() - out_merged.float())
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    parity = bool(torch.allclose(out_adapter.float(), out_merged.float(), rtol=rtol, atol=atol))

    report = {
        "parity": parity,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rtol": rtol,
        "atol": atol,
    }

    if not parity and raise_on_failure:
        raise ValueError(
            f"Merge parity verification failed: max_abs_diff={max_abs_diff:.6e}, "
            f"mean_abs_diff={mean_abs_diff:.6e} exceeds rtol={rtol}, atol={atol}"
        )

    return report
