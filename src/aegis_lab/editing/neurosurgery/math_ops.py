from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F

EPS = 1e-8


def contrast_direction(keep_mean: torch.Tensor, drop_mean: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return normalized DROP-KEEP direction and the pre-normalization separation norm."""
    raw = drop_mean.float() - keep_mean.float()
    separation = float(raw.norm().item())
    if separation <= EPS:
        return torch.zeros_like(raw), 0.0
    return F.normalize(raw, dim=0), separation


def preservation_basis(samples: torch.Tensor, rank: int) -> torch.Tensor:
    """Orthonormal output-space basis spanning dominant KEEP activations.

    samples: [n_samples, hidden]
    returns: [hidden, r]
    """
    if samples.ndim != 2:
        raise ValueError("samples must be [n_samples, hidden]")
    x = samples.float()
    max_rank = min(x.shape)
    r = max(0, min(rank, max_rank))
    if r == 0:
        return x.new_zeros((x.shape[1], 0))
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[:r].T.contiguous()


def constrain_update_to_preserve_basis(delta: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    """Remove update components that lie inside a preservation subspace.

    For a weight update [out,in] and basis [out,r], project in output space:
      delta <- delta - Q(Q^T delta)

    If basis matches input dimension, project on the right instead.
    """
    if basis is None or basis.numel() == 0:
        return delta
    q = basis.to(device=delta.device, dtype=torch.float32)
    d = delta.float()
    if q.shape[0] == d.shape[0]:
        return d - q @ (q.T @ d)
    if q.shape[0] == d.shape[1]:
        return d - (d @ q) @ q.T
    raise ValueError(f"basis {tuple(q.shape)} incompatible with delta {tuple(d.shape)}")


def projected_weight_update(weight: torch.Tensor, direction: torch.Tensor, strength: float) -> torch.Tensor:
    """Directional projection update generalized to either weight axis.

    Returns delta, not the modified weight.
    """
    w = weight.float()
    v = F.normalize(direction.float().to(w.device), dim=0)
    if v.numel() == w.shape[0]:
        return -strength * v[:, None] * (v @ w)[None, :]
    if v.numel() == w.shape[1]:
        return -strength * (w @ v)[:, None] * v[None, :]
    raise ValueError(f"direction {v.numel()} incompatible with weight {tuple(w.shape)}")


def restore_row_norms(original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
    """Restore each row's L2 norm after surgery."""
    o = original.float()
    m = modified.float()
    before = o.norm(dim=1, keepdim=True)
    after = m.norm(dim=1, keepdim=True).clamp_min(EPS)
    scaled = m * (before / after)
    return scaled.to(original.dtype)


def apply_constrained_directional_surgery(
    weight: torch.Tensor,
    direction: torch.Tensor,
    strength: float,
    preserve_basis: Optional[torch.Tensor] = None,
    norm_preserve: bool = True,
) -> torch.Tensor:
    delta = projected_weight_update(weight, direction, strength)
    delta = constrain_update_to_preserve_basis(delta, preserve_basis)
    out = weight.float() + delta
    if norm_preserve and preserve_basis is not None and preserve_basis.numel() > 0:
        for _ in range(32):
            out = restore_row_norms(weight, out).float()
            constrained = constrain_update_to_preserve_basis(out - weight.float(), preserve_basis)
            out = weight.float() + constrained
        return out.to(weight.dtype)
    if norm_preserve:
        return restore_row_norms(weight, out)
    return out.to(weight.dtype)
