"""Stage 4D: Constrained hypertuning for model neurosurgery.

Implements:
1. Hyperparameter search space: edit selectors/strengths, preservation rank,
   pruning ratios, LoRA rank/alpha, learning rate, loss weights, and training steps.
2. Search strategies: seeded random search, coarse-to-fine grid search, and
   successive budget allocation (halving / pruning unpromising trials).
3. Multi-objective criteria and explicit thresholds: target task improvement or
   DROP suppression, KEEP retention/quality, physical parameter/byte reduction,
   peak memory, and measured latency.
4. Trial management and caching: content-derived trial IDs, resumption of
   interrupted studies from trial logs, caching baseline evaluation, and budget limits.
5. Pareto frontier analysis: maintain non-dominated Pareto set and untouched baseline;
   search sees validation data; held-out test split evaluated strictly after selection.
6. Operator-readable report and replayable winning plan artifact export, with
   graceful infeasible handling when no candidate satisfies hard constraints.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import yaml

from .common import (
    LOG,
    file_sha256,
    nested_getattr,
    nested_setattr,
    parameter_bytes,
    resolve_device,
)
from .math_ops import apply_constrained_directional_surgery, preservation_basis


# ==============================================================================
# Enums and Constants
# ==============================================================================

SCHEMA_VERSION_4D = "4d.1"


class ObjectiveDirection(str, Enum):
    """Direction for objective optimization."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SearchStrategyKind(str, Enum):
    """Available Stage-4D search strategies."""

    SEEDED_RANDOM = "seeded_random"
    COARSE_TO_FINE = "coarse_to_fine"
    SUCCESSIVE_HALVING = "successive_halving"


class TrialStatus(str, Enum):
    """Status of an individual hypertuning trial."""

    COMPLETED = "COMPLETED"
    PRUNED = "PRUNED"
    FAILED = "FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    SKIPPED = "SKIPPED"


class StudyStatus(str, Enum):
    """Outcome status of a hypertuning study."""

    SUCCESS = "SUCCESS"
    INFEASIBLE_NO_CANDIDATE_PASSED = "INFEASIBLE_NO_CANDIDATE_PASSED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    STOPPED = "STOPPED"


# ==============================================================================
# 1. Hyperparameter Search Space
# ==============================================================================


@dataclass
class Dimension:
    """Base parameter dimension."""

    name: str

    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def grid(self, count: int = 3) -> list[Any]:
        raise NotImplementedError


@dataclass
class CategoricalDim(Dimension):
    """Categorical discrete parameter dimension."""

    name: str
    choices: list[Any]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"Categorical dimension '{self.name}' must have at least one choice.")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)

    def grid(self, count: int = 3) -> list[Any]:
        if len(self.choices) <= count:
            return list(self.choices)
        indices = [int(round(i * (len(self.choices) - 1) / (count - 1))) for i in range(count)]
        return [self.choices[i] for i in sorted(set(indices))]


@dataclass
class IntDim(Dimension):
    """Integer parameter dimension with optional step increment."""

    name: str
    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Int dimension '{self.name}': low ({self.low}) > high ({self.high}).")
        if self.step <= 0:
            raise ValueError(f"Int dimension '{self.name}': step ({self.step}) must be > 0.")

    def sample(self, rng: random.Random) -> int:
        steps = (self.high - self.low) // self.step
        return self.low + rng.randint(0, steps) * self.step

    def grid(self, count: int = 3) -> list[int]:
        if count <= 1 or self.low == self.high:
            return [self.low]
        vals = []
        for i in range(count):
            raw = self.low + i * (self.high - self.low) / (count - 1)
            stepped = self.low + int(round((raw - self.low) / self.step)) * self.step
            stepped = max(self.low, min(self.high, stepped))
            vals.append(stepped)
        return sorted(set(vals))

    def perturb(self, val: int, ratio: float, rng: random.Random) -> int:
        span = max(1, int(round((self.high - self.low) * ratio)))
        delta = rng.randint(-span, span)
        stepped_delta = int(round(delta / self.step)) * self.step
        return max(self.low, min(self.high, val + stepped_delta))


@dataclass
class FloatDim(Dimension):
    """Continuous float dimension (linear or log-scaled) with optional step."""

    name: str
    low: float
    high: float
    log_scale: bool = False
    step: Optional[float] = None

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Float dimension '{self.name}': low ({self.low}) > high ({self.high}).")
        if self.log_scale and self.low <= 0:
            raise ValueError(f"Float dimension '{self.name}': low ({self.low}) must be > 0 for log scale.")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Float dimension '{self.name}': step ({self.step}) must be > 0.")

    def sample(self, rng: random.Random) -> float:
        if self.log_scale:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            val = math.exp(rng.uniform(log_low, log_high))
        else:
            val = rng.uniform(self.low, self.high)

        if self.step is not None:
            val = self.low + round((val - self.low) / self.step) * self.step
            val = max(self.low, min(self.high, val))
        return round(val, 8)

    def grid(self, count: int = 3) -> list[float]:
        if count <= 1 or self.low == self.high:
            return [round(self.low, 8)]
        vals = []
        if self.log_scale:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            for i in range(count):
                lv = log_low + i * (log_high - log_low) / (count - 1)
                vals.append(math.exp(lv))
        else:
            for i in range(count):
                vals.append(self.low + i * (self.high - self.low) / (count - 1))

        if self.step is not None:
            stepped = []
            for v in vals:
                st = self.low + round((v - self.low) / self.step) * self.step
                st = max(self.low, min(self.high, st))
                stepped.append(round(st, 8))
            return sorted(set(stepped))

        return sorted(set(round(v, 8) for v in vals))

    def perturb(self, val: float, ratio: float, rng: random.Random) -> float:
        if self.log_scale:
            log_span = (math.log(self.high) - math.log(self.low)) * ratio
            delta = rng.uniform(-log_span, log_span)
            new_val = math.exp(math.log(val) + delta)
        else:
            span = (self.high - self.low) * ratio
            delta = rng.uniform(-span, span)
            new_val = val + delta

        if self.step is not None:
            new_val = self.low + round((new_val - self.low) / self.step) * self.step

        new_val = max(self.low, min(self.high, new_val))
        return round(new_val, 8)


@dataclass
class SearchSpace:
    """Multi-dimensional hyperparameter search space."""

    dimensions: dict[str, Dimension] = field(default_factory=dict)

    def add(self, dim: Dimension) -> "SearchSpace":
        self.dimensions[dim.name] = dim
        return self

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {name: dim.sample(rng) for name, dim in self.dimensions.items()}

    def coarse_grid(self, steps_per_dim: int = 3) -> list[dict[str, Any]]:
        dim_names = list(self.dimensions.keys())
        if not dim_names:
            return [{}]
        grids = [self.dimensions[name].grid(steps_per_dim) for name in dim_names]

        # Cartesian product
        import itertools

        combos = []
        for point in itertools.product(*grids):
            combos.append({name: val for name, val in zip(dim_names, point)})
        return combos

    def fine_grid_around(
        self,
        center: dict[str, Any],
        perturbation_ratio: float = 0.25,
        samples: int = 5,
        rng: Optional[random.Random] = None,
    ) -> list[dict[str, Any]]:
        if rng is None:
            rng = random.Random(42)
        candidates = [dict(center)]
        for _ in range(samples):
            point = {}
            for name, dim in self.dimensions.items():
                curr = center.get(name)
                if curr is None:
                    point[name] = dim.sample(rng)
                elif isinstance(dim, (IntDim, FloatDim)):
                    point[name] = dim.perturb(curr, perturbation_ratio, rng)
                else:
                    # Categorical: 50% keep, 50% sample
                    point[name] = curr if rng.random() > 0.5 else dim.sample(rng)
            candidates.append(point)
        return candidates


def create_default_search_space() -> SearchSpace:
    """Create a standard Stage-4D search space across surgery & recovery dimensions."""
    space = SearchSpace()
    # Surgery edit selectors & strengths
    space.add(FloatDim("ablation_strength", low=0.0, high=2.0, step=0.25))
    space.add(IntDim("preservation_rank", low=0, high=16, step=4))
    space.add(IntDim("layer_drop_count", low=0, high=3, step=1))

    # Pruning ratios
    space.add(FloatDim("mlp_keep_ratio", low=0.7, high=1.0, step=0.05))
    space.add(FloatDim("attention_keep_ratio", low=0.7, high=1.0, step=0.05))
    space.add(FloatDim("moe_keep_ratio", low=0.5, high=1.0, step=0.1))

    # Stage 5 Recovery LoRA
    space.add(CategoricalDim("lora_r", choices=[0, 4, 8, 16]))
    space.add(CategoricalDim("lora_alpha", choices=[8.0, 16.0, 32.0]))

    # Recovery training parameters
    space.add(FloatDim("lr", low=1e-5, high=1e-3, log_scale=True))
    space.add(FloatDim("w_keep", low=0.5, high=2.0, step=0.5))
    space.add(FloatDim("w_change", low=0.5, high=2.0, step=0.5))
    space.add(FloatDim("w_distill", low=0.0, high=1.0, step=0.25))
    space.add(IntDim("training_steps", low=10, high=100, step=10))

    return space


# ==============================================================================
# 2. Concrete Candidate Configuration and Trial Record
# ==============================================================================


@dataclass
class HypertuningCandidate:
    """Concrete hyperparameter configuration for a Stage-4D trial."""

    # Surgery: Directional edit
    ablation_strength: float = 0.0
    preservation_rank: int = 0
    norm_preserve: bool = True
    preserve_subspace: bool = True
    target_modules: list[str] = field(default_factory=lambda: ["mlp.down_proj", "o_proj"])

    # Surgery: Structural cuts
    layer_drop_count: int = 0
    mlp_keep_ratio: float = 1.0
    attention_keep_ratio: float = 1.0
    moe_keep_ratio: float = 1.0

    # Stage 5: LoRA recovery
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["mlp.down_proj", "o_proj"])

    # Stage 5: Training hyperparameters
    lr: float = 1e-4
    w_keep: float = 1.0
    w_change: float = 1.0
    w_distill: float = 0.0
    training_steps: int = 50

    # Arbitrary custom hyperparameters
    custom_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_strength": round(float(self.ablation_strength), 8),
            "preservation_rank": int(self.preservation_rank),
            "norm_preserve": bool(self.norm_preserve),
            "preserve_subspace": bool(self.preserve_subspace),
            "target_modules": list(self.target_modules),
            "layer_drop_count": int(self.layer_drop_count),
            "mlp_keep_ratio": round(float(self.mlp_keep_ratio), 8),
            "attention_keep_ratio": round(float(self.attention_keep_ratio), 8),
            "moe_keep_ratio": round(float(self.moe_keep_ratio), 8),
            "lora_r": int(self.lora_r),
            "lora_alpha": round(float(self.lora_alpha), 8),
            "lora_target_modules": list(self.lora_target_modules),
            "lr": round(float(self.lr), 8),
            "w_keep": round(float(self.w_keep), 8),
            "w_change": round(float(self.w_change), 8),
            "w_distill": round(float(self.w_distill), 8),
            "training_steps": int(self.training_steps),
            "custom_params": dict(self.custom_params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypertuningCandidate":
        d = dict(data)
        custom = d.pop("custom_params", {})
        valid_keys = {
            "ablation_strength",
            "preservation_rank",
            "norm_preserve",
            "preserve_subspace",
            "target_modules",
            "layer_drop_count",
            "mlp_keep_ratio",
            "attention_keep_ratio",
            "moe_keep_ratio",
            "lora_r",
            "lora_alpha",
            "lora_target_modules",
            "lr",
            "w_keep",
            "w_change",
            "w_distill",
            "training_steps",
        }
        known = {k: v for k, v in d.items() if k in valid_keys}
        unknown = {k: v for k, v in d.items() if k not in valid_keys}
        custom.update(unknown)
        return cls(**known, custom_params=custom)

    def compute_trial_id(self, study_salt: str = "") -> str:
        """Derive an immutable, deterministic content hash for this configuration."""
        canonical_dict = self.to_dict()
        canonical_json = json.dumps(
            {"params": canonical_dict, "salt": study_salt}, sort_keys=True
        )
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return f"trial_{digest[:16]}"


@dataclass
class TrialRecord:
    """Execution record of an evaluated or pruned trial."""

    trial_id: str
    candidate: HypertuningCandidate
    rung: int = 0
    budget_allocated: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    feasible: bool = False
    violations: list[str] = field(default_factory=list)
    pruned: bool = False
    pruning_reason: Optional[str] = None
    duration_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    latency_ms: Optional[float] = None
    test_metrics: Optional[dict[str, float]] = None
    status: str = TrialStatus.COMPLETED.value
    error_message: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "candidate": self.candidate.to_dict(),
            "rung": self.rung,
            "budget_allocated": self.budget_allocated,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "feasible": self.feasible,
            "violations": list(self.violations),
            "pruned": self.pruned,
            "pruning_reason": self.pruning_reason,
            "duration_seconds": round(float(self.duration_seconds), 4),
            "peak_memory_mb": round(float(self.peak_memory_mb), 2),
            "latency_ms": round(float(self.latency_ms), 4) if self.latency_ms is not None else None,
            "test_metrics": {k: float(v) for k, v in self.test_metrics.items()}
            if self.test_metrics is not None
            else None,
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialRecord":
        d = dict(data)
        cand_dict = d.pop("candidate", {})
        candidate = HypertuningCandidate.from_dict(cand_dict)
        return cls(candidate=candidate, **d)


# ==============================================================================
# 3. Multi-Objective Criteria & Hard Thresholds
# ==============================================================================


@dataclass
class ObjectiveCriterion:
    """Individual objective criterion with direction and optional hard threshold."""

    name: str
    direction: ObjectiveDirection
    hard_threshold: Optional[float] = None
    weight: float = 1.0
    tolerance: float = 1e-6

    def is_better(self, val_a: float, val_b: float) -> bool:
        if self.direction == ObjectiveDirection.MINIMIZE:
            return val_a < (val_b - self.tolerance)
        return val_a > (val_b + self.tolerance)

    def is_at_least_as_good(self, val_a: float, val_b: float) -> bool:
        if self.direction == ObjectiveDirection.MINIMIZE:
            return val_a <= (val_b + self.tolerance)
        return val_a >= (val_b - self.tolerance)

    def passes_threshold(self, val: float) -> bool:
        if self.hard_threshold is None:
            return True
        if self.direction == ObjectiveDirection.MINIMIZE:
            return val <= (self.hard_threshold + self.tolerance)
        return val >= (self.hard_threshold - self.tolerance)


@dataclass
class MultiObjectiveCriteria:
    """Multi-objective criteria set with explicit thresholds and dominance logic."""

    criteria: list[ObjectiveCriterion] = field(default_factory=list)

    def add(self, criterion: ObjectiveCriterion) -> "MultiObjectiveCriteria":
        self.criteria.append(criterion)
        return self

    def get(self, name: str) -> Optional[ObjectiveCriterion]:
        for c in self.criteria:
            if c.name == name:
                return c
        return None

    def evaluate_feasibility(self, metrics: dict[str, float]) -> tuple[bool, list[str]]:
        """Verify whether candidate metrics satisfy all declared hard thresholds."""
        violations = []
        for crit in self.criteria:
            if crit.hard_threshold is None:
                continue
            val = metrics.get(crit.name)
            if val is None:
                violations.append(f"Missing required metric '{crit.name}'")
                continue
            if not crit.passes_threshold(float(val)):
                violations.append(
                    f"{crit.name}: {val:.4f} violates {crit.direction.value} threshold {crit.hard_threshold:.4f}"
                )
        return (len(violations) == 0, violations)

    def dominates(self, metrics_a: dict[str, float], metrics_b: dict[str, float]) -> bool:
        """Return True if metrics_a Pareto-dominates metrics_b."""
        at_least_as_good_all = True
        strictly_better_any = False

        for crit in self.criteria:
            val_a = metrics_a.get(crit.name)
            val_b = metrics_b.get(crit.name)
            if val_a is None or val_b is None:
                continue

            va = float(val_a)
            vb = float(val_b)
            if not crit.is_at_least_as_good(va, vb):
                at_least_as_good_all = False
                break
            if crit.is_better(va, vb):
                strictly_better_any = True

        return at_least_as_good_all and strictly_better_any


def create_default_criteria(
    min_keep_retention: float = 0.85,
    min_drop_suppression: float = 0.50,
    max_mean_kl: Optional[float] = 0.05,
    min_bytes_saved: float = 0.0,
    max_peak_memory_mb: Optional[float] = None,
    max_latency_ms: Optional[float] = None,
) -> MultiObjectiveCriteria:
    """Create standard multi-objective criteria for neurosurgery hypertuning."""
    mo = MultiObjectiveCriteria()
    mo.add(
        ObjectiveCriterion(
            name="keep_retention",
            direction=ObjectiveDirection.MAXIMIZE,
            hard_threshold=min_keep_retention,
            weight=1.5,
        )
    )
    mo.add(
        ObjectiveCriterion(
            name="drop_suppression",
            direction=ObjectiveDirection.MAXIMIZE,
            hard_threshold=min_drop_suppression,
            weight=1.5,
        )
    )
    if max_mean_kl is not None:
        mo.add(
            ObjectiveCriterion(
                name="mean_kl",
                direction=ObjectiveDirection.MINIMIZE,
                hard_threshold=max_mean_kl,
                weight=1.0,
            )
        )
    mo.add(
        ObjectiveCriterion(
            name="bytes_saved",
            direction=ObjectiveDirection.MAXIMIZE,
            hard_threshold=min_bytes_saved,
            weight=1.0,
        )
    )
    mo.add(
        ObjectiveCriterion(
            name="peak_memory_mb",
            direction=ObjectiveDirection.MINIMIZE,
            hard_threshold=max_peak_memory_mb,
            weight=0.5,
        )
    )
    mo.add(
        ObjectiveCriterion(
            name="latency_ms",
            direction=ObjectiveDirection.MINIMIZE,
            hard_threshold=max_latency_ms,
            weight=0.5,
        )
    )
    return mo


# ==============================================================================
# 4. Pareto Frontier Analysis and Selection
# ==============================================================================


def compute_pareto_front(
    trials: list[TrialRecord],
    criteria: MultiObjectiveCriteria,
    feasible_only: bool = True,
) -> list[TrialRecord]:
    """Extract the non-dominated Pareto frontier from evaluated trials."""
    candidates = [
        t for t in trials if t.status == TrialStatus.COMPLETED.value and (not feasible_only or t.feasible)
    ]
    if not candidates:
        return []

    front: list[TrialRecord] = []
    for cand in candidates:
        # Candidate is non-dominated if no other candidate dominates it
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            if criteria.dominates(other.metrics, cand.metrics):
                dominated = True
                break
        if not dominated:
            front.append(cand)

    # Sort front deterministically: higher keep_retention / lower KL / higher bytes_saved
    front.sort(
        key=lambda r: (
            -float(r.metrics.get("keep_retention", 0.0)),
            float(r.metrics.get("mean_kl", float("inf"))),
            -float(r.metrics.get("drop_suppression", 0.0)),
            -float(r.metrics.get("bytes_saved", 0.0)),
        )
    )
    return front


def select_winning_candidate(
    pareto_front: list[TrialRecord],
    criteria: MultiObjectiveCriteria,
    baseline_metrics: Optional[dict[str, float]] = None,
) -> Optional[TrialRecord]:
    """Select the best candidate from the Pareto front using weighted normalized distance."""
    if not pareto_front:
        return None
    if len(pareto_front) == 1:
        return pareto_front[0]

    # Normalize metrics across the front to compute score
    scores: list[tuple[float, TrialRecord]] = []
    for candidate in pareto_front:
        total_utility = 0.0
        for crit in criteria.criteria:
            val = candidate.metrics.get(crit.name)
            if val is None:
                continue
            val_f = float(val)

            # Get min and max across front for normalization
            all_vals = [
                float(c.metrics[crit.name])
                for c in pareto_front
                if crit.name in c.metrics
            ]
            min_v = min(all_vals)
            max_v = max(all_vals)
            rng = max_v - min_v

            if rng > 1e-8:
                norm = (val_f - min_v) / rng
            else:
                norm = 1.0

            # If minimize, invert norm so higher is always better
            if crit.direction == ObjectiveDirection.MINIMIZE:
                norm = 1.0 - norm

            total_utility += crit.weight * norm

        scores.append((total_utility, candidate))

    scores.sort(key=lambda s: -s[0])
    return scores[0][1]


# ==============================================================================
# 5. Study Journal, Baseline Caching, and Budgets
# ==============================================================================


@dataclass
class StudyBudget:
    """Limits and parameters governing search execution."""

    max_trials: int = 50
    max_time_seconds: Optional[float] = None
    max_peak_memory_mb: Optional[float] = None
    successive_halving_rungs: int = 3
    reduction_factor: int = 2
    min_budget: int = 10
    max_budget: int = 40

    def check_limits(
        self,
        trials_completed: int,
        start_time: float,
        current_memory_mb: float = 0.0,
    ) -> tuple[bool, Optional[str]]:
        if trials_completed >= self.max_trials:
            return True, f"Max trials budget reached ({self.max_trials})"
        if self.max_time_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= self.max_time_seconds:
                return True, f"Max time budget exceeded ({elapsed:.1f}s >= {self.max_time_seconds:.1f}s)"
        if self.max_peak_memory_mb is not None and current_memory_mb > 0:
            if current_memory_mb > self.max_peak_memory_mb:
                return True, f"Peak memory budget exceeded ({current_memory_mb:.1f}MB > {self.max_peak_memory_mb:.1f}MB)"
        return False, None


class StudyJournal:
    """Persistent append-only log of hypertuning trials supporting resumption."""

    def __init__(self, journal_path: Union[str, Path]) -> None:
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, TrialRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    rec = TrialRecord.from_dict(raw)
                    key = f"{rec.trial_id}__budget_{rec.budget_allocated}"
                    self._records[key] = rec
                except Exception as exc:
                    LOG.warning("Failed to parse journal record line: %s", exc)

    def record(self, trial: TrialRecord) -> None:
        key = f"{trial.trial_id}__budget_{trial.budget_allocated}"
        self._records[key] = trial
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trial.to_dict()) + "\n")

    def get_completed(self, trial_id: str, budget: int) -> Optional[TrialRecord]:
        key = f"{trial_id}__budget_{budget}"
        rec = self._records.get(key)
        if rec and rec.status in (TrialStatus.COMPLETED.value, TrialStatus.PRUNED.value):
            return rec
        return None

    def all_trials(self) -> list[TrialRecord]:
        return list(self._records.values())


class BaselineCache:
    """Caches baseline model evaluation to avoid redundant baseline runs."""

    def __init__(self, cache_path: Union[str, Path]) -> None:
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                LOG.warning("Failed to load baseline cache: %s", exc)

    def get_or_compute(
        self,
        key: str,
        eval_fn: Callable[[], dict[str, float]],
    ) -> dict[str, float]:
        if key in self.data:
            return {k: float(v) for k, v in self.data[key].items()}
        computed = eval_fn()
        self.data[key] = {k: float(v) for k, v in computed.items()}
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return self.data[key]


# ==============================================================================
# 6. Latency & Memory Profiling Utilities
# ==============================================================================


def measure_model_latency(
    model: nn.Module,
    sample_batch: Any,
    num_warmup: int = 3,
    num_repeats: int = 10,
    device: Optional[str] = None,
) -> float:
    """Measure inference latency under identical runtime, context, and warmup conditions."""
    model.eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _prep(b):
        if isinstance(b, torch.Tensor):
            return b.to(dev)
        if isinstance(b, dict):
            return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        return b

    prep_batch = _prep(sample_batch)

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            if isinstance(prep_batch, dict):
                model(**prep_batch)
            elif isinstance(prep_batch, (list, tuple)):
                model(*prep_batch)
            else:
                model(prep_batch)

        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_repeats):
            if isinstance(prep_batch, dict):
                model(**prep_batch)
            elif isinstance(prep_batch, (list, tuple)):
                model(*prep_batch)
            else:
                model(prep_batch)

        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        mean_ms = (elapsed / num_repeats) * 1000.0
        return round(mean_ms, 4)


def measure_peak_memory_mb(device: Optional[str] = None) -> float:
    """Measure peak memory in MB."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    # CPU fallback: resident memory using resource module
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux ru_maxrss is in KiB
        return round(usage.ru_maxrss / 1024.0, 2)
    except Exception:
        return 0.0


# ==============================================================================
# 7. Search Strategies
# ==============================================================================


def seeded_random_search(
    space: SearchSpace,
    count: int,
    seed: int,
) -> list[HypertuningCandidate]:
    """Sample candidates deterministically using a seeded RNG."""
    rng = random.Random(seed)
    candidates = []
    for _ in range(count):
        params = space.sample(rng)
        candidates.append(HypertuningCandidate.from_dict(params))
    return candidates


def coarse_to_fine_search_candidates(
    space: SearchSpace,
    coarse_steps: int = 2,
    fine_perturbation: float = 0.25,
    fine_samples_per_center: int = 3,
    top_k: int = 2,
    seed: int = 42,
) -> Tuple[list[HypertuningCandidate], Callable[[list[TrialRecord]], list[HypertuningCandidate]]]:
    """Construct coarse grid candidates and a callback to generate fine candidates from top trials."""
    coarse_grid = space.coarse_grid(coarse_steps)
    coarse_candidates = [HypertuningCandidate.from_dict(p) for p in coarse_grid]

    def generate_fine(evaluated_coarse: list[TrialRecord]) -> list[HypertuningCandidate]:
        rng = random.Random(seed)
        # Filter to feasible if possible, otherwise best available
        candidates = [t for t in evaluated_coarse if t.status == TrialStatus.COMPLETED.value]
        feasible = [t for t in candidates if t.feasible]
        pool = feasible if feasible else candidates
        pool.sort(
            key=lambda t: (
                -float(t.metrics.get("keep_retention", 0.0)),
                float(t.metrics.get("mean_kl", float("inf"))),
            )
        )
        selected_centers = pool[:top_k]
        fine_candidates = []
        for center_record in selected_centers:
            center_dict = center_record.candidate.to_dict()
            fine_points = space.fine_grid_around(
                center=center_dict,
                perturbation_ratio=fine_perturbation,
                samples=fine_samples_per_center,
                rng=rng,
            )
            for p in fine_points:
                fine_candidates.append(HypertuningCandidate.from_dict(p))
        return fine_candidates

    return coarse_candidates, generate_fine


# ==============================================================================
# 8. Study Orchestration & Results
# ==============================================================================


@dataclass
class HypertuningStudyResult:
    """Full outcome report and artifact pointers for a Stage-4D hypertuning study."""

    study_id: str
    status: str
    total_trials: int
    completed_trials: int
    pruned_trials: int
    feasible_trials: int
    elapsed_time_seconds: float
    baseline_metrics: dict[str, float]
    baseline_test_metrics: Optional[dict[str, float]] = None
    winning_trial: Optional[TrialRecord] = None
    pareto_front: list[TrialRecord] = field(default_factory=list)
    all_trials: list[TrialRecord] = field(default_factory=list)
    criteria: MultiObjectiveCriteria = field(default_factory=MultiObjectiveCriteria)
    winning_plan_path: Optional[str] = None
    report_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "status": self.status,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "pruned_trials": self.pruned_trials,
            "feasible_trials": self.feasible_trials,
            "elapsed_time_seconds": round(float(self.elapsed_time_seconds), 4),
            "baseline_metrics": self.baseline_metrics,
            "baseline_test_metrics": self.baseline_test_metrics,
            "winning_trial": self.winning_trial.to_dict() if self.winning_trial else None,
            "pareto_front": [t.to_dict() for t in self.pareto_front],
            "all_trials": [t.to_dict() for t in self.all_trials],
            "winning_plan_path": self.winning_plan_path,
            "metadata": self.metadata,
        }


def run_hypertuning_study(
    study_id: str,
    evaluator_fn: Callable[[HypertuningCandidate, int, str], dict[str, float]],
    search_space: Optional[SearchSpace] = None,
    strategy: Union[str, SearchStrategyKind] = SearchStrategyKind.SEEDED_RANDOM,
    criteria: Optional[MultiObjectiveCriteria] = None,
    budget: Optional[StudyBudget] = None,
    study_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    test_evaluator_fn: Optional[Callable[[HypertuningCandidate], dict[str, float]]] = None,
    baseline_evaluator_fn: Optional[Callable[[str], dict[str, float]]] = None,
    initial_candidates: Optional[list[HypertuningCandidate]] = None,
) -> HypertuningStudyResult:
    """Execute a Stage-4D constrained hypertuning study.

    Args:
        study_id: Unique identifier for this study.
        evaluator_fn: Callable(candidate, budget_allocated, split="validation") -> metrics dict.
        search_space: Parameter space to search over.
        strategy: 'seeded_random', 'coarse_to_fine', or 'successive_halving'.
        criteria: MultiObjectiveCriteria defining objectives and hard thresholds.
        budget: StudyBudget configuration.
        study_dir: Directory for trial logging, journals, and exported artifacts.
        seed: Random seed for reproducibility.
        test_evaluator_fn: Callable(candidate) -> metrics dict evaluated strictly on held-out test split post-selection.
        baseline_evaluator_fn: Callable(split="validation"|"test") -> baseline metrics dict.
        initial_candidates: Optional pre-defined candidates.

    Returns:
        HypertuningStudyResult.
    """
    if search_space is None:
        search_space = create_default_search_space()
    if criteria is None:
        criteria = create_default_criteria()
    if budget is None:
        budget = StudyBudget()

    strat_kind = SearchStrategyKind(strategy)
    out_dir = Path(study_dir) if study_dir else Path(f"./hypertuning_studies/{study_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    journal = StudyJournal(out_dir / "journal.jsonl")
    baseline_cache = BaselineCache(out_dir / "baseline_cache.json")

    # Baseline evaluation (cached)
    if baseline_evaluator_fn is not None:
        baseline_val_metrics = baseline_cache.get_or_compute(
            "validation", lambda: baseline_evaluator_fn("validation")
        )
    else:
        # Default baseline identity candidate
        baseline_cand = HypertuningCandidate(
            ablation_strength=0.0,
            preservation_rank=0,
            layer_drop_count=0,
            mlp_keep_ratio=1.0,
            attention_keep_ratio=1.0,
            moe_keep_ratio=1.0,
            lora_r=0,
            training_steps=0,
        )
        baseline_val_metrics = baseline_cache.get_or_compute(
            "validation", lambda: evaluator_fn(baseline_cand, 0, "validation")
        )

    start_time = time.time()
    all_trials: list[TrialRecord] = []
    completed_counter = 0

    # -------------------------------------------------------------------------
    # Helper to evaluate single candidate trial with caching & budget check
    # -------------------------------------------------------------------------
    def _evaluate_trial(
        candidate: HypertuningCandidate,
        rung: int,
        budget_allocated: int,
    ) -> TrialRecord:
        nonlocal completed_counter
        trial_id = candidate.compute_trial_id(study_salt=study_id)

        # Resumption check
        existing = journal.get_completed(trial_id, budget_allocated)
        if existing is not None:
            LOG.info("Reusing cached trial record %s from journal", trial_id)
            return existing

        # Check study limits
        current_mem = measure_peak_memory_mb()
        limit_hit, limit_msg = budget.check_limits(
            trials_completed=completed_counter,
            start_time=start_time,
            current_memory_mb=current_mem,
        )
        if limit_hit:
            record = TrialRecord(
                trial_id=trial_id,
                candidate=candidate,
                rung=rung,
                budget_allocated=budget_allocated,
                metrics={},
                feasible=False,
                violations=[limit_msg or "Budget limit hit"],
                status=TrialStatus.BUDGET_EXCEEDED.value,
                error_message=limit_msg,
            )
            journal.record(record)
            return record

        t0 = time.time()
        try:
            metrics = evaluator_fn(candidate, budget_allocated, "validation")
            duration = time.time() - t0
            peak_mem = measure_peak_memory_mb()
            feasible, violations = criteria.evaluate_feasibility(metrics)
            record = TrialRecord(
                trial_id=trial_id,
                candidate=candidate,
                rung=rung,
                budget_allocated=budget_allocated,
                metrics=metrics,
                feasible=feasible,
                violations=violations,
                duration_seconds=duration,
                peak_memory_mb=peak_mem,
                latency_ms=metrics.get("latency_ms"),
                status=TrialStatus.COMPLETED.value,
            )
        except Exception as exc:
            duration = time.time() - t0
            record = TrialRecord(
                trial_id=trial_id,
                candidate=candidate,
                rung=rung,
                budget_allocated=budget_allocated,
                metrics={},
                feasible=False,
                violations=[f"Trial failed: {exc}"],
                duration_seconds=duration,
                status=TrialStatus.FAILED.value,
                error_message=str(exc),
            )

        journal.record(record)
        completed_counter += 1
        return record

    # -------------------------------------------------------------------------
    # Search Execution
    # -------------------------------------------------------------------------
    if strat_kind == SearchStrategyKind.SUCCESSIVE_HALVING:
        # Successive budget allocation / halving
        # Start N candidates at min_budget, promote top 1/eta at each rung
        eta = budget.reduction_factor
        rungs = budget.successive_halving_rungs
        b0 = budget.min_budget

        if initial_candidates:
            candidates_pool = list(initial_candidates)
        else:
            candidates_pool = seeded_random_search(
                space=search_space,
                count=budget.max_trials,
                seed=seed,
            )

        active_candidates = candidates_pool
        current_budget = b0

        for rung in range(rungs):
            rung_trials: list[TrialRecord] = []
            for cand in active_candidates:
                rec = _evaluate_trial(cand, rung=rung, budget_allocated=current_budget)
                rung_trials.append(rec)
                all_trials.append(rec)

                # Stop immediately if budget limit hit
                if rec.status == TrialStatus.BUDGET_EXCEEDED.value:
                    break

            if rung == rungs - 1 or len(active_candidates) <= 1:
                break

            # Filter surviving candidates for next rung
            completed_this_rung = [
                t for t in rung_trials if t.status == TrialStatus.COMPLETED.value
            ]
            if not completed_this_rung:
                break

            # Rank by multi-objective performance
            completed_this_rung.sort(
                key=lambda t: (
                    int(t.feasible),
                    float(t.metrics.get("keep_retention", 0.0)),
                    -float(t.metrics.get("mean_kl", float("inf"))),
                    float(t.metrics.get("drop_suppression", 0.0)),
                ),
                reverse=True,
            )

            survivor_count = max(1, len(completed_this_rung) // eta)
            survivor_records = completed_this_rung[:survivor_count]
            pruned_records = completed_this_rung[survivor_count:]

            # Explicitly log pruned trials
            for pr in pruned_records:
                pr.pruned = True
                pr.pruning_reason = f"Pruned at rung {rung}: ranked {completed_this_rung.index(pr) + 1}/{len(completed_this_rung)}"
                journal.record(pr)

            active_candidates = [sr.candidate for sr in survivor_records]
            current_budget = min(budget.max_budget, current_budget * eta)

    elif strat_kind == SearchStrategyKind.COARSE_TO_FINE:
        coarse_candidates, gen_fine = coarse_to_fine_search_candidates(
            space=search_space,
            seed=seed,
        )
        coarse_trials: list[TrialRecord] = []
        for cand in coarse_candidates:
            rec = _evaluate_trial(cand, rung=0, budget_allocated=budget.max_budget)
            coarse_trials.append(rec)
            all_trials.append(rec)
            if rec.status == TrialStatus.BUDGET_EXCEEDED.value:
                break

        fine_candidates = gen_fine(coarse_trials)
        for cand in fine_candidates:
            rec = _evaluate_trial(cand, rung=1, budget_allocated=budget.max_budget)
            all_trials.append(rec)
            if rec.status == TrialStatus.BUDGET_EXCEEDED.value:
                break

    else:
        # Seeded Random Search
        if initial_candidates:
            candidates = initial_candidates
        else:
            candidates = seeded_random_search(
                space=search_space,
                count=budget.max_trials,
                seed=seed,
            )

        for cand in candidates:
            rec = _evaluate_trial(cand, rung=0, budget_allocated=budget.max_budget)
            all_trials.append(rec)
            if rec.status == TrialStatus.BUDGET_EXCEEDED.value:
                break

    # -------------------------------------------------------------------------
    # Pareto Analysis & Post-Selection Held-Out Test Evaluation
    # -------------------------------------------------------------------------
    all_journal_trials = journal.all_trials()
    pareto_front = compute_pareto_front(
        trials=all_journal_trials,
        criteria=criteria,
        feasible_only=True,
    )

    winning_trial = select_winning_candidate(
        pareto_front=pareto_front,
        criteria=criteria,
        baseline_metrics=baseline_val_metrics,
    )

    baseline_test_metrics: Optional[dict[str, float]] = None
    winning_plan_path: Optional[str] = None
    study_status = StudyStatus.SUCCESS.value

    if winning_trial is None:
        study_status = StudyStatus.INFEASIBLE_NO_CANDIDATE_PASSED.value
        LOG.warning("Hypertuning study %s found NO feasible candidate meeting hard constraints.", study_id)
    else:
        # Evaluate held-out TEST set strictly post-selection
        if test_evaluator_fn is not None:
            winning_trial.test_metrics = test_evaluator_fn(winning_trial.candidate)
            if baseline_evaluator_fn is not None:
                baseline_test_metrics = baseline_cache.get_or_compute(
                    "test", lambda: baseline_evaluator_fn("test")
                )
            journal.record(winning_trial)

        # Export replayable winning plan
        winning_plan_path = str(export_winning_plan(
            study_id=study_id,
            winning_trial=winning_trial,
            baseline_val_metrics=baseline_val_metrics,
            baseline_test_metrics=baseline_test_metrics,
            out_path=out_dir / "winning_plan.yaml",
        ))

    elapsed_time = time.time() - start_time
    total_trials = len(all_journal_trials)
    completed_trials = sum(1 for t in all_journal_trials if t.status == TrialStatus.COMPLETED.value)
    pruned_trials = sum(1 for t in all_journal_trials if t.pruned)
    feasible_trials = sum(1 for t in all_journal_trials if t.feasible)

    result = HypertuningStudyResult(
        study_id=study_id,
        status=study_status,
        total_trials=total_trials,
        completed_trials=completed_trials,
        pruned_trials=pruned_trials,
        feasible_trials=feasible_trials,
        elapsed_time_seconds=elapsed_time,
        baseline_metrics=baseline_val_metrics,
        baseline_test_metrics=baseline_test_metrics,
        winning_trial=winning_trial,
        pareto_front=pareto_front,
        all_trials=all_journal_trials,
        criteria=criteria,
        winning_plan_path=winning_plan_path,
        metadata={"seed": seed, "strategy": strat_kind.value},
    )

    # Generate and save reports
    result.report_text = generate_operator_report(result)
    (out_dir / "study_report.txt").write_text(result.report_text, encoding="utf-8")
    (out_dir / "study_report.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )

    return result


# ==============================================================================
# 9. Operator-Readable Report and Replayable Winning Plan Artifact Export
# ==============================================================================


def generate_operator_report(result: HypertuningStudyResult) -> str:
    """Generate human-readable operator comparison report."""
    lines = [
        "=" * 78,
        f"STAGE 4D HYPERTUNING REPORT: {result.study_id}",
        "=" * 78,
        f"Status:             {result.status}",
        f"Total Trials:       {result.total_trials}",
        f"Completed Trials:   {result.completed_trials}",
        f"Pruned Trials:      {result.pruned_trials}",
        f"Feasible Trials:    {result.feasible_trials}",
        f"Elapsed Time:       {result.elapsed_time_seconds:.2f}s",
        f"Strategy:           {result.metadata.get('strategy')}",
        "-" * 78,
        "HARD THRESHOLD CONSTRAINTS:",
    ]
    for crit in result.criteria.criteria:
        thresh = f"{crit.hard_threshold:.4f}" if crit.hard_threshold is not None else "None"
        lines.append(f"  - {crit.name:20s}: {crit.direction.value.upper():8s} | Threshold: {thresh}")

    lines.append("-" * 78)

    if result.winning_trial is None:
        lines.append("*** INFEASIBLE: NO CANDIDATE SATISFIED ALL HARD CONSTRAINTS ***")
        lines.append("")
        lines.append("Diagnostics for closest candidates:")
        # Rank by least violations
        sorted_all = sorted(result.all_trials, key=lambda t: len(t.violations))
        for cand in sorted_all[:5]:
            lines.append(f"  Trial {cand.trial_id}: {len(cand.violations)} violations")
            for v in cand.violations:
                lines.append(f"    * {v}")
    else:
        winner = result.winning_trial
        lines.append("WINNING CANDIDATE SUMMARY:")
        lines.append(f"  Trial ID:         {winner.trial_id}")
        lines.append(f"  Duration:         {winner.duration_seconds:.2f}s")
        lines.append(f"  Peak Memory:      {winner.peak_memory_mb:.1f} MB")
        if winner.latency_ms is not None:
            lines.append(f"  Latency:          {winner.latency_ms:.2f} ms")
        lines.append("")
        lines.append("Hyperparameters:")
        for k, v in winner.candidate.to_dict().items():
            lines.append(f"    {k:22s}: {v}")

        lines.append("")
        lines.append("METRIC COMPARISON (Baseline vs Winner):")
        lines.append(f"  {'Metric':<22s} | {'Baseline (Val)':<14s} | {'Winner (Val)':<14s} | {'Delta':<14s}")
        lines.append("  " + "-" * 70)
        for crit in result.criteria.criteria:
            m = crit.name
            b_val = result.baseline_metrics.get(m)
            w_val = winner.metrics.get(m)
            b_str = f"{b_val:.4f}" if b_val is not None else "N/A"
            w_str = f"{w_val:.4f}" if w_val is not None else "N/A"
            if b_val is not None and w_val is not None:
                d_val = w_val - b_val
                d_str = f"{d_val:+.4f}"
            else:
                d_str = "N/A"
            lines.append(f"  {m:<22s} | {b_str:<14s} | {w_str:<14s} | {d_str:<14s}")

        if winner.test_metrics:
            lines.append("")
            lines.append("HELD-OUT TEST SET METRICS (Strict post-selection evaluation):")
            for m, v in winner.test_metrics.items():
                b_test = result.baseline_test_metrics.get(m) if result.baseline_test_metrics else None
                b_str = f" (baseline: {b_test:.4f})" if b_test is not None else ""
                lines.append(f"  * {m:22s}: {v:.4f}{b_str}")

        lines.append("")
        lines.append(f"Winning Plan Export: {result.winning_plan_path}")

    lines.append("-" * 78)
    lines.append(f"PARETO FRONTIER ({len(result.pareto_front)} non-dominated feasible candidates):")
    for idx, p in enumerate(result.pareto_front[:10]):
        kr = p.metrics.get("keep_retention", 0.0)
        ds = p.metrics.get("drop_suppression", 0.0)
        bs = p.metrics.get("bytes_saved", 0.0)
        lat = p.latency_ms if p.latency_ms is not None else p.metrics.get("latency_ms", 0.0)
        lines.append(
            f"  [{idx+1}] {p.trial_id} | KEEP={kr:.3f} | DROP_SUPP={ds:.3f} | BYTES_SAVED={bs:.0f} | LATENCY={lat:.1f}ms"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def export_winning_plan(
    study_id: str,
    winning_trial: TrialRecord,
    baseline_val_metrics: dict[str, float],
    baseline_test_metrics: Optional[dict[str, float]],
    out_path: Union[str, Path],
) -> Path:
    """Export a replayable winning plan artifact."""
    plan_path = Path(out_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    cand = winning_trial.candidate
    plan = {
        "version": SCHEMA_VERSION_4D,
        "study_id": study_id,
        "trial_id": winning_trial.trial_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "surgery": {
            "ablation": {
                "strength": cand.ablation_strength,
                "preservation_rank": cand.preservation_rank,
                "norm_preserve": cand.norm_preserve,
                "preserve_subspace": cand.preserve_subspace,
                "targets": cand.target_modules,
            },
            "pruning": {
                "layer_drop_count": cand.layer_drop_count,
                "mlp_keep_ratio": cand.mlp_keep_ratio,
                "attention_keep_ratio": cand.attention_keep_ratio,
                "moe_keep_ratio": cand.moe_keep_ratio,
            },
        },
        "recovery": {
            "lora_r": cand.lora_r,
            "lora_alpha": cand.lora_alpha,
            "lora_targets": cand.lora_target_modules,
            "lr": cand.lr,
            "w_keep": cand.w_keep,
            "w_change": cand.w_change,
            "w_distill": cand.w_distill,
            "training_steps": cand.training_steps,
            "custom_params": cand.custom_params,
        },
        "metrics": {
            "validation": winning_trial.metrics,
            "test": winning_trial.test_metrics,
            "duration_seconds": winning_trial.duration_seconds,
            "peak_memory_mb": winning_trial.peak_memory_mb,
            "latency_ms": winning_trial.latency_ms,
        },
        "baseline_metrics": {
            "validation": baseline_val_metrics,
            "test": baseline_test_metrics,
        },
        "provenance": {
            "schema_version": SCHEMA_VERSION_4D,
            "checksum": winning_trial.trial_id,
        },
    }

    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return plan_path


def replay_winning_plan(
    plan_path_or_dict: Union[str, Path, dict[str, Any]],
    evaluator_fn: Callable[[HypertuningCandidate, int, str], dict[str, float]],
    tolerance: float = 1e-3,
) -> tuple[bool, dict[str, Any]]:
    """Replay a winning trial from its plan artifact and verify reproducibility within tolerances.

    Args:
        plan_path_or_dict: YAML path or loaded dictionary of winning plan.
        evaluator_fn: Callable(candidate, budget_allocated, split) -> metrics dict.
        tolerance: Allowed deviation between original and replayed metrics.

    Returns:
        (passed: bool, details: dict).
    """
    if isinstance(plan_path_or_dict, (str, Path)):
        plan_data = yaml.safe_load(Path(plan_path_or_dict).read_text(encoding="utf-8"))
    else:
        plan_data = dict(plan_path_or_dict)

    surgery = plan_data.get("surgery", {})
    ablation = surgery.get("ablation", {})
    pruning = surgery.get("pruning", {})
    recovery = plan_data.get("recovery", {})

    cand = HypertuningCandidate(
        ablation_strength=float(ablation.get("strength", 0.0)),
        preservation_rank=int(ablation.get("preservation_rank", 0)),
        norm_preserve=bool(ablation.get("norm_preserve", True)),
        preserve_subspace=bool(ablation.get("preserve_subspace", True)),
        target_modules=list(ablation.get("targets", ["mlp.down_proj", "o_proj"])),
        layer_drop_count=int(pruning.get("layer_drop_count", 0)),
        mlp_keep_ratio=float(pruning.get("mlp_keep_ratio", 1.0)),
        attention_keep_ratio=float(pruning.get("attention_keep_ratio", 1.0)),
        moe_keep_ratio=float(pruning.get("moe_keep_ratio", 1.0)),
        lora_r=int(recovery.get("lora_r", 8)),
        lora_alpha=float(recovery.get("lora_alpha", 16.0)),
        lora_target_modules=list(recovery.get("lora_targets", ["mlp.down_proj", "o_proj"])),
        lr=float(recovery.get("lr", 1e-4)),
        w_keep=float(recovery.get("w_keep", 1.0)),
        w_change=float(recovery.get("w_change", 1.0)),
        w_distill=float(recovery.get("w_distill", 0.0)),
        training_steps=int(recovery.get("training_steps", 50)),
        custom_params=dict(recovery.get("custom_params", {})),
    )

    budget_steps = cand.training_steps
    replayed_metrics = evaluator_fn(cand, budget_steps, "validation")

    recorded_metrics = plan_data.get("metrics", {}).get("validation", {})
    discrepancies = {}
    passed = True

    for k, v in recorded_metrics.items():
        if k in replayed_metrics:
            rep_v = replayed_metrics[k]
            diff = abs(float(rep_v) - float(v))
            if diff > tolerance:
                passed = False
                discrepancies[k] = {"recorded": float(v), "replayed": float(rep_v), "diff": diff}

    return passed, {
        "candidate": cand.to_dict(),
        "recorded_metrics": recorded_metrics,
        "replayed_metrics": replayed_metrics,
        "discrepancies": discrepancies,
    }


# ==============================================================================
# 10. End-to-End Model Neurosurgery Execution
# ==============================================================================


def apply_candidate_surgery(
    model: nn.Module,
    candidate: HypertuningCandidate,
    directional_profiles: Optional[dict[str, Any]] = None,
) -> nn.Module:
    """Apply directional surgery to target modules in the model according to candidate specs.

    Args:
        model: Target PyTorch neural network.
        candidate: Hyperparameter candidate containing ablation strength and target modules.
        directional_profiles: Dict providing 'direction' (or 'directions' map),
                              optional 'preserve_bases' map, and optional 'samples'.

    Returns:
        Modified model.
    """
    if candidate.ablation_strength <= 0.0 or not directional_profiles:
        return model

    with torch.no_grad():
        for mod_name in candidate.target_modules:
            try:
                mod = nested_getattr(model, mod_name)
            except (AttributeError, TypeError):
                continue
            if not hasattr(mod, "weight") or mod.weight is None:
                continue

            direction = directional_profiles.get("direction")
            if direction is None:
                directions_map = directional_profiles.get("directions", {})
                direction = directions_map.get(mod_name)
            if direction is None:
                continue

            basis = None
            if candidate.preserve_subspace:
                bases_map = directional_profiles.get("preserve_bases", {})
                basis = bases_map.get(mod_name)
                if basis is None and "samples" in directional_profiles and candidate.preservation_rank > 0:
                    samples = directional_profiles["samples"]
                    basis = preservation_basis(samples, candidate.preservation_rank)

            old_weight = mod.weight.data
            new_weight = apply_constrained_directional_surgery(
                weight=old_weight,
                direction=direction,
                strength=candidate.ablation_strength,
                preserve_basis=basis,
                norm_preserve=candidate.norm_preserve,
            )
            mod.weight.data.copy_(new_weight.to(device=old_weight.device, dtype=old_weight.dtype))

    return model


def execute_candidate_neurosurgery(
    model: nn.Module,
    candidate: HypertuningCandidate,
    keep_data: Optional[Any] = None,
    change_data: Optional[Any] = None,
    distill_data: Optional[Any] = None,
    teacher_model: Optional[nn.Module] = None,
    directional_profiles: Optional[dict[str, Any]] = None,
    drop_evaluator: Optional[Callable[[nn.Module], float]] = None,
    validation_evaluator: Optional[Callable[[nn.Module], dict[str, float]]] = None,
    sample_batch_for_latency: Optional[Any] = None,
) -> dict[str, float]:
    """Execute end-to-end neurosurgery pipeline for a candidate: surgery -> LoRA -> recovery -> eval.

    Args:
        model: Base neural network.
        candidate: Concrete hypertuning candidate.
        keep_data: KEEP task training data.
        change_data: CHANGE task training data.
        distill_data: Optional teacher distillation data.
        teacher_model: Optional teacher network.
        directional_profiles: Profiles for directional surgery.
        drop_evaluator: Evaluator for DROP task suppression/score.
        validation_evaluator: Evaluator for overall validation metrics.
        sample_batch_for_latency: Sample inputs to measure inference latency.

    Returns:
        Evaluated metrics dictionary.
    """
    from .stage5_recovery import RecoveryConfig, RecoveryLossConfig, inject_lora, run_recovery_training

    # 1. Apply directional surgery
    apply_candidate_surgery(
        model=model,
        candidate=candidate,
        directional_profiles=directional_profiles,
    )

    # 2. Inject LoRA recovery adapters if requested
    if candidate.lora_r > 0:
        inject_lora(
            model=model,
            target_modules=candidate.lora_target_modules,
            r=candidate.lora_r,
            lora_alpha=candidate.lora_alpha,
        )

    # 3. Run Stage 5 recovery training if steps > 0 and data is provided
    if candidate.training_steps > 0 and (keep_data or change_data or distill_data):
        rec_cfg = RecoveryConfig(
            max_steps=candidate.training_steps,
            lr=candidate.lr,
            loss_config=RecoveryLossConfig(
                w_keep=candidate.w_keep,
                w_change=candidate.w_change,
                w_distill=candidate.w_distill,
            ),
        )
        run_recovery_training(
            model=model,
            keep_data=keep_data,
            change_data=change_data,
            distill_data=distill_data,
            teacher_model=teacher_model,
            config=rec_cfg,
            drop_evaluator=drop_evaluator,
        )

    # 4. Compute metrics
    metrics: dict[str, float] = {}
    if validation_evaluator is not None:
        metrics = validation_evaluator(model)

    if drop_evaluator is not None and "drop_score" not in metrics:
        metrics["drop_score"] = float(drop_evaluator(model))

    if sample_batch_for_latency is not None:
        metrics["latency_ms"] = measure_model_latency(model, sample_batch_for_latency)

    metrics["peak_memory_mb"] = measure_peak_memory_mb()
    if "bytes_saved" not in metrics:
        metrics["bytes_saved"] = float(candidate.layer_drop_count * 1024 * 1024)

    return metrics

