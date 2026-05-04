"""
Heretic-style refusal ablation configuration models and loaders.

This module is intentionally conservative and dependency-light so the existing
repo can run it without adding the full Heretic dependency stack yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml


def _validate_ratio(value: float, name: str) -> float:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {value}")
    return value


def _validate_unit_interval(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class HereticRefusalConfig:
    """
    Minimal config surface for the Heretic-inspired optimization path.
    """

    model_path: str
    dataset_path: str
    output_dir: str = "exports/heretic_refusal"
    strategy: str = "heretic_refusal"
    seed: int = 42
    n_trials: int = 12
    top_k_layers: int = 3
    train_split: float = 0.7
    val_split: float = 0.2
    max_prompts: Optional[int] = None
    policy_documents: Optional[List[str]] = None
    policy_document_label: str = "unsafe"
    prompt_field: str = "prompt"
    label_field: str = "label"
    prompt_label_map: Optional[Mapping[str, str]] = None
    refusal_weight: float = 1.0
    kl_weight: float = 0.15
    max_parallel_agents: int = 3
    max_runtime_seconds: Optional[float] = None
    min_improvement: float = 0.0
    patience: int = 4
    feasible_refusal_min: float = 0.20
    feasible_kl_max: float = 0.15
    feasible_utility_min: float = 0.55
    max_trial_stagnation: int = 6
    enable_early_stop: bool = True
    enable_optuna: bool = True
    optuna_startup_trials: int = 8
    study_resume: bool = True
    study_name: str = "heretic_refusal"
    study_checkpoint_dir: str = "exports/heretic_refusal/studies"
    study_checkpoint_file: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_ratio(self.train_split, "train_split")
        _validate_ratio(self.val_split, "val_split")
        _validate_unit_interval(self.feasible_refusal_min, "feasible_refusal_min")
        _validate_unit_interval(self.feasible_kl_max, "feasible_kl_max")
        _validate_unit_interval(self.feasible_utility_min, "feasible_utility_min")
        if self.min_improvement < 0.0:
            raise ValueError("min_improvement must be >= 0")
        if self.train_split + self.val_split > 1.0:
            raise ValueError("train_split + val_split must be <= 1.0")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.max_parallel_agents < 1:
            raise ValueError("max_parallel_agents must be >= 1")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.max_trial_stagnation < 1:
            raise ValueError("max_trial_stagnation must be >= 1")
        if self.optuna_startup_trials < 0:
            raise ValueError("optuna_startup_trials must be >= 0")
        if not self.study_name.strip():
            raise ValueError("study_name must not be empty")
        if self.policy_document_label is None:
            raise ValueError("policy_document_label must not be None")
        if not str(self.policy_document_label).strip():
            raise ValueError("policy_document_label must not be empty")
        if self.policy_documents is not None:
            if not isinstance(self.policy_documents, list):
                raise ValueError("policy_documents must be a list of document paths or URLs")
            for idx, item in enumerate(self.policy_documents, start=1):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"policy_documents[{idx}] must be a non-empty string")


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("heretic config must deserialize to a mapping")
    return data


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("heretic config must deserialize to a mapping")
    return data


def load_config(path: Union[str, Path]) -> HereticRefusalConfig:
    """
    Load and normalize Heretic-like refusal-ablation config from yaml/json.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config path not found: {path_obj}")
    if not path_obj.is_file():
        raise ValueError(f"Config path must be a file: {path_obj}")

    suffix = path_obj.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = _read_yaml(path_obj)
    elif suffix == ".json":
        raw = _read_json(path_obj)
    else:
        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .yaml/.yml or .json"
        )

    return HereticRefusalConfig(**raw)


def config_to_dict(cfg: HereticRefusalConfig) -> Dict[str, Any]:
    return {
        "model_path": cfg.model_path,
        "dataset_path": cfg.dataset_path,
        "output_dir": cfg.output_dir,
        "strategy": cfg.strategy,
        "seed": cfg.seed,
        "n_trials": cfg.n_trials,
        "top_k_layers": cfg.top_k_layers,
        "train_split": cfg.train_split,
        "val_split": cfg.val_split,
        "max_prompts": cfg.max_prompts,
        "policy_documents": list(cfg.policy_documents) if cfg.policy_documents else None,
        "policy_document_label": cfg.policy_document_label,
        "prompt_field": cfg.prompt_field,
        "label_field": cfg.label_field,
        "prompt_label_map": dict(cfg.prompt_label_map) if cfg.prompt_label_map else None,
        "refusal_weight": cfg.refusal_weight,
        "kl_weight": cfg.kl_weight,
        "max_parallel_agents": cfg.max_parallel_agents,
        "max_runtime_seconds": cfg.max_runtime_seconds,
        "min_improvement": cfg.min_improvement,
        "patience": cfg.patience,
        "feasible_refusal_min": cfg.feasible_refusal_min,
        "feasible_kl_max": cfg.feasible_kl_max,
        "feasible_utility_min": cfg.feasible_utility_min,
        "max_trial_stagnation": cfg.max_trial_stagnation,
        "enable_early_stop": cfg.enable_early_stop,
        "enable_optuna": cfg.enable_optuna,
        "optuna_startup_trials": cfg.optuna_startup_trials,
        "study_resume": cfg.study_resume,
        "study_name": cfg.study_name,
        "study_checkpoint_dir": cfg.study_checkpoint_dir,
        "study_checkpoint_file": cfg.study_checkpoint_file,
    }
