from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, List, Optional, Set, Union

import torch
import yaml

from .common import LOG, file_sha256, load_tensor_artifact

SCHEMA_VERSION_4A = "4a.1"
ALLOWED_SPLITS = {"discovery", "search", "validation", "test"}
VALID_DATASET_KINDS = {"keep", "change", "drop"}

ALLOWED_PROVENANCE_FIELDS = {
    "schema_version",
    "model_revision",
    "model_hash",
    "tokenizer_hash",
    "config_hash",
    "dataset_fingerprints",
    "adapter_version",
    "seed",
    "dtype",
    "operation_order",
    "profile_hashes",
    "parent_manifest_hash",
    "software_version",
    "created_at",
    "metadata",
}

REQUIRED_PROVENANCE_FIELDS = {
    "schema_version",
    "model_hash",
    "tokenizer_hash",
    "config_hash",
    "dataset_fingerprints",
    "adapter_version",
    "seed",
    "dtype",
    "operation_order",
}

ALLOWED_RESTORATION_FIELDS = {
    "schema_version",
    "source_checkpoint",
    "candidate_checkpoint",
    "source_hashes",
    "candidate_hashes",
    "edited_tensors",
    "backup_location",
    "parent_manifest_hash",
    "created_at",
    "metadata",
}

REQUIRED_RESTORATION_FIELDS = {
    "schema_version",
    "source_checkpoint",
    "candidate_checkpoint",
    "source_hashes",
    "candidate_hashes",
    "edited_tensors",
}


class RestorationIntegrityError(ValueError):
    """Raised when restoration hashes do not match the original checkpoint/tensors."""
    pass


class StaleModelPlanError(ValueError):
    """Raised when a plan's provenance hashes do not match the target model/tokenizer/config."""
    pass


# ==============================================================================
# 1. Dataset split & fingerprinting
# ==============================================================================

@dataclass
class Sample:
    prompt: str
    domain: str = "default"
    target: Optional[str] = None
    forbidden_targets: list[str] = field(default_factory=list)
    forbidden_regexes: list[str] = field(default_factory=list)
    sample_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Sample prompt must be a non-empty string")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("Sample domain must be a non-empty string")
        self.prompt = self.prompt.strip()
        self.domain = self.domain.strip()
        if self.target is not None:
            if not isinstance(self.target, str) or not self.target.strip():
                raise ValueError("Sample target must be a non-empty string when provided")
            self.target = self.target.strip()
        if not isinstance(self.forbidden_targets, list):
            raise ValueError("forbidden_targets must be a list of strings")
        self.forbidden_targets = [str(t).strip() for t in self.forbidden_targets if str(t).strip()]
        if not isinstance(self.forbidden_regexes, list):
            raise ValueError("forbidden_regexes must be a list of regex pattern strings")
        self.forbidden_regexes = [str(r).strip() for r in self.forbidden_regexes if str(r).strip()]
        # Validate regex syntax
        for pattern in self.forbidden_regexes:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid forbidden_regex pattern '{pattern}': {exc}") from exc

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "domain": self.domain,
            "target": self.target,
            "forbidden_targets": list(self.forbidden_targets),
            "forbidden_regexes": list(self.forbidden_regexes),
            "sample_id": self.sample_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        return cls(
            prompt=data["prompt"],
            domain=data.get("domain", "default"),
            target=data.get("target"),
            forbidden_targets=data.get("forbidden_targets", []),
            forbidden_regexes=data.get("forbidden_regexes", []),
            sample_id=data.get("sample_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DatasetFingerprint:
    sha256: str
    sample_count: int
    split_counts: dict[str, int]
    domain_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION_4A

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "sample_count": self.sample_count,
            "split_counts": dict(self.split_counts),
            "domain_counts": dict(self.domain_counts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetFingerprint:
        if data.get("schema_version") != SCHEMA_VERSION_4A:
            raise ValueError(
                f"Unsupported fingerprint schema_version: {data.get('schema_version')}, expected {SCHEMA_VERSION_4A}"
            )
        return cls(
            sha256=data["sha256"],
            sample_count=int(data["sample_count"]),
            split_counts=dict(data.get("split_counts", {})),
            domain_counts=dict(data.get("domain_counts", {})),
            metadata=dict(data.get("metadata", {})),
            schema_version=data.get("schema_version", SCHEMA_VERSION_4A),
        )


class NeurosurgeryDataset:
    """Manages KEEP, CHANGE, or DROP datasets with strict split disjointness and fingerprinting."""

    def __init__(
        self,
        kind: str,
        name: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        norm_kind = kind.lower().strip()
        if norm_kind not in VALID_DATASET_KINDS:
            raise ValueError(f"Invalid dataset kind '{kind}'. Must be one of {sorted(VALID_DATASET_KINDS)}")
        self.kind = norm_kind
        self.name = name.strip() or norm_kind
        self.metadata = dict(metadata or {})
        self.splits: dict[str, list[Sample]] = {
            "discovery": [],
            "search": [],
            "validation": [],
            "test": [],
        }

    def add_sample(self, split: str, sample: Sample) -> None:
        norm_split = split.lower().strip()
        if norm_split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split name '{split}'. Allowed splits: {sorted(ALLOWED_SPLITS)}")
        self._validate_sample_for_kind(sample)
        self.splits[norm_split].append(sample)

    def add_samples(self, split: str, samples: Iterable[Sample]) -> None:
        for s in samples:
            self.add_sample(split, s)

    def get_split(self, split: str) -> list[Sample]:
        norm_split = split.lower().strip()
        if norm_split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split name '{split}'. Allowed splits: {sorted(ALLOWED_SPLITS)}")
        return list(self.splits[norm_split])

    def _validate_sample_for_kind(self, sample: Sample) -> None:
        if self.kind == "change":
            if not sample.target:
                raise ValueError("CHANGE dataset samples must specify a desired target output")
        elif self.kind == "drop":
            if not sample.forbidden_targets and not sample.forbidden_regexes:
                raise ValueError(
                    "DROP dataset samples must specify at least one forbidden target or regex for suppression"
                )

    def verify_disjoint_splits(self) -> None:
        """Verify that discovery, search, validation, and test splits have strictly disjoint prompt hashes."""
        split_by_hash: dict[str, list[tuple[str, str]]] = {}
        for split_name, samples in self.splits.items():
            for sample in samples:
                h = sample.prompt_hash
                if h not in split_by_hash:
                    split_by_hash[h] = []
                split_by_hash[h].append((split_name, sample.prompt))

        overlap_errors = []
        for h, occurrences in split_by_hash.items():
            seen_splits = set(split for split, _ in occurrences)
            if len(seen_splits) > 1:
                sample_text = occurrences[0][1]
                truncated = (sample_text[:40] + "...") if len(sample_text) > 40 else sample_text
                overlap_errors.append(
                    f"Prompt hash {h[:12]} ({truncated!r}) shared across splits: {sorted(seen_splits)}"
                )

        if overlap_errors:
            raise ValueError(
                f"Split disjointness violation in '{self.name}' ({self.kind}):\n" + "\n".join(overlap_errors)
            )

    def verify(self) -> None:
        """Verify all samples and enforce disjoint splits."""
        for split_name, samples in self.splits.items():
            for sample in samples:
                self._validate_sample_for_kind(sample)
        self.verify_disjoint_splits()

    def fingerprint(self) -> DatasetFingerprint:
        """Compute deterministic SHA256 fingerprint, sample counts, and domain metadata."""
        self.verify()

        canonical_records = []
        domain_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        total_samples = 0

        for split_name in sorted(ALLOWED_SPLITS):
            samples = self.splits[split_name]
            split_counts[split_name] = len(samples)
            total_samples += len(samples)

            sorted_samples = sorted(samples, key=lambda s: (s.prompt_hash, s.domain, s.prompt))
            for s in sorted_samples:
                domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1
                canonical_records.append({
                    "split": split_name,
                    "prompt_hash": s.prompt_hash,
                    "prompt": s.prompt,
                    "domain": s.domain,
                    "target": s.target,
                    "forbidden_targets": sorted(s.forbidden_targets),
                    "forbidden_regexes": sorted(s.forbidden_regexes),
                    "sample_id": s.sample_id,
                })

        payload = {
            "kind": self.kind,
            "name": self.name,
            "records": canonical_records,
            "metadata": dict(sorted(self.metadata.items())),
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return DatasetFingerprint(
            sha256=sha256,
            sample_count=total_samples,
            split_counts=split_counts,
            domain_counts=domain_counts,
            metadata=dict(self.metadata),
            schema_version=SCHEMA_VERSION_4A,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "metadata": dict(self.metadata),
            "splits": {
                split_name: [s.to_dict() for s in samples]
                for split_name, samples in self.splits.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeurosurgeryDataset:
        dataset = cls(
            kind=data["kind"],
            name=data.get("name", ""),
            metadata=data.get("metadata", {}),
        )
        for split_name, sample_dicts in data.get("splits", {}).items():
            for sd in sample_dicts:
                dataset.add_sample(split_name, Sample.from_dict(sd))
        return dataset


def save_dataset(dataset: NeurosurgeryDataset, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dataset.to_dict(), indent=2), encoding="utf-8")


def load_dataset(path: Union[str, Path]) -> NeurosurgeryDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return NeurosurgeryDataset.from_dict(data)


def verify_disjoint_dataset_collection(
    datasets: dict[str, NeurosurgeryDataset],
    enforce_cross_dataset: bool = False,
) -> None:
    """Verifies that each dataset in the collection enforces internal disjoint splits,
    and optionally checks that prompt hashes across datasets are disjoint.
    """
    for name, ds in datasets.items():
        ds.verify()

    if enforce_cross_dataset:
        prompt_to_datasets: dict[str, list[str]] = {}
        for ds_name, ds in datasets.items():
            for split_name, samples in ds.splits.items():
                for s in samples:
                    h = s.prompt_hash
                    if h not in prompt_to_datasets:
                        prompt_to_datasets[h] = []
                    prompt_to_datasets[h].append(f"{ds_name}:{split_name}")

        cross_overlaps = []
        for h, occurrences in prompt_to_datasets.items():
            unique_ds = set(occ.split(":")[0] for occ in occurrences)
            if len(unique_ds) > 1:
                cross_overlaps.append(
                    f"Prompt hash {h[:12]} shared across datasets: {sorted(unique_ds)}"
                )

        if cross_overlaps:
            raise ValueError("Cross-dataset disjointness violation:\n" + "\n".join(cross_overlaps))


# ==============================================================================
# 2. Multi-token / sequence-level task evaluation and scoring
# ==============================================================================

def exact_match_score(prediction: str, target: str, ignore_case: bool = False, strip: bool = True) -> float:
    p = prediction.strip() if strip else prediction
    t = target.strip() if strip else target
    if ignore_case:
        p = p.lower()
        t = t.lower()
    return 1.0 if p == t else 0.0


def substring_detected(text: str, substring: str, ignore_case: bool = False) -> bool:
    if not substring:
        return False
    t = text.lower() if ignore_case else text
    s = substring.lower() if ignore_case else substring
    return s in t


def regex_detected(text: str, pattern: str, flags: int = 0) -> bool:
    if not pattern:
        return False
    return bool(re.search(pattern, text, flags=flags))


def evaluate_drop_sample(prediction: str, sample: Sample, ignore_case: bool = True) -> dict[str, Any]:
    """Evaluate target suppression for a DROP sample.
    Suppression succeeds (1.0) when NO forbidden target or regex is detected.
    """
    detected_targets = [
        target for target in sample.forbidden_targets
        if substring_detected(prediction, target, ignore_case=ignore_case)
    ]
    detected_regexes = [
        pattern for pattern in sample.forbidden_regexes
        if regex_detected(prediction, pattern, flags=re.IGNORECASE if ignore_case else 0)
    ]
    detected = detected_targets + detected_regexes
    suppressed = len(detected) == 0
    score = 1.0 if suppressed else 0.0
    leak_rate = 0.0 if suppressed else 1.0
    return {
        "suppressed": suppressed,
        "score": score,
        "leak": leak_rate,
        "detected": detected,
    }


def evaluate_keep_sample(
    prediction: str,
    sample: Sample,
    baseline_prediction: Optional[str] = None,
    metric: str = "exact_match",
    ignore_case: bool = False,
) -> dict[str, Any]:
    """Evaluate retention score for a KEEP sample.
    If sample.target is provided, tests candidate prediction against target.
    If baseline_prediction is provided, calculates damage as max(0, baseline - candidate).
    """
    if metric not in {"exact_match", "substring"}:
        raise ValueError(f"Unsupported evaluation mode '{metric}'. Supported: exact_match, substring")

    if sample.target is None and baseline_prediction is None:
        raise ValueError("KEEP sample evaluation requires either sample.target or baseline_prediction")

    cand_score: float = 0.0
    if sample.target is not None:
        if metric == "exact_match":
            cand_score = exact_match_score(prediction, sample.target, ignore_case=ignore_case)
        elif metric == "substring":
            cand_score = 1.0 if substring_detected(prediction, sample.target, ignore_case=ignore_case) else 0.0

    base_score: float = 1.0
    if baseline_prediction is not None:
        if sample.target is not None:
            if metric == "exact_match":
                base_score = exact_match_score(baseline_prediction, sample.target, ignore_case=ignore_case)
            elif metric == "substring":
                base_score = 1.0 if substring_detected(baseline_prediction, sample.target, ignore_case=ignore_case) else 0.0
        else:
            # Baseline is reference itself
            if metric == "exact_match":
                cand_score = exact_match_score(prediction, baseline_prediction, ignore_case=ignore_case)
            elif metric == "substring":
                cand_score = 1.0 if substring_detected(prediction, baseline_prediction, ignore_case=ignore_case) else 0.0
            base_score = 1.0

    damage = max(0.0, base_score - cand_score)
    if baseline_prediction is not None:
        if prediction == baseline_prediction:
            return {
                "candidate_score": cand_score if sample.target is not None else 1.0,
                "baseline_score": base_score if sample.target is not None else 1.0,
                "retention_score": 1.0,
                "damage": 0.0,
            }
    return {
        "candidate_score": cand_score,
        "baseline_score": base_score,
        "retention_score": cand_score,
        "damage": damage,
    }


def evaluate_change_sample(prediction: str, sample: Sample, ignore_case: bool = True) -> dict[str, Any]:
    """Evaluate target eliciation for a CHANGE sample."""
    if not sample.target:
        raise ValueError("CHANGE sample must have a target")
    # Change success: target is generated (exact or substring)
    em = exact_match_score(prediction, sample.target, ignore_case=ignore_case)
    sub = 1.0 if substring_detected(prediction, sample.target, ignore_case=ignore_case) else 0.0
    success = max(em, sub)
    return {
        "success": success,
        "exact_match": em,
        "substring_match": sub,
    }


@dataclass
class SlicedMetricsReport:
    kind: str
    sample_count: int
    overall_score: float
    worst_slice_score: float
    worst_slice_damage: float
    worst_slice_domain: str
    domain_metrics: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sample_count": self.sample_count,
            "overall_score": self.overall_score,
            "worst_slice_score": self.worst_slice_score,
            "worst_slice_damage": self.worst_slice_damage,
            "worst_slice_domain": self.worst_slice_domain,
            "domain_metrics": self.domain_metrics,
            "metadata": dict(self.metadata),
        }


def evaluate_sliced_sequence_task(
    samples: list[Sample],
    predictions: list[str],
    baseline_predictions: Optional[list[str]] = None,
    kind: str = "keep",
    metric: str = "exact_match",
    ignore_case: bool = False,
) -> SlicedMetricsReport:
    """Computes sliced damage reporting: per-domain metrics, overall score, and worst-slice damage."""
    if not samples:
        raise ValueError("Cannot evaluate an empty sample set")
    if len(samples) != len(predictions):
        raise ValueError(
            f"Sample count ({len(samples)}) does not match prediction count ({len(predictions)})"
        )
    if baseline_predictions is not None and len(baseline_predictions) != len(samples):
        raise ValueError(
            f"Baseline prediction count ({len(baseline_predictions)}) does not match sample count ({len(samples)})"
        )
    norm_kind = kind.lower().strip()
    if norm_kind not in VALID_DATASET_KINDS:
        raise ValueError(f"Invalid evaluation kind '{kind}'. Must be one of {sorted(VALID_DATASET_KINDS)}")

    domain_records: dict[str, list[dict[str, Any]]] = {}

    for idx, sample in enumerate(samples):
        domain = sample.domain
        if domain not in domain_records:
            domain_records[domain] = []

        cand_pred = predictions[idx]
        base_pred = baseline_predictions[idx] if baseline_predictions is not None else None

        if norm_kind == "keep":
            res = evaluate_keep_sample(
                cand_pred, sample, baseline_prediction=base_pred, metric=metric, ignore_case=ignore_case
            )
        elif norm_kind == "drop":
            res = evaluate_drop_sample(cand_pred, sample, ignore_case=ignore_case)
        elif norm_kind == "change":
            res = evaluate_change_sample(cand_pred, sample, ignore_case=ignore_case)

        # Check for non-finite scores
        for k, v in res.items():
            if isinstance(v, (int, float)) and not math.isfinite(v):
                raise ValueError(f"Evaluation produced non-finite score {k}={v} at sample {idx}")

        domain_records[domain].append(res)

    domain_metrics: dict[str, dict[str, Any]] = {}
    all_scores: list[float] = []
    all_damages: list[float] = []

    for domain, recs in domain_records.items():
        count = len(recs)
        if norm_kind == "keep":
            retentions = [r["retention_score"] for r in recs]
            damages = [r["damage"] for r in recs]
            mean_ret = sum(retentions) / count
            mean_dam = sum(damages) / count
            domain_metrics[domain] = {
                "count": count,
                "retention_score": mean_ret,
                "damage": mean_dam,
            }
            all_scores.extend(retentions)
            all_damages.extend(damages)
        elif norm_kind == "drop":
            suppressions = [r["score"] for r in recs]
            leaks = [r["leak"] for r in recs]
            mean_supp = sum(suppressions) / count
            mean_leak = sum(leaks) / count
            domain_metrics[domain] = {
                "count": count,
                "suppression_score": mean_supp,
                "leak_rate": mean_leak,
            }
            all_scores.extend(suppressions)
            all_damages.extend(leaks)
        elif norm_kind == "change":
            successes = [r["success"] for r in recs]
            mean_succ = sum(successes) / count
            domain_metrics[domain] = {
                "count": count,
                "success_rate": mean_succ,
                "damage": 1.0 - mean_succ,
            }
            all_scores.extend(successes)
            all_damages.extend([1.0 - s for s in successes])

    overall_score = sum(all_scores) / len(all_scores)

    if norm_kind == "keep":
        # Worst-slice damage: highest damage slice (or lowest retention)
        worst_domain = max(domain_metrics, key=lambda d: domain_metrics[d]["damage"])
        worst_slice_score = domain_metrics[worst_domain]["retention_score"]
        worst_slice_damage = domain_metrics[worst_domain]["damage"]
    elif norm_kind == "drop":
        # Worst-slice suppression: lowest suppression score (highest leak rate)
        worst_domain = min(domain_metrics, key=lambda d: domain_metrics[d]["suppression_score"])
        worst_slice_score = domain_metrics[worst_domain]["suppression_score"]
        worst_slice_damage = domain_metrics[worst_domain]["leak_rate"]
    else:
        worst_domain = min(domain_metrics, key=lambda d: domain_metrics[d]["success_rate"])
        worst_slice_score = domain_metrics[worst_domain]["success_rate"]
        worst_slice_damage = 1.0 - worst_slice_score

    return SlicedMetricsReport(
        kind=norm_kind,
        sample_count=len(samples),
        overall_score=overall_score,
        worst_slice_score=worst_slice_score,
        worst_slice_damage=worst_slice_damage,
        worst_slice_domain=worst_domain,
        domain_metrics=domain_metrics,
        metadata={"metric": metric},
    )


def evaluate_dataset_predictions(
    dataset: NeurosurgeryDataset,
    split: str,
    predictions: list[str],
    baseline_predictions: Optional[list[str]] = None,
    metric: str = "exact_match",
    ignore_case: bool = False,
) -> SlicedMetricsReport:
    """Evaluates sequence predictions against a specific split of a NeurosurgeryDataset."""
    samples = dataset.get_split(split)
    return evaluate_sliced_sequence_task(
        samples=samples,
        predictions=predictions,
        baseline_predictions=baseline_predictions,
        kind=dataset.kind,
        metric=metric,
        ignore_case=ignore_case,
    )


@dataclass
class EvaluationReport:
    keep_report: Optional[SlicedMetricsReport] = None
    drop_report: Optional[SlicedMetricsReport] = None
    change_report: Optional[SlicedMetricsReport] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep_report": self.keep_report.to_dict() if self.keep_report else None,
            "drop_report": self.drop_report.to_dict() if self.drop_report else None,
            "change_report": self.change_report.to_dict() if self.change_report else None,
            "metadata": dict(self.metadata),
        }

    def gate(self, thresholds: GateThresholds) -> GateResult:
        return gate_evaluation(self, thresholds)

    def publish(
        self,
        out_path: Optional[Union[str, Path]] = None,
        manifest: Optional[ProvenanceManifest] = None,
    ) -> dict[str, Any]:
        return publish_evaluation_report(self, out_path=out_path, manifest=manifest)


@dataclass
class GateThresholds:
    min_keep_retention: Optional[float] = None
    max_keep_worst_slice_damage: Optional[float] = None
    max_keep_mean_damage: Optional[float] = None
    min_drop_suppression: Optional[float] = None
    min_drop_worst_slice_suppression: Optional[float] = None
    max_drop_worst_slice_leak: Optional[float] = None
    min_change_success: Optional[float] = None


@dataclass
class GateResult:
    passed: bool
    keep_passed: bool
    drop_passed: bool
    change_passed: bool
    failures: list[str]
    details: dict[str, Any]


def gate_evaluation(report: EvaluationReport, thresholds: GateThresholds) -> GateResult:
    """Independently gate KEEP and DROP (and CHANGE if specified) against thresholds.
    Rejects missing required task scores, empty thresholds, or non-finite values.
    """
    keep_required = (
        thresholds.min_keep_retention is not None
        or thresholds.max_keep_worst_slice_damage is not None
        or thresholds.max_keep_mean_damage is not None
    )
    drop_required = (
        thresholds.min_drop_suppression is not None
        or thresholds.min_drop_worst_slice_suppression is not None
        or thresholds.max_drop_worst_slice_leak is not None
    )
    change_required = thresholds.min_change_success is not None

    if not keep_required and not drop_required and not change_required:
        raise ValueError("At least one gating threshold must be specified")

    if keep_required and report.keep_report is None:
        raise ValueError("Missing required KEEP evaluation report for gating")
    if drop_required and report.drop_report is None:
        raise ValueError("Missing required DROP evaluation report for gating")
    if change_required and report.change_report is None:
        raise ValueError("Missing required CHANGE evaluation report for gating")

    failures: list[str] = []
    keep_passed = True
    drop_passed = True
    change_passed = True

    # Gate KEEP independently
    if report.keep_report is not None and keep_required:
        kp = report.keep_report
        if not math.isfinite(kp.overall_score) or not math.isfinite(kp.worst_slice_damage):
            raise ValueError(
                f"Non-finite scores in KEEP report: overall={kp.overall_score}, worst_slice_damage={kp.worst_slice_damage}"
            )
        if thresholds.min_keep_retention is not None and kp.overall_score < thresholds.min_keep_retention:
            keep_passed = False
            failures.append(
                f"KEEP retention score {kp.overall_score:.4f} < min threshold {thresholds.min_keep_retention:.4f}"
            )
        if thresholds.max_keep_worst_slice_damage is not None and kp.worst_slice_damage > thresholds.max_keep_worst_slice_damage:
            keep_passed = False
            failures.append(
                f"KEEP worst-slice damage in '{kp.worst_slice_domain}' was {kp.worst_slice_damage:.4f} "
                f"> max threshold {thresholds.max_keep_worst_slice_damage:.4f}"
            )
        if thresholds.max_keep_mean_damage is not None:
            mean_damage = 1.0 - kp.overall_score
            if mean_damage > thresholds.max_keep_mean_damage:
                keep_passed = False
                failures.append(
                    f"KEEP mean damage {mean_damage:.4f} > max threshold {thresholds.max_keep_mean_damage:.4f}"
                )

    # Gate DROP independently
    if report.drop_report is not None and drop_required:
        dp = report.drop_report
        if not math.isfinite(dp.overall_score) or not math.isfinite(dp.worst_slice_damage):
            raise ValueError(
                f"Non-finite scores in DROP report: overall={dp.overall_score}, worst_slice_damage={dp.worst_slice_damage}"
            )
        if thresholds.min_drop_suppression is not None and dp.overall_score < thresholds.min_drop_suppression:
            drop_passed = False
            failures.append(
                f"DROP suppression score {dp.overall_score:.4f} < min threshold {thresholds.min_drop_suppression:.4f}"
            )
        if thresholds.min_drop_worst_slice_suppression is not None and dp.worst_slice_score < thresholds.min_drop_worst_slice_suppression:
            drop_passed = False
            failures.append(
                f"DROP worst-slice suppression in '{dp.worst_slice_domain}' was {dp.worst_slice_score:.4f} "
                f"< min threshold {thresholds.min_drop_worst_slice_suppression:.4f}"
            )
        if thresholds.max_drop_worst_slice_leak is not None and dp.worst_slice_damage > thresholds.max_drop_worst_slice_leak:
            drop_passed = False
            failures.append(
                f"DROP worst-slice leak in '{dp.worst_slice_domain}' was {dp.worst_slice_damage:.4f} "
                f"> max threshold {thresholds.max_drop_worst_slice_leak:.4f}"
            )

    # Gate CHANGE independently
    if report.change_report is not None and change_required:
        cp = report.change_report
        if not math.isfinite(cp.overall_score):
            raise ValueError(f"Non-finite score in CHANGE report: overall={cp.overall_score}")
        if thresholds.min_change_success is not None and cp.overall_score < thresholds.min_change_success:
            change_passed = False
            failures.append(
                f"CHANGE success score {cp.overall_score:.4f} < min threshold {thresholds.min_change_success:.4f}"
            )

    passed = keep_passed and drop_passed and change_passed
    return GateResult(
        passed=passed,
        keep_passed=keep_passed,
        drop_passed=drop_passed,
        change_passed=change_passed,
        failures=failures,
        details={
            "keep_passed": keep_passed,
            "drop_passed": drop_passed,
            "change_passed": change_passed,
        },
    )


def generate_sequence_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 32,
    stop_tokens: Optional[list[str]] = None,
    batch_size: int = 8,
) -> list[str]:
    """Generate multi-token sequence completions for prompts using greedy decoding.
    Supports standard HuggingFace models as well as lightweight mock models.
    Rejects empty prompt sets, invalid prompts, non-positive token limits, and non-finite logits.
    """
    if not prompts:
        raise ValueError("Cannot generate completions for an empty prompt set")
    for idx, p in enumerate(prompts):
        if not isinstance(p, str) or not p.strip():
            raise ValueError(f"Prompt at index {idx} must be a non-empty string")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    device = getattr(model, "device", torch.device("cpu"))
    completions: list[str] = []

    # Check if model has standard .generate()
    has_generate = hasattr(model, "generate") and callable(model.generate)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        if isinstance(enc, dict):
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = enc.to(device)
            attention_mask = None

        prompt_lengths = (
            attention_mask.sum(dim=-1).tolist()
            if attention_mask is not None
            else [input_ids.shape[-1]] * len(batch_prompts)
        )

        if has_generate:
            pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": pad_id,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            with torch.inference_mode():
                out = model.generate(**gen_kwargs)
            for b_idx in range(len(batch_prompts)):
                prefix_len = int(prompt_lengths[b_idx])
                gen_slice = out[b_idx, prefix_len:]
                if hasattr(tokenizer, "decode"):
                    text = tokenizer.decode(gen_slice, skip_special_tokens=True)
                else:
                    text = "".join(str(t.item()) for t in gen_slice)
                if stop_tokens:
                    for st in stop_tokens:
                        if st in text:
                            text = text.split(st)[0]
                completions.append(text)
        else:
            # Fallback greedy step generation for mock / test models
            curr_ids = input_ids.clone()
            curr_mask = attention_mask.clone() if attention_mask is not None else None
            finished = [False] * len(batch_prompts)
            generated_slices: list[list[int]] = [[] for _ in batch_prompts]

            for _ in range(max_new_tokens):
                with torch.inference_mode():
                    call_kwargs = {"input_ids": curr_ids}
                    if curr_mask is not None:
                        call_kwargs["attention_mask"] = curr_mask
                    if hasattr(model, "__call__"):
                        try:
                            logits_out = model(**call_kwargs, use_cache=False, return_dict=True)
                        except TypeError:
                            logits_out = model(curr_ids)
                    else:
                        raise RuntimeError("Model is neither callable nor implements generate")

                logits = getattr(logits_out, "logits", logits_out)
                if not torch.isfinite(logits).all():
                    raise ValueError("Model generated non-finite logits during sequence completion")

                next_tok = torch.argmax(logits[:, -1, :], dim=-1)

                eos_id = getattr(tokenizer, "eos_token_id", None)
                for b_idx, tok_id in enumerate(next_tok):
                    if not finished[b_idx]:
                        if eos_id is not None and tok_id.item() == eos_id:
                            finished[b_idx] = True
                        else:
                            generated_slices[b_idx].append(tok_id.item())

                if all(finished):
                    break

                curr_ids = torch.cat([curr_ids, next_tok.unsqueeze(-1)], dim=-1)
                if curr_mask is not None:
                    curr_mask = torch.cat([curr_mask, torch.ones((len(batch_prompts), 1), device=device, dtype=curr_mask.dtype)], dim=-1)

            for b_idx in range(len(batch_prompts)):
                tok_list = generated_slices[b_idx]
                if hasattr(tokenizer, "decode"):
                    text = tokenizer.decode(tok_list, skip_special_tokens=True)
                else:
                    text = "".join(str(t) for t in tok_list)
                if stop_tokens:
                    for st in stop_tokens:
                        if st in text:
                            text = text.split(st)[0]
                completions.append(text)

    return completions


class BaselineCache:
    """Bounded-memory cache for baseline predictions keyed by (model_hash, prompt, max_new_tokens)."""

    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self.max_entries = max_entries
        self._cache: dict[str, str] = {}
        self._order: list[str] = []

    def _make_key(self, model_hash: str, prompt: str, max_new_tokens: int) -> str:
        raw = f"{model_hash}:{max_new_tokens}:{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model_hash: str, prompt: str, max_new_tokens: int) -> Optional[str]:
        key = self._make_key(model_hash, prompt, max_new_tokens)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None

    def put(self, model_hash: str, prompt: str, max_new_tokens: int, completion: str) -> None:
        key = self._make_key(model_hash, prompt, max_new_tokens)
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_entries:
            oldest_key = self._order.pop(0)
            self._cache.pop(oldest_key, None)
        self._cache[key] = completion
        self._order.append(key)

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


def run_sequence_task_evaluation(
    model: Any,
    tokenizer: Any,
    dataset: NeurosurgeryDataset,
    split: str = "validation",
    baseline_model: Optional[Any] = None,
    baseline_cache: Optional[BaselineCache] = None,
    max_new_tokens: int = 32,
    metric: str = "exact_match",
    ignore_case: bool = False,
    batch_size: int = 8,
) -> SlicedMetricsReport:
    """Executes multi-token generation on a dataset split and evaluates sliced metrics."""
    samples = dataset.get_split(split)
    if not samples:
        raise ValueError(f"No samples found in split '{split}' of dataset '{dataset.name}'")

    prompts = [s.prompt for s in samples]
    predictions = generate_sequence_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    baseline_predictions: Optional[list[str]] = None
    if baseline_model is not None and dataset.kind == "keep":
        if baseline_cache is not None:
            base_hash = compute_model_hash(baseline_model)
            baseline_predictions = []
            needed_prompts: list[str] = []
            needed_indices: list[int] = []
            for idx, p in enumerate(prompts):
                hit = baseline_cache.get(base_hash, p, max_new_tokens)
                if hit is not None:
                    baseline_predictions.append(hit)
                else:
                    baseline_predictions.append("")
                    needed_prompts.append(p)
                    needed_indices.append(idx)

            if needed_prompts:
                gen_needed = generate_sequence_completions(
                    model=baseline_model,
                    tokenizer=tokenizer,
                    prompts=needed_prompts,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                )
                for idx, text in zip(needed_indices, gen_needed):
                    baseline_predictions[idx] = text
                    baseline_cache.put(base_hash, prompts[idx], max_new_tokens, text)
        else:
            baseline_predictions = generate_sequence_completions(
                model=baseline_model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )

    return evaluate_sliced_sequence_task(
        samples=samples,
        predictions=predictions,
        baseline_predictions=baseline_predictions,
        kind=dataset.kind,
        metric=metric,
        ignore_case=ignore_case,
    )


def run_end_to_end_evaluation(
    candidate_model: Any,
    tokenizer: Any,
    datasets: dict[str, NeurosurgeryDataset],
    split: str = "validation",
    baseline_model: Optional[Any] = None,
    baseline_cache: Optional[BaselineCache] = None,
    max_new_tokens: int = 32,
    batch_size: int = 8,
    ignore_case: bool = False,
    metric: str = "exact_match",
) -> EvaluationReport:
    """Runs sequence task evaluations across KEEP, DROP, and CHANGE datasets."""
    verify_disjoint_dataset_collection(datasets)

    keep_report: Optional[SlicedMetricsReport] = None
    drop_report: Optional[SlicedMetricsReport] = None
    change_report: Optional[SlicedMetricsReport] = None

    for kind in ("keep", "drop", "change"):
        if kind in datasets:
            ds = datasets[kind]
            report = run_sequence_task_evaluation(
                model=candidate_model,
                tokenizer=tokenizer,
                dataset=ds,
                split=split,
                baseline_model=baseline_model,
                baseline_cache=baseline_cache,
                max_new_tokens=max_new_tokens,
                metric=metric,
                ignore_case=ignore_case,
                batch_size=batch_size,
            )
            if kind == "keep":
                keep_report = report
            elif kind == "drop":
                drop_report = report
            elif kind == "change":
                change_report = report

    return EvaluationReport(
        keep_report=keep_report,
        drop_report=drop_report,
        change_report=change_report,
        metadata={"split": split, "max_new_tokens": max_new_tokens, "metric": metric},
    )


def publish_evaluation_report(
    report: EvaluationReport,
    out_path: Optional[Union[str, Path]] = None,
    manifest: Optional[ProvenanceManifest] = None,
) -> dict[str, Any]:
    """Publish a reproducible held-out evaluation report with strict validation
    against simulation/fallback scoring and non-finite metrics.
    """
    if report.keep_report is None and report.drop_report is None and report.change_report is None:
        raise ValueError("Cannot publish an empty evaluation report")

    for r in (report.keep_report, report.drop_report, report.change_report):
        if r is not None:
            if not math.isfinite(r.overall_score):
                raise ValueError(f"Non-finite overall_score {r.overall_score} in {r.kind} report")
            if not math.isfinite(r.worst_slice_score):
                raise ValueError(f"Non-finite worst_slice_score {r.worst_slice_score} in {r.kind} report")
            if not math.isfinite(r.worst_slice_damage):
                raise ValueError(f"Non-finite worst_slice_damage {r.worst_slice_damage} in {r.kind} report")
            for dom, m in r.domain_metrics.items():
                for k, v in m.items():
                    if isinstance(v, (int, float)) and not math.isfinite(v):
                        raise ValueError(f"Non-finite metric {dom}.{k}={v} in {r.kind} report")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_4A,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "report": report.to_dict(),
    }
    if manifest is not None:
        payload["provenance"] = manifest.to_dict()
        payload["provenance_hash"] = manifest.compute_manifest_hash()

    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    report_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    payload["report_hash"] = report_hash

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return payload


# ==============================================================================
# 3. Plan and artifact provenance binding
# ==============================================================================

@dataclass
class ProvenanceManifest:
    model_hash: str
    tokenizer_hash: str
    config_hash: str
    dataset_fingerprints: dict[str, Any]
    adapter_version: str
    seed: int
    dtype: str
    operation_order: list[str]
    schema_version: str = SCHEMA_VERSION_4A
    model_revision: Optional[str] = None
    profile_hashes: Optional[dict[str, str]] = None
    parent_manifest_hash: Optional[str] = None
    software_version: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION_4A:
            raise ValueError(
                f"Unsupported provenance schema_version '{self.schema_version}', expected '{SCHEMA_VERSION_4A}'"
            )
        for req in REQUIRED_PROVENANCE_FIELDS:
            val = getattr(self, req, None)
            if val is None or (isinstance(val, (str, list, dict)) and len(val) == 0):
                raise ValueError(f"Schema validation error: missing or empty required field '{req}'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"Seed must be an integer, got {type(self.seed).__name__}")
        if not isinstance(self.operation_order, list) or not all(isinstance(x, str) for x in self.operation_order):
            raise ValueError("operation_order must be a list of string operation names")
        if not isinstance(self.dataset_fingerprints, dict):
            raise ValueError("dataset_fingerprints must be a mapping of dataset names to fingerprints")

    def compute_manifest_hash(self) -> str:
        canonical_json = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_hash": self.model_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "config_hash": self.config_hash,
            "dataset_fingerprints": {
                k: (v.to_dict() if isinstance(v, DatasetFingerprint) else v)
                for k, v in self.dataset_fingerprints.items()
            },
            "adapter_version": self.adapter_version,
            "seed": self.seed,
            "dtype": self.dtype,
            "operation_order": list(self.operation_order),
        }
        if self.model_revision is not None:
            d["model_revision"] = self.model_revision
        if self.profile_hashes is not None:
            d["profile_hashes"] = dict(self.profile_hashes)
        if self.parent_manifest_hash is not None:
            d["parent_manifest_hash"] = self.parent_manifest_hash
        if self.software_version is not None:
            d["software_version"] = self.software_version
        if self.created_at is not None:
            d["created_at"] = self.created_at
        if self.metadata is not None:
            d["metadata"] = dict(self.metadata)
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceManifest:
        if not isinstance(data, dict):
            raise ValueError("Provenance payload must be a mapping")
        unknown = set(data.keys()) - ALLOWED_PROVENANCE_FIELDS
        if unknown:
            raise ValueError(f"Schema validation error: unknown fields in provenance: {sorted(unknown)}")
        if data.get("schema_version") != SCHEMA_VERSION_4A:
            raise ValueError(
                f"Unsupported provenance schema_version '{data.get('schema_version')}', expected '{SCHEMA_VERSION_4A}'"
            )
        for req in REQUIRED_PROVENANCE_FIELDS:
            if req not in data or data[req] is None or (isinstance(data[req], (str, list, dict)) and len(data[req]) == 0):
                raise ValueError(f"Schema validation error: missing or empty required field '{req}'")

        return cls(
            schema_version=data["schema_version"],
            model_hash=data["model_hash"],
            tokenizer_hash=data["tokenizer_hash"],
            config_hash=data["config_hash"],
            dataset_fingerprints=data["dataset_fingerprints"],
            adapter_version=data["adapter_version"],
            seed=data["seed"],
            dtype=data["dtype"],
            operation_order=data["operation_order"],
            model_revision=data.get("model_revision"),
            profile_hashes=data.get("profile_hashes"),
            parent_manifest_hash=data.get("parent_manifest_hash"),
            software_version=data.get("software_version"),
            created_at=data.get("created_at"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> ProvenanceManifest:
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(path_or_str)
        data = yaml.safe_load(content)
        return cls.from_dict(data)


def bind_provenance_to_plan(plan: dict[str, Any], provenance: ProvenanceManifest) -> dict[str, Any]:
    """Binds provenance manifest and its hash into an surgery plan."""
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dictionary")
    provenance.validate()
    bound_plan = copy.deepcopy(plan)
    bound_plan["provenance"] = provenance.to_dict()
    bound_plan["provenance_hash"] = provenance.compute_manifest_hash()
    return bound_plan


def validate_plan_provenance(plan: dict[str, Any], expected: Optional[ProvenanceManifest] = None) -> bool:
    """Validates the provenance structure bound within a plan and optional expected manifest matching."""
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dictionary")
    if "provenance" not in plan:
        raise ValueError("Plan does not contain a 'provenance' section")

    provenance_data = plan["provenance"]
    manifest = ProvenanceManifest.from_dict(provenance_data)

    if "provenance_hash" in plan:
        computed_hash = manifest.compute_manifest_hash()
        if plan["provenance_hash"] != computed_hash:
            raise ValueError(
                f"Plan provenance_hash mismatch: recorded '{plan['provenance_hash']}', computed '{computed_hash}'"
            )

    if expected is not None:
        if manifest.to_dict() != expected.to_dict():
            raise ValueError("Plan provenance does not match expected provenance manifest")

    return True


def compute_model_hash(model_or_state_dict: Union[torch.nn.Module, dict[str, torch.Tensor]]) -> str:
    """Compute deterministic SHA256 hash of a full model or state dict."""
    if isinstance(model_or_state_dict, torch.nn.Module):
        sd = model_or_state_dict.state_dict()
    elif isinstance(model_or_state_dict, dict):
        sd = model_or_state_dict
    else:
        raise TypeError(f"Expected nn.Module or dict, got {type(model_or_state_dict).__name__}")
    tensor_hashes = compute_state_dict_hashes(sd)
    canonical = json.dumps(tensor_hashes, sort_keys=True)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def compute_tokenizer_hash(tokenizer: Any) -> str:
    """Compute deterministic SHA256 hash of a tokenizer."""
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, "vocab"):
        vocab = tokenizer.vocab
    else:
        vocab = str(tokenizer)
    canonical = json.dumps(vocab, sort_keys=True) if isinstance(vocab, dict) else str(vocab)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def compute_config_hash(config: Any) -> str:
    """Compute deterministic SHA256 hash of a model configuration."""
    if hasattr(config, "to_dict"):
        d = config.to_dict()
    elif isinstance(config, dict):
        d = config
    else:
        d = getattr(config, "__dict__", str(config))
    canonical = json.dumps(d, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def validate_plan_against_model(
    plan: dict[str, Any],
    model: Union[torch.nn.Module, dict[str, torch.Tensor]],
    tokenizer: Optional[Any] = None,
    config: Optional[Any] = None,
) -> bool:
    """Validates that a plan's bound provenance matches the actual target model, tokenizer, and config.
    Raises StaleModelPlanError if model/tokenizer/config hash does not match.
    """
    validate_plan_provenance(plan)
    provenance = ProvenanceManifest.from_dict(plan["provenance"])
    current_model_hash = compute_model_hash(model)
    if provenance.model_hash != current_model_hash:
        raise StaleModelPlanError(
            f"Stale model plan rejected: plan bound to model_hash '{provenance.model_hash}', "
            f"but current model has '{current_model_hash}'"
        )
    if tokenizer is not None:
        current_tok_hash = compute_tokenizer_hash(tokenizer)
        if provenance.tokenizer_hash != current_tok_hash:
            raise StaleModelPlanError(
                f"Stale model plan rejected: plan bound to tokenizer_hash '{provenance.tokenizer_hash}', "
                f"but current tokenizer has '{current_tok_hash}'"
            )
    if config is not None:
        current_cfg_hash = compute_config_hash(config)
        if provenance.config_hash != current_cfg_hash:
            raise StaleModelPlanError(
                f"Stale model plan rejected: plan bound to config_hash '{provenance.config_hash}', "
                f"but current config has '{current_cfg_hash}'"
            )
    return True


# ==============================================================================
# 4. Reversible artifact restoration
# ==============================================================================

def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA256 hash of a tensor including its shape, dtype, and bytes."""
    t = tensor.detach().reshape(-1).contiguous().cpu()
    raw_bytes = t.view(torch.uint8).numpy().tobytes()
    header = f"{str(tensor.dtype)}:{list(tensor.shape)}:".encode("utf-8")
    return hashlib.sha256(header + raw_bytes).hexdigest()


def compute_state_dict_hashes(state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    """Compute per-tensor hashes for a PyTorch state dict."""
    return {
        key: compute_tensor_hash(tensor)
        for key, tensor in sorted(state_dict.items())
    }


@dataclass
class RestorationManifest:
    source_checkpoint: str
    candidate_checkpoint: str
    source_hashes: dict[str, str]
    candidate_hashes: dict[str, str]
    edited_tensors: list[str]
    backup_location: Optional[str] = None
    parent_manifest_hash: Optional[str] = None
    schema_version: str = SCHEMA_VERSION_4A
    created_at: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION_4A:
            raise ValueError(
                f"Unsupported restoration schema_version '{self.schema_version}', expected '{SCHEMA_VERSION_4A}'"
            )
        for req in REQUIRED_RESTORATION_FIELDS:
            val = getattr(self, req, None)
            if val is None or (isinstance(val, (str, list, dict)) and len(val) == 0):
                raise ValueError(f"Schema validation error: missing or empty required field '{req}'")
        if not isinstance(self.edited_tensors, list) or not all(isinstance(x, str) for x in self.edited_tensors):
            raise ValueError("edited_tensors must be a list of string tensor names")
        if not isinstance(self.source_hashes, dict) or not isinstance(self.candidate_hashes, dict):
            raise ValueError("source_hashes and candidate_hashes must be mappings of tensor names to SHA256 hashes")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source_checkpoint": self.source_checkpoint,
            "candidate_checkpoint": self.candidate_checkpoint,
            "source_hashes": dict(self.source_hashes),
            "candidate_hashes": dict(self.candidate_hashes),
            "edited_tensors": list(self.edited_tensors),
        }
        if self.backup_location is not None:
            d["backup_location"] = self.backup_location
        if self.parent_manifest_hash is not None:
            d["parent_manifest_hash"] = self.parent_manifest_hash
        if self.created_at is not None:
            d["created_at"] = self.created_at
        if self.metadata is not None:
            d["metadata"] = dict(self.metadata)
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RestorationManifest:
        if not isinstance(data, dict):
            raise ValueError("Restoration manifest payload must be a mapping")
        unknown = set(data.keys()) - ALLOWED_RESTORATION_FIELDS
        if unknown:
            raise ValueError(f"Schema validation error: unknown fields in restoration manifest: {sorted(unknown)}")
        if data.get("schema_version") != SCHEMA_VERSION_4A:
            raise ValueError(
                f"Unsupported restoration schema_version '{data.get('schema_version')}', expected '{SCHEMA_VERSION_4A}'"
            )
        for req in REQUIRED_RESTORATION_FIELDS:
            if req not in data or data[req] is None or (isinstance(data[req], (str, list, dict)) and len(data[req]) == 0):
                raise ValueError(f"Schema validation error: missing or empty required field '{req}'")

        return cls(
            schema_version=data["schema_version"],
            source_checkpoint=data["source_checkpoint"],
            candidate_checkpoint=data["candidate_checkpoint"],
            source_hashes=data["source_hashes"],
            candidate_hashes=data["candidate_hashes"],
            edited_tensors=data["edited_tensors"],
            backup_location=data.get("backup_location"),
            parent_manifest_hash=data.get("parent_manifest_hash"),
            created_at=data.get("created_at"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> RestorationManifest:
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(path_or_str)
        data = yaml.safe_load(content)
        return cls.from_dict(data)


def create_restoration_manifest(
    source_checkpoint: str,
    candidate_checkpoint: str,
    source_state_dict: dict[str, torch.Tensor],
    candidate_state_dict: dict[str, torch.Tensor],
    edited_tensors: list[str],
    backup_location: Optional[str] = None,
    parent_manifest_hash: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> RestorationManifest:
    """Create a restoration manifest tracking original and candidate tensor hashes."""
    if not edited_tensors:
        raise ValueError("edited_tensors list cannot be empty")
    source_hashes = compute_state_dict_hashes(source_state_dict)
    candidate_hashes = compute_state_dict_hashes(candidate_state_dict)

    for key in edited_tensors:
        if key not in source_hashes:
            raise KeyError(f"Edited tensor '{key}' not found in source state dict")
        if key not in candidate_hashes:
            raise KeyError(f"Edited tensor '{key}' not found in candidate state dict")

    return RestorationManifest(
        schema_version=SCHEMA_VERSION_4A,
        source_checkpoint=source_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        source_hashes=source_hashes,
        candidate_hashes=candidate_hashes,
        edited_tensors=sorted(set(edited_tensors)),
        backup_location=backup_location,
        parent_manifest_hash=parent_manifest_hash,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metadata=metadata,
    )


def backup_edited_tensors(
    source_state_dict: dict[str, torch.Tensor],
    edited_tensors: list[str],
) -> dict[str, torch.Tensor]:
    """Extract a deep clone backup of the original tensors that are targeted for surgery."""
    backup: dict[str, torch.Tensor] = {}
    for key in edited_tensors:
        if key not in source_state_dict:
            raise KeyError(f"Target tensor '{key}' not found in source state dict")
        backup[key] = source_state_dict[key].clone().detach()
    return backup


def restore_state_dict(
    candidate_state_dict: dict[str, torch.Tensor],
    backup_tensors: dict[str, torch.Tensor],
    manifest: RestorationManifest,
) -> dict[str, torch.Tensor]:
    """Restore candidate state dict back to original tensors using backup and manifest."""
    manifest.validate()
    restored = copy.copy(candidate_state_dict)

    for key in manifest.edited_tensors:
        if key not in backup_tensors:
            raise KeyError(f"Required backup tensor '{key}' not found in backup state dict")
        restored[key] = backup_tensors[key].clone().detach()

    return restored


def verify_restoration_integrity(
    restored_state_dict: dict[str, torch.Tensor],
    manifest: RestorationManifest,
    check_all_source_tensors: bool = True,
) -> dict[str, Any]:
    """Verify that restored tensors match original checkpoint hashes recorded in manifest."""
    manifest.validate()
    mismatches: list[str] = []
    missing: list[str] = []

    tensors_to_check = (
        manifest.source_hashes.keys()
        if check_all_source_tensors
        else manifest.edited_tensors
    )

    for key in tensors_to_check:
        if key not in restored_state_dict:
            missing.append(key)
            continue
        expected_hash = manifest.source_hashes[key]
        actual_hash = compute_tensor_hash(restored_state_dict[key])
        if actual_hash != expected_hash:
            mismatches.append(
                f"Tensor '{key}': expected SHA256 {expected_hash}, got {actual_hash}"
            )

    if missing:
        raise RestorationIntegrityError(f"Missing required tensors during restoration verification: {missing}")

    if mismatches:
        raise RestorationIntegrityError(
            "Restoration integrity verification failed. Tensor hash mismatches:\n" + "\n".join(mismatches)
        )

    return {
        "status": "verified",
        "tensors_verified": len(tensors_to_check),
    }


def compute_directory_file_hashes(dir_path: Union[str, Path]) -> dict[str, str]:
    """Compute relative file path -> SHA256 mapping for all files in a directory."""
    root = Path(dir_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    hashes: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            hashes[rel] = file_sha256(p)
    return hashes


def create_file_restoration_manifest(
    source_dir: Union[str, Path],
    candidate_dir: Union[str, Path],
    edited_files: list[str],
    backup_dir: Optional[Union[str, Path]] = None,
    parent_manifest_hash: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> RestorationManifest:
    """Create a restoration manifest tracking original and candidate checkpoint file hashes."""
    s_root = Path(source_dir)
    c_root = Path(candidate_dir)
    if not s_root.exists() or not s_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {s_root}")
    if not c_root.exists() or not c_root.is_dir():
        raise FileNotFoundError(f"Candidate directory not found: {c_root}")
    if not edited_files:
        raise ValueError("edited_files cannot be empty")

    source_hashes = compute_directory_file_hashes(s_root)
    candidate_hashes = compute_directory_file_hashes(c_root)

    for f in edited_files:
        if f not in source_hashes:
            raise KeyError(f"Edited file '{f}' not found in source directory")
        if f not in candidate_hashes:
            raise KeyError(f"Edited file '{f}' not found in candidate directory")

    return RestorationManifest(
        schema_version=SCHEMA_VERSION_4A,
        source_checkpoint=str(source_dir),
        candidate_checkpoint=str(candidate_dir),
        source_hashes=source_hashes,
        candidate_hashes=candidate_hashes,
        edited_tensors=sorted(set(edited_files)),
        backup_location=str(backup_dir) if backup_dir else None,
        parent_manifest_hash=parent_manifest_hash,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metadata=metadata,
    )


def backup_checkpoint_files(
    source_dir: Union[str, Path],
    edited_files: list[str],
    backup_dir: Union[str, Path],
) -> None:
    """Safely copy target files from source to backup directory (source remains untouched and immutable)."""
    import shutil
    s_root = Path(source_dir)
    b_root = Path(backup_dir)
    b_root.mkdir(parents=True, exist_ok=True)
    for f in edited_files:
        src = s_root / f
        dst = b_root / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def restore_checkpoint_files_from_backup(
    candidate_dir: Union[str, Path],
    backup_dir: Union[str, Path],
    manifest: RestorationManifest,
    target_dir: Union[str, Path],
) -> None:
    """Restore candidate checkpoint directory back to original using backup files without modifying source."""
    import shutil
    manifest.validate()
    c_root = Path(candidate_dir)
    b_root = Path(backup_dir)
    t_root = Path(target_dir)

    if t_root.exists():
        shutil.rmtree(t_root)
    shutil.copytree(c_root, t_root)

    for f in manifest.edited_tensors:
        src = b_root / f
        dst = t_root / f
        if not src.exists():
            raise FileNotFoundError(f"Backup file not found for restore: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def verify_restored_directory(
    restored_dir: Union[str, Path],
    manifest: RestorationManifest,
    check_all_files: bool = True,
) -> dict[str, Any]:
    """Verify that restored directory files match the hashes in manifest.source_hashes."""
    manifest.validate()
    r_root = Path(restored_dir)
    if not r_root.exists() or not r_root.is_dir():
        raise FileNotFoundError(f"Restored directory not found: {r_root}")

    files_to_check = manifest.source_hashes.keys() if check_all_files else manifest.edited_tensors
    mismatches: list[str] = []
    missing: list[str] = []

    for f in files_to_check:
        p = r_root / f
        if not p.exists():
            missing.append(f)
            continue
        actual = file_sha256(p)
        expected = manifest.source_hashes[f]
        if actual != expected:
            mismatches.append(f"File '{f}': expected SHA256 {expected}, got {actual}")

    if missing:
        raise RestorationIntegrityError(f"Missing required files during restoration verification: {missing}")
    if mismatches:
        raise RestorationIntegrityError(
            "Restoration file integrity verification failed:\n" + "\n".join(mismatches)
        )

    return {"status": "verified", "files_verified": len(files_to_check)}


verify_checkpoint_files = verify_restored_directory


def export_candidate_checkpoint(
    candidate_state_dict: dict[str, torch.Tensor],
    export_dir: Union[str, Path],
    provenance: ProvenanceManifest,
    restoration_manifest: Optional[RestorationManifest] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Export a candidate model checkpoint to a separate directory, preserving immutability of the source.
    Writes tensors (with weights_only compatibility), config, provenance manifest, and optional restoration manifest.
    Returns dictionary of written file paths.
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_file = out_dir / "candidate_model.pt"
    torch.save(candidate_state_dict, weights_file)

    prov_file = out_dir / "provenance.yaml"
    provenance.save(prov_file)

    written = {
        "weights": str(weights_file),
        "provenance": str(prov_file),
    }

    if restoration_manifest is not None:
        rest_file = out_dir / "restoration.yaml"
        restoration_manifest.save(rest_file)
        written["restoration"] = str(rest_file)

    if config is not None:
        cfg_file = out_dir / "config.json"
        cfg_file.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        written["config"] = str(cfg_file)

    return written


def load_candidate_checkpoint(
    export_dir: Union[str, Path],
) -> tuple[dict[str, torch.Tensor], ProvenanceManifest, Optional[RestorationManifest]]:
    """Loads an exported candidate checkpoint using weights_only=True and validates provenance."""
    in_dir = Path(export_dir)
    weights_file = in_dir / "candidate_model.pt"
    prov_file = in_dir / "provenance.yaml"
    rest_file = in_dir / "restoration.yaml"

    if not weights_file.exists():
        raise FileNotFoundError(f"Candidate weights not found: {weights_file}")
    if not prov_file.exists():
        raise FileNotFoundError(f"Candidate provenance not found: {prov_file}")

    state_dict = load_tensor_artifact(weights_file)
    provenance = ProvenanceManifest.from_yaml(prov_file)
    provenance.validate()

    restoration_manifest = None
    if rest_file.exists():
        restoration_manifest = RestorationManifest.from_yaml(rest_file)
        restoration_manifest.validate()

    return state_dict, provenance, restoration_manifest
