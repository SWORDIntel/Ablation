"""
Hardware-aware model suggestion helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aegis_lab.hardware.discovery import HardwareDiscovery


@dataclass(frozen=True)
class ModelSuggestion:
    model_id: str
    reason: str
    task: str
    recommended_quantization: str
    estimated_memory_gb: float
    score: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "reason": self.reason,
            "recommended_quantization": self.recommended_quantization,
            "estimated_memory_gb": self.estimated_memory_gb,
            "score": round(self.score, 4),
        }


_MODEL_CATALOG = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Low-memory, CPU-first",
        "task": "general",
        "memory_gb": 1.0,
        "base_score": 0.55,
        "preferred_devices": {"CPU"},
        "accelerator_preference": {"CPU"},
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Balanced entry model",
        "task": "general",
        "memory_gb": 2.0,
        "base_score": 0.70,
        "preferred_devices": {"CPU", "iGPU"},
        "accelerator_preference": {"NPU", "iGPU", "CPU"},
    },
    {
        "id": "google/gemma-2-9b-it",
        "label": "Quality-first iGPU/NPU profile",
        "task": "ablation",
        "memory_gb": 4.5,
        "base_score": 0.82,
        "preferred_devices": {"NPU", "iGPU", "CPU"},
        "accelerator_preference": {"NPU", "iGPU"},
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "High-capability for strong hardware",
        "task": "chat",
        "memory_gb": 8.0,
        "base_score": 0.95,
        "preferred_devices": {"NPU", "CUDA", "CPU_AMX"},
        "accelerator_preference": {"NPU", "iGPU", "VPU"},
    },
    {
        "id": "microsoft/Phi-3.5-Mini-Instruct",
        "label": "Small reasoning-friendly benchmark",
        "task": "general",
        "memory_gb": 3.1,
        "base_score": 0.78,
        "preferred_devices": {"CPU", "iGPU", "NPU"},
        "accelerator_preference": {"CPU", "NPU", "iGPU"},
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Throughput-oriented ablation baseline",
        "task": "ablation",
        "memory_gb": 5.2,
        "base_score": 0.84,
        "preferred_devices": {"NPU", "CUDA", "iGPU", "CPU"},
        "accelerator_preference": {"NPU", "CUDA", "iGPU"},
    },
]


def _is_powerful_hardware(profile: Dict[str, Any]) -> bool:
    if profile.get("hardware_tier") == "HIGH_PERF_SERVER":
        return True
    return profile.get("cpu_amx") or profile.get("cpu_avx512") or profile.get("cpu_vnni")


def _preferred_quantization(profile: Dict[str, Any]) -> str:
    if profile.get("npu_present") or profile.get("vpu_present"):
        return "int8"
    if profile.get("cpu_amx") or profile.get("cpu_avx512"):
        return "bf16"
    if profile.get("cpu_avx2") or profile.get("cpu_vnni"):
        return "int8"
    return "fp16"


def suggest_models(
    hardware_profile: Optional[Dict[str, Any]] = None,
    task: str = "general",
    top_k: int = 3,
) -> List[ModelSuggestion]:
    """
    Return ordered model recommendations for the current environment.
    """
    profile = hardware_profile or HardwareDiscovery.discover()
    target_task = task.lower().strip() or "general"
    quantization = _preferred_quantization(profile)
    score_penalty = 0.15 if _is_powerful_hardware(profile) else 0.0
    result: List[ModelSuggestion] = []

    for item in _MODEL_CATALOG:
        task_match = item["task"] == target_task or item["task"] == "general"
        if not task_match:
            continue

        score = item["base_score"]
        preferred_devices = set(item["accelerator_preference"])
        if profile.get("npu_present") and "NPU" in preferred_devices:
            score += 0.20
        if profile.get("igpu_present") and "iGPU" in preferred_devices:
            score += 0.12
        if profile.get("vpu_present") and "VPU" in preferred_devices:
            score += 0.08
        if profile.get("accel_available") is False:
            if "CPU" in preferred_devices:
                score += 0.10
            else:
                score -= 0.20

        if not _is_powerful_hardware(profile) and item["memory_gb"] > 6.0:
            score -= 0.15

        score += score_penalty
        score = max(0.0, score)
        result.append(
            ModelSuggestion(
                model_id=item["id"],
                reason=item["label"],
                task=target_task,
                recommended_quantization=quantization,
                estimated_memory_gb=float(item["memory_gb"]),
                score=score,
            )
        )

    result.sort(key=lambda item: item.score, reverse=True)
    return result[: max(0, top_k)]


def build_selector_output(
    hardware_profile: Optional[Dict[str, Any]] = None,
    task: str = "general",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    return [suggestion.as_dict() for suggestion in suggest_models(hardware_profile, task, top_k)]
