from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DeviceCapability:
    device_id: str
    device_class: str
    physically_present: bool
    runtime_usable: bool
    backend: str
    supported_precisions: List[str]
    supported_stage_types: List[str]
    memory_budget: Dict[str, Any]
    thermal_state: str
    runtime_notes: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_class": self.device_class,
            "physically_present": self.physically_present,
            "runtime_usable": self.runtime_usable,
            "backend": self.backend,
            "supported_precisions": list(self.supported_precisions),
            "supported_stage_types": list(self.supported_stage_types),
            "memory_budget": dict(self.memory_budget),
            "thermal_state": self.thermal_state,
            "runtime_notes": list(self.runtime_notes),
            "confidence_score": self.confidence_score,
        }


@dataclass
class HardwareCapabilityMatrix:
    devices: List[DeviceCapability]
    supported_stage_types: Dict[str, List[str]]
    precision_support: Dict[str, List[str]]
    runtime_health: Dict[str, str]
    thermal_state: Dict[str, str]
    preferred_assignments: Dict[str, str]
    fallback_paths: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "supported_stage_types": dict(self.supported_stage_types),
            "precision_support": dict(self.precision_support),
            "runtime_health": dict(self.runtime_health),
            "thermal_state": dict(self.thermal_state),
            "preferred_assignments": dict(self.preferred_assignments),
            "fallback_paths": dict(self.fallback_paths),
        }
