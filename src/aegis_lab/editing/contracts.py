from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EditPlan:
    target: str
    protected_capabilities: List[str]
    edit_method: str
    target_loci: List[str]
    required_capture_types: List[str]
    required_validation_suites: List[str]
    reversible: bool
    delta_expected: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "protected_capabilities": list(self.protected_capabilities),
            "edit_method": self.edit_method,
            "target_loci": list(self.target_loci),
            "required_capture_types": list(self.required_capture_types),
            "required_validation_suites": list(self.required_validation_suites),
            "reversible": self.reversible,
            "delta_expected": self.delta_expected,
            "notes": list(self.notes),
        }

    @classmethod
    def from_request(cls, target: str, request: Dict[str, Any]) -> "EditPlan":
        return cls(
            target=target,
            protected_capabilities=list(request.get("protected_capabilities", [])),
            edit_method=request.get("edit_method", "runtime_masking"),
            target_loci=list(request.get("target_loci", [])),
            required_capture_types=list(request.get("required_capture_types", ["hidden_state_capture"])),
            required_validation_suites=list(request.get("required_validation_suites", ["semantic_retention"])),
            reversible=bool(request.get("reversible", True)),
            delta_expected=bool(request.get("delta_expected", True)),
            notes=list(request.get("notes", [])),
        )
