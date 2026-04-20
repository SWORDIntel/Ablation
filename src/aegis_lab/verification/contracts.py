from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationReport:
    job_id: str
    stage: str
    execution_mode: str
    metrics: Dict[str, Any]
    gate_results: Dict[str, bool]
    passed: bool
    blocking_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "execution_mode": self.execution_mode,
            "metrics": dict(self.metrics),
            "gate_results": dict(self.gate_results),
            "passed": self.passed,
            "blocking_reasons": list(self.blocking_reasons),
        }
