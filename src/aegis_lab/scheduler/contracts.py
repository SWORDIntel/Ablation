from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StageAssignment:
    stage_name: str
    stage_device: str
    stage_precision: str
    stage_batch_strategy: str
    stage_memory_budget: Dict[str, Any]
    stage_fallback_chain: List[str]
    stage_execution_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "stage_device": self.stage_device,
            "stage_precision": self.stage_precision,
            "stage_batch_strategy": self.stage_batch_strategy,
            "stage_memory_budget": dict(self.stage_memory_budget),
            "stage_fallback_chain": list(self.stage_fallback_chain),
            "stage_execution_mode": self.stage_execution_mode,
        }


@dataclass
class ExecutionPlan:
    job_id: str
    stage_assignments: List[StageAssignment]
    promotion_target_precision: str
    fallback_chain: List[str]
    execution_mode_summary: Dict[str, str]
    scheduler_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage_assignments": [assignment.to_dict() for assignment in self.stage_assignments],
            "promotion_target_precision": self.promotion_target_precision,
            "fallback_chain": list(self.fallback_chain),
            "execution_mode_summary": dict(self.execution_mode_summary),
            "scheduler_notes": list(self.scheduler_notes),
        }
