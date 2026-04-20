from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class PromotionManifest:
    artifact_hash: str
    lineage: Dict[str, Any]
    model_profile_hash: str
    execution_plan_hash: str
    validation_report_hash: str
    quantization_report_hash: str
    promoted_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "lineage": dict(self.lineage),
            "model_profile_hash": self.model_profile_hash,
            "execution_plan_hash": self.execution_plan_hash,
            "validation_report_hash": self.validation_report_hash,
            "quantization_report_hash": self.quantization_report_hash,
            "promoted_at": self.promoted_at,
        }

    @classmethod
    def create(
        cls,
        artifact_hash: str,
        lineage: Dict[str, Any],
        model_profile_hash: str,
        execution_plan_hash: str,
        validation_report_hash: str,
        quantization_report_hash: str,
    ) -> "PromotionManifest":
        return cls(
            artifact_hash=artifact_hash,
            lineage=lineage,
            model_profile_hash=model_profile_hash,
            execution_plan_hash=execution_plan_hash,
            validation_report_hash=validation_report_hash,
            quantization_report_hash=quantization_report_hash,
            promoted_at=datetime.now(timezone.utc).isoformat(),
        )
