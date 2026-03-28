import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ExecutionMode(str, Enum):
    NATIVE = "native"
    FALLBACK = "fallback"
    MOCK = "mock"


def normalize_execution_mode(value: Optional[str]) -> ExecutionMode:
    if not value:
        return ExecutionMode.FALLBACK
    try:
        return ExecutionMode(value.lower())
    except ValueError:
        return ExecutionMode.FALLBACK


def stable_json_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExecutionContract:
    operation: str
    mode: ExecutionMode
    requested_mode: ExecutionMode
    native_available: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "mode": self.mode.value,
            "requested_mode": self.requested_mode.value,
            "native_available": self.native_available,
            "reason": self.reason,
            "details": self.details,
        }


def resolve_execution_contract(
    operation: str,
    requested_mode: Optional[str] = None,
    native_available: bool = False,
    reason: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> ExecutionContract:
    requested = normalize_execution_mode(requested_mode)
    native_capable = native_available and requested == ExecutionMode.NATIVE

    if requested == ExecutionMode.MOCK:
        selected = ExecutionMode.MOCK
        reason_text = reason or f"{operation} executed in synthetic mock mode."
    elif native_capable:
        selected = ExecutionMode.NATIVE
        reason_text = reason or f"{operation} executed with native runtime."
    else:
        selected = ExecutionMode.FALLBACK
        reason_text = reason or f"{operation} executed with deterministic fallback behavior."

    return ExecutionContract(
        operation=operation,
        mode=selected,
        requested_mode=requested,
        native_available=native_available,
        reason=reason_text,
        details=details or {},
    )


def write_json_artifact(work_dir: Path, prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    payload_hash = stable_json_hash(payload)
    path = work_dir / f"{prefix}-{payload_hash[:12]}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "content_hash": payload_hash,
        "size": path.stat().st_size,
    }
