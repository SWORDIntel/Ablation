import time
import logging

logger = logging.getLogger(__name__)

class LatencyMonitor:
    """Monitors TTFT for Section 4 steering operations."""
    def __init__(self, ttft_limit_ms: float = 50.0):
        self.ttft_limit_ms = ttft_limit_ms
        self.start_times = {}

    def start_op(self, op_id: str):
        self.start_times[op_id] = time.perf_counter()

    def end_op(self, op_id: str):
        end = time.perf_counter()
        if op_id in self.start_times:
            duration_ms = (end - self.start_times[op_id]) * 1000
            if duration_ms > self.ttft_limit_ms:
                logger.warning(f"TTFT Violation for {op_id}: {duration_ms:.2f}ms (limit {self.ttft_limit_ms}ms)")
            else:
                logger.info(f"Steering operation {op_id} TTFT: {duration_ms:.2f}ms")
            del self.start_times[op_id]



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
