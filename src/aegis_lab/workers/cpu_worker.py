import logging
import time
from typing import Dict, Any
from aegis_lab.workers.base import WorkerBase

logger = logging.getLogger(__name__)

class CpuWorker(WorkerBase):
    def __init__(self, orchestrator_url: str = "tcp://localhost:5555"):
        super().__init__(orchestrator_url, worker_type="cpu")

    def execute_stage(self, task: Dict[str, Any]) -> Dict[str, Any]:
        stage_name = task["stage_name"]
        logger.info(f"CPU Worker executing stage: {stage_name}")
        
        # Simulate work
        time.sleep(2)
        
        return {
            "success": True,
            "message": f"Successfully executed {stage_name} on CPU",
            "telemetry": {
                "duration": 2.0,
                "peak_memory_mb": 150
            }
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = CpuWorker()
    worker.connect()
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop()
