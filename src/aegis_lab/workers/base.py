import zmq
import json
import time
import uuid
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from aegis_lab.hardware.discovery import HardwareDiscovery

logger = logging.getLogger(__name__)

class ZmqLogHandler(logging.Handler):
    """
    Log handler that publishes logs to a ZMQ PUB socket.
    Used for streaming logs back to the orchestrator.
    """
    def __init__(self, host="localhost", port=5556, worker_id=None):
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(f"tcp://{host}:{port}")
        self.worker_id = worker_id
        self.job_id = None
        self.stage_id = None

    def emit(self, record):
        try:
            log_entry = {
                "worker_id": self.worker_id,
                "job_id": self.job_id,
                "stage_id": self.stage_id,
                "level": record.levelname,
                "name": record.name,
                "msg": self.format(record),
                "timestamp": record.created
            }
            self.socket.send_json(log_entry)
        except Exception:
            self.handleError(record)

class WorkerBase(ABC):
    """
    Base class for all AEGIS-LAB workers (CPU, iGPU, NPU).
    Handles IPC registration, heartbeats, and stage execution lifecycle.
    """
    
    def __init__(self, orchestrator_url: str = "tcp://localhost:5555", worker_type: str = "generic", log_port: int = 5556):
        self.worker_id = f"worker-{worker_type}-{uuid.uuid4().hex[:8]}"
        self.worker_type = worker_type
        self.orchestrator_url = orchestrator_url
        self.log_port = log_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.running = False
        self.capabilities = HardwareDiscovery.discover()
        self.log_handler = None
        
    def connect(self):
        logger.info(f"Worker {self.worker_id} connecting to {self.orchestrator_url}")
        self.socket.connect(self.orchestrator_url)
        self.running = True
        
        # Setup ZMQ logging
        try:
            import urllib.parse
            # Basic parsing of tcp://host:port
            url = self.orchestrator_url
            if "//" in url:
                host = url.split("//")[1].split(":")[0]
            else:
                host = "localhost"
            
            self.log_handler = ZmqLogHandler(host=host, port=self.log_port, worker_id=self.worker_id)
            logging.getLogger().addHandler(self.log_handler)
            logger.info("ZMQ Log Handler attached.")
        except Exception as e:
            logger.error(f"Failed to setup ZMQ logging: {e}")

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        
        # Initial registration
        self._send_command({
            "type": "register",
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "capabilities": self.capabilities
        })

    def _send_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.socket.send_json(message)
            return self.socket.recv_json()
        except Exception as e:
            logger.error(f"Failed to send command to orchestrator: {e}")
            return {"status": "error", "error": str(e)}

    def _heartbeat_loop(self):
        while self.running:
            self._send_command({
                "type": "heartbeat",
                "worker_id": self.worker_id,
                "timestamp": time.time()
            })
            time.sleep(5) # 5-second heartbeat interval

    def run(self):
        """
        Main worker loop to pull tasks from orchestrator.
        In this implementation, we'll use a polling REP/REQ or PULL/PUSH pattern.
        For simplicity, let's assume the worker asks for work.
        """
        logger.info(f"Worker {self.worker_id} starting main loop.")
        while self.running:
            try:
                response = self._send_command({
                    "type": "request_task",
                    "worker_id": self.worker_id
                })
                
                if response.get("status") == "task_assigned":
                    task = response.get("task")
                    
                    # Set log context
                    if self.log_handler:
                        self.log_handler.job_id = task["job_id"]
                        self.log_handler.stage_id = task["stage_id"]
                        
                    logger.info(f"Task assigned: {task['stage_name']} for job {task['job_id']}")
                    result = self.execute_stage(task)
                    self._send_command({
                        "type": "task_complete",
                        "worker_id": self.worker_id,
                        "job_id": task["job_id"],
                        "stage_id": task["stage_id"],
                        "result": result
                    })
                    
                    # Clear log context
                    if self.log_handler:
                        self.log_handler.job_id = None
                        self.log_handler.stage_id = None
                else:
                    # No work available, back off
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(5)

    @abstractmethod
    def execute_stage(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        To be implemented by hardware-specific workers.
        """
        pass

    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()
