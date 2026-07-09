import zmq
import json
import time
import uuid
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from framewerx.aegis_lab.hardware.discovery import HardwareDiscovery
from framewerx.aegis_lab.workers.p2p import PeerStreamer

logger = logging.getLogger(__name__)

AEGIS_AUTH_TOKEN = os.getenv("AEGIS_AUTH_TOKEN", "aegis-secret-token-2024")

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

    def close(self):
        try:
            if getattr(self, "socket", None) is not None:
                self.socket.close()
        finally:
            try:
                if getattr(self, "context", None) is not None:
                    self.context.destroy(linger=0)
            finally:
                super().close()

class WorkerBase(ABC):
    """
    Base class for all AEGIS-LAB workers (CPU, iGPU, NPU).
    Handles IPC registration, heartbeats, and stage execution lifecycle.
    """
    
    def __init__(self, orchestrator_url: str = "tcp://localhost:5555", worker_type: str = "generic", log_port: int = 5556, auth_token: str = AEGIS_AUTH_TOKEN):
        self.worker_id = f"worker-{worker_type}-{uuid.uuid4().hex[:8]}"
        self.worker_type = worker_type
        self.orchestrator_url = orchestrator_url
        self.log_port = log_port
        self.auth_token = auth_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket_lock = threading.Lock()
        self.running = False
        self.capabilities = HardwareDiscovery.discover()
        self.log_handler = None
        self.p2p_streamer: Optional[PeerStreamer] = None
        self.p2p_endpoint: Optional[str] = None
        self.peer_connections: Dict[str, PeerStreamer] = {}
        
    def start_p2p_server(self, mode: str = "push", bind_address: str = "tcp://0.0.0.0:*"):
        """Starts this worker's P2P server to allow peers to connect/pull data."""
        self.p2p_streamer = PeerStreamer(self.context, self.auth_token)
        self.p2p_endpoint = self.p2p_streamer.start_sender(mode=mode, bind_address=bind_address)
        logger.info(f"Worker {self.worker_id} started P2P server at {self.p2p_endpoint}")
        return self.p2p_endpoint

    def discover_peer(self, target_worker_id: str = None, target_worker_type: str = None) -> Optional[str]:
        """Queries the orchestrator for a peer's P2P endpoint."""
        response = self._send_command({
            "type": "discover_peer",
            "target_worker_id": target_worker_id,
            "target_worker_type": target_worker_type
        })
        if response.get("status") == "ok":
            return response.get("p2p_endpoint")
        return None

    def connect_to_peer(self, worker_id: str, endpoint: str, mode: str = "pull"):
        """Connects to a peer's P2P server."""
        streamer = PeerStreamer(self.context, self.auth_token)
        streamer.start_receiver(mode=mode, connect_address=endpoint)
        self.peer_connections[worker_id] = streamer
        logger.info(f"Worker {self.worker_id} connected to peer {worker_id} at {endpoint}")
        return streamer

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

        # Initial registration
        self._send_command({
            "type": "register",
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "capabilities": self.capabilities,
            "p2p_endpoint": self.p2p_endpoint
        })

        # Start heartbeat thread after registration
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _send_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        with self.socket_lock:
            try:
                # Inject auth token
                if self.auth_token:
                    message["auth_token"] = self.auth_token
                    
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
        heartbeat_thread = getattr(self, "_heartbeat_thread", None)
        if heartbeat_thread and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=1)

        if self.log_handler is not None:
            try:
                logging.getLogger().removeHandler(self.log_handler)
            except ValueError:
                pass
            self.log_handler.close()
            self.log_handler = None

        if self.p2p_streamer:
            self.p2p_streamer.close()
            self.p2p_streamer = None

        for streamer in self.peer_connections.values():
            streamer.close()
        self.peer_connections.clear()

        self.socket.close()
        self.context.destroy(linger=0)
