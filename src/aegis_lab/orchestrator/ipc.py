import zmq
import json
import logging
import threading
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

class IPCServer:
    """
    ZeroMQ-based IPC server for the Orchestrator.
    Manages communication with distributed workers.
    """
    
    def __init__(self, port: int = 5555, auth_token: Optional[str] = None):
        self.port = port
        self.auth_token = auth_token
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.running = False
        self._handlers: Dict[str, Callable] = {}

    def register_handler(self, message_type: str, handler: Callable):
        self._handlers[message_type] = handler

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"IPC Server started on port {self.port}")

    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()

    def _run(self):
        while self.running:
            try:
                # Polling with timeout to allow checking self.running
                if self.socket.poll(1000):
                    message = self.socket.recv_json()
                    
                    # Authentication Check
                    if self.auth_token:
                        if message.get("auth_token") != self.auth_token:
                            logger.warning(f"Authentication failed for request: {message.get('type')}")
                            self.socket.send_json({"status": "error", "error": "Authentication failed"})
                            continue

                    msg_type = message.get("type")
                    
                    if msg_type in self._handlers:
                        response = self._handlers[msg_type](message)
                    else:
                        response = {"status": "error", "error": f"Unknown message type: {msg_type}"}
                    
                    self.socket.send_json(response)
            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ Error: {e}")
                break
            except Exception as e:
                logger.error(f"IPC Error: {e}")
                if self.running:
                    self.socket.send_json({"status": "error", "error": str(e)})

class LogServer:
    """
    ZeroMQ-based Log Server for the Orchestrator.
    Subscribes to log streams from distributed workers.
    """
    
    def __init__(self, port: int = 5556):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.running = False

    def start(self, callback: Callable[[Dict[str, Any]], None]):
        self.running = True
        self._thread = threading.Thread(target=self._run, args=(callback,), daemon=True)
        self._thread.start()
        logger.info(f"Log Server started on port {self.port}")

    def stop(self):
        self.running = False
        self.socket.close()

    def _run(self, callback: Callable[[Dict[str, Any]], None]):
        while self.running:
            try:
                if self.socket.poll(1000):
                    message = self.socket.recv_json()
                    callback(message)
            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ Log Error: {e}")
                break
            except Exception as e:
                if self.running:
                    logger.error(f"Log processing error: {e}")

class EventPublisher:
    """
    ZeroMQ-based Event Publisher for the Orchestrator.
    Publishes high-level events (atom updates, job progress) to subscribers (GUI).
    """
    def __init__(self, port: int = 5557):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        logger.info(f"Event Publisher bound to port {self.port}")

    def publish(self, topic: str, data: Dict[str, Any]):
        try:
            # We can use multipart messages [topic, json_data]
            self.socket.send_string(topic, zmq.SNDMORE)
            self.socket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")

    def stop(self):
        self.socket.close()
