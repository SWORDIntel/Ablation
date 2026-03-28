import zmq
import logging
import pickle
import time
from typing import Optional, Any, Union

logger = logging.getLogger(__name__)

class PeerStreamer:
    """
    Handles high-bandwidth representation data streaming between workers.
    Supports PUSH/PULL and PUB/SUB patterns with mandatory authentication.
    """
    def __init__(self, context: zmq.Context, auth_token: str):
        self.context = context
        self.auth_token = auth_token
        self.socket: Optional[zmq.Socket] = None
        self.endpoint: Optional[str] = None
        self.mode: Optional[str] = None

    def start_sender(self, mode: str = "push", bind_address: str = "tcp://0.0.0.0:*") -> str:
        """
        Starts a sender socket (PUSH or PUB) and binds to an address.
        Returns the bound endpoint.
        """
        if mode.lower() in {"push", "pub"}:
            self.socket = self.context.socket(zmq.PUB)
        else:
            raise ValueError(f"Unsupported sender mode: {mode}. Use 'push' or 'pub'.")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(bind_address)
        self.endpoint = self.socket.get_string(zmq.LAST_ENDPOINT)
        self.mode = mode.lower()
        logger.info(f"PeerStreamer sender started on {self.endpoint} (mode={self.mode})")
        return self.endpoint

    def start_receiver(self, mode: str = "pull", connect_address: str = None):
        """
        Starts a receiver socket (PULL or SUB) and connects to a peer.
        """
        if not connect_address:
            raise ValueError("connect_address must be provided for receiver")

        if mode.lower() in {"pull", "sub"}:
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        else:
            raise ValueError(f"Unsupported receiver mode: {mode}. Use 'pull' or 'sub'.")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.connect(connect_address)
        if self.socket.type == zmq.SUB:
            time.sleep(0.1)
        self.mode = mode.lower()
        logger.info(f"PeerStreamer receiver connected to {connect_address} (mode={self.mode})")

    def send_data(self, data: Any, metadata: Optional[dict] = None):
        """
        Sends data with authentication token and optional metadata.
        Uses multipart messages: [auth_token, metadata_json, data_bytes]
        """
        if self.socket is None:
            raise RuntimeError("Socket not initialized. Call start_sender first.")

        meta = metadata or {}
        meta_json = pickle.dumps(meta) # Using pickle for flexibility, or json
        
        if not isinstance(data, bytes):
            data_bytes = pickle.dumps(data)
        else:
            data_bytes = data

        self.socket.send_multipart([
            self.auth_token.encode(),
            meta_json,
            data_bytes
        ])

    def receive_data(self) -> tuple[Any, dict]:
        """
        Receives data and verifies authentication token.
        Returns (data, metadata).
        """
        if self.socket is None:
            raise RuntimeError("Socket not initialized. Call start_receiver first.")

        try:
            parts = self.socket.recv_multipart()
        except zmq.Again as exc:
            raise TimeoutError("Timed out waiting for peer data") from exc
        if len(parts) != 3:
            logger.error(f"Invalid P2P message format: expected 3 parts, got {len(parts)}")
            raise ValueError("Invalid P2P message format")

        received_token = parts[0].decode()
        if received_token != self.auth_token:
            logger.warning("P2P Authentication failed: token mismatch")
            raise PermissionError("P2P Authentication failed")

        metadata = pickle.loads(parts[1])
        
        try:
            data = pickle.loads(parts[2])
        except Exception:
            data = parts[2] # Return raw bytes if not unpicklable

        return data, metadata

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
