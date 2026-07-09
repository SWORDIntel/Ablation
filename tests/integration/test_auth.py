import zmq
import json
import time
import threading
import logging
import socket
from framewerx.aegis_lab.orchestrator.ipc import IPCServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def test_auth():
    auth_token = "correct-token"
    port = _free_port()
    server = IPCServer(port=port, auth_token=auth_token)
    
    # Define a simple handler
    def handle_ping(msg):
        return {"status": "ok", "message": "pong"}
    
    server.register_handler("ping", handle_ping)
    server.start()
    
    time.sleep(1) # Wait for server to start
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://localhost:{port}")
    try:
        # 1. Test with correct token
        logger.info("Testing with correct token...")
        socket.send_json({"type": "ping", "auth_token": auth_token})
        resp = socket.recv_json()
        logger.info(f"Response: {resp}")
        assert resp["status"] == "ok"

        # 2. Test with incorrect token
        logger.info("Testing with incorrect token...")
        socket.send_json({"type": "ping", "auth_token": "wrong-token"})
        resp = socket.recv_json()
        logger.info(f"Response: {resp}")
        assert resp["status"] == "error"
        assert "Authentication failed" in resp["error"]

        # 3. Test without token
        logger.info("Testing without token...")
        socket.send_json({"type": "ping"})
        resp = socket.recv_json()
        logger.info(f"Response: {resp}")
        assert resp["status"] == "error"
        assert "Authentication failed" in resp["error"]

        # 4. Test if server is still alive after failure
        logger.info("Testing if server is still alive...")
        socket.send_json({"type": "ping", "auth_token": auth_token})
        resp = socket.recv_json()
        logger.info(f"Response: {resp}")
        assert resp["status"] == "ok"
    finally:
        socket.close()
        context.destroy(linger=0)
        server.stop()
        logger.info("Authentication test passed!")

if __name__ == "__main__":
    test_auth()
