import zmq
import json
import time
import threading
import logging
from aegis_lab.orchestrator.ipc import IPCServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_auth():
    auth_token = "correct-token"
    server = IPCServer(port=5557, auth_token=auth_token)
    
    # Define a simple handler
    def handle_ping(msg):
        return {"status": "ok", "message": "pong"}
    
    server.register_handler("ping", handle_ping)
    server.start()
    
    time.sleep(1) # Wait for server to start
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5557")
    
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
    
    server.stop()
    logger.info("Authentication test passed!")

if __name__ == "__main__":
    test_auth()
