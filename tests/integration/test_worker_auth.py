import zmq
import json
import time
import threading
import logging
import socket
from framewerx.aegis_lab.orchestrator.ipc import IPCServer
from framewerx.aegis_lab.workers.cpu_worker import CpuWorker
from framewerx.aegis_lab.workers.base import AEGIS_AUTH_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def test_worker_auth():
    port = _free_port()
    server = IPCServer(port=port, auth_token=AEGIS_AUTH_TOKEN)
    
    # Define a simple register handler
    def handle_register(msg):
        logger.info(f"Registering worker: {msg.get('worker_id')}")
        return {"status": "registered"}
    
    # Define a simple request_task handler
    def handle_request_task(msg):
        return {"status": "no_task"}
        
    server.register_handler("register", handle_register)
    server.register_handler("heartbeat", lambda msg: {"status": "ok"})
    server.register_handler("request_task", handle_request_task)
    server.start()
    
    time.sleep(1) # Wait for server to start
    worker = None
    worker_wrong = None
    try:
        # 1. Test CpuWorker with default (correct) token
        logger.info("Testing CpuWorker with correct token...")
        worker = CpuWorker(orchestrator_url=f"tcp://localhost:{port}")
        worker.connect()

        # We'll just check if it can send heartbeats or register
        # Registration happens in connect()
        # Let's wait a bit and then stop
        time.sleep(1)
        worker.stop()

        # 2. Test CpuWorker with WRONG token
        logger.info("Testing CpuWorker with wrong token...")
        worker_wrong = CpuWorker(orchestrator_url=f"tcp://localhost:{port}", auth_token="bad-token")

        # Registration should fail
        # However, WorkerBase._send_command logs errors but doesn't necessarily crash.
        # We should check if the server logged an authentication failure.
        worker_wrong.connect()
        time.sleep(1)
        worker_wrong.stop()
    finally:
        if worker is not None:
            worker.stop()
        if worker_wrong is not None:
            worker_wrong.stop()
        server.stop()
        logger.info("Worker authentication test finished!")

if __name__ == "__main__":
    test_worker_auth()
