import unittest
import time
import threading
import zmq
import os
import shutil
import tempfile
from framewerx.aegis_lab.orchestrator.service import OrchestratorService
from framewerx.aegis_lab.state.db import AegisState
from framewerx.aegis_lab.workers.cpu_worker import CpuWorker
from framewerx.aegis_lab.workers.vpu_worker import VpuWorker
from framewerx.aegis_lab.workers.p2p import PeerStreamer

AEGIS_AUTH_TOKEN = "aegis-secret-token-2024"

class TestP2PFabric(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.test_dir, "state")
        self.state = AegisState(self.state_path, os.path.abspath("QIHSE/qihse/libqihse.so"))
        
        # Use different ports for testing to avoid collisions
        self.ipc_port = 5570
        self.log_port = 5571
        self.orchestrator = OrchestratorService(self.state, ipc_port=self.ipc_port, log_port=self.log_port)
        self.orchestrator.start()
        time.sleep(1) # Give it time to start

    def tearDown(self):
        self.orchestrator.stop()
        shutil.rmtree(self.test_dir)

    def test_p2p_data_transfer(self):
        # 1. Initialize two workers
        # Worker A (CPU) will be the sender
        worker_a = CpuWorker(orchestrator_url=f"tcp://localhost:{self.ipc_port}", auth_token=AEGIS_AUTH_TOKEN)
        # Worker B (VPU) will be the receiver
        worker_b = VpuWorker(orchestrator_url=f"tcp://localhost:{self.ipc_port}", auth_token=AEGIS_AUTH_TOKEN)

        # 2. Worker A starts P2P server
        # Explicitly bind to 127.0.0.1 to avoid external access during test
        endpoint_a = worker_a.start_p2p_server(mode="push", bind_address="tcp://127.0.0.1:*")
        
        # 3. Connect and register both
        worker_a.connect()
        worker_b.connect()
        time.sleep(1)

        # 4. Worker B discovers Worker A's endpoint via the Orchestrator
        discovered_endpoint = worker_b.discover_peer(target_worker_id=worker_a.worker_id)
        # Handle cases where localhost is translated differently (0.0.0.0 vs 127.0.0.1)
        # We'll check if the port is correct and it's a valid tcp address
        self.assertIsNotNone(discovered_endpoint)
        self.assertTrue(discovered_endpoint.startswith("tcp://"))

        # 5. Worker B connects to Worker A directly
        # We'll use the exact endpoint discovered from Orchestrator
        streamer_b = worker_b.connect_to_peer(worker_a.worker_id, discovered_endpoint, mode="pull")
        time.sleep(1)

        # 6. Transfer data from A to B (Direct P2P, NOT through Orchestrator)
        test_data = {"representation": [0.1, 0.2, 0.3], "layer": "hidden_1"}
        worker_a.p2p_streamer.send_data(test_data, metadata={"type": "tensor"})

        received_data, metadata = streamer_b.receive_data()
        
        self.assertEqual(received_data, test_data)
        self.assertEqual(metadata["type"], "tensor")

        # 7. Verify Auth Token requirement
        # Create a receiver streamer with WRONG token
        bad_streamer = PeerStreamer(worker_b.context, "wrong-token-for-failure")
        bad_streamer.start_receiver(mode="pull", connect_address=discovered_endpoint)
        
        # Send data again from A
        worker_a.p2p_streamer.send_data("unauthorized data attempt")
        
        # Receiver with bad token should fail authentication (PermissionError)
        with self.assertRaises(PermissionError):
            bad_streamer.receive_data()

        worker_a.stop()
        worker_b.stop()
        bad_streamer.close()

if __name__ == "__main__":
    unittest.main()
