import unittest
from unittest.mock import patch

from aegis_lab.workers.vpu_worker import MultiVpuWorker, VpuWorker


class TestVpuWorkerContract(unittest.TestCase):
    def test_alias_is_preserved(self):
        self.assertIs(VpuWorker, MultiVpuWorker)

    @patch("aegis_lab.workers.base.HardwareDiscovery.discover", return_value={"npu_type": None})
    @patch("aegis_lab.workers.vpu_worker.Core", new=None)
    def test_worker_constructs_without_hardware(self, _discover_mock):
        worker = VpuWorker(orchestrator_url="tcp://127.0.0.1:1")
        try:
            self.assertEqual(worker.worker_type, "vpu")
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
