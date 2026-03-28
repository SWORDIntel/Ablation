import unittest
from unittest.mock import patch

from aegis_lab.workers.vpu_worker import MultiVpuWorker, VpuWorker


class FakeCore:
    def __init__(self):
        self.available_devices = ["CPU"]

    def set_property(self, *_args, **_kwargs):
        return None


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

    @patch("aegis_lab.workers.base.HardwareDiscovery.discover", return_value={"npu_type": None})
    @patch("aegis_lab.workers.vpu_worker.HardwareDiscovery.list_usb_myriad_devices", return_value=[
        {
            "sysfs_name": "3-1",
            "vendor_id": "03e7",
            "product_id": "2485",
            "busnum": "3",
            "devnum": "95",
            "serial": "03e72485",
            "manufacturer": "Movidius Ltd.",
            "product": "Movidius MyriadX",
        }
    ])
    @patch("aegis_lab.workers.vpu_worker.Core", new=FakeCore)
    def test_worker_only_registers_physically_matched_profiles(self, _discover_mock, _usb_devices):
        worker = VpuWorker(
            orchestrator_url="tcp://127.0.0.1:1",
            config_dir="configs/hardware",
        )
        try:
            self.assertEqual(list(worker.vpus.keys()), ["vpu_stick_17"])
            self.assertEqual(worker.vpus["vpu_stick_17"]["ov_device"], "SIMULATION")
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
