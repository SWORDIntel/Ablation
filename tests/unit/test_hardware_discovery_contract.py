import unittest
from unittest.mock import patch

from aegis_lab.hardware.discovery import HardwareDiscovery


class FakeCore:
    def __init__(self):
        self.available_devices = ["CPU", "MYRIAD.0"]

    def get_property(self, device, name):
        if device == "CPU" and name == "FULL_DEVICE_NAME":
            return "Fake CPU"
        return "Fake Device"


class TestHardwareDiscoveryContract(unittest.TestCase):
    @patch.object(HardwareDiscovery, "list_usb_myriad_devices", return_value=[
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
    @patch.object(HardwareDiscovery, "_load_openvino_core", return_value=(FakeCore, "openvino.Core"))
    def test_discovery_distinguishes_usb_and_runtime_vpu(self, _core_loader, _usb_devices):
        devices = HardwareDiscovery.check_openvino_devices()

        self.assertTrue(devices["openvino_available"])
        self.assertEqual(devices["openvino_import"], "openvino.Core")
        self.assertTrue(devices["vpu"])
        self.assertTrue(devices["vpu_runtime"])
        self.assertTrue(devices["vpu_usb"])
        self.assertEqual(devices["vpu_runtime_count"], 1)
        self.assertEqual(devices["vpu_usb_count"], 1)
        self.assertEqual(devices["vpu_details"], ["MYRIAD.0"])

    @patch.object(HardwareDiscovery, "check_cpu_features", return_value={"amx": True, "avx512": True})
    @patch.object(HardwareDiscovery, "get_system_memory_gb", return_value=64.0)
    @patch.object(HardwareDiscovery, "check_openvino_devices", return_value={
        "openvino_available": True,
        "igpu": True,
        "igpu_type": "xe-lpg",
        "npu": True,
        "npu_type": "intel_ai_boost",
        "vpu": True,
        "vpu_runtime": True,
        "vpu_runtime_count": 1,
        "vpu_usb_count": 1,
        "vpu_count": 1,
    })
    def test_build_capability_matrix_contract(self, _ov_devices, _mem_gb, _cpu_features):
        matrix = HardwareDiscovery.build_capability_matrix()

        self.assertIn("devices", matrix)
        self.assertIn("supported_stage_types", matrix)
        self.assertIn("precision_support", matrix)
        self.assertIn("runtime_health", matrix)

        device_ids = {device["device_id"] for device in matrix["devices"]}
        self.assertIn("cpu-0", device_ids)
        self.assertIn("igpu-0", device_ids)
        self.assertIn("npu-0", device_ids)
        self.assertIn("vpu-0", device_ids)


if __name__ == "__main__":
    unittest.main()
