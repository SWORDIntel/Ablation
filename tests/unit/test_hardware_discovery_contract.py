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


if __name__ == "__main__":
    unittest.main()
