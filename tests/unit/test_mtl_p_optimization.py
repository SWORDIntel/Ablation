import unittest
import os
from framewerx.aegis_lab.scheduler.engine import SchedulerEngine

class TestMTLPOptimization(unittest.TestCase):
    def test_mtl_p_activation_capture(self):
        # Mock MTL-P hardware: Hybrid CPU, Xe-LPG iGPU, NPU
        hw_caps = {
            "cpu_amx": False,
            "cpu_avx512": False,
            "cpu_vnni": True,
            "cpu_hybrid": True,
            "igpu_present": True,
            "igpu_type": "xe-lpg",
            "npu_present": True,
            "npu_type": "intel_ai_boost"
        }
        
        scheduler = SchedulerEngine(hw_caps)
        
        # Activation capture should prefer P-cores
        placement = scheduler.determine_placement("activation_capture", {})
        self.assertEqual(placement[0], "CPU_P_CORE")
        self.assertIn("CPU_VNNI", placement)

    def test_mtl_p_ridge_regression(self):
        # Mock MTL-P hardware
        hw_caps = {
            "cpu_amx": False,
            "cpu_avx512": False,
            "cpu_vnni": True,
            "cpu_hybrid": True,
            "igpu_present": True,
            "igpu_type": "xe-lpg",
            "npu_present": True,
            "npu_type": "intel_ai_boost"
        }
        
        scheduler = SchedulerEngine(hw_caps)
        
        # Ridge regression should prefer NPU and Xe-LPG iGPU
        placement = scheduler.determine_placement("ridge_regression", {})
        self.assertIn("NPU", placement)
        self.assertIn("iGPU", placement)
        self.assertIn("CPU_E_CORE", placement)
        
        # Priority should be NPU > iGPU for high throughput
        self.assertEqual(placement[0], "NPU")
        self.assertEqual(placement[1], "iGPU")

    def test_cross_compatibility_fallback(self):
        # Mock older hardware (e.g. Haswell-like but with AVX2/VNNI)
        hw_caps = {
            "cpu_amx": False,
            "cpu_avx512": False,
            "cpu_vnni": True,
            "cpu_hybrid": False,
            "igpu_present": True,
            "igpu_type": "standard",
            "npu_present": False,
            "npu_type": "none"
        }
        
        scheduler = SchedulerEngine(hw_caps)
        
        # Activation capture should fall back to standard CPU_VNNI
        placement = scheduler.determine_placement("activation_capture", {})
        self.assertEqual(placement[0], "CPU_VNNI")
        
        # Ridge regression should fall back to iGPU if present
        placement = scheduler.determine_placement("ridge_regression", {})
        self.assertEqual(placement[0], "iGPU")

if __name__ == "__main__":
    unittest.main()
