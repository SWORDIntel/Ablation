import unittest
import os
from aegis_lab.scheduler.engine import SchedulerEngine

class TestSchedulerFailSafe(unittest.TestCase):
    def test_failsafe_activation(self):
        # Mock hardware with no high-perf CPU features but with NPU/iGPU
        hw_caps = {
            "cpu_amx": False,
            "cpu_avx512": False,
            "igpu_present": True,
            "npu_present": True
        }
        
        scheduler = SchedulerEngine(hw_caps)
        
        # Clear env var if it exists
        if "QIHSE_HPU_CACHE_MB" in os.environ:
            del os.environ["QIHSE_HPU_CACHE_MB"]
            
        # Any stage should trigger the fail-safe
        placement = scheduler.determine_placement("atom_clean", {})
        
        # Verify NPU/iGPU are prioritized
        self.assertIn("NPU", placement)
        self.assertIn("iGPU", placement)
        
        # Verify 4x cache environment variable is set
        self.assertEqual(os.environ.get("QIHSE_HPU_CACHE_MB"), "512")

    def test_standard_placement(self):
        # Mock hardware with AMX
        hw_caps = {
            "cpu_amx": True,
            "cpu_avx512": True,
            "igpu_present": True,
            "npu_present": True
        }
        
        scheduler = SchedulerEngine(hw_caps)
        
        # This stage should prefer CPU_AMX when available
        placement = scheduler.determine_placement("atom_clean", {})
        self.assertEqual(placement, ["CPU_AMX"])

    def test_thermal_throttling(self):
        # High-perf CPU is present, but thermal level is HIGH
        hw_caps = {
            "cpu_amx": True,
            "cpu_avx512": True,
            "cpu_vnni": True,
            "igpu_present": True,
            "npu_present": True
        }
        thermal_status = {
            "safe_to_compute": True,
            "throttling_recommended": True
        }
        
        scheduler = SchedulerEngine(hw_caps, thermal_status)
        placement = scheduler.determine_placement("atom_clean", {})
        
        # Should avoid CPU_AMX and prefer NPU due to throttling
        self.assertEqual(placement, ["NPU"])

    def test_thermal_critical(self):
        hw_caps = {"cpu_amx": True, "npu_present": True}
        thermal_status = {"safe_to_compute": False}
        
        scheduler = SchedulerEngine(hw_caps, thermal_status)
        placement = scheduler.determine_placement("atom_clean", {})
        
        # Should return empty list (suspend compute)
        self.assertEqual(placement, [])

    def test_vnni_prioritization(self):
        # No AMX/AVX512, but VNNI present
        hw_caps = {
            "cpu_amx": False,
            "cpu_avx512": False,
            "cpu_vnni": True,
            "igpu_present": False,
            "npu_present": False
        }
        
        scheduler = SchedulerEngine(hw_caps)
        placement = scheduler.determine_placement("atom_clean", {})
        
        # Should pick CPU_VNNI
        self.assertIn("CPU_VNNI", placement)

if __name__ == "__main__":
    unittest.main()
