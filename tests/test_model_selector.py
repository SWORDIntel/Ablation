#!/usr/bin/env python3
"""
Tests for hardware-aware model suggestion helpers.
"""

from __future__ import annotations

import unittest

from framewerx.aegis_lab.hardware.model_selector import suggest_models


class TestModelSelector(unittest.TestCase):
    def test_selector_prefers_cpu_for_generic_hw(self):
        profile = {
            "hardware_tier": "GENERIC",
            "accel_available": False,
            "cpu_avx2": True,
            "npu_present": False,
            "vpu_present": False,
            "igpu_present": False,
            "cpu_amx": False,
            "cpu_avx512": False,
            "cpu_vnni": False,
        }
        suggestions = suggest_models(profile, task="general", top_k=2)
        self.assertEqual(len(suggestions), 2)
        self.assertGreaterEqual(suggestions[0].score, suggestions[1].score)
        self.assertEqual(suggestions[0].recommended_quantization, "int8")

    def test_selector_uses_accel_preferences(self):
        profile = {
            "hardware_tier": "HIGH_PERF_SERVER",
            "accel_available": True,
            "cpu_avx2": True,
            "cpu_amx": True,
            "npu_present": True,
            "vpu_present": False,
            "igpu_present": True,
            "cpu_avx512": False,
            "cpu_vnni": False,
        }
        suggestions = suggest_models(profile, task="ablation", top_k=3)
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions[0].recommended_quantization, "int8")


if __name__ == "__main__":
    unittest.main()
