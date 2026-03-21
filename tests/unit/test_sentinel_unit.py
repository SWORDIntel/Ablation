import unittest
from unittest.mock import MagicMock, patch
from aegis_lab.sentinel.cascade import NPUSentinel

class TestSentinelUnit(unittest.TestCase):
    def setUp(self):
        self.hw_caps = {
            "npu_type": "intel_ai_boost",
            "npu_bar_found": True
        }
        # Patch out mmap and open/dev/mem during init to avoid errors
        with patch('os.path.exists', return_value=False):
            self.sentinel = NPUSentinel(self.hw_caps)

    def test_stage0_pass(self):
        features = {"drift_summary": 0.1, "routing_stats": 0.8}
        verdict = self.sentinel.run_stage0_statistical(features)
        self.assertEqual(verdict, "pass")

    def test_stage0_suspicious(self):
        features = {"drift_summary": 0.5, "routing_stats": 0.8}
        verdict = self.sentinel.run_stage0_statistical(features)
        self.assertEqual(verdict, "suspicious")

    def test_stage0_catastrophic(self):
        features = {"drift_summary": 0.9, "routing_stats": 0.8}
        verdict = self.sentinel.run_stage0_statistical(features)
        self.assertEqual(verdict, "catastrophic")

    def test_stage1_pass(self):
        metadata = {"anchor_delta": 0.1}
        verdict = self.sentinel.run_stage1_semantic(None, metadata)
        self.assertEqual(verdict, "pass")

    def test_stage1_fail(self):
        metadata = {"anchor_delta": 0.8}
        verdict = self.sentinel.run_stage1_semantic(None, metadata)
        self.assertEqual(verdict, "fail")

    def test_evaluate_cascade_pass(self):
        stage0_data = {"drift_summary": 0.1, "routing_stats": 0.8}
        result = self.sentinel.evaluate_cascade(stage0_data)
        self.assertEqual(result, "pass")

    def test_evaluate_cascade_escalate_s0(self):
        stage0_data = {"drift_summary": 0.9, "routing_stats": 0.8}
        result = self.sentinel.evaluate_cascade(stage0_data)
        self.assertEqual(result, "escalate")

    def test_evaluate_cascade_escalate_s1(self):
        stage0_data = {"drift_summary": 0.5, "routing_stats": 0.8}
        stage1_data = {"projections": None, "metadata": {"anchor_delta": 0.8}}
        result = self.sentinel.evaluate_cascade(stage0_data, stage1_data)
        self.assertEqual(result, "escalate")

    def test_evaluate_cascade_pass_after_s1(self):
        stage0_data = {"drift_summary": 0.5, "routing_stats": 0.8}
        stage1_data = {"projections": None, "metadata": {"anchor_delta": 0.1}}
        result = self.sentinel.evaluate_cascade(stage0_data, stage1_data)
        self.assertEqual(result, "pass")

if __name__ == "__main__":
    unittest.main()
