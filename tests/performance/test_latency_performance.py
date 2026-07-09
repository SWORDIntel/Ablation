import unittest
import time
import os
from unittest.mock import patch
from framewerx.aegis_lab.sentinel.cascade import NPUSentinel
from framewerx.aegis_lab.state.db import AegisState

class TestLatencyPerformance(unittest.TestCase):
    def setUp(self):
        self.hw_caps = {
            "npu_type": "intel_ai_boost",
            "npu_bar_found": True
        }
        with patch('os.path.exists', return_value=False):
            self.sentinel = NPUSentinel(self.hw_caps)
            
        self.state = AegisState("/tmp/aegis_perf_test", os.path.abspath("QIHSE/qihse/libqihse.so"))

    def test_sentinel_stage0_latency(self):
        # Stage 0 (Statistical) must be < 5ms
        features = {"drift_summary": 0.1, "routing_stats": 0.8}
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            self.sentinel.run_stage0_statistical(features)
            latencies.append((time.perf_counter() - start) * 1000)
            
        avg_latency = sum(latencies) / len(latencies)
        print(f"Sentinel Stage 0 Average Latency: {avg_latency:.4f} ms")
        self.assertLess(avg_latency, 5.0, "Sentinel Stage 0 exceeds 5ms latency")

    def test_sentinel_stage1_latency(self):
        # Stage 1 (Semantic) must be < 20ms
        metadata = {"anchor_delta": 0.1}
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            self.sentinel.run_stage1_semantic(None, metadata)
            latencies.append((time.perf_counter() - start) * 1000)
            
        avg_latency = sum(latencies) / len(latencies)
        print(f"Sentinel Stage 1 Average Latency: {avg_latency:.4f} ms")
        self.assertLess(avg_latency, 20.0, "Sentinel Stage 1 exceeds 20ms latency")

    def test_qihse_lookup_latency(self):
        # Benchmark QIHSE lookups (using AegisState wrapper)
        # Note: In mock mode it might be too fast, but we should measure it.
        # Ensure some data exists
        self.state.create_job("job-perf", "proj-perf", "ablation")
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            self.state.get_jobs()
            latencies.append((time.perf_counter() - start) * 1000)
            
        avg_latency = sum(latencies) / len(latencies)
        print(f"QIHSE Lookup Average Latency: {avg_latency:.4f} ms")
        # Spec doesn't specify hard cap for lookup but let's say < 10ms for now
        self.assertLess(avg_latency, 10.0, "QIHSE Lookup exceeds 10ms latency")

if __name__ == "__main__":
    unittest.main()
