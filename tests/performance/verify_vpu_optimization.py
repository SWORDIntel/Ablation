import time
import logging
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from aegis_lab.workers.vpu_worker import VpuWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vpu_perf_test")

def run_benchmark(num_requests=100):
    worker = VpuWorker()
    
    # Mock task
    task = {
        "stage_name": "sentinel_stage0_guard",
        "job_id": "job-123",
        "stage_id": "stage-456",
        "model_path": "models/dummy_model.xml",
        "input_data": {
            "input": np.random.rand(1, 1024).astype(np.float32)
        }
    }
    
    logger.info(f"Starting benchmark with {num_requests} requests...")
    latencies = []
    
    # Warm-up (especially important for caching)
    logger.info("Warm-up request...")
    worker.execute_stage(task)
    
    for i in range(num_requests):
        start_time = time.time()
        result = worker.execute_stage(task)
        end_time = time.time()
        
        if not result.get("success"):
            logger.error(f"Request {i} failed: {result.get('error')}")
            continue
            
        latencies.append(end_time - start_time)
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i+1}/{num_requests} requests")
            
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print("\n--- VPU Performance Report ---")
    print(f"Total Requests: {len(latencies)}")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"P95 Latency:     {p95_latency:.4f}s")
    print(f"Throughput:      {1/avg_latency:.2f} req/s")
    print("------------------------------\n")
    
    return avg_latency

if __name__ == "__main__":
    run_benchmark(100)
