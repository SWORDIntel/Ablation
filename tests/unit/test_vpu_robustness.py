import logging
from aegis_lab.scheduler.engine import SchedulerEngine

def test_vpu_fail_fast_logic():
    """
    Verifies that the scheduler marks VPU workers as INACTIVE after 3 failures
    and re-routes tasks to CPU/AMX backend.
    """
    # Setup hardware discovery with VPU and AMX
    hw = {
        "accel_available": True,
        "vpu_present": True,
        "cpu_amx": True,
        "npu_present": False,
        "igpu_present": False
    }
    
    # Reset scheduler state for a clean test
    SchedulerEngine.VPU_WORKER_STATS = {
        "vpu-0": {"failures": 0, "active": True},
        "vpu-1": {"failures": 0, "active": True}
    }
    
    scheduler = SchedulerEngine(hw)
    
    # Sentinel stage should prefer VPU initially for an active VPU worker
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-vpu-0")
    assert "VPU" in placement
    
    # Simulate 3 failures for vpu-0
    SchedulerEngine.report_failure("worker-vpu-0")
    SchedulerEngine.report_failure("worker-vpu-0")
    SchedulerEngine.report_failure("worker-vpu-0")
    
    assert SchedulerEngine.VPU_WORKER_STATS["vpu-0"]["active"] is False
    
    # vpu-0 should no longer get VPU tasks if it asks (Round-Robin will skip it)
    # Actually, in my implementation, if it's INACTIVE it won't even be in active_vpus
    
    # vpu-1 should still be active and get tasks
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-vpu-1")
    assert "VPU" in placement
    
    # Simulate 3 failures for vpu-1
    SchedulerEngine.report_failure("worker-vpu-1")
    SchedulerEngine.report_failure("worker-vpu-1")
    SchedulerEngine.report_failure("worker-vpu-1")
    
    assert SchedulerEngine.VPU_WORKER_STATS["vpu-1"]["active"] is False
    
    # Now all VPUs are inactive. Re-routing should trigger for any worker asking.
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-cpu-any")
    assert "CPU_AMX" in placement
    assert "VPU" not in placement

def test_vpu_round_robin_scheduling():
    """
    Verifies that the scheduler distributes tasks across active VPU workers
    using a Round-Robin strategy.
    """
    hw = {"accel_available": True, "vpu_present": True, "cpu_amx": True}
    
    # Reset scheduler state
    SchedulerEngine.VPU_WORKER_STATS = {
        "vpu-0": {"failures": 0, "active": True},
        "vpu-1": {"failures": 0, "active": True}
    }
    SchedulerEngine.VPU_ROUND_ROBIN_IDX = 0
    
    scheduler = SchedulerEngine(hw)
    
    # First turn: vpu-0 should be selected
    # vpu-1 asks: should not get VPU task if it's vpu-0's turn
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-vpu-1")
    assert "VPU" not in placement
    
    # vpu-0 asks: should get it
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-vpu-0")
    assert "VPU" in placement
    assert SchedulerEngine.VPU_ROUND_ROBIN_IDX == 1
    
    # Second turn: vpu-1 should be selected
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-vpu-0")
    assert "VPU" not in placement
    
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-vpu-1")
    assert "VPU" in placement
    assert SchedulerEngine.VPU_ROUND_ROBIN_IDX == 2
    
    # Third turn: back to vpu-0
    placement = scheduler.determine_placement("sentinel_stage0_guard", {}, worker_id="worker-vpu-0")
    assert "VPU" in placement
    assert SchedulerEngine.VPU_ROUND_ROBIN_IDX == 3

if __name__ == "__main__":
    # Manual run if needed
    test_vpu_fail_fast_logic()
    test_vpu_round_robin_scheduling()
    print("All VPU robustness tests passed.")
