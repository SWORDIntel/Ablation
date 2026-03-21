import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SchedulerEngine:
    """
    Implements the core device placement logic per stage,
    enforcing the Aegis-Lab execution doctrine.
    Includes robustness (Fail-Fast) and load balancing (Round-Robin).
    """

    # Class-level state to persist worker health and scheduling across instances
    VPU_WORKER_STATS = {
        "vpu-0": {"failures": 0, "active": True},
        "vpu-1": {"failures": 0, "active": True}
    }
    VPU_ROUND_ROBIN_IDX = 0

    # Stages that legally require CPU/AMX semantic authority.
    CPU_AMX_MANDATORY_STAGES = {
        "atom_clean",
        "edit_generate",
        "verify_high_precision",
        "verify_post_quant"
    }

    # Stages representing NPU sentinels.
    SENTINEL_STAGES = {
        "sentinel_stage0_guard",
        "sentinel_stage1_semantic",
        "SENTINEL_VALIDATION"
    }

    # Stages eligible to fulfill the iGPU participation mandate.
    IGPU_ELIGIBLE_STAGES = {
        "probe", 
        "verify_high_precision_slice",
        "verify_post_quant_slice",
        "ridge_regression"
    }

    # Low-latency activation capture stages.
    ACTIVATION_CAPTURE_STAGES = {
        "activation_capture",
        "probe_live"
    }

    # High-throughput ridge regression stages.
    RIDGE_REGRESSION_STAGES = {
        "ridge_regression",
        "batch_verify"
    }

    def __init__(self, hardware_discovery: Dict[str, Any], thermal_status: Optional[Dict[str, Any]] = None):
        """
        Initializes the scheduler with hardware capabilities and optional thermal status.
        """
        self.hw = hardware_discovery
        self.thermal = thermal_status or {"safe_to_compute": True, "throttling_recommended": False}
        self.tier = self.hw.get("hardware_tier", "GENERIC")

    def get_sharding_config(self) -> Dict[str, Any]:
        """
        Returns dynamic work-sharding configuration based on hardware tier.
        """
        config = {
            "search_resolution": 1.0,
            "vpu_offload_ratio": 0.2,
            "batch_size": 32,
            "parallel_tasks": 4
        }
        
        if self.tier == "HIGH_PERF_SERVER":
            config.update({
                "search_resolution": 2.0, # High precision search
                "batch_size": 128,
                "parallel_tasks": 16
            })
        elif self.tier == "MODERN_MTL":
            config.update({
                "search_resolution": 1.5,
                "batch_size": 64,
                "parallel_tasks": 8,
                "vpu_offload_ratio": 0.4 # Leverage NPU/VPU more
            })
        elif self.tier == "LEGACY_AVX2":
            config.update({
                "search_resolution": 0.5, # Reduce resolution to maintain latency
                "vpu_offload_ratio": 0.8, # Aggressively offload to VPU
                "batch_size": 16,
                "parallel_tasks": 2
            })
            
        return config

    @classmethod
    def report_failure(cls, worker_id: str):
        """
        Reports a worker failure for Fail-Fast tracking.
        """
        # Normalize worker_id to match our keys (e.g., vpu-0)
        norm_id = None
        if "vpu-0" in worker_id.lower(): norm_id = "vpu-0"
        elif "vpu-1" in worker_id.lower(): norm_id = "vpu-1"
        
        if norm_id:
            cls.VPU_WORKER_STATS[norm_id]["failures"] += 1
            if cls.VPU_WORKER_STATS[norm_id]["failures"] >= 3:
                cls.VPU_WORKER_STATS[norm_id]["active"] = False
                logger.error(f"FAIL-FAST: VPU worker {norm_id} marked INACTIVE after 3 failures.")

    @classmethod
    def report_success(cls, worker_id: str):
        """
        Resets failure count on success.
        """
        norm_id = None
        if "vpu-0" in worker_id.lower(): norm_id = "vpu-0"
        elif "vpu-1" in worker_id.lower(): norm_id = "vpu-1"
        
        if norm_id:
            cls.VPU_WORKER_STATS[norm_id]["failures"] = 0

    def determine_placement(self, stage_name: str, runtime_profile: Dict[str, Any], worker_id: Optional[str] = None) -> List[str]:
        """
        Determines the ordered list of target devices for a DAG stage.
        Supports load balancing and fail-fast re-routing.
        """
        # Thermal Safety: Emergency shutdown of compute if critical
        if not self.thermal.get("safe_to_compute", True):
            logger.error("THERMAL CRITICAL: Suspending all compute operations.")
            return []

        # DYNAMIC ACCELERATION CHECK: 
        # If the discovery layer flagged accelerators as non-functional/unsupported
        if not self.hw.get("accel_available", True):
            logger.info("No functional hardware accelerators detected. Using optimized CPU path.")
            return ["CPU_AVX2"] if self.hw.get("cpu_vnni") or self.hw.get("cpu_avx512") else ["CPU"]

        # Fail-Fast Check for Sentinel Stages
        vpu_any_active = any(s["active"] for s in self.VPU_WORKER_STATS.values())
        if stage_name in self.SENTINEL_STAGES and not vpu_any_active:
            logger.warning(f"VPU Fail-Fast active. Re-routing sentinel task {stage_name} to CPU/AMX backend.")
            if self.hw.get("cpu_amx"):
                return ["CPU_AMX"]
            return ["CPU_AVX512", "CPU"]

        # MTL-P Optimization: Prefer NPU/Xe-LPG for high throughput if available
        is_mtl_p = self.hw.get("igpu_type") == "xe-lpg" and self.hw.get("npu_present")
        
        # AVX-512/AMX/VNNI Fail-Safe: If no modern accelerators, shift all operations to Flex Fabric (NPU/iGPU)
        has_high_perf_cpu = self.hw.get("cpu_amx") or self.hw.get("cpu_avx512") or self.hw.get("cpu_vnni")
        
        # Thermal Throttling: If HIGH, avoid power-hungry CPU_AMX/iGPU, prefer NPU
        is_throttling = self.thermal.get("throttling_recommended", False)
        
        if not has_high_perf_cpu or is_throttling:
            import os
            reason = "Thermal Throttling" if is_throttling else "No modern CPU accelerators"
            logger.warning(f"{reason}. Fail-safe active: Shifting to Flex Fabric (NPU/iGPU/VPU).")
            # Configure NPU for 4x cache
            os.environ["QIHSE_HPU_CACHE_MB"] = "512"
            
            devices = []
            if self.hw.get("npu_present"):
                devices.append("NPU")
            
            if self.hw.get("vpu_present") and vpu_any_active:
                for vid, s in self.VPU_WORKER_STATS.items():
                    if s["active"]:
                        devices.append("VPU") # For generic matching
                        devices.append("VPU_0" if "0" in vid else "VPU_1")
            
            # Avoid iGPU in thermal throttling if NPU/VPU is present
            if self.hw.get("igpu_present") and not is_throttling:
                devices.append("iGPU")
            
            if devices:
                return devices
            
            return ["CPU"]

        devices = []
        
        # 1. Evaluate NPU/VPU Sentinel Requirements
        if stage_name in self.SENTINEL_STAGES:
            npu_required = runtime_profile.get("npu_required_if_present", True)
            vpu_required = runtime_profile.get("vpu_required_if_present", True)
            
            # VPU is prioritized for stage0_guard for low-power operation
            if stage_name == "sentinel_stage0_guard" and self.hw.get("vpu_present") and vpu_required:
                active_vpus = [vid for vid, s in self.VPU_WORKER_STATS.items() if s["active"]]
                if active_vpus:
                    # Round-Robin Load Balancing across physical sticks
                    target_vpu_id = active_vpus[SchedulerEngine.VPU_ROUND_ROBIN_IDX % len(active_vpus)]
                    # Map vpu-0 to VPU_0 and vpu-1 to VPU_1
                    target_device = "VPU_0" if "0" in target_vpu_id else "VPU_1"
                    
                    if worker_id:
                        if any(target_vpu_id in worker_id.lower() for target_vpu_id in active_vpus):
                            if target_vpu_id in worker_id.lower():
                                SchedulerEngine.VPU_ROUND_ROBIN_IDX += 1
                                devices.append("VPU")
                                devices.append(target_device)
                            else:
                                return []
                        else:
                            devices.append("VPU")
                            devices.append(target_device)
                    else:
                        devices.append("VPU")
                        devices.append(target_device)
                else:
                    # Fallback if VPU present but none active in stats (should not happen normally)
                    devices.append("VPU")
                    devices.append("VPU_0")
            
            if self.hw.get("npu_present") and npu_required:
                devices.append("NPU")
            
            if devices:
                return devices
            else:
                logger.warning(f"NPU/VPU not available for {stage_name}. Escalating to CPU/AMX.")

        # 2. MTL-P Specific: Low-latency activation capture (Prefer P-cores)
        if stage_name in self.ACTIVATION_CAPTURE_STAGES:
            if self.tier == "LEGACY_AVX2" and self.hw.get("vpu_present"):
                # Increase VPU offloading for probing on legacy hardware
                devices.append("VPU")
                
            if self.hw.get("cpu_hybrid"):
                devices.append("CPU_P_CORE") # Worker will set affinity to P-cores
            if self.hw.get("cpu_amx"):
                devices.append("CPU_AMX")
            elif self.hw.get("cpu_avx512"):
                devices.append("CPU_AVX512")
            else:
                devices.append("CPU_VNNI" if self.hw.get("cpu_vnni") else "CPU")
            return devices

        # 3. MTL-P Specific: High-throughput ridge regression (Prefer NPU/iGPU)
        if stage_name in self.RIDGE_REGRESSION_STAGES:
            if self.hw.get("npu_present"):
                devices.append("NPU")
            if self.hw.get("igpu_type") == "xe-lpg":
                devices.append("iGPU") # Xe-LPG is excellent for ridge regression
            elif self.hw.get("igpu_present"):
                devices.append("iGPU")
            
            # Fallback to E-cores if hybrid
            if self.hw.get("cpu_hybrid"):
                devices.append("CPU_E_CORE")
                
            if devices:
                return devices

        # 4. Evaluate CPU Mandatory Authority Stages
        if stage_name in self.CPU_AMX_MANDATORY_STAGES:
            if self.hw.get("cpu_amx"):
                devices.append("CPU_AMX")
            elif self.hw.get("cpu_avx512"):
                devices.append("CPU_AVX512")
            elif self.hw.get("cpu_vnni"):
                devices.append("CPU_VNNI")
            else:
                devices.append("CPU")
            return devices

        # 3. Opportunistic iGPU Placement (for compliance or speed)
        if stage_name in self.IGPU_ELIGIBLE_STAGES:
            # On LEGACY_AVX2, prioritize VPU for probing
            if self.tier == "LEGACY_AVX2" and stage_name == "probe" and self.hw.get("vpu_present"):
                return ["VPU", "CPU"]

            # Prefer Virtual CUDA if ZLUDA is present for enhanced performance
            if self.hw.get("cuda_compat") and self.hw.get("igpu_present"):
                devices.append("CUDA") # Routes to ZLUDA-wrapped iGPU
                return devices

            if runtime_profile.get("igpu_required", False) and self.hw.get("igpu_present"):
                devices.append("iGPU")
                return devices


        # 6. Default execution (fallback to best available CPU)
        if self.hw.get("cpu_amx"):
            devices.append("CPU_AMX")
        elif self.hw.get("cpu_avx512"):
            devices.append("CPU_AVX512")
        elif self.hw.get("cpu_vnni"):
            devices.append("CPU_VNNI")
        else:
            devices.append("CPU")
            
        return devices
