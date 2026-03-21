import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SchedulerEngine:
    """
    Implements the core device placement logic per stage,
    enforcing the Aegis-Lab execution doctrine.
    """

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
        "sentinel_stage1_semantic"
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

    def determine_placement(self, stage_name: str, runtime_profile: Dict[str, Any]) -> List[str]:
        """
        Determines the ordered list of target devices for a DAG stage.
        """
        # Thermal Safety: Emergency shutdown of compute if critical
        if not self.thermal.get("safe_to_compute", True):
            logger.error("THERMAL CRITICAL: Suspending all compute operations.")
            return []

        # MTL-P Optimization: Prefer NPU/Xe-LPG for high throughput if available
        is_mtl_p = self.hw.get("igpu_type") == "xe-lpg" and self.hw.get("npu_present")
        
        # AVX-512/AMX/VNNI Fail-Safe: If no modern accelerators, shift all operations to Flex Fabric (NPU/iGPU)
        has_high_perf_cpu = self.hw.get("cpu_amx") or self.hw.get("cpu_avx512") or self.hw.get("cpu_vnni")
        
        # Thermal Throttling: If HIGH, avoid power-hungry CPU_AMX/iGPU, prefer NPU
        is_throttling = self.thermal.get("throttling_recommended", False)
        
        if not has_high_perf_cpu or is_throttling:
            import os
            reason = "Thermal Throttling" if is_throttling else "No modern CPU accelerators"
            logger.warning(f"{reason}. Fail-safe active: Shifting to Flex Fabric (NPU/iGPU).")
            # Configure NPU for 4x cache
            os.environ["QIHSE_HPU_CACHE_MB"] = "512"
            
            devices = []
            if self.hw.get("npu_present"):
                devices.append("NPU")
            
            # Avoid iGPU in thermal throttling if NPU is present
            if self.hw.get("igpu_present") and not is_throttling:
                devices.append("iGPU")
            
            if devices:
                return devices
            
            return ["CPU"]

        devices = []
        
        # 1. Evaluate NPU Sentinel Requirements
        if stage_name in self.SENTINEL_STAGES:
            npu_required = runtime_profile.get("npu_required_if_present", True)
            if self.hw.get("npu_present") and npu_required:
                devices.append("NPU")
                return devices
            else:
                logger.warning("NPU not available for sentinel stage. Escalating to CPU/AMX.")

        # 2. MTL-P Specific: Low-latency activation capture (Prefer P-cores)
        if stage_name in self.ACTIVATION_CAPTURE_STAGES:
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

        # 5. Opportunistic iGPU Placement (for compliance or speed)
        if stage_name in self.IGPU_ELIGIBLE_STAGES:
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
