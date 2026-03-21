import platform
import os
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HardwareDiscovery:
    """
    Discovers hardware capabilities including CPU features (AMX/AVX512),
    iGPU presence, and NPU availability via OpenVINO or system queries.
    """

    @staticmethod
    def check_cpu_features() -> Dict[str, bool]:
        features = {"avx512": False, "amx": False, "avx_vnni": False, "hybrid": False}
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                    # Check for AVX-512 foundation
                    if "avx512f" in content:
                        features["avx512"] = True
                    # Check for AMX components (tile, int8, bf16)
                    if "amx_tile" in content or "amx_int8" in content or "amx_bf16" in content:
                        features["amx"] = True
                    # Check for AVX-VNNI (AVX2-based VNNI)
                    if "avx_vnni" in content or "avx512_vnni" in content:
                        features["avx_vnni"] = True
                
                # Check for hybrid architecture (P/E cores)
                cpu_dir = "/sys/devices/system/cpu"
                if os.path.exists(cpu_dir):
                    # Intel hybrid detection via sysfs core_type
                    core_types = []
                    for d in os.listdir(cpu_dir):
                        if d.startswith("cpu") and d[3:].isdigit():
                            core_type_path = os.path.join(cpu_dir, d, "topology/core_type")
                            if os.path.exists(core_type_path):
                                with open(core_type_path, "r") as f:
                                    core_types.append(f.read().strip())
                    
                    if len(set(core_types)) > 1:
                        features["hybrid"] = True
                    elif not core_types:
                        # Fallback for older kernels or non-sysfs detection
                        # Check if we have different L3 cache sharing or different max frequencies
                        pass

            except Exception as e:
                logger.warning(f"Failed to read CPU topology: {e}")
        return features

    @staticmethod
    def check_openvino_devices() -> Dict[str, Any]:
        devices = {"igpu": False, "npu": False, "igpu_type": "standard", "npu_type": "none"}
        try:
            from openvino.runtime import Core
            core = Core()
            available_devices = core.available_devices
            for dev in available_devices:
                if "GPU" in dev:
                    devices["igpu"] = True
                    try:
                        full_name = core.get_property(dev, "FULL_DEVICE_NAME").lower()
                        if "arc" in full_name or "xe-lpg" in full_name:
                            devices["igpu_type"] = "xe-lpg"
                    except:
                        pass
                if "NPU" in dev:
                    devices["npu"] = True
                    devices["npu_type"] = "intel_ai_boost"
        except ImportError:
            logger.info("OpenVINO not installed. Checking lspci as fallback.")
            if platform.system() == "Linux":
                try:
                    output = subprocess.check_output(["lspci"], text=True).lower()
                    if "npu" in output or "neural processing unit" in output or "7b40" in output: # 7b40 is MTL NPU
                        devices["npu"] = True
                        devices["npu_type"] = "intel_ai_boost"
                    if "vga" in output and "intel" in output:
                        devices["igpu"] = True
                        if "arc" in output or "meteor lake" in output:
                            devices["igpu_type"] = "xe-lpg"
                except Exception as e:
                    logger.debug(f"Failed to run lspci: {e}")
        return devices

    @staticmethod
    def check_npu_bar() -> Dict[str, Any]:
        """
        Check for 0x5010000000 (NPU BAR) and its status in /proc/iomem to avoid BIOS/E820 conflicts.
        """
        npu_bar = 0x5010000000
        result = {"found": False, "protected": False, "conflict": False}
        if os.path.exists("/proc/iomem"):
            try:
                with open("/proc/iomem", "r") as f:
                    for line in f:
                        if "-" in line and ":" in line:
                            range_part, name = line.split(":", 1)
                            try:
                                start_str, end_str = range_part.strip().split("-")
                                start_addr = int(start_str, 16)
                                end_addr = int(end_str, 16)
                                if start_addr <= npu_bar <= end_addr:
                                    result["found"] = True
                                    name = name.strip().lower()
                                    if "system ram" in name:
                                        result["conflict"] = True
                                    elif "reserved" in name or "npu" in name or "pci" in name:
                                        result["protected"] = True
                            except ValueError:
                                continue
            except Exception as e:
                logger.warning(f"Failed to read /proc/iomem: {e}")
        return result

    @classmethod
    def discover(cls) -> Dict[str, Any]:
        """
        Discover system hardware capabilities.

        Returns:
            Dict containing boolean flags for cpu_amx, cpu_avx512, cpu_vnni,
            cpu_hybrid, igpu_present, igpu_type, npu_present, npu_type,
            and npu_bar_protected.
        """
        cpu_features = cls.check_cpu_features()
        ov_devices = cls.check_openvino_devices()
        npu_bar_status = cls.check_npu_bar()
        
        return {
            "cpu_amx": cpu_features["amx"],
            "cpu_avx512": cpu_features["avx512"],
            "cpu_vnni": cpu_features["avx_vnni"],
            "cpu_hybrid": cpu_features["hybrid"],
            "igpu_present": ov_devices["igpu"],
            "igpu_type": ov_devices["igpu_type"],
            "npu_present": ov_devices["npu"],
            "npu_type": ov_devices["npu_type"],
            "npu_bar_found": npu_bar_status["found"],
            "npu_bar_protected": npu_bar_status["protected"],
            "npu_bar_conflict": npu_bar_status["conflict"]
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Hardware Discovery Output:", HardwareDiscovery.discover())
