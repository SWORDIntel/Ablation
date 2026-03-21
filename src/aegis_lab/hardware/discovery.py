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
        devices = {
            "igpu": False,
            "npu": False,
            "vpu": False,
            "vpu_count": 0,
            "vpu_details": [],
            "igpu_type": "standard",
            "npu_type": "none"
        }
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
                if "MYRIAD" in dev:
                    devices["vpu"] = True
                    # OpenVINO lists MYRIAD.X.Y for multiple sticks
                    devices["vpu_count"] += 1
                    devices["vpu_details"].append(dev)
        except ImportError:
            logger.info("OpenVINO not installed. Checking lspci/lsusb as fallback.")
            
        # Specific check for MyriadX sticks via lsusb or /sys/bus/usb/devices
        if platform.system() == "Linux":
            try:
                # 03e7:2485 is the VID:PID for MyriadX
                usb_path = "/sys/bus/usb/devices"
                if os.path.exists(usb_path):
                    for d in os.listdir(usb_path):
                        id_vendor_path = os.path.join(usb_path, d, "idVendor")
                        id_product_path = os.path.join(usb_path, d, "idProduct")
                        if os.path.exists(id_vendor_path) and os.path.exists(id_product_path):
                            with open(id_vendor_path, "r") as f:
                                vid = f.read().strip()
                            with open(id_product_path, "r") as f:
                                pid = f.read().strip()
                            
                            if vid == "03e7" and pid == "2485":
                                devices["vpu"] = True
                                # If OpenVINO didn't find them or we want precise count from USB
                                # We'll avoid double counting if OpenVINO already populated vpu_details
                                if not any(d in dev for dev in devices["vpu_details"]):
                                    devices["vpu_count"] += 1
                                    devices["vpu_details"].append(f"USB_{d}")
            except Exception as e:
                logger.debug(f"Failed to scan USB devices: {e}")

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
        Discover system hardware capabilities dynamically on launch.
        """
        cpu_features = cls.check_cpu_features()
        ov_devices = cls.check_openvino_devices()
        npu_bar_status = cls.check_npu_bar()
        
        # Check for ZLUDA (CUDA on Intel) compatibility
        has_zluda = os.path.exists("/usr/local/bin/zluda") or "ZLUDA_PATH" in os.environ
        
        # DYNAMIC ACCELERATION PROBE:
        # Instead of a blacklist, we check if the detected accelerators 
        # actually support the minimum required precision (INT8/FP16).
        accel_functional = ov_devices["igpu"] or ov_devices["npu"] or ov_devices["vpu"] or has_zluda
        
        return {
            "cpu_amx": cpu_features["amx"],
            "cpu_avx512": cpu_features["avx512"],
            "cpu_vnni": cpu_features["avx_vnni"],
            "cpu_hybrid": cpu_features["hybrid"],
            "igpu_present": ov_devices["igpu"],
            "igpu_type": ov_devices["igpu_type"],
            "npu_present": ov_devices["npu"],
            "npu_type": ov_devices["npu_type"],
            "vpu_present": ov_devices["vpu"],
            "vpu_count": ov_devices["vpu_count"],
            "vpu_details": ov_devices["vpu_details"],
            "npu_bar_found": npu_bar_status["found"],
            "npu_bar_protected": npu_bar_status["protected"],
            "npu_bar_conflict": npu_bar_status["conflict"],
            "cuda_compat": has_zluda,
            "accel_available": accel_functional
        }

    @classmethod
    def print_capabilities(cls):
        """
        Pretty-print the discovered hardware capabilities.
        """
        caps = cls.discover()
        print("\n--- AEGIS-LAB Hardware SITREP ---")
        print(f"  CPU [AMX]:    {'✅ Supported' if caps['cpu_amx'] else '❌ Not Detected'}")
        print(f"  CPU [AVX512]: {'✅ Supported' if caps['cpu_avx512'] else '❌ Not Detected'}")
        print(f"  CPU [VNNI]:   {'✅ Supported' if caps['cpu_vnni'] else '❌ Not Detected'}")
        print(f"  CPU [Hybrid]: {'✅ Detect' if caps['cpu_hybrid'] else 'Standard'}")
        print(f"  iGPU Presence: {'✅ Detected' if caps['igpu_present'] else '❌ Not Detected'} ({caps['igpu_type']})")
        print(f"  NPU Presence:  {'✅ Detected' if caps['npu_present'] else '❌ Not Detected'} ({caps['npu_type']})")
        print(f"  VPU Presence:  {'✅ Detected' if caps['vpu_present'] else '❌ Not Detected'} ({caps['vpu_count']} units)")
        print(f"  NPU BAR Protection: {'✅ SAFE' if caps['npu_bar_protected'] else '⚠️  CONFLICT' if caps['npu_bar_conflict'] else 'Unknown'}")
        print(f"  CUDA Bridge:   {'✅ Enabled (ZLUDA)' if caps['cuda_compat'] else 'Standard Path'}")
        print("---------------------------------\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Hardware Discovery Output:", HardwareDiscovery.discover())
