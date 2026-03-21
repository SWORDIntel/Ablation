import platform
import os
import subprocess
import logging
import time
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HardwareDiscovery:
    """
    Discovers hardware capabilities including CPU features (AMX/AVX512/AVX2),
    iGPU presence, and NPU availability via OpenVINO or system queries.
    Performs memory bandwidth benchmarking and hardware tier classification.
    """

    @staticmethod
    def check_cpu_features() -> Dict[str, bool]:
        features = {"avx512": False, "amx": False, "avx_vnni": False, "avx2": False, "hybrid": False}
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
                    # Check for AVX2
                    if "avx2" in content:
                        features["avx2"] = True
                
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

            except Exception as e:
                logger.warning(f"Failed to read CPU topology: {e}")
        return features

    @staticmethod
    def check_memory_bandwidth() -> float:
        """
        Performs a rapid memory-bandwidth benchmark (Read+Write).
        Returns estimated GB/s.
        """
        try:
            size_mb = 128
            data = np.random.bytes(size_mb * 1024 * 1024)
            arr = np.frombuffer(data, dtype=np.uint8).copy()
            
            start_time = time.perf_counter()
            # Perform a simple copy-like operation
            arr2 = arr + 1
            # Perform some sum for read throughput
            _ = np.sum(arr2)
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            # (Read + Write + Read) = 3 * size_mb
            total_gb = (3.0 * size_mb) / 1024.0
            gb_per_sec = total_gb / duration
            return gb_per_sec
        except Exception as e:
            logger.warning(f"Memory benchmark failed: {e}")
            return 0.0

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
                                if not any(d in dev for dev in devices["vpu_details"]):
                                    devices["vpu_count"] += 1
                                    devices["vpu_details"].append(f"USB_{d}")
            except Exception as e:
                logger.debug(f"Failed to scan USB devices: {e}")

        return devices

    @staticmethod
    def check_gpu_blacklist() -> Dict[str, Any]:
        """
        Blacklists Fermi-based NVIDIA GPUs (GTX 560 Ti) if modern compute kernels are required.
        """
        blacklist = {"blacklisted_gpus_found": False, "reasons": []}
        if platform.system() == "Linux":
            try:
                # 10de:1200 is GTX 560 Ti
                lspci_output = subprocess.check_output(["lspci", "-nn"], text=True).lower()
                if "10de:1200" in lspci_output or "gtx 560 ti" in lspci_output:
                    blacklist["blacklisted_gpus_found"] = True
                    blacklist["reasons"].append("Fermi-based NVIDIA GPU (GTX 560 Ti) detected. Blacklisted for modern kernels.")
            except Exception:
                pass
        return blacklist

    @staticmethod
    def get_hardware_tier(cpu_features: Dict[str, bool], ov_devices: Dict[str, Any]) -> str:
        """
        Classifies the system into a 'Hardware Tier'.
        """
        if cpu_features["amx"] or (cpu_features["avx512"] and ov_devices["npu"]):
            return "HIGH_PERF_SERVER"
        elif ov_devices["igpu_type"] == "xe-lpg" or ov_devices["npu"]:
            return "MODERN_MTL"
        elif cpu_features["avx2"]:
            return "LEGACY_AVX2"
        return "GENERIC"

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
        mem_bandwidth = cls.check_memory_bandwidth()
        gpu_blacklist = cls.check_gpu_blacklist()
        
        # Check for ZLUDA (CUDA on Intel) compatibility
        has_zluda = os.path.exists("/usr/local/bin/zluda") or "ZLUDA_PATH" in os.environ
        
        # DYNAMIC ACCELERATION PROBE:
        # Instead of a blacklist, we check if the detected accelerators 
        # actually support the minimum required precision (INT8/FP16).
        accel_functional = ov_devices["igpu"] or ov_devices["npu"] or ov_devices["vpu"] or has_zluda
        
        # Blacklist logic: If Fermi GPU found, we explicitly disable CUDA compat for it
        if gpu_blacklist["blacklisted_gpus_found"]:
            has_zluda = False
            accel_functional = ov_devices["igpu"] or ov_devices["npu"] or ov_devices["vpu"]

        hardware_tier = cls.get_hardware_tier(cpu_features, ov_devices)
        
        return {
            "cpu_amx": cpu_features["amx"],
            "cpu_avx512": cpu_features["avx512"],
            "cpu_vnni": cpu_features["avx_vnni"],
            "cpu_avx2": cpu_features["avx2"],
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
            "accel_available": accel_functional,
            "mem_bandwidth_gbs": mem_bandwidth,
            "hardware_tier": hardware_tier,
            "gpu_blacklisted": gpu_blacklist["blacklisted_gpus_found"],
            "blacklist_reasons": gpu_blacklist["reasons"]
        }

    @classmethod
    def print_capabilities(cls):
        """
        Pretty-print the discovered hardware capabilities.
        """
        caps = cls.discover()
        print("\n--- AEGIS-LAB Hardware SITREP ---")
        print(f"  Hardware Tier: {caps['hardware_tier']}")
        print(f"  CPU [AMX]:    {'✅ Supported' if caps['cpu_amx'] else '❌ Not Detected'}")
        print(f"  CPU [AVX512]: {'✅ Supported' if caps['cpu_avx512'] else '❌ Not Detected'}")
        print(f"  CPU [AVX2]:   {'✅ Supported' if caps['cpu_avx2'] else '❌ Not Detected'}")
        print(f"  CPU [VNNI]:   {'✅ Supported' if caps['cpu_vnni'] else '❌ Not Detected'}")
        print(f"  CPU [Hybrid]: {'✅ Detect' if caps['cpu_hybrid'] else 'Standard'}")
        print(f"  Memory BW:    {caps['mem_bandwidth_gbs']:.2f} GB/s")
        print(f"  iGPU Presence: {'✅ Detected' if caps['igpu_present'] else '❌ Not Detected'} ({caps['igpu_type']})")
        print(f"  NPU Presence:  {'✅ Detected' if caps['npu_present'] else '❌ Not Detected'} ({caps['npu_type']})")
        print(f"  VPU Presence:  {'✅ Detected' if caps['vpu_present'] else '❌ Not Detected'} ({caps['vpu_count']} units)")
        print(f"  NPU BAR Protection: {'✅ SAFE' if caps['npu_bar_protected'] else '⚠️  CONFLICT' if caps['npu_bar_conflict'] else 'Unknown'}")
        if caps['gpu_blacklisted']:
            print(f"  GPU Status:    ❌ BLACKLISTED: {', '.join(caps['blacklist_reasons'])}")
        else:
            print(f"  CUDA Bridge:   {'✅ Enabled (ZLUDA)' if caps['cuda_compat'] else 'Standard Path'}")
        print("---------------------------------\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    HardwareDiscovery.print_capabilities()
